# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import Withable

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@dataclass
class ReaLBPlan:
    """Per-layer ReaLB scheduling decision.

    ReaLB operates at EP-rank granularity.  The plan marks ranks that are both
    hot (above the capacity-factor load target) and vision-heavy.  A downstream
    mixed-precision MoE implementation can use ``low_precision_rank_mask`` to
    select an FP4/NVFP4 expert path for the current EP rank.
    """

    layer_id: int
    activated: bool
    reason: str
    rank: int
    global_num_tokens: torch.Tensor
    load_per_rank: torch.Tensor
    vision_load_per_rank: torch.Tensor
    vision_ratio_per_rank: torch.Tensor
    hot_rank_mask: torch.Tensor
    vision_heavy_rank_mask: torch.Tensor
    low_precision_rank_mask: torch.Tensor

    @property
    def current_rank_low_precision(self) -> bool:
        if self.low_precision_rank_mask.numel() == 0:
            return False
        return bool(self.low_precision_rank_mask[self.rank].item())


_current_forward_batch: Withable["ForwardBatch"] = Withable()


@contextmanager
def reallb_forward_context(forward_batch: "ForwardBatch"):
    """Expose the current ForwardBatch to MoE layers without changing model APIs."""

    with _current_forward_batch.with_value(forward_batch):
        yield


def get_current_reallb_forward_batch() -> Optional["ForwardBatch"]:
    return _current_forward_batch.value


def build_vision_token_mask(
    batch: "ScheduleBatch",
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build a flattened token mask for image/video tokens in the current forward.

    Multimodal item offsets are absolute prompt positions and inclusive
    ``[start, end]`` spans.  For extend/chunked-prefill batches, the flattened
    forward tokens are ordered by request, from each request's prefix length to
    prefix length + extend length.  Decode batches return ``None`` because their
    next tokens are text continuations for ReaLB's purpose.
    """

    if batch.forward_mode.is_decode_or_idle():
        return None
    if not batch.multimodal_inputs or batch.extend_lens is None:
        return None
    if not isinstance(batch.extend_lens, list) or not isinstance(batch.prefix_lens, list):
        return None

    mask: list[bool] = []
    any_vision = False
    for req_index, extend_len in enumerate(batch.extend_lens):
        prefix_len = int(batch.prefix_lens[req_index])
        mm_input = (
            batch.multimodal_inputs[req_index]
            if req_index < len(batch.multimodal_inputs)
            else None
        )

        vision_spans: list[tuple[int, int]] = []
        if mm_input is not None:
            for item in mm_input.mm_items:
                if item is None or not item.offsets:
                    continue
                if not (item.is_image() or item.is_video()):
                    continue
                for start, end in item.offsets:
                    vision_spans.append((int(start), int(end)))

        for pos in range(prefix_len, prefix_len + int(extend_len)):
            is_vision = any(start <= pos <= end for start, end in vision_spans)
            mask.append(is_vision)
            any_vision = any_vision or is_vision

    if not mask or not any_vision:
        return None

    return torch.tensor(mask, dtype=torch.bool, device=device)


def maybe_build_reallb_plan(
    *,
    layer_id: int,
    topk_ids: Optional[torch.Tensor],
    num_physical_routed_experts: int,
    hidden_states_num_tokens: int,
    vision_token_mask: Optional[torch.Tensor],
    force_all_tokens_vision: bool,
) -> Optional[ReaLBPlan]:
    """Create the ReaLB plan for a MoE layer if the feature is enabled."""

    from sglang.srt.server_args import get_global_server_args

    server_args = get_global_server_args()
    if not server_args.enable_reallb:
        return None

    parallel = get_parallel()
    ep_size = int(parallel.moe_ep_size)
    ep_rank = int(parallel.moe_ep_rank)
    if ep_size <= 1:
        return _empty_plan(
            layer_id,
            ep_rank,
            "ep_size<=1",
            device=topk_ids.device if topk_ids is not None else torch.device("cpu"),
            ep_size=ep_size,
        )
    if topk_ids is None:
        return _empty_plan(
            layer_id,
            ep_rank,
            "topk-output-not-materialized",
            device=torch.device("cpu"),
            ep_size=ep_size,
        )
    if num_physical_routed_experts <= 0 or num_physical_routed_experts % ep_size != 0:
        return _empty_plan(
            layer_id,
            ep_rank,
            "invalid-physical-expert-layout",
            device=topk_ids.device,
            ep_size=ep_size,
        )

    global_num_tokens = _all_reduce_ep(
        torch.tensor([hidden_states_num_tokens], dtype=torch.int64, device=topk_ids.device)
    )
    if int(global_num_tokens.item()) < server_args.reallb_global_batch_threshold:
        return _empty_plan(
            layer_id,
            ep_rank,
            "below-global-batch-threshold",
            device=topk_ids.device,
            global_num_tokens=global_num_tokens,
            ep_size=ep_size,
        )

    if force_all_tokens_vision:
        vision_token_mask = torch.ones(
            topk_ids.shape[0], dtype=torch.bool, device=topk_ids.device
        )
    elif vision_token_mask is not None:
        vision_token_mask = vision_token_mask.to(device=topk_ids.device, non_blocking=True)
        if vision_token_mask.numel() != topk_ids.shape[0]:
            vision_token_mask = None

    load_per_rank, vision_load_per_rank = compute_rank_loads(
        topk_ids=topk_ids,
        vision_token_mask=vision_token_mask,
        num_physical_routed_experts=num_physical_routed_experts,
        ep_size=ep_size,
    )
    load_per_rank = _all_reduce_ep(load_per_rank)
    vision_load_per_rank = _all_reduce_ep(vision_load_per_rank)

    load_float = load_per_rank.float()
    ideal_load = load_float.mean()
    hot_rank_mask = load_float > ideal_load * float(server_args.reallb_capacity_factor)

    vision_ratio_per_rank = torch.where(
        load_per_rank > 0,
        vision_load_per_rank.float() / load_float.clamp_min(1.0),
        torch.zeros_like(load_float),
    )
    vision_heavy_rank_mask = (
        vision_ratio_per_rank > float(server_args.reallb_modality_threshold)
    )
    low_precision_rank_mask = hot_rank_mask & vision_heavy_rank_mask

    return ReaLBPlan(
        layer_id=layer_id,
        activated=bool(low_precision_rank_mask.any().item()),
        reason="activated" if bool(low_precision_rank_mask.any().item()) else "no-rank-selected",
        rank=ep_rank,
        global_num_tokens=global_num_tokens,
        load_per_rank=load_per_rank,
        vision_load_per_rank=vision_load_per_rank,
        vision_ratio_per_rank=vision_ratio_per_rank,
        hot_rank_mask=hot_rank_mask,
        vision_heavy_rank_mask=vision_heavy_rank_mask,
        low_precision_rank_mask=low_precision_rank_mask,
    )


def compute_rank_loads(
    *,
    topk_ids: torch.Tensor,
    vision_token_mask: Optional[torch.Tensor],
    num_physical_routed_experts: int,
    ep_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count routed expert-token load and vision load per EP rank."""

    if topk_ids.numel() == 0:
        zeros = torch.zeros(ep_size, dtype=torch.int64, device=topk_ids.device)
        return zeros, zeros.clone()

    num_local_routed = num_physical_routed_experts // ep_size
    valid = (topk_ids >= 0) & (topk_ids < num_physical_routed_experts)
    rank_ids = torch.div(
        topk_ids.masked_fill(~valid, 0).long(),
        num_local_routed,
        rounding_mode="floor",
    )

    load_per_rank = torch.zeros(ep_size, dtype=torch.int64, device=topk_ids.device)
    load_per_rank.scatter_add_(
        0,
        rank_ids.flatten(),
        valid.to(torch.int64).flatten(),
    )

    vision_load_per_rank = torch.zeros_like(load_per_rank)
    if vision_token_mask is None:
        return load_per_rank, vision_load_per_rank

    vision_token_mask = vision_token_mask.to(device=topk_ids.device, dtype=torch.bool)
    if vision_token_mask.numel() != topk_ids.shape[0]:
        return load_per_rank, vision_load_per_rank

    vision_valid = valid & vision_token_mask.view(-1, 1)
    vision_load_per_rank.scatter_add_(
        0,
        rank_ids.flatten(),
        vision_valid.to(torch.int64).flatten(),
    )
    return load_per_rank, vision_load_per_rank


def _all_reduce_ep(tensor: torch.Tensor) -> torch.Tensor:
    parallel = get_parallel()
    if int(parallel.moe_ep_size) <= 1:
        return tensor

    ep_group = parallel.moe_ep_group
    if ep_group is not None and hasattr(ep_group, "all_reduce"):
        reduced = ep_group.all_reduce(tensor)
        return reduced if reduced is not None else tensor

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        reduced = tensor.clone()
        torch.distributed.all_reduce(reduced)
        return reduced

    return tensor


def _empty_plan(
    layer_id: int,
    rank: int,
    reason: str,
    *,
    device: torch.device,
    global_num_tokens: Optional[torch.Tensor] = None,
    ep_size: Optional[int] = None,
) -> ReaLBPlan:
    ep_size = ep_size or max(1, int(get_parallel().moe_ep_size))
    zeros = torch.zeros(ep_size, dtype=torch.int64, device=device)
    bools = torch.zeros(ep_size, dtype=torch.bool, device=device)
    ratios = torch.zeros(ep_size, dtype=torch.float32, device=device)
    if global_num_tokens is None:
        global_num_tokens = torch.zeros(1, dtype=torch.int64, device=device)
    return ReaLBPlan(
        layer_id=layer_id,
        activated=False,
        reason=reason,
        rank=rank,
        global_num_tokens=global_num_tokens,
        load_per_rank=zeros,
        vision_load_per_rank=zeros.clone(),
        vision_ratio_per_rank=ratios,
        hot_rank_mask=bools,
        vision_heavy_rank_mask=bools.clone(),
        low_precision_rank_mask=bools.clone(),
    )
