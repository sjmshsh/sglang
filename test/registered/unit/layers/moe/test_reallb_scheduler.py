"""Unit tests for ReaLB scheduling helpers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

import types
import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.layers.moe.reallb import (
    build_vision_token_mask,
    compute_rank_loads,
    maybe_build_reallb_plan,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase


class _FakeMmItem:
    def __init__(self, offsets, *, image=True, video=False):
        self.offsets = offsets
        self._image = image
        self._video = video

    def is_image(self):
        return self._image

    def is_video(self):
        return self._video


class TestReaLBScheduler(CustomTestCase):
    def tearDown(self):
        set_global_server_args_for_scheduler(None)

    def test_compute_rank_loads_counts_vision_per_routed_rank(self):
        topk_ids = torch.tensor(
            [
                [0, 2],
                [3, 6],
                [4, 7],
                [8, 1],  # 8 is outside routed expert range and is ignored.
            ],
            dtype=torch.int64,
        )
        vision_mask = torch.tensor([True, True, False, True])

        load, vision_load = compute_rank_loads(
            topk_ids=topk_ids,
            vision_token_mask=vision_mask,
            num_physical_routed_experts=8,
            ep_size=4,
        )

        self.assertEqual(load.tolist(), [2, 2, 1, 2])
        self.assertEqual(vision_load.tolist(), [2, 2, 0, 1])

    def test_plan_selects_hot_and_vision_heavy_ranks(self):
        server_args = types.SimpleNamespace(
            enable_reallb=True,
            reallb_capacity_factor=1.0,
            reallb_modality_threshold=0.7,
            reallb_global_batch_threshold=0,
        )
        set_global_server_args_for_scheduler(server_args)

        topk_ids = torch.tensor(
            [[0, 2], [3, 6], [4, 7], [8, 1]],
            dtype=torch.int64,
        )
        vision_mask = torch.tensor([True, True, False, True])

        with get_parallel().override(
            moe_ep_size=4,
            moe_ep_rank=1,
            moe_ep_group=None,
        ):
            plan = maybe_build_reallb_plan(
                layer_id=17,
                topk_ids=topk_ids,
                num_physical_routed_experts=8,
                hidden_states_num_tokens=4,
                vision_token_mask=vision_mask,
                force_all_tokens_vision=False,
            )

        self.assertTrue(plan.activated)
        self.assertEqual(plan.hot_rank_mask.tolist(), [True, True, False, True])
        self.assertEqual(
            plan.vision_heavy_rank_mask.tolist(), [True, True, False, False]
        )
        self.assertEqual(
            plan.low_precision_rank_mask.tolist(), [True, True, False, False]
        )
        self.assertTrue(plan.current_rank_low_precision)

    def test_plan_respects_global_batch_threshold(self):
        server_args = types.SimpleNamespace(
            enable_reallb=True,
            reallb_capacity_factor=1.0,
            reallb_modality_threshold=0.7,
            reallb_global_batch_threshold=100,
        )
        set_global_server_args_for_scheduler(server_args)

        with get_parallel().override(
            moe_ep_size=4,
            moe_ep_rank=0,
            moe_ep_group=None,
        ):
            plan = maybe_build_reallb_plan(
                layer_id=0,
                topk_ids=torch.tensor([[0], [1]], dtype=torch.int64),
                num_physical_routed_experts=8,
                hidden_states_num_tokens=2,
                vision_token_mask=torch.tensor([True, True]),
                force_all_tokens_vision=False,
            )

        self.assertFalse(plan.activated)
        self.assertEqual(plan.reason, "below-global-batch-threshold")

    def test_build_vision_token_mask_from_multimodal_offsets(self):
        batch = types.SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            prefix_lens=[1, 5],
            extend_lens=[4, 2],
            multimodal_inputs=[
                types.SimpleNamespace(
                    mm_items=[_FakeMmItem(offsets=[(2, 4)], image=True)]
                ),
                types.SimpleNamespace(
                    mm_items=[_FakeMmItem(offsets=[(5, 6)], image=False, video=False)]
                ),
            ],
        )

        mask = build_vision_token_mask(batch, device=torch.device("cpu"))

        self.assertEqual(mask.tolist(), [False, True, True, True, False, False])


if __name__ == "__main__":
    unittest.main()
