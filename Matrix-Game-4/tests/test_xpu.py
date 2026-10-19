"""XPU admission and precision/cache checks; device cases require explicit opt-in."""
import argparse
import os
import unittest
from unittest.mock import patch

import torch

from matrix_game.scenes import add_model_arguments, resolve_model_arguments


class XpuArgumentsTests(unittest.TestCase):
    def test_metadata_policy_preserves_other_devices(self):
        from matrix_game.inference.devices import prepare_kv_metadata
        index = torch.tensor([13], dtype=torch.int64)
        cache = dict(global_end_index=index, local_end_index=index)
        with patch.dict(os.environ, MATRIX_XPU_HOST_KV_METADATA="invalid"):
            for device in ("cpu", "cuda"):
                prepare_kv_metadata([cache], device)
                self.assertIs(cache["global_end_index"], index)

    def resolve(self, *arguments):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        return resolve_model_arguments(parser.parse_args(arguments), require_model=False)

    def test_cuda_defaults_retained(self):
        args = self.resolve()
        self.assertEqual((args.device, args.dit_runtime, args.t5_runtime), ("cuda", "ring-fused", "graph"))
        self.assertTrue(args.vae_graph)

    def test_xpu_keeps_real_time_model_contract(self):
        for arguments in (("--preset", "xpu"), ("--device", "xpu")):
            args = self.resolve(*arguments)
            self.assertEqual((args.device, args.dit_runtime, args.t5_runtime), ("xpu", "native", "native"))
            self.assertEqual((args.vae_backend, args.steps, args.first_steps), ("taehv", (1000,), (1000,)))
            self.assertFalse(args.vae_graph)

    def test_cuda_only_runtime_is_rejected_on_xpu(self):
        for extra in (("--vae-graph",), ("--t5-runtime", "graph"), ("--dit-runtime", "ring-fused"),
                      ("--dit-runtime", "rtx5070ti")):
            with self.assertRaisesRegex(ValueError, "XPU currently requires"):
                self.resolve("--preset", "xpu", *extra)


@unittest.skipUnless(os.environ.get("MATRIX_RUN_XPU_TESTS") == "1", "requires explicit XPU test admission")
class XpuDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.xpu.is_available() or torch.xpu.device_count() != 1:
            raise RuntimeError("XPU device tests require exactly one affinity-visible device")
        torch.set_num_threads(8)

    def test_host_metadata_preserves_int64_values_and_kv_storage(self):
        from matrix_game.inference.devices import prepare_kv_metadata
        keys = torch.ones(1, 2, 24, 128, dtype=torch.bfloat16, device="xpu")
        cache = dict(k=keys, v=keys.clone(), global_end_index=torch.tensor([2**40 + 405], device="xpu"),
                     local_end_index=torch.tensor([8505], device="xpu"))
        with patch.dict(os.environ, MATRIX_XPU_HOST_KV_METADATA="1"):
            prepare_kv_metadata([cache], torch.device("xpu"))
        self.assertIs(cache["k"], keys)
        self.assertEqual(cache["v"].device.type, "xpu")
        for name, value in (("global_end_index", 2**40 + 405), ("local_end_index", 8505)):
            self.assertEqual(cache[name].device.type, "cpu")
            self.assertEqual(cache[name].dtype, torch.int64)
            self.assertEqual(cache[name].tolist(), [value])

    def test_metadata_opt_out_and_invalid_setting(self):
        from matrix_game.inference.devices import prepare_kv_metadata
        index = torch.tensor([405], device="xpu")
        cache = dict(global_end_index=index, local_end_index=index)
        with patch.dict(os.environ, MATRIX_XPU_HOST_KV_METADATA="0"):
            prepare_kv_metadata([cache], "xpu")
        self.assertIs(cache["global_end_index"], index)
        with patch.dict(os.environ, MATRIX_XPU_HOST_KV_METADATA="invalid"):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                prepare_kv_metadata([cache], "xpu")

    def test_rotary_preserves_cpu_fp64_reference(self):
        from matrix_game.inference.wan.modules.model import rope_params
        from matrix_game.inference.wan.modules.causal_model import causal_rope_apply, rope_apply_frames
        generator = torch.Generator().manual_seed(731)
        x = torch.randn(1, 405, 24, 128, generator=generator).to(torch.bfloat16)
        freqs = torch.cat([rope_params(8192, 44), rope_params(8192, 42), rope_params(8192, 42)], dim=1)
        grid = torch.tensor([[1, 15, 27]])
        for position in (0, 21, 8191):
            with self.subTest(position=position):
                expected = causal_rope_apply(x, grid, freqs, position)
                actual = causal_rope_apply(x.to("xpu"), grid, freqs.to("xpu"), position).cpu()
                self.assertTrue(torch.equal(actual, expected))
        expected = rope_apply_frames(x, grid, freqs, torch.tensor([-3]))
        actual = rope_apply_frames(x.to("xpu"), grid, freqs.to("xpu"), torch.tensor([-3])).cpu()
        self.assertTrue(torch.equal(actual, expected))

    def test_rectangular_sdpa_matches_float_reference(self):
        from matrix_game.inference.wan.modules.attention import attention
        generator = torch.Generator().manual_seed(732)
        for key_length in (405, 512, 8505):
            with self.subTest(key_length=key_length):
                q = torch.randn(1, 405, 24, 128, generator=generator).to(torch.bfloat16)
                k = torch.randn(1, key_length, 24, 128, generator=generator).to(torch.bfloat16)
                v = torch.randn(k.shape, generator=generator).to(torch.bfloat16)
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)
                ).transpose(1, 2)
                actual = attention(q.to("xpu"), k.to("xpu"), v.to("xpu")).cpu().float()
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.02)

    def test_scheduler_retains_fp64_arithmetic(self):
        from matrix_game.inference.scheduler import FlowMatchScheduler
        from matrix_game.inference.wrappers import WanDiffusionWrapper
        from types import SimpleNamespace
        scheduler = FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)
        scheduler.set_timesteps(1000, training=True)
        owner = SimpleNamespace(scheduler=scheduler)
        generator = torch.Generator().manual_seed(733)
        flow = torch.randn(1, 48, 30, 54, generator=generator).to(torch.bfloat16)
        latent = torch.randn(flow.shape, generator=generator).to(torch.bfloat16)
        convert = WanDiffusionWrapper._convert_flow_pred_to_x0
        for step in (0., 625., 937.5, 1000.):
            t = torch.tensor([step])
            expected = convert(owner, flow, latent, t)
            actual = convert(owner, flow.to("xpu"), latent.to("xpu"), t.to("xpu")).cpu()
            self.assertTrue(torch.equal(actual, expected))
        scheduler.sigmas = scheduler.sigmas.to("xpu")
        scheduler.timesteps = scheduler.timesteps.to("xpu")
        actual = convert(owner, flow.to("xpu"), latent.to("xpu"), torch.tensor([937.5], device="xpu"))
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
