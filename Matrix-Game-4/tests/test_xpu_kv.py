"""Fixed-sink storage ownership, fallbacks, and opt-in XPU numerical checks."""
import os
import unittest
from unittest.mock import patch

import torch

from matrix_game.inference.xpu import kv as backend


class SinkOverlayTests(unittest.TestCase):
    def inputs(self, device="cpu", length=6, sink=2, heads=2, dim=4):
        shape = (1, length, heads, dim)
        cache = {"k": torch.randn(shape, device=device),
                 "v": torch.randn(shape, device=device),
                 "local_end_index": torch.tensor([length]),
                 "global_end_index": torch.tensor([length + 17])}
        query = torch.randn((1, 2, heads, dim), device=device)
        k_sink = torch.randn((1, sink, heads, dim), device=device)
        return query, k_sink, cache

    def test_cpu_is_not_admitted(self):
        query, sink, cache = self.inputs()
        with torch.no_grad():
            self.assertIsNone(backend.try_sink_attention(
                lambda *args: self.fail("reader called"), query, sink, cache,
                2, 2, 6, simple=True))

    def test_borrowed_views_and_all_cache_fields_restored(self):
        query, sink, cache = self.inputs()
        original = {k: v.clone() for k, v in cache.items()}
        def reader(q, k, v):
            self.assertEqual(k.data_ptr(), cache["k"].data_ptr())
            self.assertEqual(v.data_ptr(), cache["v"].data_ptr())
            self.assertTrue(torch.equal(k[:, :2], sink))
            self.assertTrue(torch.equal(k[:, 2:], original["k"][:, 2:]))
            return q.clone()
        # Exercise the actual ownership algorithm on CPU; real XPU admission
        # and attention are covered below, rather than inferred from this mock.
        with patch.object(backend, "_can_overlay", return_value=True), torch.no_grad():
            actual = backend.try_sink_attention(reader, query, sink, cache, 2, 2, 6, simple=True)
        self.assertTrue(torch.equal(actual, query))
        for name, value in original.items():
            self.assertTrue(torch.equal(cache[name], value), name)

    def test_reader_exception_restores_sink(self):
        query, sink, cache = self.inputs()
        original = cache["k"].clone()
        def fail(*args):
            raise RuntimeError("reader failed")
        with patch.object(backend, "_can_overlay", return_value=True), torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "reader failed"):
                backend.try_sink_attention(fail, query, sink, cache, 2, 2, 6, simple=True)
        self.assertTrue(torch.equal(cache["k"], original))

    def test_opt_out_and_invalid_setting(self):
        query, sink, cache = self.inputs()
        for setting in ("0", "invalid"):
            with patch.dict(os.environ, MATRIX_XPU_KV_SINK_OVERLAY=setting):
                if setting == "0":
                    self.assertIsNone(backend.try_sink_attention(None, query, sink, cache, 2, 2, 6, simple=True))
                else:
                    with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                        backend.try_sink_attention(None, query, sink, cache, 2, 2, 6, simple=True)


@unittest.skipUnless(os.environ.get("MATRIX_RUN_XPU_TESTS") == "1", "requires explicit XPU test admission")
class SinkOverlayDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.xpu.is_available() or torch.xpu.device_count() != 1:
            raise RuntimeError("KV device tests require one affinity-visible XPU")
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        cls.addClassCleanup(torch.use_deterministic_algorithms, previous)

    def inputs(self):
        torch.manual_seed(734)
        query = torch.randn(1, 405, 24, 128, device="xpu", dtype=torch.bfloat16)
        cache = {name: torch.randn(1, 8505, 24, 128, device="xpu", dtype=torch.bfloat16)
                 for name in ("k", "v")}
        sink = torch.randn_like(cache["k"][:, :405])
        return query, sink, cache

    def test_actual_attention_and_final_storage_are_bitwise(self):
        from matrix_game.inference.wan.modules.attention import attention
        query, sink, cache = self.inputs()
        original = {k: v.clone() for k, v in cache.items()}
        with torch.no_grad():
            expected = attention(query, torch.cat([sink, cache["k"][:, 405:]], 1), cache["v"].clone())
            actual = backend.try_sink_attention(attention, query, sink, cache, 405, 405, 8505, simple=True)
        self.assertIsNotNone(actual)
        self.assertTrue(torch.equal(actual, expected))
        for k, v in original.items():
            self.assertTrue(torch.equal(cache[k], v))

    def test_admission_rejects_other_cache_contracts(self):
        query, sink, cache = self.inputs()
        with torch.no_grad():
            self.assertTrue(backend._can_overlay(sink, cache, 405, 405, 8505, True))
            for sink_tokens, recent, end, simple in [(405, 405, 8100, True),
                    (405, 810, 8505, True), (0, 0, 8505, True), (405, 405, 8505, False)]:
                self.assertFalse(backend._can_overlay(sink, cache, sink_tokens, recent, end, simple))
            noncontiguous = {k: v.transpose(2, 3) for k, v in cache.items()}
            self.assertFalse(backend._can_overlay(sink.transpose(2, 3), noncontiguous, 405, 405, 8505, True))
            mixed = {"k": cache["k"], "v": cache["v"].half()}
            self.assertFalse(backend._can_overlay(sink, mixed, 405, 405, 8505, True))
        with torch.enable_grad():
            self.assertFalse(backend._can_overlay(sink, cache, 405, 405, 8505, True))

    def test_failed_device_reader_preserves_cache(self):
        query, sink, cache = self.inputs()
        saved = cache["k"].clone()
        def fail(*args):
            raise RuntimeError("intentional device reader failure")
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, "intentional device reader failure"):
            backend.try_sink_attention(fail, query, sink, cache, 405, 405, 8505, simple=True)
        self.assertTrue(torch.equal(saved, cache["k"]))


if __name__ == "__main__":
    unittest.main()
