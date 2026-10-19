"""Optional TLA admission, real dispatch, and preserved native fallbacks."""
import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from matrix_game.inference.xpu import attention as backend
from matrix_game.inference.wan.modules.attention import attention


class AttentionPolicyTests(unittest.TestCase):
    def inputs(self, length=405, heads=2):
        return [torch.zeros(1, n, heads, 128, dtype=torch.bfloat16) for n in (length, 21*length, 21*length)]

    def test_structural_window_contract_and_unsupported_semantics(self):
        q, k, v = self.inputs()
        self.assertTrue(backend._supports_layout(q, k, v))
        for options in ({"causal": True}, {"dropout_p": .1}, {"softmax_scale": .5},
                        {"q_scale": 2}, {"q_lens": torch.tensor([405])}, {"window_size": (5, 5)}, {"window_size": None}):
            self.assertFalse(backend._supports_layout(q, k, v, **options))
        self.assertFalse(backend._supports_layout(q, k[:, :512], v[:, :512]))
        self.assertFalse(backend._supports_layout(q.float(), k.float(), v.float()))
        self.assertFalse(backend._supports_layout(q, k, v.transpose(1, 2)))
        self.assertFalse(backend._supports_layout(q.requires_grad_(), k, v))
        for n in (257, 512):
            self.assertTrue(backend._supports_layout(*self.inputs(n)))
        for n in (256, 513):
            self.assertFalse(backend._supports_layout(*self.inputs(n)))

    def test_cpu_route_does_not_load_xpu_library(self):
        q = torch.randn(1, 5, 2, 8)
        with patch.dict(os.environ, MATRIX_XPU_ATTENTION="tla", MATRIX_XPU_TLA_LIBRARY="/missing.so"):
            with patch.object(backend, "_load_library", side_effect=AssertionError("CPU must not load TLA")):
                actual = attention(q, q, q, causal=True, softmax_scale=.2)
        expected = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),q.transpose(1,2),q.transpose(1,2),is_causal=True,scale=.2).transpose(1,2).contiguous()
        self.assertTrue(torch.equal(actual, expected))

    def test_missing_library_auto_warns_and_strict_raises(self):
        path="/nonexistent/matrix-tla-policy-test.so"
        backend._resolve_library.cache_clear()
        with self.assertWarnsRegex(RuntimeWarning, "using Torch SDPA"):
            self.assertIsNone(backend._resolve_library(path, False))
        with self.assertRaisesRegex(RuntimeError, "Cannot load Matrix4 TLA"):
            backend._resolve_library(path, True)
        backend._resolve_library.cache_clear()

    def test_incompatible_manifest_rejected_before_native_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            library=Path(directory)/'wrong.so';library.write_bytes(b'not a library')
            metadata={'contract':backend.CONTRACT,'torch':str(torch.__version__),
                      'torch_git':torch.version.git_version,'cxx11_abi':not bool(torch.compiled_with_cxx11_abi())}
            library.with_suffix('.so.json').write_text(json.dumps(metadata))
            with patch.object(torch.ops,'load_library',side_effect=AssertionError('must reject before loading')):
                with self.assertRaisesRegex(RuntimeError,'cxx11_abi mismatch'):
                    backend._load_library(str(library))


@unittest.skipUnless(os.environ.get("MATRIX_RUN_XPU_TESTS")=="1" and os.environ.get("MATRIX_XPU_TLA_LIBRARY"), "requires admitted XPU and built TLA library")
class AttentionDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.xpu.device_count()!=1 or torch.xpu.get_device_properties(0).device_id!=0xE223:
            raise RuntimeError("TLA tests require one affinity-visible B70")
        torch.set_num_threads(8)

    def test_integrated_kernel_and_native_fallbacks(self):
        g=torch.Generator().manual_seed(734)
        with torch.no_grad(), patch.dict(os.environ, MATRIX_XPU_ATTENTION="tla"):
            for length,heads in ((257,2),(405,24),(512,2)):
                cpu=[torch.randn(1,n,heads,128,generator=g).to(torch.bfloat16) for n in (length,21*length,21*length)]
                ts=[t.to('xpu') for t in cpu]
                before=backend.backend_status()['tla_calls']
                actual=attention(*ts)
                repeat=attention(*ts)
                expected=torch.nn.functional.scaled_dot_product_attention(*[t.float().transpose(1,2) for t in cpu]).transpose(1,2)
                self.assertEqual(backend.backend_status()['tla_calls']-before,2)
                self.assertTrue(torch.equal(actual,repeat))
                torch.testing.assert_close(actual.cpu().float(),expected,atol=.008,rtol=.02)
            q=ts[0];short=[q,ts[1][:,:512],ts[2][:,:512]]
            before=backend.backend_status()['tla_calls']
            with patch.dict(os.environ, MATRIX_XPU_ATTENTION="sdpa"):expected=attention(*short)
            self.assertTrue(torch.equal(attention(*short),expected))
            # A supported shape with causal/custom scale and a strided input must
            # retain the complete native API semantics, without entering TLA.
            for opts in ({'causal':True},{'softmax_scale':.2}):
                with patch.dict(os.environ, MATRIX_XPU_ATTENTION="sdpa"):expected=attention(*ts,**opts)
                torch.testing.assert_close(attention(*ts,**opts),expected,rtol=0,atol=0)
            strided=[t[...,::2] for t in ts]
            with patch.dict(os.environ, MATRIX_XPU_ATTENTION="sdpa"):expected=attention(*strided)
            torch.testing.assert_close(attention(*strided),expected,rtol=0,atol=0)
            self.assertEqual(backend.backend_status()['tla_calls'],before)
