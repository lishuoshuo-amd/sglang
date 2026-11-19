"""On gfx942 the DeepSeek-V4.1 fused Triton paths match the torch forms they replace."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.layers.attention.dsv4.dsv41_sparse as dsv41_sparse
import sglang.srt.layers.engram as engram
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_gfx942_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd")

ROPE_DIM = 64
BF16_TOL = dict(rtol=2**-7, atol=2**-7)


def _torch_and_fused(module, gate: str, fn):
    """``fn`` with the arch gate forced off (torch forms), then on (fused kernels)."""
    outs = []
    for enabled in (False, True):
        with mock.patch.object(module, gate, lambda *args, **kwargs: enabled):
            outs.append(fn())
    return outs


def _assert_fp4_close(test, out: torch.Tensor, ref: torch.Tensor):
    # A pre-quant value on an FP4 rounding tie can land one FP4 step apart (about one
    # value in 1e8), so all but 1e-6 of the values must match exactly.
    mismatched = (out.float() != ref.float()).sum().item()
    test.assertLessEqual(mismatched, max(1, ref.numel() // 10**6))


def _freqs(rows: int) -> torch.Tensor:
    angle = torch.rand(rows, ROPE_DIM // 2, device="cuda") * 100
    return torch.polar(torch.ones_like(angle), angle)


@unittest.skipUnless(
    is_gfx942_supported(), "gfx942 enablement of the fused DSv4.1 kernels"
)
class TestDsv41FusedKernelsGfx942(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_gates_open_on_gfx942(self):
        x = torch.zeros(1, device="cuda")
        self.assertTrue(dsv41_sparse._fused_dsv41_kernels(x))
        self.assertTrue(engram._cuda_kernels(x))

    def test_rope_fake_quant_matches_torch_form(self):
        table = _freqs(8192)
        for shape in ((512,), (8, 512), (128,), (32, 128)):
            for t in (1, 135, 4096):
                x = (torch.randn(t, *shape, device="cuda") * 3).to(torch.bfloat16)
                freqs = _freqs(t)
                positions = torch.randint(0, table.shape[0], (t,), device="cuda")
                for compressed_kv in (False, True) if len(shape) == 1 else (False,):
                    with self.subTest(shape=shape, t=t, compressed_kv=compressed_kv):
                        ref, out = _torch_and_fused(
                            dsv41_sparse,
                            "_fused_dsv41_kernels",
                            lambda: dsv41_sparse._rope_fq4(
                                x, freqs, ROPE_DIM, compressed_kv=compressed_kv
                            ),
                        )
                        _assert_fp4_close(self, out, ref)
                        ref, out = _torch_and_fused(
                            dsv41_sparse,
                            "_fused_dsv41_kernels",
                            lambda: dsv41_sparse._rope_fq4(
                                x,
                                table,
                                ROPE_DIM,
                                compressed_kv=compressed_kv,
                                positions=positions,
                            ),
                        )
                        _assert_fp4_close(self, out, ref)

    def test_rmsnorm_matches_torch_form(self):
        # A few outputs differ from the torch form by one ulp.
        for dim in (128, 512):
            for weight_dtype in (torch.bfloat16, torch.float32):
                norm = dsv41_sparse.RMSNorm(dim, 1e-6).cuda()
                norm.weight.data = (torch.rand(dim, device="cuda") + 0.5).to(
                    weight_dtype
                )
                for x_dtype in (torch.bfloat16, torch.float32):
                    for shape in ((1, dim), (135, dim), (4096, dim), (135, 8, dim)):
                        x = (torch.randn(*shape, device="cuda") * 4).to(x_dtype)
                        with self.subTest(
                            dim=dim, w=weight_dtype, x=x_dtype, shape=shape
                        ):
                            ref, out = _torch_and_fused(
                                dsv41_sparse, "_fused_dsv41_kernels", lambda: norm(x)
                            )
                            tol = BF16_TOL if x_dtype == torch.bfloat16 else {}
                            torch.testing.assert_close(out, ref, **tol)

    def test_engram_gate_matches_torch_form(self):
        hc, dim, image_id = 4, 5120, 999
        for t in (1, 135, 4096):
            x = torch.randn(t, hc, dim, device="cuda").to(torch.bfloat16)
            kv = torch.randn(t, (hc + 1) * dim, device="cuda").to(torch.bfloat16)
            q_weight = (torch.rand(hc, dim, device="cuda") + 0.5).to(torch.bfloat16)
            k_weight = (torch.rand(hc, dim, device="cuda") + 0.5).to(torch.bfloat16)
            ids = torch.randint(0, image_id, (t,), device="cuda")
            ids[::7] = image_id
            for with_ids in (False, True):
                with self.subTest(t=t, with_ids=with_ids):
                    ref, out = _torch_and_fused(
                        engram,
                        "_cuda_kernels",
                        lambda: engram.engram_gate(
                            x,
                            kv,
                            q_weight,
                            k_weight,
                            1e-6,
                            1e-6,
                            ids if with_ids else None,
                            image_id if with_ids else None,
                        ),
                    )
                    torch.testing.assert_close(out, ref, **BF16_TOL)

    def _hasher(self, slots: int):
        vocab, compressed_vocab = 20000, 5000
        config = SimpleNamespace(
            engram_layer_ids=[1, 14],
            engram_max_ngram_size=4,
            engram_n_heads=8,
            engram_vocab_size=1_000_000,
            engram_num_embeddings=[1 << 20, 1 << 20],
            engram_head_dim=256,
        )
        token_map = [i % compressed_vocab for i in range(vocab)]
        with mock.patch.object(
            engram,
            "build_compressed_token_map",
            lambda tokenizer: (token_map, compressed_vocab),
        ):
            hasher = engram.EngramHasher(
                engram.EngramLayout.from_config(config), None, 2, compressed_vocab
            ).cuda()
        hasher.init_history(slots, "cuda")
        return hasher, vocab

    def test_engram_hash_and_history_are_bitwise(self):
        slots = 64
        hasher, vocab = self._hasher(slots)
        base = torch.randint(
            0, vocab, hasher.history.shape, device="cuda", dtype=torch.int32
        )
        for mode, block in ((ForwardMode.DECODE, 1), (ForwardMode.TARGET_VERIFY, 5)):
            for bs in (1, 25, 64):
                tokens = bs * block
                ids = torch.randint(0, vocab, (tokens,), device="cuda")
                req = torch.randperm(slots, device="cuda")[:bs]
                positions = torch.randint(0, 3, (bs,), device="cuda").repeat_interleave(
                    block
                ) + torch.arange(block, device="cuda").repeat(bs)
                out_cache_loc = torch.randint(1, 1 << 20, (tokens,), device="cuda")
                out_cache_loc[-block:] = 0 if bs > 1 else 1
                batch = SimpleNamespace(
                    forward_mode=mode,
                    positions=positions,
                    req_pool_indices=req,
                    out_cache_loc=out_cache_loc,
                    extend_seq_lens_cpu=None,
                    ragged_verify_layout=None,
                    engram_history=None,
                    spec_info=SimpleNamespace(draft_token_num=block),
                )

                def run():
                    hasher.history.copy_(base)
                    return hasher(ids, batch), hasher.history.clone()

                with self.subTest(mode=mode.name, bs=bs):
                    (ref_ids, ref_hist), (out_ids, out_hist) = _torch_and_fused(
                        engram, "_cuda_kernels", run
                    )
                    torch.testing.assert_close(out_ids, ref_ids, rtol=0, atol=0)
                    # The torch form also writes graph padding into the spare pad row,
                    # which no request reads.
                    live = slice(0, hasher.pad_row)
                    torch.testing.assert_close(
                        out_hist[live], ref_hist[live], rtol=0, atol=0
                    )

    def test_engram_commit_after_verify_is_bitwise(self):
        slots = 64
        hasher, vocab = self._hasher(slots)
        base = torch.randint(
            0, vocab, hasher.history.shape, device="cuda", dtype=torch.int32
        )
        for bs in (1, 25, 64):
            verify_ids = torch.randint(0, vocab, (bs, 6), device="cuda")
            req = torch.randperm(slots, device="cuda")[:bs]
            commit_lens = torch.randint(1, 7, (bs,), device="cuda", dtype=torch.int32)

            def commit():
                hasher.history.copy_(base)
                hasher.commit_after_verify(verify_ids, req, commit_lens)
                return hasher.history.clone()

            with self.subTest(bs=bs):
                ref, out = _torch_and_fused(engram, "_cuda_kernels", commit)
                torch.testing.assert_close(out, ref, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
