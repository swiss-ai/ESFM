"""Explicit attention for short sequences (`perceiver.short_sequence_attention`).

The encoder's variable self-attention (`AxialAttention`, 5 or 9 variables) and variable aggregation
(`PerceiverAttention`, 1 query over 5 or 9) ran flash attention, whose 128-wide tiles are almost all
padding at these lengths, in 40,000-row slices. Below `SHORT_SEQUENCE_MAX` they compute attention
explicitly: fp32 softmax, bf16 products, dropout on the probabilities.
"""
import unittest
from unittest import mock

import torch

from esfm.model import perceiver
from esfm.model.encoder import AxialAttention
from esfm.model.perceiver import PerceiverAttention, short_sequence_attention


def fp32_attention(q, k, v):
    q, k, v = (t.float().transpose(1, 2) for t in (q, k, v))  # (b, h, l, d)
    probs = (q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5).softmax(-1)
    return (probs @ v).transpose(1, 2)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def qkv(b=64, lq=5, lk=5, h=4, d=32, dtype=torch.bfloat16):
    torch.manual_seed(0)
    return [torch.randn(b, n, h, d, dtype=dtype, requires_grad=True) for n in (lq, lk, lk)]


class ShortSequenceAttentionTest(unittest.TestCase):
    def test_matches_fp32_reference(self):
        q, k, v = qkv()
        out = short_sequence_attention(q, k, v, dropout_p=0.0, training=False)
        out.float().square().sum().backward()
        grads = [t.grad.clone() for t in (q, k, v)]
        q32, k32, v32 = (t.detach().float().requires_grad_(True) for t in (q, k, v))
        ref = fp32_attention(q32, k32, v32)
        ref.square().sum().backward()
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertLess(rel(out, ref), 1e-2)
        for g, r in zip(grads, (q32.grad, k32.grad, v32.grad)):
            self.assertLess(rel(g, r), 1e-2)

    def test_cross_attention_shapes(self):
        q, k, v = qkv(lq=1, lk=9)
        self.assertEqual(short_sequence_attention(q, k, v, 0.0, False).shape, (64, 1, 4, 32))

    def test_row_chunks_match_single_call(self):
        q, k, v = (t.detach() for t in qkv(b=100))
        whole = short_sequence_attention(q, k, v, 0.0, False)
        with mock.patch.object(perceiver, "_SHORT_SEQUENCE_CHUNK_ROWS", 7):
            chunked = short_sequence_attention(q, k, v, 0.0, False)
        torch.testing.assert_close(chunked, whole)

    def test_dropout_only_in_training(self):
        q, k, v = qkv()
        no_drop = short_sequence_attention(q, k, v, 0.0, False)
        torch.testing.assert_close(short_sequence_attention(q, k, v, 0.5, False), no_drop)
        self.assertFalse(torch.equal(short_sequence_attention(q, k, v, 0.5, True), no_drop))


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA autocast")
class LargeLogitsUnderAutocastTest(unittest.TestCase):
    def test_logits_stay_fp32_under_autocast(self):
        # Trained level aggregators reach |logits| ~ 1e5; under autocast a bf16 einsum would round
        # them in steps of 512 and change which keys the near-argmax softmax picks.
        q, k, v = (t.detach().cuda() for t in qkv(b=4096, lq=3, lk=13))
        q, k = q * 64, k * 64
        with torch.autocast("cuda", torch.bfloat16):
            out = short_sequence_attention(q, k, v, dropout_p=0.0, training=False)
        self.assertLess(rel(out, fp32_attention(q, k, v)), 1e-2)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA autocast")
class ShortPathWiringTest(unittest.TestCase):
    def run_both(self, module, *inputs):
        module = module.cuda().eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            with mock.patch("flash_attn.flash_attn_qkvpacked_func", side_effect=AssertionError("flash used")), \
                    mock.patch.object(module, "flash_attn_qkvpacked_func", side_effect=AssertionError("flash used"), create=True), \
                    mock.patch.object(module, "flash_attn_func", side_effect=AssertionError("flash used"), create=True):
                short = module(*inputs)
            with mock.patch.object(perceiver, "SHORT_SEQUENCE_MAX", 0):
                long = module(*inputs)
        return short, long

    def test_axial_attention_uses_short_path(self):
        torch.manual_seed(0)
        x = torch.randn(1000, 5, 64, device="cuda")
        short, long = self.run_both(AxialAttention(64, 4, heads_dim=16, attn_drop_rate=0.0), x)
        self.assertLess(rel(short, long), 1e-2)

    def test_axial_attention_separate_kv(self):
        torch.manual_seed(0)
        x, kv = torch.randn(1000, 5, 64, device="cuda"), torch.randn(1000, 5, 64, device="cuda")
        short, long = self.run_both(AxialAttention(64, 4, heads_dim=16, attn_drop_rate=0.0), x, kv)
        self.assertLess(rel(short, long), 1e-2)

    def test_perceiver_attention_uses_short_path(self):
        torch.manual_seed(0)
        latents, x = torch.randn(1000, 1, 64, device="cuda"), torch.randn(1000, 9, 64, device="cuda")
        module = PerceiverAttention(latent_dim=64, context_dim=64, head_dim=16, num_heads=4, ln_k_q=True, attn_drop_rate=0.0)
        short, long = self.run_both(module, latents.bfloat16(), x.bfloat16())
        self.assertLess(rel(short, long), 1e-2)


class Fp32ShortPathTest(unittest.TestCase):
    def test_fp32_perceiver_attention_matches_sdpa(self):
        # The decoder runs outside autocast: its level de-aggregation (13 queries over 3 keys) takes the
        # short path in fp32 and must match fp32 SDPA up to summation order.
        torch.manual_seed(0)
        latents, x = torch.randn(512, 13, 64), torch.randn(512, 3, 64)
        module = PerceiverAttention(latent_dim=64, context_dim=64, head_dim=16, num_heads=4,
                                    attn_drop_rate=0.0, disable_flashattention=True).eval()
        with torch.no_grad():
            with mock.patch.object(perceiver, "short_sequence_attention",
                                   wraps=perceiver.short_sequence_attention) as spy:
                short = module(latents, x)
            self.assertEqual(spy.call_count, 1)
            with mock.patch.object(perceiver, "SHORT_SEQUENCE_MAX", 0):
                long = module(latents, x)
        self.assertEqual(short.dtype, torch.float32)
        self.assertLess(rel(short, long), 1e-5)


if __name__ == "__main__":
    unittest.main()
