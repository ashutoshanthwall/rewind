"""Vendored LanguageBind model (rewind/embed/languagebind_vendor) on a tiny random model: the two ways it could be
wrong without an error. CPU, fp32, no weights."""

import numpy as np
import pytest
import torch

from rewind.embed.languagebind import LanguageBind
from rewind.embed.languagebind_vendor.configuration_video import LanguageBindVideoConfig
from rewind.embed.languagebind_vendor.modeling_video import LanguageBindVideo

T = 4   # frames per window in the tiny model


def tiny_model() -> LanguageBindVideo:
    torch.manual_seed(0)
    cfg = LanguageBindVideoConfig(
        text_config=dict(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, max_position_embeddings=16, eos_token_id=2),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                           image_size=28, patch_size=14, num_frames=T, add_time_attn=True, clip_type="vl_new",
                           lora_r=0, force_patch_dropout=0.0),
        projection_dim=8)
    m = LanguageBindVideo(cfg).eval()
    with torch.no_grad():   # random temporal embeddings: frames at different positions are treated differently
        for layer in m.vision_model.encoder.layers:
            layer.temporal_embedding.normal_(std=1.0)
    return m


def wrapper(m: LanguageBindVideo) -> LanguageBind:
    lb = object.__new__(LanguageBind)        # skip weight loading: tiny random model on the CPU
    lb.model, lb.num_frames, lb.dim = m, T, 8
    return lb


@torch.no_grad()
def test_text_tower_is_causal():
    """With the causal mask, the hidden state at position i depends only on tokens <= i. Newer transformers'
    CLIPAttention ignores causal_attention_mask; this fails if that code path is used."""
    m = tiny_model()
    ids = torch.tensor([[1, 5, 6, 7, 2, 2, 2, 2]])
    ids2 = ids.clone()
    ids2[0, 3] = 9                                # change a LATER token
    h1 = m.text_model(input_ids=ids).last_hidden_state
    h2 = m.text_model(input_ids=ids2).last_hidden_state
    assert torch.allclose(h1[0, :3], h2[0, :3], atol=1e-6)      # positions 0..2 unchanged
    assert not torch.allclose(h1[0, 3:], h2[0, 3:], atol=1e-4)  # from position 3 on, changed


@torch.no_grad()
def test_windows_are_embedded_independently():
    """encode_windows takes (B, T, 3, S, S): a window's vector must not depend on the other windows of the batch
    (frames mis-grouped across windows, e.g. a wrong permute before the model's '(b t)' reshape, would break this).
    Frame order within a window is not asserted: on a random model the time attention barely moves the
    frame-averaged output (measured: max |diff| 2e-4 .. 8e-3), so such a test would not be reliable."""
    lb = wrapper(tiny_model())
    torch.manual_seed(1)
    a, b = torch.randn(T, 3, 28, 28), torch.randn(T, 3, 28, 28)
    za = lb.encode_windows(a[None]).astype(np.float32)
    zab = lb.encode_windows(torch.stack([a, b])).astype(np.float32)
    zba = lb.encode_windows(torch.stack([b, a])).astype(np.float32)
    assert np.allclose(za[0], zab[0], atol=2e-3) and np.allclose(za[0], zba[1], atol=2e-3)
    assert np.linalg.norm(zab, axis=1) == pytest.approx([1, 1], abs=2e-3)
    with pytest.raises(ValueError, match="expected"):
        lb.encode_windows(torch.randn(1, T + 1, 3, 28, 28))
