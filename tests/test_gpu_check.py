"""Tests for the model/weights compatibility guard in scripts/gpu_check.py.

Expected answers come from open_clip 3.3.0's bundled metadata, listed by hand with
open_clip.list_pretrained_tags_by_model() on 2026-10-24: 'openai', 'dfn2b' and 'metaclip_*' are
QuickGELU-trained; 'laion2b_s34b_b88k' and 'datacomp_xl_s13b_b90k' are not; the
ViT-B-16-quickgelu model has no laion2b tag. No download and no GPU needed.
"""

import pytest

from scripts.gpu_check import check_pretrained_compatible


@pytest.mark.parametrize("model_name, tag", [
    ("ViT-B-16", "laion2b_s34b_b88k"),
    ("ViT-B-16", "datacomp_xl_s13b_b90k"),
    ("ViT-B-16-quickgelu", "openai"),
    ("ViT-B-16-quickgelu", "dfn2b"),
])
def test_matching_pairs_pass(model_name, tag):
    check_pretrained_compatible(model_name, tag)


@pytest.mark.parametrize("tag", ["openai", "dfn2b", "metaclip_400m"])
def test_quickgelu_weights_in_gelu_model_rejected(tag):
    # The case that only produced a warning in the first gpu_check run.
    with pytest.raises(ValueError, match="QuickGELU mismatch"):
        check_pretrained_compatible("ViT-B-16", tag)


def test_tag_not_available_for_model_rejected():
    with pytest.raises(ValueError, match="no pretrained tag"):
        check_pretrained_compatible("ViT-B-16-quickgelu", "laion2b_s34b_b88k")


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unknown open_clip model"):
        check_pretrained_compatible("ViT-B-99", "openai")
