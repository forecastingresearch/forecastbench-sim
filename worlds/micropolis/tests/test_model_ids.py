"""Slug vs model id, and the filename form both caches share."""

from micropolis_world.model_ids import filename_slug, to_model_id


def test_to_model_id_strips_the_suffix():
    assert to_model_id("openai/o3:lowef") == "openai/o3"
    assert to_model_id("openai/o3") == "openai/o3"
    # Only the first colon separates; the suffix may contain more.
    assert to_model_id("prov/m:a:b") == "prov/m"


def test_filename_slug_escapes_slash_and_colon():
    assert filename_slug("openai/o3:lowef") == "openai_o3+lowef"
    # Unsuffixed slugs keep the filenames the existing caches were written with.
    assert filename_slug("openai/gpt-4o-2024-05-13") == "openai_gpt-4o-2024-05-13"
    assert filename_slug("qwen/qwen3.5-flash-02-23") == "qwen_qwen3.5-flash-02-23"
