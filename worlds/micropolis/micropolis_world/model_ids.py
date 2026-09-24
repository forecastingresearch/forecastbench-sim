"""Slug vs model id, and the filename form of a slug.

A *slug* is the canonical model id used everywhere in this world: configs,
`model_specs.json5` keys, cache filenames, data.json, labels. It is an
OpenRouter model id optionally followed by `:suffix` ("openai/o3:lowef"), where
the suffix picks a different ModelSpec for the same underlying model. The
*model id* is the part before the colon — what is actually sent to the API and
what external scores are keyed on. No suffix: slug == model id.

Dependency-free so analysis code can use it without importing the LLM client.
"""

import re

SUFFIX_SEP = ":"
# ':' is escaped in filenames: scp/rsync read "a:b" as host:path and Windows
# forbids it. '+' is legal everywhere and never appears in an OpenRouter id, so
# the mapping is invertible.
FILENAME_SUFFIX_SEP = "+"


def to_model_id(slug: str) -> str:
    """The model id for a slug: everything before the first ':'."""
    return slug.split(SUFFIX_SEP, 1)[0]


def filename_slug(slug: str) -> str:
    """A slug flattened to one filename component.

    '/' (and anything else odd) becomes '_', ':' becomes '+'. Both response
    caches name files with this, so a usage sidecar lands beside its response.
    """
    escaped = slug.replace(SUFFIX_SEP, FILENAME_SUFFIX_SEP)
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", escaped).strip("_")
