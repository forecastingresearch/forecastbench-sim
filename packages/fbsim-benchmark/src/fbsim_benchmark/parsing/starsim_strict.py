"""Strict public binary response parsing; historical parsing stays separate."""
import json
import math
import re


def parse(text, keys):
    """Read requested probabilities from the last flat JSON object.

    Reject invalid/missing values rather than retrying numeric prefixes or earlier
    answers. Only JSON numbers are accepted, excluding booleans and strings.
    Surrounding prose is allowed; historical unbraced fallbacks are not.
    """
    if not isinstance(keys, (list, tuple)) or not keys or any(not isinstance(k, str) or not k for k in keys) or len(set(keys)) != len(keys):
        raise ValueError("keys must be a nonempty list of distinct strings")
    objects = re.findall(r"\{[^{}]*\}", text)
    if not objects:
        raise ValueError("Expected a JSON object containing the requested probabilities")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON constant")
    record = json.loads(objects[-1], object_pairs_hook=pairs, parse_constant=invalid)
    values = {}
    for key in keys:
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError("Each requested probability must be a finite JSON number in [0,1]")
        values[key] = float(value)
    return values
