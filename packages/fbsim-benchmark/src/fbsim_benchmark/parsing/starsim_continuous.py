"""Historical administered Starsim parser; not a simulator or truth repair."""
import json,re

def parse(txt: str, keys) -> dict | None:
    """Last JSON object with all quantile keys as numbers ≥ 0; sorted non-decreasing
    (a model listing them out of order is re-sorted, not rejected)."""
    txt = txt.replace("\\{", "{").replace("\\}", "}")
    for o in reversed(re.findall(r"\{[^{}]*\}", txt)):
        try:
            d = json.loads(o)
            vals = [float(str(d[k]).replace(",", "")) for k in keys]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if all(v >= 0 for v in vals):
            return dict(zip(keys, sorted(vals)))
    return None
