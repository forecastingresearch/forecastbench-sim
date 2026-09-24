"""Historical administered Starsim parser; not a simulator or truth repair."""
import json,re

def parse(txt: str, keys: tuple[str, ...]) -> dict | None:
    """Last well-formed JSON object in the reply with all keys in [0,1].
    Lenient fallbacks (v2 pilot): LaTeX-escaped braces (\\{ "p": 0.7 \\}) and a
    bare trailing `"p": 0.98` with no braces both count."""
    txt = txt.replace("\\{", "{").replace("\\}", "}")
    for o in reversed(re.findall(r"\{[^{}]*\}", txt)):
        try:
            d = json.loads(o)
            vals = {k: float(d[k]) for k in keys}
            if all(0.0 <= v <= 1.0 for v in vals.values()):
                return vals
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    vals = {}
    for k in keys:
        ms = re.findall(rf'"{k}"\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)', txt)
        if not ms:
            return None
        vals[k] = float(ms[-1])
    return vals
