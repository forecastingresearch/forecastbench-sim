"""Jaeho: numbered FreeCiv batch parser."""
import re
_num = r"([0-9]*\.?[0-9]+)\s*%?"

def _prob(v):
    v = float(v)
    v = v / 100 if v > 1 else v
    return v if 0 <= v <= 1 else None

def parse_block(text, positions, kind):
    """-> ({position: value}, mode). Strict: the last <<<PROBABILITIES>>>/<<<PERCENTILES>>> block. Lenient: 'Qk: ...' lines anywhere
    (last occurrence wins). Positions not found are absent from the dict."""
    text = text or ""
    tag = "PERCENTILES" if kind == "cont" else "PROBABILITIES"
    blocks = re.findall(rf"<<<{tag}>>>(.*?)<<<END>>>", text, re.S)
    src, mode = (blocks[-1], "block") if blocks else (text[-6000:], "lenient")
    out = {}
    if kind == "cont":
        def five(rest):
            """Five labelled percentiles, or (Micropolis's last resort) five bare numbers on the line."""
            vals = dict(re.findall(r"\b(p5|p25|p50|p75|p95)\s*[=:]\s*(-?[0-9][0-9,]*\.?[0-9]*)", rest))
            try:
                return [float(vals[q].replace(",", "")) for q in ("p5", "p25", "p50", "p75", "p95")]
            except (KeyError, ValueError):
                pass
            if not re.search(r"\bp\d", rest):
                nums = re.findall(r"(?<![\w.])-?\d[\d,]*\.?\d*", rest)
                if len(nums) == 5:
                    try:
                        return [float(x.replace(",", "")) for x in nums]
                    except ValueError:
                        return None
            return None
        src_lines = src.splitlines()
        for li, line in enumerate(src_lines):
            m = re.match(r"\s*Q(\d+)\s*(?:\([^)]*\))?\s*[:.)-]?\s*(.*)", line)
            if not m:
                continue
            k = int(m.group(1))
            if k not in positions:
                continue
            v = five(m.group(2))
            if v is None:               # "Q5: <question restated>" with the percentiles on the next line
                nxt = next((x for x in src_lines[li + 1:li + 3] if x.strip()), "")
                if not re.match(r"\s*Q\d+", nxt):
                    v = five(nxt)
            if v is not None:
                out[k] = v
        if len(out) < len(positions):   # answers written as numbered lines outside the block ("16) ... p5=0, p25=1, ..." or "Q1 (Greek cities): p5=5, ...")
            for m in re.finditer(r"(?m)^\s*(?:Q|Question\s*)?(\d+)\s*(?:\([^)]*\))?\s*[).:-]?\s*(.*?\bp5\s*[=:].*)$", text):
                k = int(m.group(1))
                if k in positions and k not in out:
                    v = five(m.group(2))
                    if v is not None:
                        out[k] = v
                        mode = "lenient"
    else:
        for m in re.finditer(rf"Q(\d+)\s*[:.)-]\s*{_num}", src):
            k = int(m.group(1))
            if k in positions:
                v = _prob(m.group(2))
                if v is not None:
                    out[k] = v
    return out, (mode if out else None)
