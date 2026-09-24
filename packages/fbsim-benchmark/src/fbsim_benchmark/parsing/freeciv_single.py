"""Jaeho: original single-question FreeCiv parser."""
import re

def _prob(v):
    v = float(v); v = v / 100 if v > 1 else v
    return v if 0 <= v <= 1 else None

def parse_prob(t, mode=False):
    """strict: the last <probability> tag. lenient: 'probability ... 0.xx' or a bare 'p = 0.xx' in the last 400 chars."""
    m = re.findall(r'<probability>\s*([0-9]*\.?[0-9]+)\s*%?\s*</probability>', t or '')
    if m: return (_prob(m[-1]), 'strict') if mode else _prob(m[-1])
    tail = (t or '')[-400:]
    t = re.sub(r'(?i)\s*</?probability>\s*$', '', tail.strip()).rstrip('.* ')          # stray closing tag / trailing period
    m = (re.findall(r'<\s*([0-9]*\.?[0-9]+)\s*%?\s*>\s*$', t)                          # a bare '<0.3>' at the end (Llama)
         or re.findall(r'(?i)probability[^0-9]{0,40}([0-9]*\.?[0-9]+)\s*%?', t)         # 'Probability: 0.3'
         or re.findall(r'(?:^|[\s:=~≈])(0?\.[0-9]+|1\.0+|1|0)\s*$', t))                  # a final bare number in [0,1] (o3 ends with '0.35')
    v = _prob(m[-1]) if m else None
    return (v, 'lenient' if v is not None else None) if mode else v

def parse_pct(t, mode=False):
    """strict: the last <percentiles> block. lenient: the last run of p5..p95 values anywhere in the text."""
    m = re.findall(r'<percentiles>(.*?)</percentiles>', t or '', re.S)
    src, how = (m[-1], 'strict') if m else ((t or '')[-1500:], 'lenient')
    found = re.findall(r'\b(p25|p50|p75|p95|p5)(?![0-9])\s*(?:[:=]|\s+(?:maybe|about|around|roughly|of|at|~|≈))?\s*(-?[0-9][0-9,]*\.?[0-9]*)', src)   # 'p5: 3', 'p5=3', 'p5 maybe 22'
    vals = {}
    for k, v in found: vals[k] = v                                   # last occurrence of each key wins
    try: out = [float(vals[k].replace(',', '')) for k in ('p5', 'p25', 'p50', 'p75', 'p95')]
    except (KeyError, ValueError): return (None, None) if mode else None
    return (out, how) if mode else out
