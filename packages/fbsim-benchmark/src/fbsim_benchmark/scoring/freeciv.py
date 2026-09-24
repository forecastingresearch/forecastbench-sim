"""Verbatim numerical functions from Jaeho's score_v2.py."""
import numpy as np
import math
TAU = np.array([.05, .25, .5, .75, .95]); EPS = 1e-3

def pinball(x, ys):
    """mean over ys and tau of the quantile approximation to CRPS for percentile vector x (len 5)."""
    x = np.asarray(x, float); d = ys[:, None] - x[None, :]
    return float((2.0 / len(TAU)) * (np.where(d >= 0, TAU * d, (TAU - 1) * d)).mean(axis=0).sum())

def truth_quantiles(ys): return np.quantile(ys, TAU, method='inverted_cdf')

def bits(p, q):
    p = min(max(p, EPS), 1 - EPS)
    ll = -(q * math.log2(p) + (1 - q) * math.log2(1 - p)) if 0 < q < 1 else -math.log2(p if q == 1 else 1 - p)
    H = -(q * math.log2(q) + (1 - q) * math.log2(1 - q)) if 0 < q < 1 else 0.0
    return ll, ll - H
