"""Verbatim numerical routines from the accepted Starsim scorer."""
import numpy as np
import math
TAU = (.10, .25, .50, .75, .90)
CLIP = 1e-3

def crps5(q, ys):
    ys = np.asarray(ys, float); tot = 0.0
    for x, t in zip(q, TAU):
        d = ys - x; tot += float(np.mean(np.where(d >= 0, t * d, (t - 1) * d)))
    return 2 * tot / len(TAU)

def kl_bits(q, p):
    p = min(max(p, CLIP), 1 - CLIP)
    ll = -(q * math.log2(p) if q > 0 else 0.0) - ((1 - q) * math.log2(1 - p) if q < 1 else 0.0)
    H = -(q * math.log2(q) if q > 0 else 0.0) - ((1 - q) * math.log2(1 - q) if q < 1 else 0.0)
    return ll - H
