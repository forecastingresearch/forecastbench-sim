"""Verbatim numerical routines from Fabio's Micropolis scorer; no production config."""
PERCENTILE_LEVELS = {"p10": .10, "p25": .25, "p50": .50, "p75": .75, "p90": .90}

def quantile_array(percentiles: dict[str, float]):
    """A forecast's five quantiles as an array, in PERCENTILE_LEVELS order."""
    import numpy as np

    return np.array([percentiles[k] for k in PERCENTILE_LEVELS], dtype=float)

def crps_distribution(quantiles, outcomes) -> float:
    """compute_crps of `quantiles` averaged over every value in `outcomes`.

    The five-quantile pinball approximation of CRPS, (2/5) sum_tau rho_tau,
    scored against the whole replay distribution rather than one draw from
    it; with a single outcome it equals compute_crps exactly.
    """
    import numpy as np

    taus = np.fromiter(PERCENTILE_LEVELS.values(), dtype=float)
    d = np.asarray(outcomes, dtype=float)[:, None] - np.asarray(quantiles)[None, :]
    loss = np.where(d >= 0, taus * d, (taus - 1) * d)
    return float(2 * loss.sum(axis=1).mean() / len(taus))

def crps_floor(outcomes) -> float:
    """The best crps_distribution five quantiles can reach on `outcomes`.

    Scores the outcomes' own quantiles at the five levels (np.quantile,
    "inverted_cdf") against the outcomes, so it is what a forecast equal to
    the replay distribution would get; the excess CRPS subtracts it.
    """
    import numpy as np

    values = np.asarray(outcomes, dtype=float)
    levels = list(PERCENTILE_LEVELS.values())
    return crps_distribution(np.quantile(values, levels, method="inverted_cdf"), values)
