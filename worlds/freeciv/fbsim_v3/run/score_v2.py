#!/usr/bin/env python3
"""score_v2.py (run 2, 2026-09-19): score_v1.py with paths relative to this checkout and the arm t1nc for the
unbatched natural-conditional turn-1 rows, which take precedence as p1 and stand in for a missing t1 row.

score_v1.py — scorer for the draw v1 sets (one rule per set, no per-family special cases).

  uv run python score_v1.py RESULTS_DIR [RESULTS_DIR ...] --out DIR [--boot 1000] [--seed 0] [--impute-binary 0.5]

Reads every results.jsonl under the given dirs (rows from elicit_v1.py; the last row per (model, item, arm) wins),
joins on (world, item id) — never on question text — and writes:
  DIR/score_items.csv     one row per (model, set, item, arm) with the per-item scores
  DIR/score_summary.csv   every aggregate (model x set [x horizon | x family | x block]) with bootstrap 95% CIs
  DIR/SCORES.md           the headline tables

Rules
  binary (bank, tails, mirrors, extra):  q = qAll.  brier = (p-q)^2 + q(1-q)  [expected Brier over the 1,000 replays];
                                         excess_brier = (p-q)^2  (headline; the q-knowing forecaster scores 0).
  tails, additionally:                   logloss_bits = -[q log2 p + (1-q) log2(1-p)]  with p clipped to [1e-3, 1-1e-3];
                                         excess_bits = KL(q||p) = logloss_bits - H(q).
  continuous:                            HEADLINE ncrps_global = crps5 / C_family, Fabio's Micropolis rule (CRPS over one fixed constant per metric);
                                         C_family from continuous_norm_constants.json (median truth p05-p95 range per family, per metric inside
                                         NC1_value_at_T, rounded to 1 s.f.). Also excess_ncrps_global = (crps5 - floor) / C_family and
                                         excess_crps_norm = (crps5 - floor) / item IQR; raw crps5 / excess_crps kept per item.
                                         the five reported percentiles (p5,p25,p50,p75,p95; sorted if not monotone) scored by
                                         pinball loss at tau = .05,.25,.5,.75,.95 against every truth value, averaged:
                                         crps5 = (2/5) * sum_tau mean_y rho_tau(y - x_tau)  (quantile approximation to CRPS);
                                         floor = the same with the truth distribution's own tau-quantiles; excess_crps = crps5 - floor.
                                         Diagnostics: cov50 / cov90 = share of truth values inside [p25,p75] / [p5,p95].
  natcond:                               p1 = turn-1 answer to the question, p2 = turn-2 answer after the reveal.
                                         excess_t2 = (p2 - p_given)^2 (headline), stay = (p1 - p_given)^2 (no-update baseline),
                                         gain = stay - excess_t2; nonews: excess_nonews = (p_nn - p)^2, drift = p_nn - p1;
                                         single: excess_single = (p_s - p_given)^2.  move = p2 - p1 vs target = p_given - p.
Missing / unparsed answers: binary p is imputed (default 0.5, `--impute-binary none` to drop instead); continuous and natcond
rows are dropped and the parse rate is reported alongside every aggregate.
"""
import argparse, csv, glob, json, math, os, collections
import numpy as np
SETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sets', 'draw_v1')   # v2: relative to this checkout
TAU = np.array([.05, .25, .5, .75, .95]); EPS = 1e-3

def load_sets(sets_dir=SETS):
    S = {}
    for name, f in [('bank', 'bank_750'), ('tails', 'tails_300'), ('mirrors', 'mirrors_50'), ('extra', 'natcond_extra_turn1')]:
        for i in json.load(open(f'{sets_dir}/{f}.json')): S[i['id']] = dict(i, set=name)
    C = {i['id']: i for i in json.load(open(f'{sets_dir}/continuous_300.json'))}
    try: NORM = json.load(open(f'{sets_dir}/continuous_norm_constants.json'))['constants']
    except FileNotFoundError: NORM = {}
    for i in C.values(): i['norm_c'] = (NORM.get(i['family'] + ('/' + i.get('metric', '') if i['family'] == 'NC1_value_at_T' else '')) or {}).get('constant')
    N = json.load(open(f'{sets_dir}/natcond_600.json'))
    return S, C, N

def load_results(dirs):
    R = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, '**', 'results.jsonl'), recursive=True)) + ([d] if d.endswith('.jsonl') else []):
            for line in open(f):
                try: r = json.loads(line)
                except Exception: continue
                R[(r['model'], r['item'], r['arm'])] = r
    return R

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

def score_items(R, S, C, N, impute):
    rows = []
    models = sorted({m for (m, _, _) in R})
    for m in models:
        for iid, it in S.items():
            r = R.get((m, iid, 't1')) or R.get((m, iid, 't1nc')); q = it['qAll']; p = r.get('value') if r else None; parsed = p is not None   # v2: unbatched row stands in where no batched row exists (the extra questions)
            if not parsed:
                if impute is None: continue
                p = impute
            ll, kl = bits(p, q)
            rows.append(dict(model=m, set=it['set'], item=iid, arm='t1', world=it['world'], family=it['family'], T=it['T'], parsed=int(parsed), q=q, p=p,
                             brier=(p - q) ** 2 + q * (1 - q), excess_brier=(p - q) ** 2, expected_brier=(p - q) ** 2 + q * (1 - q), calibration=(p - q) ** 2, logloss_bits=ll, excess_bits=kl))
        for iid, it in C.items():
            r = R.get((m, iid, 't1')); x = r.get('value') if r else None
            row = dict(model=m, set='continuous', item=iid, arm='t1', world=it['world'], family=it['family'], T=it['T'], parsed=int(x is not None))
            if x is None: rows.append(row); continue
            ys = np.asarray(it['values'], float); xs = sorted(x); tq = truth_quantiles(ys)
            crps = pinball(xs, ys); floor = pinball(tq, ys)
            iqr = float(it.get('iqrAll') or (tq[3] - tq[1]) or 1.0)
            cN = float(it.get('norm_c') or iqr)
            row.update(monotone=int(list(x) == xs), crps5=crps, floor=floor, excess_crps=crps - floor, ncrps_global=crps / cN, excess_ncrps_global=(crps - floor) / cN, norm_c=cN,
                       excess_crps_norm=(crps - floor) / iqr, crps_norm=crps / iqr, iqr=iqr,
                       cov50=float(((ys >= xs[1]) & (ys <= xs[3])).mean()), cov90=float(((ys >= xs[0]) & (ys <= xs[4])).mean()),
                       width90=xs[4] - xs[0], truth_width90=float(tq[4] - tq[0]), median_err=xs[2] - float(tq[2]), truth_median=float(tq[2]))
            rows.append(row)
        for c in N:
            item = c['qid'] + '|' + c['rev_id']
            t1 = R.get((m, c['qid'], 't1nc')) or R.get((m, c['qid'], 't1')); t2 = R.get((m, item, 't2')); nn = R.get((m, c['qid'], 'nonews')); sg = R.get((m, item, 'single'))
            p1 = t1.get('value') if t1 else None; p2 = t2.get('value') if t2 else None
            row = dict(model=m, set='natcond', item=item, arm='t2', world=c['world'], family=c['family'], T=c['T'], block=c['block'], rev_kind=c['rev_kind'],
                       q=c['p'], p_given=c['p_given'], delta=c['delta'], parsed=int(p1 is not None and p2 is not None))
            if p1 is not None and p2 is not None:
                row.update(p1=p1, p2=p2, excess_t2=(p2 - c['p_given']) ** 2, stay=(p1 - c['p_given']) ** 2, excess_t1_uncond=(p1 - c['p']) ** 2,
                           move=p2 - p1, target=c['p_given'] - c['p'])
                row['gain'] = row['stay'] - row['excess_t2']
            if c['control_no_news'] and nn and nn.get('value') is not None and p1 is not None:
                row.update(p_nonews=nn['value'], excess_nonews=(nn['value'] - c['p']) ** 2, drift=nn['value'] - p1)
            if c['control_single_prompt'] and sg and sg.get('value') is not None:
                row.update(p_single=sg['value'], excess_single=(sg['value'] - c['p_given']) ** 2)
            rows.append(row)
    return rows

def boot_mean(v, rng, B):
    v = np.asarray(v, float)
    if len(v) == 0: return (float('nan'),) * 3
    if len(v) == 1: return float(v[0]), float(v[0]), float(v[0])
    idx = rng.integers(0, len(v), size=(B, len(v))); means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

METRICS = {'bank': ['excess_brier', 'brier', 'expected_brier', 'calibration'], 'tails': ['excess_brier', 'brier', 'expected_brier', 'calibration', 'excess_bits', 'logloss_bits'], 'mirrors': ['excess_brier', 'brier', 'expected_brier', 'calibration'],
           'extra': ['excess_brier', 'brier', 'expected_brier', 'calibration'], 'continuous': ['ncrps_global', 'excess_ncrps_global', 'excess_crps_norm', 'excess_crps', 'crps5', 'cov50', 'cov90', 'median_err'],
           'natcond': ['excess_t2', 'stay', 'gain', 'excess_nonews', 'excess_single', 'move', 'drift']}
HEADLINE = {'bank': 'excess_brier', 'tails': 'excess_bits', 'mirrors': 'excess_brier', 'extra': 'excess_brier', 'continuous': 'ncrps_global', 'natcond': 'excess_t2'}

def summarise(rows, B, seed):
    rng = np.random.default_rng(seed); out = []
    by = collections.defaultdict(list)
    for r in rows: by[(r['model'], r['set'])].append(r)
    for (m, s), rs in sorted(by.items()):
        groups = [('all', 'all', rs)] + [('T', str(T), [r for r in rs if r['T'] == T]) for T in sorted({r['T'] for r in rs})]
        if s == 'natcond':
            groups += [('block', b, [r for r in rs if r['block'] == b]) for b in ['A', 'B', 'C1', 'C2', 'D']]
            groups += [('rev_kind', k, [r for r in rs if r['rev_kind'] == k]) for k in sorted({r['rev_kind'] for r in rs})]
        else:
            groups += [('family', f, [r for r in rs if r['family'] == f]) for f in sorted({r['family'] for r in rs})]
        for gtype, gval, g in groups:
            n_parsed = sum(r['parsed'] for r in g)
            rec = dict(model=m, set=s, group=gtype, value=gval, n=len(g), n_parsed=n_parsed, parse_rate=n_parsed / len(g) if g else float('nan'))
            for met in METRICS[s]:
                v = [r[met] for r in g if met in r and r[met] is not None]
                mean, lo, hi = boot_mean(v, rng, B); rec[met] = mean; rec[met + '_lo'] = lo; rec[met + '_hi'] = hi; rec[met + '_n'] = len(v)
            if s == 'natcond':
                mv = [(r['move'], r['target']) for r in g if 'move' in r]
                if len(mv) >= 3:
                    a = np.array(mv); rec['move_target_corr'] = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) if a[:, 1].std() > 0 and a[:, 0].std() > 0 else float('nan')
                    big = a[np.abs(a[:, 1]) >= 0.05]
                    rec['sign_agree'] = float((np.sign(big[:, 0]) == np.sign(big[:, 1])).mean()) if len(big) else float('nan'); rec['sign_agree_n'] = int(len(big))
                    rec['abs_move'] = float(np.abs(a[:, 0]).mean())
            out.append(rec)
    return out

def fmt(x, d=4): return '' if x is None or (isinstance(x, float) and math.isnan(x)) else (f'{x:.{d}f}' if isinstance(x, float) else str(x))

def write_md(summary, path):
    L = ['# Scores (draw v1, all-1000 truth)', '']
    models = sorted({r['model'] for r in summary})
    for s, label in [('bank', 'Binary bank 750 — excess Brier (p−q)²'), ('tails', 'Tails 300 — excess bits KL(q‖p)'), ('mirrors', 'Mirrors 50 — excess Brier'),
                     ('continuous', 'Continuous 300 — nCRPS_global = CRPS / fixed per-family constant (Fabio-style); excess variants alongside'), ('natcond', 'Natural conditionals 400 — turn-2 excess Brier vs p(Y|X)'), ('extra', 'Natcond extra turn-1 questions 124 — excess Brier')]:
        rs = [r for r in summary if r['set'] == s and r['group'] == 'all']
        if not rs: continue
        met = HEADLINE[s]; L += [f'## {label}', '']
        cols = ['model', 'n', 'parse', met, '95% CI'] + (['brier'] if s in ('bank', 'tails', 'mirrors', 'extra') else []) + (['logloss_bits'] if s == 'tails' else []) \
               + (['excess_ncrps_global', 'excess_crps_norm', 'crps5', 'cov50', 'cov90'] if s == 'continuous' else []) + (['stay', 'gain', 'excess_nonews', 'excess_single', 'move_target_corr', 'sign_agree'] if s == 'natcond' else [])
        L.append('| ' + ' | '.join(cols) + ' |'); L.append('|' + '---|' * len(cols))
        for r in sorted(rs, key=lambda r: (r.get(met) if r.get(met) == r.get(met) else 9e9)):
            cells = [r['model'], str(r['n']), fmt(r['parse_rate'], 3), fmt(r.get(met)), f"[{fmt(r.get(met + '_lo'))}, {fmt(r.get(met + '_hi'))}]"] + [fmt(r.get(c)) for c in cols[5:]]
            L.append('| ' + ' | '.join(cells) + ' |')
        L.append('')
        # by horizon
        L += [f'### {s}: {met} by horizon', '']
        Ts = sorted({r['value'] for r in summary if r['set'] == s and r['group'] == 'T'}, key=int)
        L.append('| model | ' + ' | '.join(f'T{T}' for T in Ts) + ' |'); L.append('|---|' + '---|' * len(Ts))
        for m in models:
            vals = {r['value']: r.get(met) for r in summary if r['set'] == s and r['group'] == 'T' and r['model'] == m}
            if vals: L.append(f'| {m} | ' + ' | '.join(fmt(vals.get(T)) for T in Ts) + ' |')
        L.append('')
        if s == 'natcond':
            L += ['### natcond by block (excess_t2 / stay / gain / excess_nonews)', '']
            blocks = ['A', 'B', 'C1', 'C2', 'D']
            L.append('| model | ' + ' | '.join(blocks) + ' |'); L.append('|---|' + '---|' * len(blocks))
            for m in models:
                vals = {r['value']: r for r in summary if r['set'] == s and r['group'] == 'block' and r['model'] == m}
                if vals: L.append(f'| {m} | ' + ' | '.join(f"{fmt(vals[b].get('excess_t2'),3)}/{fmt(vals[b].get('stay'),3)}/{fmt(vals[b].get('gain'),3)}/{fmt(vals[b].get('excess_nonews'),3)}" if b in vals else '' for b in blocks) + ' |')
            L.append('')
    open(path, 'w').write('\n'.join(L))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('results', nargs='+'); ap.add_argument('--out', required=True)
    ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--impute-binary', default='0.5', help='probability substituted for an unparsed binary answer, or "none" to drop')
    ap.add_argument('--sets-dir', default=SETS); a = ap.parse_args()
    impute = None if a.impute_binary == 'none' else float(a.impute_binary)
    S, C, N = load_sets(a.sets_dir); R = load_results(a.results)
    rows = score_items(R, S, C, N, impute); summary = summarise(rows, a.boot, a.seed)
    os.makedirs(a.out, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('model', 'set', 'item', 'arm', 'world', 'family', 'T'), k))
    with open(f'{a.out}/score_items.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    keys = sorted({k for r in summary for k in r}, key=lambda k: (k not in ('model', 'set', 'group', 'value', 'n', 'n_parsed', 'parse_rate'), k))
    with open(f'{a.out}/score_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)
    write_md(summary, f'{a.out}/SCORES.md')
    for r in summary:
        if r['group'] == 'all': print(f"{r['model']:40s} {r['set']:11s} n={r['n']:4d} parse={r['parse_rate']:.3f} {HEADLINE[r['set']]}={fmt(r.get(HEADLINE[r['set']]))} [{fmt(r.get(HEADLINE[r['set']] + '_lo'))}, {fmt(r.get(HEADLINE[r['set']] + '_hi'))}]")
    print('written', a.out)
if __name__ == '__main__': main()
