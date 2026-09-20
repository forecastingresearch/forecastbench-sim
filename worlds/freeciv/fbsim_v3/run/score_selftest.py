#!/usr/bin/env python3
"""score_selftest.py — synthetic forecasters through score_v1.py; no API calls.

  oracle   binary p = q; continuous = the truth distribution's own tau-quantiles; natcond t1 = p, t2 = p_given, nonews = p, single = p_given
  flat     0.5 everywhere; continuous = the truth median for all five percentiles
  noisy    oracle + N(0, 0.1) noise (clipped); continuous = truth quantiles * lognormal jitter
  broken   oracle with a third of the answers unparsed (value None)
Checks: oracle excess = 0 on every set; flat excess = closed form; noisy excess >= 0 item-wise; broken parse rate = 2/3 and
binary imputation at 0.5 gives (0.5 - q)^2 on the unparsed items.
"""
import json, os, sys, tempfile, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_v1 as sv
S, C, N = sv.load_sets(); rng = np.random.default_rng(1)
rows = []
def emit(model, item, arm, kind, value): rows.append(dict(model=model, item=item, arm=arm, kind=kind, value=value))
def noise(p): return float(min(1 - 1e-6, max(1e-6, p + rng.normal(0, 0.1))))
for i, (iid, it) in enumerate(S.items()):
    q = it['qAll']
    emit('oracle', iid, 't1', 'bin', q); emit('flat', iid, 't1', 'bin', 0.5); emit('noisy', iid, 't1', 'bin', noise(q)); emit('broken', iid, 't1', 'bin', None if i % 3 == 0 else q)
for i, (iid, it) in enumerate(C.items()):
    ys = np.asarray(it['values'], float); tq = sv.truth_quantiles(ys).tolist()
    emit('oracle', iid, 't1', 'cont', tq); emit('flat', iid, 't1', 'cont', [float(np.median(ys))] * 5)
    emit('noisy', iid, 't1', 'cont', sorted((np.array(tq) * np.exp(rng.normal(0, 0.2, 5))).tolist())); emit('broken', iid, 't1', 'cont', None if i % 3 == 0 else tq)
for i, c in enumerate(N):
    item = c['qid'] + '|' + c['rev_id']
    emit('oracle', item, 't2', 'bin', c['p_given']); emit('flat', item, 't2', 'bin', 0.5); emit('noisy', item, 't2', 'bin', noise(c['p_given'])); emit('broken', item, 't2', 'bin', None if i % 3 == 0 else c['p_given'])
    if c['control_no_news']:
        for m, v in [('oracle', c['p']), ('flat', 0.5), ('noisy', noise(c['p'])), ('broken', c['p'])]: emit(m, c['qid'], 'nonews', 'bin', v)
    if c['control_single_prompt']:
        for m, v in [('oracle', c['p_given']), ('flat', 0.5), ('noisy', noise(c['p_given'])), ('broken', c['p_given'])]: emit(m, item, 'single', 'bin', v)
d = tempfile.mkdtemp(dir='/private/tmp/claude-501/-Users-jaeholee0404/eec6d9ea-59b9-4f86-a6bf-66180c2461cb/scratchpad')
with open(f'{d}/results.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
R = sv.load_results([d]); items = sv.score_items(R, S, C, N, 0.5); summ = sv.summarise(items, 200, 0)
def get(model, s, met, group='all', value='all'): return next(r for r in summ if r['model'] == model and r['set'] == s and r['group'] == group and r['value'] == value)[met]
ok = True
def check(name, cond, detail=''):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name + (f'  {detail}' if detail else ''))
for s in ['bank', 'tails', 'mirrors', 'extra']: check(f'oracle {s} excess_brier == 0', abs(get('oracle', s, 'excess_brier')) < 1e-12, f"{get('oracle', s, 'excess_brier'):.2e}")
check('oracle tails excess_bits == 0', abs(get('oracle', 'tails', 'excess_bits')) < 1e-9, f"{get('oracle', 'tails', 'excess_bits'):.2e}")
check('oracle continuous excess_crps == 0', abs(get('oracle', 'continuous', 'excess_crps')) < 1e-9, f"{get('oracle', 'continuous', 'excess_crps'):.2e}")
check('oracle continuous excess_ncrps_global == 0, ncrps_global == floor/C > 0', abs(get('oracle', 'continuous', 'excess_ncrps_global')) < 1e-9 and get('oracle', 'continuous', 'ncrps_global') > 0, f"oracle nCRPS_global {get('oracle', 'continuous', 'ncrps_global'):.3f} flat {get('flat', 'continuous', 'ncrps_global'):.3f}")
check('oracle continuous excess_crps_norm == 0 and flat > 0', abs(get('oracle', 'continuous', 'excess_crps_norm')) < 1e-9 and get('flat', 'continuous', 'excess_crps_norm') > 0, f"flat {get('flat', 'continuous', 'excess_crps_norm'):.3f} IQR units")
check('oracle natcond excess_t2 == 0', abs(get('oracle', 'natcond', 'excess_t2')) < 1e-12)
has_single = any(c['control_single_prompt'] for c in N)
check('oracle natcond excess_nonews == 0' + (' and excess_single == 0' if has_single else ' (no single-prompt cells in v1.8)'),
      abs(get('oracle', 'natcond', 'excess_nonews')) < 1e-12 and (not has_single or abs(get('oracle', 'natcond', 'excess_single')) < 1e-12))
gain_expect = float(np.mean([(c['p'] - c['p_given']) ** 2 for c in N]))
check('oracle natcond gain == mean (p - p_given)^2', abs(get('oracle', 'natcond', 'gain') - gain_expect) < 1e-12, f"{get('oracle', 'natcond', 'gain'):.5f} vs {gain_expect:.5f}")
check('oracle natcond move/target corr == 1', abs(get('oracle', 'natcond', 'move_target_corr') - 1) < 1e-9)
flat_expect = float(np.mean([(0.5 - it['qAll']) ** 2 for it in S.values() if it['set'] == 'bank']))
check('flat bank excess_brier == mean (0.5-q)^2', abs(get('flat', 'bank', 'excess_brier') - flat_expect) < 1e-12, f"{get('flat', 'bank', 'excess_brier'):.5f}")
q_t = [it['qAll'] for it in S.values() if it['set'] == 'tails']
check('flat tails excess_bits == mean KL(q||0.5)', abs(get('flat', 'tails', 'excess_bits') - np.mean([sv.bits(0.5, q)[1] for q in q_t])) < 1e-12, f"{get('flat', 'tails', 'excess_bits'):.4f} bits")
cont_noisy = [r for r in items if r['model'] == 'noisy' and r['set'] == 'continuous']
check('noisy continuous excess_crps >= 0 item-wise', all(r['excess_crps'] >= -1e-9 for r in cont_noisy), f"min {min(r['excess_crps'] for r in cont_noisy):.2e}")
check('noisy binary excess > 0, natcond excess_t2 > 0', get('noisy', 'bank', 'excess_brier') > 0 and get('noisy', 'natcond', 'excess_t2') > 0, f"bank {get('noisy', 'bank', 'excess_brier'):.4f} natcond {get('noisy', 'natcond', 'excess_t2'):.4f}")
check('flat continuous cov50/cov90 == 0 or the median mass', get('flat', 'continuous', 'cov90') == get('flat', 'continuous', 'cov50'))
check('broken bank parse_rate == 2/3', abs(get('broken', 'bank', 'parse_rate') - 2 / 3) < 0.01, f"{get('broken', 'bank', 'parse_rate'):.3f}")
b_items = [r for r in items if r['model'] == 'broken' and r['set'] == 'bank' and not r['parsed']]
check('broken bank unparsed imputed at 0.5', all(abs(r['excess_brier'] - (0.5 - r['q']) ** 2) < 1e-12 for r in b_items), f"{len(b_items)} items")
check('broken continuous parse_rate == 2/3 and excess == 0 on parsed', abs(get('broken', 'continuous', 'parse_rate') - 2 / 3) < 0.01 and abs(get('broken', 'continuous', 'excess_crps')) < 1e-9)
t1_none = {r['item'] for r in rows if r['model'] == 'broken' and r['arm'] == 't1' and r['value'] is None}
nc_expect = np.mean([(c['qid'] not in t1_none) and (i % 3 != 0) for i, c in enumerate(N)])   # a cell needs both its t1 and its t2 parsed
check('broken natcond parse_rate == share of cells with t1 and t2 parsed', abs(get('broken', 'natcond', 'parse_rate') - nc_expect) < 1e-9, f"{get('broken', 'natcond', 'parse_rate'):.3f} vs {nc_expect:.3f}")
lo, hi = get('noisy', 'bank', 'excess_brier_lo'), get('noisy', 'bank', 'excess_brier_hi'); mean = get('noisy', 'bank', 'excess_brier')
check('bootstrap CI brackets the mean', lo <= mean <= hi, f"{mean:.4f} in [{lo:.4f}, {hi:.4f}]")
out = f'{d}/scores'; os.makedirs(out, exist_ok=True); sv.write_md(summ, f'{out}/SCORES.md')
print('SCORES.md sample written to', out)
print('ALL PASS' if ok else 'SOME CHECKS FAILED'); sys.exit(0 if ok else 1)
