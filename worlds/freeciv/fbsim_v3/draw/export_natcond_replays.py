#!/usr/bin/env python3
"""Export the per-replay outcome vectors behind the 400 natural-conditional cells.

Each cell pairs a question (qid = <world>:b<i>, row i of the world's bank pickle 'Y') with a revealed fact
(rev_id = <world>:r<j>, row j of the pickle 'X').  Both rows are boolean vectors over the world's 1,000 replays,
in the replay order of consolidate_v4.py (rng tags sorted numeric-aware; the same order for every cell of a world).
The export keeps all 1,000 columns, so p = Y.mean(), p(Y|X) = Y[X].mean() and nx = X.sum() can be recomputed and
resampled jointly across cells of the same world.

Usage:
  python export_natcond_replays.py --bank-dir  .../fbsim_v3_corpus/bank_v1 \
                                   --consol-dir .../fbsim_v3_corpus/consol_v4   (optional: replay tags and halves)
                                   --cells     .../sets/draw_v1/natcond_600.json \
                                   --out-dir   .../sets/draw_v1

Writes natcond_replays.npz (boolean arrays plus ids), natcond_replays_cells.csv (one row per cell, in npz row
order) and natcond_replay_tags.csv (world, column, rng tag, split half).  Every cell's p, p_given and nx are
checked against the cell file before writing.
"""
import argparse, csv, json, os, pickle, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--bank-dir', required=True)
ap.add_argument('--consol-dir', default=None, help='consolidate_v4 pickles; supplies replay tags and the split halves')
ap.add_argument('--cells', required=True, help='natcond_600.json (the 400 scored cells)')
ap.add_argument('--out-dir', required=True)
ap.add_argument('--tol', type=float, default=1e-9, help='tolerance for p, p_given against the cell file')
a = ap.parse_args()

cells = json.load(open(a.cells))
worlds = sorted({c['world'] for c in cells})
bank = {w: pickle.load(open(os.path.join(a.bank_dir, f'{w}.pkl'), 'rb')) for w in worlds}
n = {w: bank[w]['n'] for w in worlds}
assert len(set(n.values())) == 1, n
n_rep = next(iter(n.values()))

tags = {}; half_a = {}; half_b = {}; reserved = {}
for w in worlds:
    B = bank[w]
    assert B['Y'].shape[1] == n_rep and B['X'].shape[1] == n_rep and len(B['halfA']) == n_rep
    if a.consol_dir:
        C = pickle.load(open(os.path.join(a.consol_dir, f'{w}.pkl'), 'rb'))
        assert len(C['tags']) == n_rep, (w, len(C['tags']))
        assert np.array_equal(np.asarray(C['halfA']), np.asarray(B['halfA'])), w   # same column order as the bank
        assert np.array_equal(np.asarray(C['halfB']), np.asarray(B['halfB'])), w
        tags[w] = [str(t) for t in C['tags']]; reserved[w] = str(C['reserved'])
    else:
        tags[w] = [f'col{k}' for k in range(n_rep)]; reserved[w] = ''
    half_a[w] = np.asarray(B['halfA'], bool); half_b[w] = np.asarray(B['halfB'], bool)

X = np.zeros((len(cells), n_rep), bool); Y = np.zeros((len(cells), n_rep), bool)
meta = []
worst = 0.0
for k, c in enumerate(cells):
    w = c['world']
    qw, qi = c['qid'].split(':'); rw, ri = c['rev_id'].split(':')
    assert qw == w and rw == w and qi[0] == 'b' and ri[0] == 'r', c['qid'] + ' ' + c['rev_id']
    qi, ri = int(qi[1:]), int(ri[1:])
    B = bank[w]
    inst, rev = B['bin'][qi], B['reveals'][ri]
    assert inst['text'] == c['question'] and inst['family'] == c['family'] and inst['T'] == c['T'], (c['qid'], inst['text'], c['question'])
    assert rev['text'] == c['reveal'] and rev['kind'] == c['rev_kind'], (c['rev_id'], rev['text'], c['reveal'])
    y, x = B['Y'][qi], B['X'][ri]
    assert int(x.sum()) == c['nx'], (c['qid'], c['rev_id'], int(x.sum()), c['nx'])
    dp = abs(float(y.mean()) - c['p']); dpg = abs(float(y[x].mean()) - c['p_given'])
    worst = max(worst, dp, dpg)
    assert dp <= a.tol and dpg <= a.tol, (c['qid'], c['rev_id'], float(y.mean()), c['p'], float(y[x].mean()), c['p_given'])
    X[k], Y[k] = x, y
    meta.append(dict(row=k, cell_id=c['qid'] + '|' + c['rev_id'], world=w, world_index=worlds.index(w), qid=c['qid'], rev_id=c['rev_id'],
                     horizon=c['T'], block=c['block'], family=c['family'], rev_kind=c['rev_kind'], from_bank=c['from_bank'],
                     control_no_news=c['control_no_news'], p=c['p'], p_given=c['p_given'], delta=c['delta'], nx=c['nx'],
                     question=c['question'], reveal=c['reveal']))
print(f'{len(cells)} cells, {len(worlds)} worlds, {n_rep} replays each; max |p - Y.mean()| or |p_given - Y[X].mean()| = {worst:.2e}')

os.makedirs(a.out_dir, exist_ok=True)
S = lambda key: np.array([m[key] for m in meta])
np.savez_compressed(os.path.join(a.out_dir, 'natcond_replays.npz'),
    X=X, Y=Y,
    cell_id=S('cell_id'), world=S('world'), world_index=S('world_index').astype(np.int16), qid=S('qid'), rev_id=S('rev_id'),
    horizon=S('horizon').astype(np.int16), block=S('block'), family=S('family'), rev_kind=S('rev_kind'),
    from_bank=S('from_bank').astype(bool), control_no_news=S('control_no_news').astype(bool),
    p=S('p').astype(float), p_given=S('p_given').astype(float), delta=S('delta').astype(float), nx=S('nx').astype(np.int16),
    question=S('question'), reveal=S('reveal'),
    worlds=np.array(worlds), replay_tags=np.array([tags[w] for w in worlds]), reserved_tag=np.array([reserved[w] for w in worlds]),
    half_a=np.array([half_a[w] for w in worlds]), half_b=np.array([half_b[w] for w in worlds]))
cols = ['row', 'cell_id', 'world', 'world_index', 'qid', 'rev_id', 'horizon', 'block', 'family', 'rev_kind', 'from_bank', 'control_no_news', 'p', 'p_given', 'delta', 'nx', 'question', 'reveal']
with open(os.path.join(a.out_dir, 'natcond_replays_cells.csv'), 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows({k: m[k] for k in cols} for m in meta)
with open(os.path.join(a.out_dir, 'natcond_replay_tags.csv'), 'w', newline='') as f:
    wr = csv.writer(f); wr.writerow(['world', 'column', 'replay_tag', 'half'])
    for w in worlds:
        for k in range(n_rep):
            wr.writerow([w, k, tags[w][k], 'A' if half_a[w][k] else ('B' if half_b[w][k] else 'reserved')])
print('wrote', os.path.join(a.out_dir, 'natcond_replays.npz'), 'and the two csv files')
