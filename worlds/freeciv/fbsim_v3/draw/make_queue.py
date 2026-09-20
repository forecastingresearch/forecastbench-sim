#!/usr/bin/env python3
"""make_queue.py --anchors 7001 7002 ... --rng 5001-6000 --pods ID:VCPU ... --out state/queue
Writes one task file per (anchor, rng) as a<anchor>-rng<rng> containing "<anchor> <rng>", dealt to pods
proportionally to vCPU with anchors interleaved (so every pod carries every anchor). Prints the plan."""
import argparse, os, random, shutil
p = argparse.ArgumentParser()
p.add_argument('--anchors', nargs='+', type=int, required=True)
p.add_argument('--rng', default='5001-6000')
p.add_argument('--pods', nargs='+', required=True, help='POD_ID:VCPU')
p.add_argument('--out', default='state/queue')
p.add_argument('--seed', type=int, default=20260902)
a = p.parse_args()
lo, hi = map(int, a.rng.split('-'))
tasks = [(A, r) for r in range(lo, hi + 1) for A in a.anchors]   # rng-major so anchors interleave
pods = [(x.split(':')[0], int(x.split(':')[1])) for x in a.pods]
tot_v = sum(v for _, v in pods)
if os.path.isdir(a.out): shutil.rmtree(a.out)
# deal in proportion to vCPU, walking the interleaved list in order
quota = {pid: len(tasks) * v / tot_v for pid, v in pods}
given = {pid: 0 for pid, _ in pods}
for A, r in tasks:
    pid = max(pods, key=lambda pv: quota[pv[0]] - given[pv[0]])[0]
    given[pid] += 1
    d = os.path.join(a.out, pid, 'todo'); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f'a{A}-rng{r}'), 'w') as f: f.write(f'{A} {r}\n')
for pid, v in pods:
    print(f'{pid} {v:>3} vcpu -> {given[pid]} tasks ({given[pid]/v:.1f}/vcpu)')
print(f'total {len(tasks)} tasks, {len(a.anchors)} anchors x {hi-lo+1} rng')
