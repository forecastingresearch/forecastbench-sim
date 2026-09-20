#!/usr/bin/env python3
"""qc_pipeline.py DRAW_DIR — downstream-aware QC of the drawn sets: report consistency, criteria coverage,
scoring-rule uniformity, natcond admissibility, plus a random reading sample."""
import json, sys, re, random, collections, glob, os
import pandas as pd
D = sys.argv[1]; rng = random.Random(5)
S = {k: json.load(open(f'{D}/{k}.json')) for k in ('bank_750', 'tails_300', 'mirrors_50', 'continuous_300', 'natcond_600', 'natcond_extra_turn1')}
bank, tails, mirrors, cont, nc, extra = (S[k] for k in ('bank_750', 'tails_300', 'mirrors_50', 'continuous_300', 'natcond_600', 'natcond_extra_turn1'))
binary = bank + tails + mirrors + extra
print('== sizes:', {k: len(v) for k, v in S.items()})
# 1. names used in questions vs names in the regenerated world reports
rep = {}
for f in glob.glob('/Users/jaeholee0404/civbench/tmp/fbsim_v3_corpus/reports/seed*/turn_060_report.txt'):
    w = os.path.basename(os.path.dirname(f)); txt = open(f).read(); rep[w] = txt
    names = re.findall(r'^\d+\s+\| ([^|]+?)\s+\|', txt.split('CURRENT STATE')[0].split('CIVILIZATIONS')[1], re.M)
    rep[w + ':names'] = [n.strip() for n in names]
bad = []
for i in binary + cont + [dict(x, text=x['question'], world=x['world']) for x in nc]:
    for n in rep.get(i['world'] + ':names', []):
        pass
    if not any(n in i['text'] for n in rep.get(i['world'] + ':names', [])) and re.search(r"Will [A-Z]", i['text']) and not re.search(r"Will (at least|any|the )", i['text']):
        bad.append((i['world'], i['text'][:80]))
print('== questions naming a civ that is not in that world\'s report:', len(bad), bad[:3])
# 2. quantities referenced vs report columns
hdr = re.search(r'ID \| Name.*', rep['seed7001']).group(0)
cols = [c.strip().lower() for c in hdr.split('|')]
need = {'score': 'score', 'population': 'population', 'number of cities': 'cities', 'territory size': 'territory', 'treasury': 'treasury', 'number of military units': 'military', 'number of technologies': 'techs', 'government': 'government'}
print('== report columns:', cols)
print('== quantity in questions -> report column present:', {k: (v in cols) for k, v in need.items()})
# 3. criteria template coverage per family
spec = json.load(open('/Users/jaeholee0404/civbench/tmp/fbsim_v2/family_specs.json'))['families']
alias = {'NB1_value_threshold': 'NB1_threshold', 'W2_directed_conquest': 'W2_directed_conquest', 'NC1_value_at_T': 'NC1_value', 'NC5_world_captures': 'NC5_count', 'NC14_world_wonders': 'NC14_wonders_built', 'S7_world_razings': 'S7_city_razing', 'S7_any_destroyed': 'S7_city_razing', 'S6_city_founding': 'S6_city_founding', 'W3_capture_k': 'NB6_event', 'W4_lose_k': 'NB6_event', 'S3_wars_at_T': 'S3_wars_at_T'}
fams = collections.Counter(i['family'] for i in binary + cont)
rows = []
for f, n in sorted(fams.items()):
    sid = alias.get(f, f); tpl = (spec.get(sid) or {}).get('criteria_template')
    rows.append((f, n, sid if sid in spec else '-', 'yes' if tpl else 'NO'))
print('== criteria template coverage (family, items, spec id, template):'); [print('  ', r) for r in rows]
# 4. scoring-rule uniformity
print('== continuous: any censoring:', any(c['censAll'] > 0 for c in cont), '| any median 0:', any(c['medAll'] == 0 for c in cont), '| any IQR 0:', any(c['iqrAll'] == 0 for c in cont), '| per horizon:', dict(collections.Counter(c['T'] for c in cont)))
print('== binary: any q outside (0,1):', any(not 0 < i['qAll'] < 1 for i in binary), '| tails 0<q<=.05:', all(0 < i['qAll'] <= .05 for i in tails), '| mirrors .95<=q<1:', all(.95 <= i['qAll'] < 1 for i in mirrors))
print('== leftover parentheticals in questions:', collections.Counter(re.findall(r'\(([^)]*)\)', ' '.join(i['text'] for i in binary + cont))).most_common(6))
# 5. natcond admissibility
print('== natcond: T>=120:', all(c['T'] >= 120 for c in nc), '| n_x>=100:', all(c['nx'] >= 100 for c in nc), '| p|X in (0.02,0.98):', all(.02 < c['p_given'] < .98 for c in nc),
      '| max reveals per question:', max(collections.Counter(c['qid'] for c in nc).values()), '| max cells per event:', max(collections.Counter(c['rev_id'] for c in nc).values()),
      '| questions from bank:', sum(c['from_bank'] for c in nc), '| extra turn-1 questions:', len(extra))
# 6. reading sample: 2 per family over all sets + 2 per natcond block
print('\n== READING SAMPLE')
for f in sorted(fams):
    pool = [i for i in binary + cont if i['family'] == f]
    for i in rng.sample(pool, min(2, len(pool))):
        v = f"q={i['qAll']:.3f}" if 'qAll' in i else f"med={i['medAll']:g} IQR={i['iqrAll']:g}"
        print(f"  [{f}] {i['text']}  ({v})")
for b in ('A', 'B', 'C1', 'C2', 'D'):
    for c in rng.sample([x for x in nc if x['block'] == b], 2): print(f"  [natcond {b}] {c['question']} | news: {c['reveal']} | {c['p']:.2f}->{c['p_given']:.2f}")
