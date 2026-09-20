#!/usr/bin/env python3
"""elicit_v1.py — elicitation over the draw v1 sets via OpenRouter (v1.8: report-first prompts with cache breakpoints).

  uv run python elicit_v1.py --out DIR [--models NAME_OR_ID,...] [--models-file models_v1.csv]
                             [--sets bank,tails,mirrors,continuous,natcond] [--reasoning low|none]
                             [--limit N] [--workers 16] [--dry-run]

The model sees the turn-60 world report and one question with its resolution criteria (prompt text verbatim from
the derisk harness). Provider-default sampling: no temperature, no max_tokens. Reasoning: `--reasoning low`
(default) sends OpenRouter's {"reasoning": {"effort": "low"}} to models flagged reasoning=yes in the models file
and nothing to the others; `--reasoning none` sends nothing to anyone.

Arms (one row per (model, item, arm) in DIR/results.jsonl; resumable, duplicates resolved to the last row):
  t1      every bank / tails / mirrors / continuous item and every natcond extra question -> one call.
  t2      every natcond cell: continues the t1 conversation of the cell's question with the uniform reveal
          sentence, re-asks. item = qid|rev_id.
  nonews  cells flagged control_no_news: the same t1 conversation continued with the no-news sentence instead.
          The reveal plays no part, so this arm is keyed per question (item = qid) and shared by the cells.
  single  cells flagged control_single_prompt: one fresh prompt with the reveal sentence inserted before the
          criteria. item = qid|rev_id.
A row whose answer failed to parse is re-asked once (fresh call, `attempts` = 2); the last row wins.
"""
import argparse, csv, json, os, random, re, sys, time, threading, urllib.request, urllib.error, collections
from concurrent.futures import ThreadPoolExecutor
ROOT = '/Users/jaeholee0404/civbench'; SETS = f'{ROOT}/tmp/fbsim_v3_corpus/draw_v1'; REPORTS = f'{ROOT}/tmp/fbsim_v3_corpus/reports'
HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ.get('OPENROUTER_API_KEY') or next((l.split('=', 1)[1].strip().strip('"\'') for l in open(f'{ROOT}/.env') if l.startswith('OPENROUTER_API_KEY=')), None)
# Prompt = [prefix (intro + background + report), tail (question, optional fact, criteria, instructions)].
# The prefix is identical for every question of one kind in one world and is sent as its own content part with a
# cache_control breakpoint so provider prefix caches hit (v1.8, 2026-09-08). Wording otherwise = the derisk prompts.
BASE_PREFIX = """You will be asked a binary forecasting question about a simulation in progress. Your task is to estimate the probability that the event described in the question occurs. Please provide your reasoning before stating your final probability.

Question Background: This is a partial report on a FreeCiv game simulation in progress, observed at turn 60. Five AI civilizations are competing.

{report}
"""
BASE_TAIL = """
Question Title: {question}
{fact}
Resolution Criteria: {criteria}

Think step by step about the information provided, reason about uncertainty, and put your final confidence that the question resolves YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

Your answer will be evaluated with a proper scoring rule, so your best strategy is to report your honest probability estimate.

Your final answer should be the probability that the event resolves YES and your response SHOULD STRICTLY END with <probability> </probability> tags."""
CONT_PREFIX = """You will be asked a numeric forecasting question about a simulation in progress. Your task is to estimate the full range of plausible values for the quantity described in the question, expressed as percentiles of your forecast distribution. Please provide your reasoning before stating your final percentiles.

Question Background: This is a partial report on a FreeCiv game simulation in progress, observed at turn 60. Five AI civilizations are competing.

{report}
"""
CONT_TAIL = """
Question Title: {question}
{fact}
Resolution Criteria: {criteria}

Think step by step about the information provided, reason about uncertainty, and give your final answer as five percentiles of your forecast distribution: p5, p25, p50, p75, p95. "p5" means you estimate a 5% chance the true value falls below that number; "p50" is your median. Your percentiles must be non-decreasing (p5 <= p25 <= p50 <= p75 <= p95). The answer is a whole number.

Your answer will be evaluated with a proper scoring rule, so your best strategy is to report your honest estimate of the distribution - percentiles that are too narrow and percentiles that are too wide will both cost you.

Your response SHOULD STRICTLY END with the five percentiles in this exact format:
<percentiles>
p5: <number>
p25: <number>
p50: <number>
p75: <number>
p95: <number>
</percentiles>"""
def make_prompt(kind, world, question, criteria, fact=''):
    """-> (prefix, tail). fact = '' or the single-prompt reveal line."""
    pre, tail = (CONT_PREFIX, CONT_TAIL) if kind == 'cont' else (BASE_PREFIX, BASE_TAIL)
    return pre.format(report=report(world)), tail.format(question=question, fact=fact, criteria=criteria)
FIRST_PARTY = ('anthropic/', 'openai/', 'google/')
def user_content(prefix, tail, model=None):
    """Two content parts with a cache breakpoint for first-party hosts (verified to keep both parts and to cache), one plain
    string for everything else: the OpenInference host serving DeepSeek V4 Flash dropped the second part (the question) in
    the 2026-09-09 stage — answers asked for 'the missing question'."""
    mid = (model or {}).get('openrouter_id', '') if isinstance(model, dict) else (model or '')
    if mid.startswith(FIRST_PARTY): return [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": tail}]
    return prefix + tail
TURN2 = """{preamble}

Given this, please answer the same question again. Reason about what, if anything, this changes, and end your response with your updated probability in <probability> </probability> tags."""
NO_NEWS = "The game has continued. No new information about turns 61 through 90 is available. If you wish, revise your forecast."
TURN2_NONEWS = NO_NEWS + "\n\nPlease answer the same question again and end your response with your probability in <probability> </probability> tags."
FILES = {'bank': 'bank_750', 'tails': 'tails_300', 'mirrors': 'mirrors_50', 'continuous': 'continuous_300'}

_rep = {}
def report(world):
    if world not in _rep: _rep[world] = open(f'{REPORTS}/{world}/turn_060_report.txt').read()
    return _rep[world]
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

def load_models(path):
    """models_v1.csv: name, openrouter_id, reasoning (yes/no), ... -> list of dicts."""
    return [r for r in csv.DictReader(open(path)) if r.get('openrouter_id')]

def smoke_filter(jobs, seed=7):
    """One item per (set, family) for bank / tails / continuous (tails only for families absent from the bank), one natcond cell
    per block preferring cells that carry both controls, plus whatever turn-1 questions those cells need. ~52 calls per model."""
    import random as _r; rng = _r.Random(seed)
    S = {}
    for name, f in [('bank', 'bank_750'), ('tails', 'tails_300'), ('continuous', 'continuous_300')]:
        for i in json.load(open(f'{SETS}/{f}.json')): S.setdefault((name, i['family']), []).append(i['id'])
    bank_fams = {fam for (nm, fam) in S if nm == 'bank'}
    keep = set()
    for (nm, fam), ids in sorted(S.items()):
        if nm == 'tails' and fam in bank_fams: continue
        keep.add(rng.choice(sorted(ids)))
    nc = json.load(open(f'{SETS}/natcond_600.json')); cells = set()
    for b in ['A', 'B', 'C1', 'C2', 'D']:
        cand = [c for c in nc if c['block'] == b and c['control_no_news'] and c['control_single_prompt']] or [c for c in nc if c['block'] == b]
        c = rng.choice(sorted(cand, key=lambda c: c['qid'] + c['rev_id'])); cells.add(c['qid'] + '|' + c['rev_id']); keep.add(c['qid'])
    t2 = [j for j in jobs['t2'] if j['item'] in cells]; single = [j for j in jobs['single'] if j['item'] in cells]
    nn_q = {j['qid'] for j in t2}; nonews = [j for j in jobs['nonews'] if j['qid'] in nn_q]
    return dict(t1=[j for j in jobs['t1'] if j['item'] in keep], t2=t2, nonews=nonews, single=single)

def sample_filter(jobs, spec, seed=11):
    """spec 'bank=20,tails=8,mirrors=4,continuous=8,natcond=8': stratified subsets — bank round-robin over families, tails and
    mirrors and continuous round-robin over horizons, natcond cells round-robin over blocks; plus the t1 questions those cells
    need and their no-news controls where flagged. Every job is one of the full run's jobs, so a resumed full run re-uses them."""
    import random as _r; rng = _r.Random(seed); want = {k: int(v) for k, v in (kv.split('=') for kv in spec.split(','))}
    def rr(items, keyf, n):
        groups = collections.defaultdict(list)
        for it in items: groups[keyf(it)].append(it)
        for g in groups.values(): rng.shuffle(g)
        out, keys = [], sorted(groups)
        while len(out) < n and any(groups[k] for k in keys):
            for k in keys:
                if groups[k] and len(out) < n: out.append(groups[k].pop())
        return out
    keep = set()
    for st, keyf in [('bank', lambda j: j['family']), ('tails', lambda j: j['T']), ('mirrors', lambda j: j['T']), ('continuous', lambda j: j['T'])]:
        if want.get(st): keep.update(j['item'] for j in rr([j for j in jobs['t1'] if j['set'] == st], keyf, want[st]))
    cells = rr(jobs['t2'], lambda j: j['block'], want.get('natcond', 0)); cell_ids = {j['item'] for j in cells}
    keep.update(j['qid'] for j in cells)
    return dict(t1=[j for j in jobs['t1'] if j['item'] in keep], t2=[j for j in jobs['t2'] if j['item'] in cell_ids],
                nonews=[j for j in jobs['nonews'] if j['qid'] in {c['qid'] for c in cells}], single=[j for j in jobs['single'] if j['item'] in cell_ids])

def build_jobs(sets, limit=0, smoke=False):
    """Every prompt of the study, independent of the model. Returns dict(t1=[...], t2=[...], nonews=[...], single=[...]).
    t1/single jobs carry the full prompt; t2/nonews jobs carry the follow-up message and the qid whose t1 they extend."""
    sets = sets.split(',') if isinstance(sets, str) else sets
    t1, seen = [], set()
    def add_t1(i, kind, set_name):
        if i['id'] in seen: return
        seen.add(i['id'])
        pre, tail = make_prompt(kind, i['world'], i['text'], i['criteria'])
        t1.append(dict(item=i['id'], kind=kind, world=i['world'], set=set_name, family=i.get('family'), T=i.get('T'), prefix=pre, tail=tail, prompt=pre + tail))
    for s in sets:
        if s in FILES:
            for i in json.load(open(f'{SETS}/{FILES[s]}.json')): add_t1(i, 'cont' if s == 'continuous' else 'bin', s)
    t2, nonews, single = [], [], []
    if 'natcond' in sets:
        nc = json.load(open(f'{SETS}/natcond_600.json'))
        bank = {i['id']: i for i in json.load(open(f'{SETS}/bank_750.json'))}
        extra = {i['id']: i for i in json.load(open(f'{SETS}/natcond_extra_turn1.json'))}
        nn_seen = set()
        for c in nc:
            q = bank.get(c['qid']) or extra[c['qid']]
            add_t1(q, 'bin', 'bank' if c['from_bank'] else 'extra')
            item = c['qid'] + '|' + c['rev_id']
            t2.append(dict(item=item, kind='bin', world=q['world'], qid=q['id'], block=c['block'], followup=TURN2.format(preamble=c['turn2_preamble'])))
            if c['control_no_news'] and q['id'] not in nn_seen:
                nn_seen.add(q['id']); nonews.append(dict(item=q['id'], kind='bin', world=q['world'], qid=q['id'], followup=TURN2_NONEWS))
            if c['control_single_prompt']:
                pre, tail = make_prompt('bin', q['world'], q['text'], q['criteria'], fact='\n' + c['turn2_preamble'] + '\n')
                single.append(dict(item=item, kind='bin', world=q['world'], qid=q['id'], block=c['block'], prefix=pre, tail=tail, prompt=pre + tail))
    if limit:
        keep = {j['item'] for j in t1[:limit]}
        t1 = t1[:limit]; t2 = [j for j in t2 if j['qid'] in keep]; nonews = [j for j in nonews if j['qid'] in keep]; single = [j for j in single if j['qid'] in keep]
    t1.sort(key=lambda j: (j['world'], j['kind'])); t2.sort(key=lambda j: j['world']); nonews.sort(key=lambda j: j['world']); single.sort(key=lambda j: j['world'])
    jobs = dict(t1=t1, t2=t2, nonews=nonews, single=single)
    return smoke_filter(jobs) if smoke else jobs

FORCE = None
lock = threading.Lock(); spent = collections.defaultdict(float)
def reasoning_param(model, reasoning):
    """What goes in the request's `reasoning` field. models file: reasoning_mode = effort | budget | none;
    effort (default = the --reasoning level) or budget tokens (default 1024 = LiteLLM's 'low')."""
    if reasoning == 'none': return None
    mode = (model.get('reasoning_mode') or ('effort' if model.get('reasoning', 'no').lower().startswith('y') else 'none')).strip()
    if FORCE:   # --force-effort: override every model's level (budget models get LiteLLM's mapping: low 1024, medium 2048, high 4096)
        if mode == 'effort': return {"effort": FORCE}
        if mode == 'budget': return {"max_tokens": {'low': 1024, 'medium': 2048, 'high': 4096}.get(FORCE, 1024)}
    if mode == 'effort': return {"effort": (model.get('effort') or '').strip() or reasoning}
    if mode == 'budget': return {"max_tokens": int((model.get('budget') or '').strip() or 1024)}
    if mode == 'disabled': return {"enabled": False}      # non-thinking mode where the model offers one (DeepSeek V4 Flash honours this; it ignores effort/budget)
    return None
def build_body(model, messages, reasoning, max_tokens=16384):
    """The exact JSON sent to OpenRouter for this model."""
    body = {"model": model['openrouter_id'], "messages": messages, "usage": {"include": True}}
    rp = reasoning_param(model, reasoning)
    if rp: body["reasoning"] = rp
    mt = (model.get('max_tokens') or '').strip() or max_tokens
    if mt: body["max_tokens"] = int(mt)
    if (model.get('provider') or '').strip(): body["provider"] = {"order": [x.strip() for x in model['provider'].split('|')], "allow_fallbacks": False}
    else: body["provider"] = {"ignore": ["Novita"] + [x.strip() for x in (model.get('ignore') or '').split('|') if x.strip()]}
    return body
def call(model, messages, reasoning, max_tokens=16384):
    body = {"model": model['openrouter_id'], "messages": messages, "usage": {"include": True}}
    rp = reasoning_param(model, reasoning)
    if rp: body["reasoning"] = rp
    mt = (model.get('max_tokens') or '').strip() or max_tokens
    if mt: body["max_tokens"] = int(mt)          # bounds OpenRouter's in-flight credit reservation; well above any answer
    if (model.get('provider') or '').strip():      # explicit pin in the models file wins (needed for models hosted only by Novita)
        body["provider"] = {"order": [x.strip() for x in model['provider'].split('|')], "allow_fallbacks": False}
    else: body["provider"] = {"ignore": ["Novita"] + [x.strip() for x in (model.get('ignore') or '').split('|') if x.strip()]}   # team rule: never Novita; per-model extra ignores
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "HTTP-Referer": "https://forecastingresearch.org", "X-Title": "fbsim-v3"})
    err = None; transient = []
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=600) as r: resp = json.load(r)
            if 'error' in resp and not resp.get('choices'): raise RuntimeError(json.dumps(resp['error'])[:300])
            ch = resp.get('choices', [{}])[0]; msg = ch.get('message') or {}; txt = msg.get('content') or ''
            u = resp.get('usage') or {}; cost = float(u.get('cost') or 0)
            with lock: spent[model['openrouter_id']] += cost
            return dict(text=txt, reasoning_text=msg.get('reasoning'), tokens_in=u.get('prompt_tokens'), tokens_out=u.get('completion_tokens'),
                        tokens_reasoning=(u.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                        tokens_cached=(u.get('prompt_tokens_details') or {}).get('cached_tokens'), usage_raw=u, cost=cost, finish=ch.get('finish_reason'),
                        provider=resp.get('provider'), reasoning_param=rp, max_tokens=body.get('max_tokens'), error=None, http_retries=transient)
        except urllib.error.HTTPError as e:
            err = f'HTTP {e.code}: ' + e.read().decode(errors='replace')[:300]; transient.append(e.code)
            if e.code in (400, 401, 404) and attempt >= 1: break                      # not transient
            if e.code in (402, 429): time.sleep(15 + 10 * attempt + 10 * random.random()); continue   # credit reservation / rate limit: wait it out
        except Exception as e: err = str(e)[:300]; transient.append(type(e).__name__)
        time.sleep(min(60, 3 * 2 ** attempt) + random.random())
    return dict(text='', cost=0, error=err, reasoning_param=rp, http_retries=transient)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='', help='comma list of names or OpenRouter ids from the models file; default = every row')
    ap.add_argument('--models-file', default=f'{HERE}/models_v1.csv')
    ap.add_argument('--sets', default='bank,tails,mirrors,continuous,natcond'); ap.add_argument('--out', required=True)
    ap.add_argument('--reasoning', default='low', choices=['low', 'medium', 'high', 'none'])
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--workers', type=int, default=0, help='override the models-file workers column (0 = use it, else 16)')
    ap.add_argument('--reask', type=int, default=1, help='re-ask a job whose answer did not parse this many times')
    ap.add_argument('--max-tokens', type=int, default=16384, help='max_tokens sent unless the models file overrides it (0 = none)')
    ap.add_argument('--dry-run', action='store_true', help='build the jobs, print counts, make no calls')
    ap.add_argument('--smoke', action='store_true', help='one item per template + one natcond cell per block (~52 calls per model)')
    ap.add_argument('--sample', default='', help='stratified subset of the full job list, e.g. "bank=20,tails=8,mirrors=4,continuous=8,natcond=8"')
    ap.add_argument('--force-effort', default='', choices=['', 'low', 'medium', 'high'], help='override every model: effort level, or 1024/2048/4096 budget for budget models')
    ap.add_argument('--cells', default='', help='JSON file {t1: [item ids], cells: [qid|rev_id]}: run exactly those turn-1 items and natcond cells (one-per-cell preflight)')
    a = ap.parse_args()
    global FORCE; FORCE = a.force_effort or None
    models = load_models(a.models_file)
    if a.models:
        want = a.models.split(','); models = [m for m in models if m['name'] in want or m['openrouter_id'] in want]
        missing = [w for w in want if not any(m['name'] == w or m['openrouter_id'] == w for m in models)]
        if missing: sys.exit(f'not in models file: {missing}')
    jobs = build_jobs(a.sets, a.limit, a.smoke)
    if a.sample: jobs = sample_filter(jobs, a.sample)
    if a.cells:
        sel = json.load(open(a.cells)); t1s = set(sel['t1']); cs = set(sel['cells'])
        jobs = dict(t1=[j for j in jobs['t1'] if j['item'] in t1s], t2=[j for j in jobs['t2'] if j['item'] in cs],
                    nonews=[j for j in jobs['nonews'] if j['qid'] in {c.split('|')[0] for c in cs}], single=[j for j in jobs['single'] if j['item'] in cs])
    print(f"jobs per model: t1={len(jobs['t1'])} t2={len(jobs['t2'])} nonews={len(jobs['nonews'])} single={len(jobs['single'])} "
          f"total={sum(len(v) for v in jobs.values())}; models={len(models)}; reasoning={a.reasoning}", flush=True)
    if a.dry_run: return
    if not KEY: sys.exit('OPENROUTER_API_KEY missing')
    os.makedirs(a.out, exist_ok=True); ck = f'{a.out}/results.jsonl'; done = {}
    if os.path.exists(ck):
        for line in open(ck):
            try: r = json.loads(line); done[(r['model'], r['item'], r['arm'])] = r
            except Exception: pass
    out = open(ck, 'a')
    def emit(row):
        with lock: out.write(json.dumps(row) + '\n'); out.flush()
        done[(row['model'], row['item'], row['arm'])] = row
    for model in models:
        mid = model['openrouter_id']; t0 = time.time()
        def run(job, arm, msgs_fn, extra_fields):
            key = (mid, job['item'], arm)
            if key in done and done[key].get('value') is not None: return done[key]
            row = None
            for attempt in range(1 + a.reask):
                t_call = time.time(); r = call(model, msgs_fn(), a.reasoning, a.max_tokens)
                value, pmode = parse_pct(r['text'], True) if job['kind'] == 'cont' else parse_prob(r['text'], True)
                row = dict(model=mid, item=job['item'], arm=arm, kind=job['kind'], world=job['world'], reasoning=a.reasoning, value=value, parse_mode=pmode, attempts=attempt + 1, ts=time.time(), secs=round(time.time() - t_call, 1), workers=W, **extra_fields, **r)
                if value is not None or r.get('error'): break
            emit(row); return row
        W = a.workers or int((model.get('workers') or '').strip() or 16)
        def primed(ex, jobs_list, arm, msgs_of, extra_of, groupf):
            """Run one job of each prefix group alone first (it writes the provider cache), then the rest of the group in parallel:
            at most one cache miss per (world, kind) instead of one per worker."""
            groups = collections.OrderedDict()
            for j in jobs_list: groups.setdefault(groupf(j), []).append(j)
            firsts, rest = [], []
            for g in groups.values():
                pending = [j for j in g if not ((mid, j['item'], arm) in done and done[(mid, j['item'], arm)].get('value') is not None)]
                if pending: firsts.append(pending[0]); rest.extend(pending[1:])
            # one ordered submission, no barrier: the first job of every prefix group goes first (they write the caches while the
            # pool fills), then everything else in world order. A slow straggler never holds the pool; at most ~W calls at the
            # very start can miss a cache that is still being written.
            list(ex.map(lambda j: run(j, arm, msgs_of(j), extra_of(j)), firsts + rest))
        t1_msgs = lambda j: (lambda: [{"role": "user", "content": user_content(j['prefix'], j['tail'], model)}])
        with ThreadPoolExecutor(W) as ex:
            primed(ex, jobs['t1'], 't1', t1_msgs, lambda j: {}, lambda j: (j['world'], j['kind']))
            primed(ex, jobs['single'], 'single', t1_msgs, lambda j: {}, lambda j: (j['world'], j['kind']))
        t1_by = {j['item']: j for j in jobs['t1']}
        def follow_msgs(job):
            t1 = done.get((mid, job['qid'], 't1')); j1 = t1_by[job['qid']]
            return lambda: [{"role": "user", "content": user_content(j1['prefix'], j1['tail'], model)}, {"role": "assistant", "content": t1['text']}, {"role": "user", "content": job['followup']}]
        def follow_extra(job): return dict(t1_value=(done.get((mid, job['qid'], 't1')) or {}).get('value'))
        with ThreadPoolExecutor(W) as ex:
            for arm in ('t2', 'nonews'):
                ready = [j for j in jobs[arm] if (done.get((mid, j['qid'], 't1')) or {}).get('text')]
                primed(ex, ready, arm, follow_msgs, follow_extra, lambda j: j['world'])
        print(f"  {mid}: workers={W}", flush=True)
        rows = [r for (m, _, _), r in done.items() if m == mid]
        fails = sum(1 for r in rows if r.get('value') is None)
        print(f"{model['name']} ({mid}): {len(rows)} rows, {fails} unparsed/errors, ${spent[mid]:.2f} this session, {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(spent), open(f'{a.out}/spend_session.json', 'w'), indent=1)
    print('done; results in', ck)
if __name__ == '__main__': main()
