#!/usr/bin/env python3
"""report_draw.py DRAW_DIR OUT_HTML — composition tables + QC samples as a single HTML page."""
import json, sys, random, html, collections, pandas as pd, numpy as np
D, OUTP = sys.argv[1], sys.argv[2]; rng = random.Random(11)
bank = pd.DataFrame(json.load(open(f'{D}/bank_750.json'))); tails = pd.DataFrame(json.load(open(f'{D}/tails_300.json')))
mirrors = pd.DataFrame(json.load(open(f'{D}/mirrors_50.json'))); cont = pd.DataFrame(json.load(open(f'{D}/continuous_300.json')))
nc = pd.DataFrame(json.load(open(f'{D}/natcond_600.json')))
BANDS = [(0.05, 0.23), (0.23, 0.41), (0.41, 0.59), (0.59, 0.77), (0.77, 0.95)]
bname = lambda q: next(f"{lo:.2f}–{hi:.2f}" for lo, hi in BANDS if lo < q <= hi)
bank['band'] = bank.qAll.apply(bname)
nc['abs'] = nc.delta.abs(); nc['stratum'] = pd.cut(nc['abs'], [0, .03, .08, .15, 1.01], labels=['null <.03', 'small .03–.08', 'medium .08–.15', 'large ≥.15'], right=False)
FAM = {'EX_comparative': 'comparative (pop/cities/territory/score)', 'EX_government_at': 'government at T', 'EX_tech_discovered': 'tech discovered', 'NB1_threshold': 'tech threshold', 'NB1_value_threshold': 'value threshold (score/treasury/pop/cities/territory/military)', 'NB4_drawdown': 'drawdown', 'NB6_event': 'world conquest count', 'NEW_civil_war': 'civil war by T', 'NW1_war_at': 'war at T', 'NW2_diplo': 'diplomatic state reached', 'NW4_survival': 'eliminated by T', 'NW5_wonder': 'wonder completed (any)', 'S2_war_end': 'war end', 'S3_wars_at_T': 'wars at T', 'S4_gov_change_count': 'gov-change count', 'S5_civ_wonders': 'civ wonders ≥ k', 'S6_city_founding': 'city founding ≥ k', 'S7_any_destroyed': 'any city razed', 'W1_wonder_race': 'wonder race', 'W2_directed_conquest': 'directed conquest', 'W3_capture_k': 'captures ≥ k', 'W4_lose_k': 'losses ≥ k', 'W5_tech_lead': 'tech lead', 'W6_peace_at': 'peace at T',
       'P1_civ_conquests': 'civ captures', 'P2_civ_losses': 'civ losses', 'P3_civ_founds': 'civ founds', 'P5_techs_at_T': 'techs at T', 'P6_world_techs': 'world techs', 'NC5_world_captures': 'world captures', 'NC14_world_wonders': 'world wonders', 'S7_world_razings': 'world razings', 'P7_first_capture': 'first capture (turn)', 'P8_first_wonder': 'first wonder (turn)', 'P9_time_to_K': 'time to K techs (turn)'}
BLOCK = {'A': 'A · irrelevant news', 'B': 'B · related, weak news', 'C1': 'C1 · mechanical, same series', 'C2': 'C2 · persistence, cross-series', 'D': 'D · inferential'}
def tbl(ct, cls='num'):
    ct = ct.copy(); ct.index = [FAM.get(i, BLOCK.get(i, str(i))) if not isinstance(i, tuple) else ' / '.join(map(str, i)) for i in ct.index]
    h = ct.to_html(classes=cls, border=0, escape=True); return f'<div class="tw">{h}</div>'
def esc(s): return html.escape(str(s))
def samp(df, k): return df.sample(min(k, len(df)), random_state=rng.randint(0, 10**6))
def crit_line(r):
    c = r.get('criteria') if hasattr(r, 'get') else None
    return f'<span class="crit">{esc(c)}</span>' if isinstance(c, str) and c else ''
def qc(iid, setname):
    iid = iid.replace(':', '_')
    return (f'<label class="flag"><input type="checkbox" data-id="{esc(iid)}" data-set="{esc(setname)}" disabled> issue</label>'
            f'<input class="why" type="text" data-id="{esc(iid)}" placeholder="why" disabled>')
parts = []
parts.append(f"""<header><p class="eyebrow">FreeCiv arm · fbsim v3 corpus · draw v1 · {pd.Timestamp.utcnow():%Y-%m-%d}</p><h1>FreeCiv Draw v1</h1>
<p class="lede">Stratified random draws from the 8 × 1,000 rerun corpus. Truth for every item is the frequency over all 1,000 replays. Strata are structural (horizon, probability band, family, reveal type, relation); no item was chosen by its measured effect.</p>
<div class="cards"><div><b>{len(bank)}</b><span>binary bank</span></div><div><b>{len(tails)}</b><span>tails q ≤ .05</span></div><div><b>{len(mirrors)}</b><span>mirrors q ≥ .95</span></div><div><b>{len(cont)}</b><span>continuous</span></div><div><b>{len(nc)}</b><span>natural conditionals</span></div></div>
<nav><a href="#bank">Bank</a><a href="#tails">Tails</a><a href="#mirrors">Mirrors</a><a href="#cont">Continuous</a><a href="#nc">Conditionals</a><a href="#qc">QC samples</a></nav></header>""")
parts.append('<section id="bank"><h2>Binary bank · 750</h2><p>150 per horizon = 5 probability bands over (0.05, 0.95) × 30. Caps: ≤ 6 per world and ≤ 6 per family in each band-cell, one item per subject per family per horizon, ≤ 150 per family overall.</p>')
parts.append('<h3>Family × horizon</h3>' + tbl(pd.crosstab(bank.family, bank['T'], margins=True, margins_name='all')))
parts.append('<h3>Band × horizon</h3>' + tbl(pd.crosstab(bank.band, bank['T'], margins=True, margins_name='all')))
parts.append('<h3>World × horizon</h3>' + tbl(pd.crosstab(bank.world, bank['T'], margins=True, margins_name='all')) + '</section>')
parts.append(f'<section id="tails"><h2>Tails · 300</h2><p>60 per horizon, 0 &lt; q ≤ 0.05, long horizons drawn first. Caps: ≤ 10 per family per horizon, ≤ 45 per family overall, ≤ 12 per world per horizon. Median q {tails.qAll.median():.3f}; 10th–90th percentile {tails.qAll.quantile(.1):.3f}–{tails.qAll.quantile(.9):.3f}. Scored in bits.</p>' + tbl(pd.crosstab(tails.family, tails['T'], margins=True, margins_name='all')) + '</section>')
parts.append('<section id="mirrors"><h2>Mirrors · 50</h2><p>10 per horizon, 0.95 ≤ q &lt; 1, same families as the tails; our own top-end calibration probe.</p>' + tbl(pd.crosstab(mirrors.family, mirrors['T'], margins=True, margins_name='all')) + '</section>')
parts.append('<section id="cont"><h2>Continuous · 300</h2><p>240 spread over horizons (6 per family per horizon, 8 families, IQR &gt; 0) plus a 60-item timing block anchored at T210 (time to K techs 40, first capture 15, first wonder 5). Elicited as percentiles, scored by CRPS against the 1,000-replay distribution.</p>' + tbl(pd.crosstab(cont.family, cont['T'], margins=True, margins_name='all')))
parts.append('<h3>Dispersion (median IQR of the truth distribution, by family)</h3>' + tbl(cont.groupby('family').iqrAll.median().round(1).to_frame('median IQR')) + '</section>')
parts.append(f'<section id="nc"><h2>Natural conditionals · {len(nc)}</h2><p>Second-turn reveals on bank questions at T120–T210. Reveal window is turns 61–90, one uniform sentence, exact turn undisclosed, at least 100 replays where the reveal holds. Blocks are structural; quota per horizon A 30 · B 20 · C1 20 · C2 50 · D 30, template-balanced inside each block, ≤ 2 distinct reveals per question, ≤ 8 cells per revealed event. Effect sizes below are reported after the draw and were never used for it.</p>')
parts.append('<h3>Block × horizon</h3>' + tbl(pd.crosstab(nc.block, nc['T'], margins=True, margins_name='all')))
parts.append('<h3>Effect size × block (post hoc)</h3>' + tbl(pd.crosstab(nc.block, nc.stratum, margins=True, margins_name='all')))
parts.append('<h3>Effect size × horizon</h3>' + tbl(pd.crosstab(nc['T'], nc.stratum, margins=True, margins_name='all')))
parts.append('<h3>Question family × block</h3>' + tbl(pd.crosstab(nc.family, nc.block, margins=True, margins_name='all')))
parts.append('<h3>Reveal type × block</h3>' + tbl(pd.crosstab(nc.rev_kind, nc.block, margins=True, margins_name='all')))
big = nc[nc['abs'] >= .08]; parts.append(f'<p class="note">Substantive effects (|Δ| ≥ 0.08): {len(big)} of {len(nc)} ({len(big)/len(nc):.0%}); {int((big.delta>0).sum())} positive, {int((big.delta<0).sum())} negative. Controls assigned: no-news {int(nc.control_no_news.sum())}, single-prompt {int(nc.control_single_prompt.sum())}. Blocks C1 and C2 are filled from the bank first and topped up from the value-series pool outside the bank (same bands and caps); those top-up questions get their own turn-1 elicitation and are listed in natcond_extra_turn1.json. Cells from the bank: {int(nc.from_bank.sum())}; from the top-up pool: {int((~nc.from_bank).sum())}.</p></section>')
# QC samples
parts.append('<section id="qc"><h2>QC samples</h2><p>Two or three items drawn at random from every cell. Numbers are the all-replay truth. Tick <b>issue</b> on anything that looks wrong and say why; your marks are saved as you type and I read them back. <span id="qcstatus" class="note">Connecting to the shared record…</span></p>')
parts.append('<h3>Bank · band × horizon</h3>')
for band in sorted(bank.band.unique()):
    for T in (90, 120, 150, 180, 210):
        cell = bank[(bank.band == band) & (bank['T'] == T)]; parts.append(f'<div class="cell"><h4>band {band} · T{T}</h4><ul>')
        for _, r in samp(cell, 3).iterrows(): parts.append(f'<li><span class="q">{esc(r.text)}</span><span class="meta">{esc(r.world)} · {esc(FAM.get(r.family, r.family))}</span><span class="val">{r.qAll:.3f}</span></li>')
        parts.append('</ul></div>')
parts.append('<h3>Tails · horizon</h3>')
for T in (90, 120, 150, 180, 210):
    cell = tails[tails['T'] == T]; parts.append(f'<div class="cell"><h4>T{T}</h4><ul>')
    for _, r in samp(cell, 3).iterrows(): parts.append(f'<li><span class="q">{esc(r.text)}</span><span class="meta">{esc(r.world)} · {esc(FAM.get(r.family, r.family))}</span><span class="val">{r.qAll:.3f}</span>{crit_line(r)}<span class="ctl">{qc(r.id, "tails")}</span></li>')
    parts.append('</ul></div>')
parts.append('<h3>Mirrors · horizon</h3>')
for T in (90, 120, 150, 180, 210):
    cell = mirrors[mirrors['T'] == T]; parts.append(f'<div class="cell"><h4>T{T}</h4><ul>')
    for _, r in samp(cell, 2).iterrows(): parts.append(f'<li><span class="q">{esc(r.text)}</span><span class="meta">{esc(r.world)} · {esc(FAM.get(r.family, r.family))}</span><span class="val">{r.qAll:.3f}</span>{crit_line(r)}<span class="ctl">{qc(r.id, "mirrors")}</span></li>')
    parts.append('</ul></div>')
parts.append('<h3>Continuous · family × horizon</h3>')
for fam in sorted(cont.family.unique(), key=lambda f: list(FAM).index(f) if f in FAM else 99):
    for T in sorted(cont[cont.family == fam]['T'].unique()):
        cell = cont[(cont.family == fam) & (cont['T'] == T)]; parts.append(f'<div class="cell"><h4>{esc(FAM.get(fam, fam))} · T{T}</h4><ul>')
        for _, r in samp(cell, 2).iterrows():
            v = f'median {r.medAll:g} · IQR {r.iqrAll:g} · p05–p95 {round(r.p05):g}–{round(r.p95):g}' + (f' · never {r.censAll:.0%}' if r.censAll > 0 else '')
            parts.append(f'<li><span class="q">{esc(r.text)}</span><span class="meta">{esc(r.world)}</span><span class="val small">{esc(v)}</span>{crit_line(r)}<span class="ctl">{qc(r.id, "continuous")}</span></li>')
        parts.append('</ul></div>')
parts.append('<h3>Natural conditionals · block × horizon</h3>')
for b in ('A', 'B', 'C1', 'C2', 'D'):
    for T in (120, 150, 180, 210):
        cell = nc[(nc.block == b) & (nc['T'] == T)]; parts.append(f'<div class="cell"><h4>{esc(BLOCK[b])} · T{T}</h4><ul>')
        for _, r in samp(cell, 3).iterrows(): parts.append(f'<li><span class="q">{esc(r.question)}</span><span class="rev">news: {esc(r.reveal)}</span><span class="meta">{esc(r.world)} · {esc(FAM.get(r.family, r.family))} · n<sub>x</sub>={r.nx}</span><span class="val">{r.p:.2f} → {r.p_given:.2f} <em>({r.delta:+.2f})</em></span>{crit_line(r)}<span class="ctl">{qc(r.qid + "__" + r.rev_id, "natcond")}</span></li>')
        parts.append('</ul></div>')
parts.append('</section><footer>Files: tmp/fbsim_v3_corpus/draw_v1/{bank_750,tails_300,mirrors_50,continuous_300,natcond_600}.json · COMPOSITION.md · QC_SAMPLES.md. Draw seed 2026; QC sample seed 11.</footer>')
CSS = """<title>FreeCiv Draw v1</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bitter:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#F3F5F7;--surface:#FFFFFF;--ink:#1C2430;--muted:#5B6774;--line:#D6DDE3;--accent:#0E6E73;--accent-soft:#D9EEEF;--head:#22303D}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121A21;--surface:#1A242D;--ink:#E6ECF0;--muted:#9AA8B4;--line:#2C3944;--accent:#4FB8BD;--accent-soft:#173A3D;--head:#F2F6F8}}
:root[data-theme="dark"]{--bg:#121A21;--surface:#1A242D;--ink:#E6ECF0;--muted:#9AA8B4;--line:#2C3944;--accent:#4FB8BD;--accent-soft:#173A3D;--head:#F2F6F8}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:15px;line-height:1.5;margin:0}
main{max-width:920px;margin:0 auto;padding:40px 24px 80px}
header{margin-bottom:36px}.eyebrow{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
h1{font-family:Bitter,Georgia,serif;font-weight:700;font-size:40px;line-height:1.1;margin:0 0 14px;color:var(--head);text-wrap:balance}
h2{font-family:Bitter,Georgia,serif;font-weight:700;font-size:26px;margin:48px 0 8px;color:var(--head);text-wrap:balance}
h3{font-family:Bitter,Georgia,serif;font-weight:500;font-size:18px;margin:26px 0 8px;color:var(--head)}
h4{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-weight:500;font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--accent);margin:0 0 6px}
.lede{font-size:17px;max-width:66ch;color:var(--ink);margin:0 0 22px}
p{max-width:70ch}.note{color:var(--muted);font-size:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:0 0 22px}
.cards div{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px 16px;display:flex;flex-direction:column;gap:2px}
.cards b{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:26px;font-weight:500;color:var(--accent);font-variant-numeric:tabular-nums}
.cards span{font-size:13px;color:var(--muted)}
nav{display:flex;gap:18px;flex-wrap:wrap;font-size:14px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:10px 0}
nav a{color:var(--accent);text-decoration:none;font-weight:500}nav a:hover,nav a:focus-visible{text-decoration:underline;outline:none}
.tw{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:6px;margin:0 0 18px}
table.num{border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums;min-width:100%}
table.num th,table.num td{padding:6px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
table.num th{font-weight:600;color:var(--muted);font-size:12px;letter-spacing:.03em}table.num tbody th{text-align:left;color:var(--ink);font-weight:500}
table.num tr:last-child td,table.num tr:last-child th{border-bottom:none}table.num tbody tr:last-child{background:var(--accent-soft)}
.cell{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px 16px;margin:0 0 10px}
.cell ul{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px}
.cell li{display:grid;grid-template-columns:1fr auto;grid-template-areas:"q val" "rev val" "meta val";column-gap:16px;align-items:start;padding-top:8px;border-top:1px dashed var(--line)}
.cell li:first-child{border-top:none;padding-top:0}
.q{grid-area:q}.rev{grid-area:rev;color:var(--accent);font-size:14px}.meta{grid-area:meta;color:var(--muted);font-size:12.5px}
.val{grid-area:val;font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:14px;color:var(--head);text-align:right;font-variant-numeric:tabular-nums}
.val.small{font-size:12.5px;max-width:200px;white-space:normal}.val em{color:var(--muted);font-style:normal}
.cell li{grid-template-columns:1fr auto;grid-template-areas:"q val" "rev val" "meta val" "crit crit" "ctl ctl"}
.crit{grid-area:crit;font-size:12.5px;color:var(--muted);line-height:1.4;max-width:80ch}
.ctl{grid-area:ctl;display:flex;gap:10px;align-items:center;margin-top:2px}
.flag{font-size:12.5px;color:var(--muted);display:inline-flex;gap:6px;align-items:center;white-space:nowrap}
.flag input{accent-color:var(--accent);width:15px;height:15px}
.why{flex:1;min-width:120px;font:inherit;font-size:13px;padding:4px 8px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--ink)}
.why:disabled{opacity:.5}.why:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
li.flagged{background:color-mix(in srgb,var(--accent-soft) 60%,transparent);border-radius:4px;padding-left:6px;padding-right:6px}
#qcstatus b{color:var(--accent)}
footer{margin-top:48px;padding-top:16px;border-top:1px solid var(--line);font-size:12.5px;color:var(--muted);font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-reduced-motion: reduce){*{scroll-behavior:auto}}
</style>"""
JS = '''<script>
(async () => {
  const status = document.getElementById('qcstatus');
  const boxes = Array.from(document.querySelectorAll('input[type=checkbox][data-id]'));
  const whys = Array.from(document.querySelectorAll('input.why[data-id]'));
  const byId = {}; boxes.forEach(b => { byId[b.dataset.id] = byId[b.dataset.id] || {}; byId[b.dataset.id].box = b; });
  whys.forEach(w => { byId[w.dataset.id] = byId[w.dataset.id] || {}; byId[w.dataset.id].why = w; });
  const textOf = (el) => { const li = el.closest('li'); return li ? (li.querySelector('.q') || {}).textContent || '' : ''; };
  const db = window.claude && window.claude.use ? await window.claude.use('db') : null;
  if (!db) { status.textContent = 'Shared record unavailable in this view; marks cannot be saved here.'; return; }
  const enable = () => { boxes.forEach(b => b.disabled = false); whys.forEach(w => w.disabled = false); };
  const paint = () => { let n = 0; Object.values(byId).forEach(o => { const on = o.box && o.box.checked; if (on) n++; const li = o.box && o.box.closest('li'); if (li) li.classList.toggle('flagged', !!on); }); status.innerHTML = '<b>' + n + '</b> item' + (n === 1 ? '' : 's') + ' flagged so far.'; };
  try {
    const snap = await db.collection('qc').limit(1000).get();
    snap.docs.forEach(d => { const v = d.data(); const o = byId[d.id]; if (!o || !v) return; if (o.box) o.box.checked = !!v.flag; if (o.why) o.why.value = v.reason || ''; });
  } catch (e) { status.textContent = 'Could not load saved marks (' + (e && e.code || 'error') + '); new marks will still be saved.'; }
  enable(); paint();
  const timers = {};
  const save = (id) => {
    const o = byId[id]; if (!o) return;
    clearTimeout(timers[id]);
    timers[id] = setTimeout(async () => {
      const body = { flag: !!(o.box && o.box.checked), reason: (o.why && o.why.value || '').slice(0, 2000), set: o.box ? o.box.dataset.set : '', text: textOf(o.box || o.why).slice(0, 400), ts: new Date().toISOString() };
      try { await db.doc('qc/' + id).set(body); paint(); }
      catch (e) { status.textContent = 'Save failed (' + (e && e.code || 'error') + '); try again.'; }
    }, 400);
  };
  boxes.forEach(b => b.addEventListener('change', () => { paint(); save(b.dataset.id); }));
  whys.forEach(w => w.addEventListener('input', () => save(w.dataset.id)));
})();
</script>'''
open(OUTP, 'w').write(CSS + '<main>' + '\n'.join(parts) + '</main>' + JS); print('wrote', OUTP, len(parts), 'parts')
