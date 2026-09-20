#!/usr/bin/env python3
"""criteria_v1.py DRAW_DIR BANK_DIR — attach resolution criteria to every item (keyed on family + params, never on
text) and export the full 1,000-replay value distribution for every continuous item.

Fairness rule: criteria state how WE resolve — which recorded quantity, the exact turn/window semantics, ties,
what counts — and nothing that teaches game mechanics a model is expected to know."""
import json, os, sys, pickle, re
D, B = sys.argv[1], sys.argv[2]
W = {fn[:-4]: pickle.load(open(os.path.join(B, fn), 'rb')) for fn in os.listdir(B) if fn.endswith('.pkl')}
def name(world, cid): return W[world]['civ_names'].get(str(cid), str(cid))
WIN = "on any turn from 61 through {T} inclusive (an event on turn {T} itself counts)"
AT = "as recorded at the end of turn {T}"
NOUN = {'scores': 'score', 'population': 'population', 'cities_count': 'number of cities', 'territory_size': 'territory size', 'treasury': 'treasury', 'military_units_count': 'number of military units', 'techs_known': 'number of technologies'}
REPORTED = "the same quantity the world report calls {label}"
LABEL = {'scores': 'Score', 'population': 'Population', 'cities_count': 'Cities', 'territory_size': 'Territory', 'treasury': 'Treasury', 'military_units_count': 'Military', 'techs_known': 'Techs'}
CAPTURE_DEF = "A capture is a city changing owner by force. Captures of cities held by barbarians count. Each capture event counts separately, so a city captured, lost and captured again counts twice. Cities that change hands when a civilization splits in a civil war are not captures."
LOSS_DEF = "Losses to any player count, including to barbarians. Each loss event counts separately, so a city lost, retaken and lost again counts twice. A city destroyed rather than captured, or lost through a civil war split, does not count."
ELIM = ""
TECHS_ANCHOR = "the count the world report calls Techs; FreeCiv's initial 'None' entry is not a technology and is not counted"
def crit(i):
    f, T, s = i['family'], i['T'], i.get('subj', []); w = i['world']; win = WIN.format(T=T); at = AT.format(T=T)
    n = [name(w, c) for c in s]
    if f == 'NW1_war_at': return f"Resolves YES if the diplomatic state between {n[0]} and {n[1]} {at} is War. Cease-fire, armistice, peace, alliance, or never having met all resolve NO."
    if f == 'W6_peace_at': return f"Resolves YES if the diplomatic state between {n[0]} and {n[1]} {at} is Peace. War, cease-fire, armistice, alliance, or never having met all resolve NO."
    if f == 'NW2_diplo':
        st = {'peace_by': ('Peace', 'war, cease-fire, armistice and alliance'), 'alliance_by': ('Alliance', 'every other state'), 'ceasefire_by': ('Cease-fire or Armistice', 'war, peace and alliance')}[i['kind']]
        return f"Resolves YES if the diplomatic state between {n[0]} and {n[1]} is {st[0]} at the end of any turn from 61 through {T} inclusive. Resolves NO if the state never occurs in that window ({st[1]} do not count)."
    if f == 'S3_wars_at_T': return f"Resolves YES if, {at}, at least {i['k']} of the ten pairs among the five civilizations are in the diplomatic state War. Exactly {i['k']} counts."
    if f == 'EX_government_at': return f"Resolves YES if {n[0]}'s government {at} is {i['gov']}. A change of government completed on turn {T} counts. Anarchy counts as a form of government. Resolves NO for any other government."
    if f == 'S4_gov_change_count': return f"Resolves YES if {n[0]}'s government changes at least {i['k']} time{'s' if i['k'] > 1 else ''} {win}. Every recorded change counts, including a change into anarchy and a change out of it, so one revolution normally counts twice. Exactly {i['k']} counts."
    if f == 'EX_tech_discovered': return f"Resolves YES if {n[0]} discovers {i['tech']} {win}. Resolves NO if it is not discovered by the end of turn {T}."
    if f == 'W5_tech_lead': return f"Resolves YES if, {at}, {n[0]} knows strictly more technologies than each of the other four civilizations, counting every technology known including those known at turn 60. This is {TECHS_ANCHOR}. Any tie for the most resolves NO."
    if f in ('NB1_threshold', 'NB1_value_threshold'):
        m = i['metric']
        if m == 'techs_known': return f"Resolves YES if {n[0]}'s number of technologies {at} is {i['x']} or more; exactly {i['x']} counts. This is {TECHS_ANCHOR}, counting technologies known at turn 60."
        return f"Resolves YES if {n[0]}'s {NOUN[m]} {at} is {i['x']} or more; exactly {i['x']} counts. This is {REPORTED.format(label=LABEL[m])}."
    if f == 'EX_comparative':
        m = i['metric']; return f"Resolves YES if {n[0]}'s {NOUN[m]} {at} is strictly greater than {n[1]}'s. Equal values resolve NO. The quantity is {REPORTED.format(label=LABEL[m])}."
    if f == 'NB4_drawdown':
        m = i['metric']; pct = int(i['f'] * 100); return f"Resolves YES if, at the end of any turn from 61 through {T} inclusive, {n[0]}'s {NOUN[m]} is below {pct}% of its value at the end of turn 60 (the value shown in the world report). Being exactly at {pct}% does not count. A later recovery does not change the result."
    if f == 'NW5_wonder': return f"Resolves YES if any civilization completes {i['wonder']} {win}. Resolves NO if it has not been completed by the end of turn {T}."
    if f == 'W1_wonder_race': return f"Resolves YES if {n[0]} completes {i['wonder']} {win}. A great wonder can be completed only once per game, so this resolves NO if another civilization completes it, or if it is not completed by the end of turn {T}."
    if f == 'S5_civ_wonders': return f"Resolves YES if {n[0]} completes at least {i['k']} great wonder{'s' if i['k'] > 1 else ''} {win}. Exactly {i['k']} counts. Small wonders such as a palace do not count."
    if f == 'S6_city_founding': return f"Resolves YES if {n[0]} founds at least {i['k']} new cit{'ies' if i['k'] > 1 else 'y'} {win}. Cities held at turn 60 and cities captured do not count. Exactly {i['k']} counts."
    if f == 'W3_capture_k': return f"Resolves YES if {n[0]} captures at least {i['k']} cit{'ies' if i['k'] > 1 else 'y'} {win}. {CAPTURE_DEF} Exactly {i['k']} counts."
    if f == 'W4_lose_k': return f"Resolves YES if {n[0]} loses at least {i['k']} cit{'ies' if i['k'] > 1 else 'y'} to capture {win}. {LOSS_DEF} Exactly {i['k']} counts."
    if f == 'W2_directed_conquest': return f"Resolves YES if {n[0]} captures at least one city whose owner at the moment of capture is {n[1]}, {win}. Recaptures count. Resolves NO otherwise."
    if f == 'NB6_event':
        if i['kind'] == 'any': return f"Resolves YES if any city is captured by any player {win}. {CAPTURE_DEF}"
        k = int(i['kind'][2:]); return f"Resolves YES if at least {k} city captures occur in total across all players {win}. {CAPTURE_DEF} Exactly {k} counts."
    if f == 'S7_any_destroyed': return f"Resolves YES if any city is destroyed {win}. A city is destroyed when it ceases to exist; a city that is captured and survives does not count."
    if f == 'NW4_survival': return f"Resolves YES if {n[0]} is eliminated {win}, meaning it ceases to exist as a civilization. Losing cities, losing the capital, or a civil war split without elimination resolve NO."
    if f == 'NEW_civil_war': return f"Resolves YES if {n[0]} splits in a civil war {win}, meaning a new civilization is created out of some of {n[0]}'s cities. Anarchy, disorder, or the loss of cities to other players without such a split resolve NO."
    # continuous
    if f == 'NC1_value_at_T':
        m = i['metric']; return f"Resolves to {n[0]}'s {NOUN[m]} {at}, {REPORTED.format(label=LABEL[m])}. A whole number."
    if f == 'P5_techs_at_T': return f"Resolves to the number of technologies {n[0]} knows {at}, counting those known at turn 60. This is {TECHS_ANCHOR}. A whole number."
    if f == 'P6_world_techs': return f"Resolves to the sum, over the five civilizations, of the number of technologies each knows {at}, counting those known at turn 60. This is {TECHS_ANCHOR}. A whole number."
    if f == 'P1_civ_conquests': return f"Resolves to the number of cities {n[0]} captures {win}. {CAPTURE_DEF}"
    if f == 'P2_civ_losses': return f"Resolves to the number of cities {n[0]} loses to capture {win}. {LOSS_DEF}"
    if f == 'P3_civ_founds': return f"Resolves to the number of new cities {n[0]} founds {win}. Cities held at turn 60 and captured cities do not count."
    if f == 'NC5_world_captures': return f"Resolves to the total number of city captures by all players {win}. {CAPTURE_DEF}"
    if f == 'NC14_world_wonders': return f"Resolves to the number of great wonders completed by all civilizations {win}. Small wonders such as a palace do not count."
    if f == 'S7_world_razings': return f"Resolves to the number of cities destroyed {win}. A city is destroyed when it ceases to exist; a captured city that survives does not count."
    raise KeyError(f)
counts = {}
for fn in ('bank_750', 'tails_300', 'mirrors_50', 'natcond_extra_turn1', 'continuous_300'):
    items = json.load(open(f'{D}/{fn}.json'))
    for i in items:
        i['criteria'] = crit(i)
        if fn == 'continuous_300':
            w, idx = i['id'].split(':c'); vals = W[w]['V'][int(idx)]
            i['values'] = [None if v != v else (int(v) if float(v).is_integer() else float(v)) for v in vals.tolist()]
    json.dump(items, open(f'{D}/{fn}.json', 'w'), indent=0); counts[fn] = len(items)
# natcond cells: criteria of the underlying question + the reveal preamble used at turn 2
by_id = {}
for fn in ('bank_750', 'natcond_extra_turn1'):
    for i in json.load(open(f'{D}/{fn}.json')): by_id[i['id']] = i
nc = json.load(open(f'{D}/natcond_600.json'))
for c in nc:
    c['criteria'] = by_id[c['qid']]['criteria']
    c['turn2_preamble'] = "One fact about turns 61 through 90 has been revealed to you: " + c['reveal']
json.dump(nc, open(f'{D}/natcond_600.json', 'w'), indent=0); counts['natcond_600'] = len(nc)
print('criteria attached:', counts)
