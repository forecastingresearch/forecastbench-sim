/**
 * Interventions applied to a running city at a chosen turn ("TURN:ACTION").
 *
 * Timing: an intervention at turn T runs after tick 16(T+1), the branch tick of
 * run_continuations.js --branch-at T+1 (one tick after turn T's stats row at
 * tick 16T+15). Its immediate events are tagged tick 16T+15 (turn T), so
 * (T, H] resolution windows count only what follows it.
 *
 * An intervention that has no effect (no river edge to flood, no nuclear plant
 * to melt down, a monster or tornado already active, nothing to demolish)
 * throws InterventionError.
 */

import { DISASTER_KINDS } from '../../src/lib/engineMessages.ts';

// Zone center tiles (micropolis.h); demolishZones() bulldozes every match.
const POWERPLANT = 750;
const NUCLEAR = 816;
const AIRPORT = 716;

const MESSAGE_BY_KIND = Object.fromEntries(Object.entries(DISASTER_KINDS).map(([num, kind]) => [kind, Number(num)]));

export class InterventionError extends Error {}

const TICKS_PER_TURN = 16;

/** Tick after which an intervention at `turn` runs. */
export const interventionTick = (turn) => (turn + 1) * TICKS_PER_TURN;

/** Tick its immediate events are tagged with: turn's stats row. */
export const interventionEventTick = (turn) => turn * TICKS_PER_TURN + TICKS_PER_TURN - 1;

// A disaster call takes effect iff it sends its own message (synchronously).
const disaster = (method, kind) => ({ run: (m) => m[method](), expectMessage: MESSAGE_BY_KIND[kind] });
const demolish = (...tiles) => ({ run: (m) => ({ demolished: tiles.reduce((n, t) => n + m.demolishZones(t), 0) }) });

const SPECS = {
	makeFire: disaster('makeFire', 'fire'),
	makeFlood: disaster('makeFlood', 'flood'),
	makeTornado: disaster('makeTornado', 'tornado'),
	makeEarthquake: disaster('makeEarthquake', 'earthquake'),
	makeMonster: disaster('makeMonster', 'monster'),
	makeMeltdown: disaster('makeMeltdown', 'meltdown'),
	makeFireBombs: disaster('makeFireBombs', 'firebombing'),
	demolishCoalPower: demolish(POWERPLANT),
	demolishNuclearPower: demolish(NUCLEAR),
	demolishPowerPlants: demolish(POWERPLANT, NUCLEAR),
	demolishAirport: demolish(AIRPORT)
};

for (const [name, spec] of Object.entries(SPECS)) {
	if ('expectMessage' in spec && spec.expectMessage === undefined) throw new Error(`${name}: no message number`);
}

export const ACTIONS = Object.keys(SPECS);

/** Parse "TURN:ACTION" into {turn, action}; throws on a bad spec. */
export function parseIntervention(spec) {
	const match = /^(\d+):(\w+)$/.exec(String(spec));
	if (!match) throw new Error(`Bad intervention "${spec}" (expected TURN:ACTION, e.g. 48:makeEarthquake)`);
	const [, turn, action] = match;
	if (!(action in SPECS)) throw new Error(`Unknown intervention action "${action}" (known: ${ACTIONS.join(', ')})`);
	return { turn: Number(turn), action };
}

/** Parse and sort a list of specs by turn (stable for equal turns). */
export function parseInterventions(specs) {
	return specs.map(parseIntervention).sort((a, b) => a.turn - b.turn);
}

/** File-name tag for a run's interventions, e.g. "-at48_demolishAirport"; "" if none. */
export function interventionsTag(interventions) {
	return interventions.map((iv) => `-at${iv.turn}_${iv.action}`).join('');
}

/**
 * Run one intervention; returns its log result or throws InterventionError.
 * sentMessages() returns the message numbers sent since the previous call.
 */
export function applyIntervention(micropolis, iv, sentMessages) {
	const spec = SPECS[iv.action];
	sentMessages();
	const result = spec.run(micropolis);
	const sent = sentMessages();
	if ('expectMessage' in spec) {
		if (!sent.includes(spec.expectMessage)) {
			throw new InterventionError(`turn ${iv.turn}: ${iv.action} had no effect (engine sent no message ${spec.expectMessage})`);
		}
		return undefined;
	}
	if (!result.demolished) throw new InterventionError(`turn ${iv.turn}: ${iv.action} found nothing to demolish`);
	return result;
}
