#!/usr/bin/env tsx

/**
 * Disaster consequence lab.
 *
 * Settles a city for a while, fires one disaster on demand, then watches the
 * aftermath turn by turn and reports the damage footprint per run: how far the
 * fire spread, how long it burned, how much permanent rubble it left, how much
 * radiation a meltdown scattered, and what the population did.
 *
 * Only the disasters exposed by the WASM bindings can be triggered:
 * makeFire, makeFlood, makeEarthquake, makeMeltdown, makeFireBombs
 * (monster and tornado have no binding). Pass --disaster none for a matched
 * control run — same city, same seed, no disaster — so damage can be read as a
 * difference rather than a level.
 *
 * Usage:
 *   pnpm tsx cli/disaster_lab.js --city haight --disaster makeFire --seeds 30
 *   pnpm tsx cli/disaster_lab.js --cities haight,linecity --disaster makeEarthquake \
 *       --settle-years 2 --observe-turns 96 --out lab.jsonl
 */

import fs from 'node:fs';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import { loadMicropolisMainModule } from '../src/lib/wasm/node.ts';
import { callbackMethodNames } from '../src/lib/wasm/callbacks.ts';
import { heapU16FromEmscriptenModule } from '../src/lib/wasm/heap.ts';

const TICKS_PER_TURN = 16;
const TURNS_PER_YEAR = 48;
const LOMASK = 0x03ff;

const argv = yargs(hideBin(process.argv))
	.scriptName('disaster_lab')
	.option('cities', { type: 'string', default: 'haight', describe: 'Comma-separated builtin city names.' })
	.option('disaster', {
		type: 'string',
		default: 'makeFire',
		choices: ['none', 'makeFire', 'makeFlood', 'makeEarthquake', 'makeMeltdown', 'makeFireBombs'],
		describe: 'Disaster to trigger after the settle period ("none" = matched control).'
	})
	.option('seeds', { type: 'number', default: 20 })
	.option('seed-base', { type: 'number', default: 0 })
	.option('settle-years', { type: 'number', default: 2, describe: 'Years to run before triggering.' })
	.option('observe-turns', { type: 'number', default: 96, describe: 'Turns to watch after the trigger.' })
	.option('out', { type: 'string', describe: 'JSONL output path (default: summary to stdout only).' })
	.help()
	.strict()
	.parseSync();

function census(map) {
	const c = { rubble: 0, flood: 0, rad: 0, fire: 0, road: 0, wire: 0, tree: 0, zones: 0 };
	for (let i = 0; i < map.length; i++) {
		const v = map[i];
		if (v & 0x0400) c.zones++;
		const t = v & LOMASK;
		if (t >= 44 && t <= 47) c.rubble++;
		else if (t >= 48 && t <= 51) c.flood++;
		else if (t === 52) c.rad++;
		else if (t >= 56 && t <= 63) c.fire++;
		else if (t >= 64 && t <= 207) c.road++;
		else if (t >= 208 && t <= 222) c.wire++;
		else if (t >= 21 && t <= 43) c.tree++;
	}
	return c;
}

const quantile = (vals, q) => {
	if (!vals.length) return null;
	const s = [...vals].sort((a, b) => a - b);
	const i = q * (s.length - 1);
	const lo = Math.floor(i), hi = Math.ceil(i);
	return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (i - lo);
};

async function main() {
	const engine = await loadMicropolisMainModule();
	const out = argv.out ? fs.createWriteStream(argv.out, { flags: 'w' }) : null;
	const cities = argv.cities.split(',').map((s) => s.trim());

	for (const city of cities) {
		const results = [];
		for (let s = 1; s <= argv.seeds; s++) {
			const seed = argv.seedBase + s;
			const m = new engine.Micropolis();
			const handlers = {};
			for (const n of callbackMethodNames) handlers[n] = () => {};
			const msgs = [];
			handlers.sendMessage = (_m, _d, num) => msgs.push(num && typeof num === 'object' ? num.value : num);
			m.setCallback(new engine.JSCallback(handlers), {});
			m.init();
			m.seedRandom(seed);
			if (!m.loadCity(`/cities/${city}.cty`)) throw new Error(`load failed: ${city}`);

			const view = () => {
				const heap = heapU16FromEmscriptenModule(engine);
				const start = m.getMapAddress() / 2;
				return heap.subarray(start, start + m.getMapSize() / 2);
			};

			for (let t = 0; t < argv.settleYears * TURNS_PER_YEAR * TICKS_PER_TURN; t++) m.simTick();

			const before = census(view());
			const popBefore = m.cityPop;
			msgs.length = 0;
			if (argv.disaster !== 'none') m[argv.disaster]();

			// Watch the aftermath: fire peak, burn duration, damage accumulation.
			let peakFire = 0, peakFlood = 0, fireEndTurn = null, maxRubble = before.rubble;
			for (let turn = 1; turn <= argv.observeTurns; turn++) {
				for (let t = 0; t < TICKS_PER_TURN; t++) m.simTick();
				const c = census(view());
				peakFire = Math.max(peakFire, c.fire);
				peakFlood = Math.max(peakFlood, c.flood);
				maxRubble = Math.max(maxRubble, c.rubble);
				if (c.fire === 0 && fireEndTurn === null && peakFire > 0) fireEndTurn = turn;
			}
			const after = census(view());

			const row = {
				city, seed, disaster: argv.disaster,
				peakFire, peakFlood, fireEndTurn,
				// fireEndTurn is null while a fire is still burning when the
				// window closes, so quantiles over it alone are censored --
				// always read them next to stillBurning.
				stillBurning: after.fire > 0,
				rubbleDelta: after.rubble - before.rubble,
				maxRubbleDelta: maxRubble - before.rubble,
				radDelta: after.rad - before.rad,
				roadDelta: after.road - before.road,
				wireDelta: after.wire - before.wire,
				treeDelta: after.tree - before.tree,
				zoneDelta: after.zones - before.zones,
				popDelta: m.cityPop - popBefore,
				popBefore, popAfter: m.cityPop,
				pollutionAfter: m.pollutionAverage,
				fireStations: m.fireStationPop,
				msgCounts: msgs.reduce((a, n) => ((a[n] = (a[n] || 0) + 1), a), {})
			};
			results.push(row);
			if (out) out.write(JSON.stringify(row) + '\n');
			m.delete();
		}

		const f = (key) => {
			const v = results.map((r) => r[key]).filter((x) => x !== null);
			return `${quantile(v, 0.1)?.toFixed(0)}/${quantile(v, 0.5)?.toFixed(0)}/${quantile(v, 0.9)?.toFixed(0)}`;
		};
		const anyFire = results.filter((r) => r.peakFire > 0).length;
		const burning = results.filter((r) => r.stillBurning).length;
		console.log(
			`${city.padEnd(12)} ${argv.disaster.padEnd(14)} n=${results.length} fireStations=${results[0].fireStations}\n` +
			`   peakFire p10/50/90:   ${f('peakFire')}   (runs with any fire: ${anyFire}/${results.length})\n` +
			`   rubbleDelta:          ${f('rubbleDelta')}   peakFlood: ${f('peakFlood')}\n` +
			`   radDelta:             ${f('radDelta')}   roadDelta: ${f('roadDelta')}   wireDelta: ${f('wireDelta')}\n` +
			`   popDelta:             ${f('popDelta')}\n` +
			`   burn-out turn (uncensored runs only): ${f('fireEndTurn')}   still burning at turn ${argv.observeTurns}: ${burning}/${results.length}`
		);
	}
	if (out) await new Promise((res, rej) => out.end((e) => (e ? rej(e) : res())));
}

await main();
