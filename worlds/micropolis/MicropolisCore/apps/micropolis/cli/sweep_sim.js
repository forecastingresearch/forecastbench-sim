#!/usr/bin/env tsx

/**
 * Monte Carlo sweep runner for the headless simulation.
 *
 * Runs many (city, seed) combinations in a single process (the WASM module is
 * loaded once, each run gets a fresh Micropolis instance), sampling a yearly
 * snapshot of city stats plus a tile census read straight out of WASM memory,
 * and recording every sendMessage/startEarthquake event. One JSON line is
 * written per run, so downstream analysis can estimate event probabilities
 * ("what fraction of seeds had a flood by year 5?") and compare intervention
 * vs. baseline distributions.
 *
 * Seeding follows run_sim.js: seedRandom() before loadCity(), which is the
 * ordering that makes runs reproducible.
 *
 * Interventions are applied at the start of a chosen turn (before any of that
 * turn's 16 ticks), e.g.:
 *
 *   --at y5:setCityTax=20          # at the start of year 5, raise tax to 20
 *   --at y2:makeEarthquake         # trigger an earthquake 2 years in
 *   --at t96:setFunds=0            # at turn 96 (2 years), empty the treasury
 *
 * Times are 'y<years>' (48 turns each, may be fractional) or 't<turns>'
 * relative to the start of the run. Repeat --at for multiple interventions.
 *
 * Usage:
 *   pnpm tsx cli/sweep_sim.js --cities haight,kobe --seeds 20 --years 20 \
 *       --out /tmp/sweep.jsonl
 *   pnpm tsx cli/sweep_sim.js --cities all --seeds 5 --years 10 --out - | head
 */

import fs from 'node:fs';
import path from 'node:path';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import { loadMicropolisMainModule } from '../src/lib/wasm/node.ts';
import { callbackMethodNames } from '../src/lib/wasm/callbacks.ts';
import { heapU16FromEmscriptenModule } from '../src/lib/wasm/heap.ts';

const TICKS_PER_TURN = 16; // one cityTime increment (see simulate.cpp phaseCycle)
const TURNS_PER_YEAR = 48; // CITYTIMES_PER_YEAR

// Mirrors run_sim.js's BUILTIN_CITIES (the cities baked into the WASM FS).
const BUILTIN_CITIES = [
	'about', 'badnews', 'bluebird', 'bruce', 'deadwood', 'finnigan', 'freds',
	'haight', 'happisle', 'joffburg', 'kamakura', 'kobe', 'kowloon', 'kyoto',
	'linecity', 'med_isle', 'ndulls', 'neatmap', 'radial', 'scenario_bern',
	'scenario_boston', 'scenario_detroit', 'scenario_dullsville',
	'scenario_hamburg', 'scenario_rio_de_janeiro', 'scenario_san_francisco',
	'scenario_tokyo', 'senri', 'southpac', 'splats', 'wetcity', 'yokohama'
];

const argv = yargs(hideBin(process.argv))
	.scriptName('sweep_sim')
	.usage('$0 --cities <a,b|all> --seeds <n> --years <n> --out <file>')
	.option('cities', { type: 'string', default: 'haight', describe: 'Comma-separated builtin city names, or "all".' })
	.option('seeds', { type: 'number', default: 10, describe: 'Number of seeds to run per city (seeds 1..N, offset by --seed-base).' })
	.option('seed-base', { type: 'number', default: 0, describe: 'Offset added to each seed (for sharding sweeps across processes).' })
	.option('years', { type: 'number', default: 20, describe: 'Game years to simulate per run (48 turns each).' })
	.option('sample-every', { type: 'number', default: 48, describe: 'Snapshot cadence in turns (48 = yearly).' })
	.option('at', { type: 'array', default: [], describe: 'Intervention "TIME:ACTION[=ARG]". TIME is y<years> or t<turns> from run start.' })
	.option('disasters', { type: 'boolean', default: true, describe: 'Random disasters on/off (--no-disasters to disable).' })
	.option('game-level', {
		type: 'number',
		describe:
			'Force game level 0..2 (0=easy, 1=medium, 2=hard). Applied AFTER loadCity(), which restores ' +
			'gameLevel from the save file (simLoadInit reads miscHist[15]) and would otherwise clobber it.'
	})
	.option('out', { type: 'string', demandOption: true, describe: 'Output JSONL path, or - for stdout.' })
	.help()
	.strict()
	.parseSync();

const cities = argv.cities === 'all' ? BUILTIN_CITIES : argv.cities.split(',').map((s) => s.trim());
for (const c of cities) {
	if (!BUILTIN_CITIES.includes(c)) {
		console.error(`Unknown city "${c}"`);
		process.exit(1);
	}
}

// setGameLevel takes an embind enum: a raw JS number is silently mis-marshalled
// (it lands on LEVEL_EASY), so numbers are mapped to the enum objects here.
let GAME_LEVELS; // filled in main() once the engine module is loaded

// Intervention actions callable per spec. Numeric-arg actions take the value
// after '='; zero-arg actions must not have one.
const ACTIONS = {
	setCityTax: (m, v) => m.setCityTax(v),
	setFunds: (m, v) => m.setFunds(v),
	setGameLevel: (m, v) => m.setGameLevel(GAME_LEVELS[v]),
	setAutoBudget: (m, v) => m.setAutoBudget(!!v),
	setEnableDisasters: (m, v) => m.setEnableDisasters(!!v),
	makeEarthquake: (m) => m.makeEarthquake(),
	makeFlood: (m) => m.makeFlood(),
	makeFire: (m) => m.makeFire(),
	makeMeltdown: (m) => m.makeMeltdown(),
	makeFireBombs: (m) => m.makeFireBombs()
};

function parseIntervention(spec) {
	const match = /^([ty])([\d.]+):(\w+)(?:=(-?\d+))?$/.exec(String(spec));
	if (!match) {
		console.error(`Bad --at spec "${spec}" (expected e.g. y5:setCityTax=20 or t96:makeEarthquake)`);
		process.exit(1);
	}
	const [, unit, timeStr, action, argStr] = match;
	if (!(action in ACTIONS)) {
		console.error(`Unknown intervention action "${action}" (known: ${Object.keys(ACTIONS).join(', ')})`);
		process.exit(1);
	}
	const turn = Math.round(unit === 'y' ? parseFloat(timeStr) * TURNS_PER_YEAR : parseFloat(timeStr));
	return { turn, action, arg: argStr === undefined ? undefined : Number(argStr) };
}

const interventions = argv.at.map(parseIntervention).sort((a, b) => a.turn - b.turn);

// Tile-census categories over the LOMASK tile number (tool.h / micropolis.h).
// Single linear pass over the 120x100 map view per snapshot.
const LOMASK = 0x03ff;
const ZONEBIT = 0x0400;
function tileCensus(mapData) {
	const c = { water: 0, shore: 0, tree: 0, rubble: 0, flood: 0, rad: 0, fire: 0, road: 0, wire: 0, rail: 0, zones: 0 };
	for (let i = 0; i < mapData.length; i++) {
		const v = mapData[i];
		if (v & ZONEBIT) c.zones++;
		const t = v & LOMASK;
		// River-edge tiles: the only tiles makeFlood() can start a flood from,
		// so flood exposure scales with shore, not water (disasters.cpp).
		if (t >= 5 && t <= 20) c.shore++;
		if (t >= 2 && t <= 20) c.water++;
		else if (t >= 21 && t <= 43) c.tree++;
		else if (t >= 44 && t <= 47) c.rubble++;
		else if (t >= 48 && t <= 51) c.flood++;
		else if (t === 52) c.rad++;
		else if (t >= 56 && t <= 63) c.fire++;
		else if (t >= 64 && t <= 207) c.road++;
		else if (t >= 208 && t <= 222) c.wire++;
		else if (t >= 224 && t <= 238) c.rail++;
	}
	return c;
}

function unwrap(value) {
	if (value && typeof value === 'object' && 'value' in value) return value.value;
	return value;
}

function snapshotRow(m, turn, mapData) {
	return {
		turn,
		cityTime: m.cityTime,
		cityYear: m.cityYear,
		cityMonth: m.cityMonth,
		cityClass: unwrap(m.cityClass),
		cityScore: m.cityScore,
		cityPop: m.cityPop,
		cityPopDelta: m.cityPopDelta,
		totalFunds: m.totalFunds,
		cashFlow: m.cashFlow,
		cityTax: m.cityTax,
		gameLevel: unwrap(m.gameLevel),
		autoBudget: m.autoBudget,
		trafficAverage: m.trafficAverage,
		pollutionAverage: m.pollutionAverage,
		crimeAverage: m.crimeAverage,
		landValueAverage: m.landValueAverage,
		resPop: m.resPop,
		comPop: m.comPop,
		indPop: m.indPop,
		totalPop: m.totalPop,
		roadTotal: m.roadTotal,
		railTotal: m.railTotal,
		policeStationPop: m.policeStationPop,
		fireStationPop: m.fireStationPop,
		hospitalPop: m.hospitalPop,
		churchPop: m.churchPop,
		stadiumPop: m.stadiumPop,
		seaportPop: m.seaportPop,
		airportPop: m.airportPop,
		coalPowerPop: m.coalPowerPop,
		nuclearPowerPop: m.nuclearPowerPop,
		poweredZoneCount: m.poweredZoneCount,
		unpoweredZoneCount: m.unpoweredZoneCount,
		externalMarket: m.externalMarket,
		roadEffect: m.roadEffect,
		policeEffect: m.policeEffect,
		fireEffect: m.fireEffect,
		resValve: m.resValve,
		comValve: m.comValve,
		indValve: m.indValve,
		resCap: m.resCap,
		comCap: m.comCap,
		indCap: m.indCap,
		census: tileCensus(mapData)
	};
}

async function main() {
	const engine = await loadMicropolisMainModule();
	GAME_LEVELS = [engine.GameLevel.LEVEL_EASY, engine.GameLevel.LEVEL_MEDIUM, engine.GameLevel.LEVEL_HARD];
	const out = argv.out === '-' ? process.stdout : fs.createWriteStream(argv.out, { flags: 'w' });
	if (argv.out !== '-') fs.mkdirSync(path.dirname(path.resolve(argv.out)), { recursive: true });

	const totalTurns = Math.round(argv.years * TURNS_PER_YEAR);
	let runCount = 0;

	for (const city of cities) {
		for (let s = 1; s <= argv.seeds; s++) {
			const seed = argv.seedBase + s;
			const micropolis = new engine.Micropolis();

			// Events captured for this run: sendMessage + startEarthquake, with
			// the turn they fired on.
			const events = [];
			let turnRef = 0;
			const handlers = {};
			for (const name of callbackMethodNames) handlers[name] = () => {};
			handlers.sendMessage = (_m, _d, messageNum, x, y, pictureFlag, important) => {
				events.push({ turn: turnRef, msg: unwrap(messageNum), x, y, important });
			};
			handlers.startEarthquake = (_m, _d, strength) => {
				events.push({ turn: turnRef, msg: 'startEarthquake', strength });
			};
			const callback = new engine.JSCallback(handlers);

			micropolis.setCallback(callback, {});
			micropolis.init();
			if (!argv.disasters) micropolis.enableDisasters = false;
			micropolis.seedRandom(seed);
			if (!micropolis.loadCity(`/cities/${city}.cty`)) {
				console.error(`Failed to load ${city}`);
				process.exit(1);
			}

			// After the load, not before: loadCity() -> simLoadInit() restores
			// gameLevel from miscHist[15], so a level set earlier is discarded.
			if (argv.gameLevel !== undefined) micropolis.setGameLevel(GAME_LEVELS[argv.gameLevel]);

			// The heap can grow, so re-derive the view each run (and, cheaply,
			// on each snapshot via the closure below).
			const mapView = () => {
				const heap = heapU16FromEmscriptenModule(engine);
				const start = micropolis.getMapAddress() / 2;
				return heap.subarray(start, start + micropolis.getMapSize() / 2);
			};

			const startCityTime = micropolis.cityTime;
			const rows = [snapshotRow(micropolis, 0, mapView())];

			let nextIntervention = 0;
			const applied = [];
			for (let turn = 1; turn <= totalTurns; turn++) {
				turnRef = turn;
				while (nextIntervention < interventions.length && interventions[nextIntervention].turn <= turn - 1) {
					const iv = interventions[nextIntervention++];
					ACTIONS[iv.action](micropolis, iv.arg);
					applied.push(iv);
				}
				for (let t = 0; t < TICKS_PER_TURN; t++) micropolis.simTick();
				if (turn % argv.sampleEvery === 0 || turn === totalTurns) {
					rows.push(snapshotRow(micropolis, turn, mapView()));
				}
			}

			out.write(
				JSON.stringify({
					city,
					seed,
					disasters: argv.disasters,
					interventions: applied,
					startCityTime,
					startYear: rows[0].cityYear,
					rows,
					events
				}) + '\n'
			);

			micropolis.delete();
			try {
				callback.delete();
			} catch {
				/* embind objects without delete() are fine to leak in a CLI */
			}
			runCount++;
			if (argv.out !== '-') process.stderr.write(`\r${runCount} runs done (${city} seed ${seed})   `);
		}
	}

	if (argv.out !== '-') {
		process.stderr.write('\n');
		await new Promise((resolve, reject) => out.end((err) => (err ? reject(err) : resolve())));
	}
	console.error(`Wrote ${runCount} runs`);
}

await main();
