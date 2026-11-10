#!/usr/bin/env tsx

/**
 * Branching headless simulation runner.
 *
 * Runs one "trunk" simulation (seeded before loadCity(), like run_sim.js) for S
 * turns, snapshots the entire WASM linear memory at that point, and then runs
 * many "continuations": each restores the snapshot byte-for-byte, reseeds the
 * engine RNG with a different seed, and ticks forward H more turns. All output
 * is streamed to stdout as JSONL so a consumer can pipe it without touching
 * disk, e.g. to estimate P(event within H turns | state at S).
 *
 * Why a heap snapshot rather than saveCity()/loadCity(): the .cty format does
 * not hold the full engine state (sprites, disaster timers, ...), and loadCity()
 * itself consumes the RNG and rewrites map tiles, so continuations would not
 * start from the true state at S. Copying linear memory restores every byte,
 * including malloc metadata, so the engine is exactly where it was at S.
 *
 * Stream protocol (one JSON object per line, every line has a "kind"):
 *   header  — once at the start: city, trunkSeed, branchAt, horizon, seed range
 *   begin   — start of one run ("seed": null for the trunk, else the branch seed)
 *   stats   — one per turn (the same fields run_sim.js writes to log-seed*.jsonl)
 *   event   — one per engine callback (same fields as events-seed*.jsonl)
 *   end     — end of one run, with the number of stats rows written
 * Every begin/stats/event/end line also carries "seed" so lines can be grouped
 * even if several producers are merged into one stream. Within one process the
 * runs are strictly sequential: trunk first (unless --no-trunk-output), then
 * each continuation in seed order.
 *
 * --intervene ACTION (repeatable) applies an intervention (lib/interventions.js)
 * in every continuation, right after the restore and reseed, so each
 * continuation draws its own realisation (disaster location, strength). Its
 * events, then an "intervention" event line, are continuation lines tagged with
 * turn branchAt-1's last tick. When one has no effect with the branch seed, the
 * continuation is restored and retried with seed + k*RETRY_STRIDE (k = 1, 2, ...)
 * and the missed seeds are listed in its begin line; after MAX_ATTEMPTS misses
 * the run stops with an error and a nonzero exit.
 *
 * Writes go through fs.writeSync on fd 1 so a slow consumer applies
 * backpressure instead of the output piling up in memory. Progress and timing
 * go to stderr.
 *
 * Usage:
 *   pnpm tsx cli/run_continuations.js --city haight --branch-at 1440 --horizon 480 \
 *       --branch-seeds 1-100 > /dev/null
 */

import fs from 'node:fs';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import { loadMicropolisMainModule } from '../src/lib/wasm/node.ts';
import { callbackMethodNames } from '../src/lib/wasm/callbacks.ts';
import { heapU16FromEmscriptenModule } from '../src/lib/wasm/heap.ts';
import { messageText } from '../src/lib/engineMessages.ts';
import {
	ACTIONS,
	InterventionError,
	applyIntervention,
	interventionEventTick,
	interventionTick,
	parseInterventions
} from './lib/interventions.js';

const TICKS_PER_TURN = 16; // one cityTime increment (see simulate.cpp phaseCycle)
// Retries of an intervention that missed: seed + k*RETRY_STRIDE, disjoint from
// every branch seed range below RETRY_STRIDE.
const RETRY_STRIDE = 1_000_000;
const MAX_ATTEMPTS = 100;

// Mirrors run_sim.js.
const LOW_SIGNAL_EVENTS = new Set(['simulateRobots', 'simulateChurch', 'updateMap', 'updateHistory', 'updateDate']);
const EVALUATION_FIELDS = ['cityPop', 'cityPopDelta', 'cityAssessedValue', 'cityClass', 'cityScore', 'cityScoreDelta', 'cityYes', 'gameLevel'];
const OPTIONS_FIELDS = ['autoGoto', 'autoBudget', 'autoBulldoze', 'enableSound', 'enableDisasters', 'doAnimation', 'doMessages', 'doNotices'];

const BUILTIN_CITIES = [
	'about', 'badnews', 'bluebird', 'bruce', 'deadwood', 'finnigan', 'freds',
	'haight', 'happisle', 'joffburg', 'kamakura', 'kobe', 'kowloon', 'kyoto',
	'linecity', 'med_isle', 'ndulls', 'neatmap', 'radial', 'scenario_bern',
	'scenario_boston', 'scenario_detroit', 'scenario_dullsville',
	'scenario_hamburg', 'scenario_rio_de_janeiro', 'scenario_san_francisco',
	'scenario_tokyo', 'senri', 'southpac', 'splats', 'wetcity', 'yokohama'
];

function parseSeedRange(text) {
	const m = /^(\d+)(?:-(\d+))?$/.exec(String(text).trim());
	if (!m) throw new Error(`--branch-seeds must be "lo-hi" or "n", got "${text}"`);
	const lo = Number(m[1]);
	const hi = m[2] === undefined ? lo : Number(m[2]);
	if (hi < lo) throw new Error(`--branch-seeds range is empty: ${text}`);
	return { lo, hi };
}

const argv = yargs(hideBin(process.argv))
	.scriptName('run_continuations')
	.usage('$0 --city <name> --branch-at <S turns> --horizon <H turns> --branch-seeds <lo-hi>')
	.option('city', { alias: 'c', type: 'string', choices: BUILTIN_CITIES, default: 'haight', describe: 'Builtin city to load.' })
	.option('seed', {
		type: 'number',
		default: 42,
		describe: 'Trunk seed: applied before loadCity(), determines the world and the trunk run up to --branch-at.'
	})
	.option('branch-at', {
		alias: 'S',
		type: 'number',
		demandOption: true,
		describe: 'Turn (16 ticks each) at which to snapshot the state and branch.'
	})
	.option('horizon', {
		alias: 'H',
		type: 'number',
		demandOption: true,
		describe: 'Number of turns each continuation runs past the branch point.'
	})
	.option('branch-seeds', {
		type: 'string',
		demandOption: true,
		describe: 'Inclusive seed range "lo-hi" (or a single "n") for the continuations, run in order.'
	})
	.option('trunk-output', {
		type: 'boolean',
		default: true,
		describe: 'Also stream the trunk run (turns 0..S) before the continuations. Pass --no-trunk-output to skip.'
	})
	.option('verbose-log', {
		type: 'boolean',
		default: false,
		describe: 'Include high-frequency, low-signal events (simulateRobots, simulateChurch, updateMap, updateHistory, updateDate).'
	})
	.option('disasters', { type: 'boolean', default: true, describe: 'Random disasters. Pass --no-disasters to turn them off.' })
	.option('intervene', {
		type: 'array',
		string: true,
		default: [],
		describe: `Intervention applied in every continuation at the branch point, after the reseed. Repeatable. ACTION is one of: ${ACTIONS.join(', ')}.`
	})
	.option('progress', { type: 'boolean', default: false, describe: 'Print a stderr line per continuation.' })
	.example('$0 --city haight -S 1440 -H 480 --branch-seeds 1-100 > /dev/null', '30 years of trunk, then 100 ten-year continuations')
	.help()
	.alias('help', 'h')
	.strict()
	.parseSync();

const seedRange = parseSeedRange(argv.branchSeeds);
if (argv.intervene.length && seedRange.hi >= RETRY_STRIDE) throw new Error(`--branch-seeds must stay below ${RETRY_STRIDE} with --intervene`);
const interventions = parseInterventions(argv.intervene.map((action) => `${argv.branchAt - 1}:${action}`));

// ---------------------------------------------------------------------------
// stdout writer: synchronous, batched. Never touch process.stdout, which would
// switch fd 1 to non-blocking mode; a blocking fd gives us pipe backpressure.
// ---------------------------------------------------------------------------
const FLUSH_BYTES = 1 << 16;
let outBuf = '';
let capture = null; // while set, emit() collects lines here (an intervention attempt)

function writeAllSync(fd, str) {
	let buf = Buffer.from(str, 'utf8');
	while (buf.length > 0) {
		let n;
		try {
			n = fs.writeSync(fd, buf);
		} catch (err) {
			if (err.code === 'EAGAIN') continue; // non-blocking pipe momentarily full; spin
			if (err.code === 'EPIPE') process.exit(0); // consumer went away
			throw err;
		}
		buf = buf.subarray(n);
	}
}

function emit(obj) {
	if (capture) {
		capture.push(obj);
		return;
	}
	outBuf += JSON.stringify(obj) + '\n';
	if (outBuf.length >= FLUSH_BYTES) flush();
}

function flush() {
	if (outBuf.length === 0) return;
	writeAllSync(1, outBuf);
	outBuf = '';
}

// ---------------------------------------------------------------------------
// Snapshot/event construction — kept field-for-field identical to run_sim.js so
// consumers can share parsing code with the file-based logs.
// ---------------------------------------------------------------------------
function unwrap(value) {
	if (value && typeof value === 'object' && 'value' in value) return value.value;
	return value;
}

const LOMASK = 0x03ff;
const ZONEBIT = 0x0400;
function tileCensus(mapData) {
	const c = { water: 0, shore: 0, tree: 0, rubble: 0, flood: 0, radioactive: 0, fire: 0, road: 0, wire: 0, rail: 0, zoneCenters: 0 };
	for (let i = 0; i < mapData.length; i++) {
		const v = mapData[i];
		if (v & ZONEBIT) c.zoneCenters++;
		const t = v & LOMASK;
		if (t >= 5 && t <= 20) c.shore++;
		if (t >= 2 && t <= 20) c.water++;
		else if (t >= 21 && t <= 43) c.tree++;
		else if (t >= 44 && t <= 47) c.rubble++;
		else if (t >= 48 && t <= 51) c.flood++;
		else if (t === 52) c.radioactive++;
		else if (t >= 56 && t <= 63) c.fire++;
		else if (t >= 64 && t <= 207) c.road++;
		else if (t >= 208 && t <= 222) c.wire++;
		else if (t >= 224 && t <= 238) c.rail++;
	}
	return c;
}

function snapshot(micropolis, seed, tick, mapData) {
	return {
		kind: 'stats',
		seed,
		tick,
		cityName: String(micropolis.cityName ?? ''),
		cityTime: micropolis.cityTime,
		cityYear: micropolis.cityYear,
		cityMonth: micropolis.cityMonth,
		cityClass: unwrap(micropolis.cityClass),
		cityScore: micropolis.cityScore,
		cityScoreDelta: micropolis.cityScoreDelta,
		cityPop: micropolis.cityPop,
		cityPopDelta: micropolis.cityPopDelta,
		totalFunds: micropolis.totalFunds,
		cashFlow: micropolis.cashFlow,
		cityTax: micropolis.cityTax,
		cityTaxAverage: micropolis.cityTaxAverage,
		gameLevel: unwrap(micropolis.gameLevel),
		autoBudget: micropolis.autoBudget,
		trafficAverage: micropolis.trafficAverage,
		pollutionAverage: micropolis.pollutionAverage,
		crimeAverage: micropolis.crimeAverage,
		landValueAverage: micropolis.landValueAverage,
		resPop: micropolis.resPop,
		comPop: micropolis.comPop,
		indPop: micropolis.indPop,
		roadTotal: micropolis.roadTotal,
		railTotal: micropolis.railTotal,
		policeStationPop: micropolis.policeStationPop,
		fireStationPop: micropolis.fireStationPop,
		hospitalPop: micropolis.hospitalPop,
		stadiumPop: micropolis.stadiumPop,
		seaportPop: micropolis.seaportPop,
		airportPop: micropolis.airportPop,
		coalPowerPop: micropolis.coalPowerPop,
		nuclearPowerPop: micropolis.nuclearPowerPop,
		poweredZoneCount: micropolis.poweredZoneCount,
		unpoweredZoneCount: micropolis.unpoweredZoneCount,
		externalMarket: micropolis.externalMarket,
		roadEffect: micropolis.roadEffect,
		policeEffect: micropolis.policeEffect,
		fireEffect: micropolis.fireEffect,
		totalPop: micropolis.totalPop,
		churchPop: micropolis.churchPop,
		cityAssessedValue: micropolis.cityAssessedValue,
		resValve: micropolis.resValve,
		comValve: micropolis.comValve,
		indValve: micropolis.indValve,
		resCap: micropolis.resCap,
		comCap: micropolis.comCap,
		indCap: micropolis.indCap,
		census: tileCensus(mapData)
	};
}

function fieldSnapshot(micropolis, fields) {
	const out = {};
	for (const field of fields) out[field] = unwrap(micropolis[field]);
	return out;
}

// See run_sim.js for the C++ call sites these names come from.
const ARG_NAMES = {
	autoGoto: ['x', 'y', 'message'],
	didGenerateMap: ['seed'],
	didLoadCity: ['fileName'],
	didLoadScenario: ['scenarioName', 'fileName'],
	didSaveCity: ['fileName'],
	didTool: ['toolName', 'x', 'y'],
	didntLoadCity: ['fileName'],
	didntSaveCity: ['fileName'],
	makeSound: ['channel', 'sound', 'x', 'y'],
	saveCityAs: ['fileName'],
	sendMessage: ['messageNum', 'x', 'y', 'pictureFlag', 'important'],
	showZoneStatus: ['tileCategoryIndex', 'populationDensityIndex', 'landValueIndex', 'crimeRateIndex', 'pollutionIndex', 'growthRateIndex', 'x', 'y'],
	simulateChurch: ['x', 'y', 'churchNumber'],
	startEarthquake: ['strength'],
	startScenario: ['scenario'],
	updateCityName: ['cityName'],
	updateDemand: ['residentialDemand', 'commercialDemand', 'industrialDemand'],
	updateFunds: ['totalFunds'],
	updateGameLevel: ['gameLevel'],
	updatePasses: ['passes'],
	updatePaused: ['paused'],
	updateSpeed: ['speed'],
	updateTaxRate: ['taxRate'],
	updateDate: ['cityYear', 'cityMonth']
};

function namedArgs(name, args) {
	const names = ARG_NAMES[name];
	const out = {};
	args.forEach((value, i) => {
		out[names?.[i] ?? `arg${i}`] = unwrap(value);
	});
	if (name === 'sendMessage') out.messageText = messageText(out.messageNum);
	return out;
}

function eventEntry(micropolis, ctx, name) {
	return {
		kind: 'event',
		seed: ctx.seed,
		tick: ctx.tick,
		event: name,
		cityTime: micropolis.cityTime,
		cityYear: micropolis.cityYear,
		cityMonth: micropolis.cityMonth
	};
}

// `ctx` is mutated between runs so one JSCallback serves the trunk and every
// continuation: ctx.seed tags the lines, ctx.tick is the script-side tick
// counter, ctx.muted drops events (used while running the trunk silently).
function makeStreamingCallback(engine, micropolis, ctx, verbose, onMessage) {
	const handlers = {};
	for (const name of callbackMethodNames) {
		if (name === 'sendMessage') {
			handlers[name] = (...allArgs) => {
				onMessage(unwrap(allArgs[2]));
				if (ctx.muted) return;
				emit({ ...eventEntry(micropolis, ctx, name), ...namedArgs(name, allArgs.slice(2)) });
			};
			continue;
		}
		if (!verbose && LOW_SIGNAL_EVENTS.has(name)) {
			handlers[name] = () => {};
			continue;
		}
		handlers[name] = (...allArgs) => {
			if (ctx.muted) return;
			const args = allArgs.slice(2); // drop Micropolis* and user-data handle
			const entry = eventEntry(micropolis, ctx, name);
			if (name === 'updateEvaluation') Object.assign(entry, fieldSnapshot(micropolis, EVALUATION_FIELDS));
			else if (name === 'updateOptions') Object.assign(entry, fieldSnapshot(micropolis, OPTIONS_FIELDS));
			else Object.assign(entry, namedArgs(name, args));
			emit(entry);
		};
	}
	return new engine.JSCallback(handlers);
}

async function main() {
	const t0 = performance.now();
	const engine = await loadMicropolisMainModule();
	const micropolis = new engine.Micropolis();
	const memory = engine.wasmMemory;
	if (!(memory instanceof WebAssembly.Memory)) throw new Error('engine.wasmMemory is not available; cannot snapshot the heap');

	const ctx = { seed: null, tick: 0, muted: !argv.trunkOutput };
	let messages = [];
	const sentMessages = () => {
		const sent = messages;
		messages = [];
		return sent;
	};
	const callback = makeStreamingCallback(engine, micropolis, ctx, argv.verboseLog, (num) => messages.push(num));
	micropolis.setCallback(callback, {});
	micropolis.init();
	if (!argv.disasters) micropolis.enableDisasters = false; // init() forces it on
	micropolis.seedRandom(argv.seed); // must precede loadCity(); see run_sim.js

	const cityPath = `/cities/${argv.city}.cty`;
	if (!micropolis.loadCity(cityPath)) {
		console.error(`Failed to load builtin city "${argv.city}" (${cityPath})`);
		micropolis.delete();
		process.exitCode = 1;
		return;
	}

	const mapView = () => {
		const heap = heapU16FromEmscriptenModule(engine);
		const start = micropolis.getMapAddress() / 2;
		return heap.subarray(start, start + micropolis.getMapSize() / 2);
	};

	// Log once per turn: phaseCycle 15 is the last phase of a cityTime value,
	// after every scan for that cityTime has run (see run_sim.js).
	const shouldLog = (m) => m.phaseCycle === 15;

	// Runs ticks (fromTick, toTick], emitting a stats line at each turn boundary.
	function runTicks(fromTick, toTick) {
		let rows = 0;
		for (let tick = fromTick + 1; tick <= toTick; tick++) {
			ctx.tick = tick;
			micropolis.simTick();
			if (shouldLog(micropolis) && !ctx.muted) {
				emit(snapshot(micropolis, ctx.seed, tick, mapView()));
				rows++;
			}
		}
		return rows;
	}

	const branchTick = argv.branchAt * TICKS_PER_TURN;
	if (interventions.length && interventionTick(argv.branchAt - 1) !== branchTick) throw new Error('intervention tick mismatch');
	const endTick = branchTick + argv.horizon * TICKS_PER_TURN;

	emit({
		kind: 'header',
		city: argv.city,
		trunkSeed: argv.seed,
		branchAt: argv.branchAt,
		horizon: argv.horizon,
		branchTick,
		endTick,
		seedLo: seedRange.lo,
		seedHi: seedRange.hi,
		disasters: argv.disasters,
		trunkOutput: argv.trunkOutput,
		interventions,
		interventionMode: 'per-continuation'
	});

	// --- trunk -------------------------------------------------------------
	const tTrunk0 = performance.now();
	ctx.seed = null;
	ctx.tick = 0;
	if (!ctx.muted) {
		emit({ kind: 'begin', seed: null, run: 'trunk', fromTick: 0, toTick: branchTick });
		if (shouldLog(micropolis)) emit(snapshot(micropolis, null, 0, mapView()));
	}
	const trunkRows = runTicks(0, branchTick);
	if (!ctx.muted) emit({ kind: 'end', seed: null, run: 'trunk', rows: trunkRows, cityTime: micropolis.cityTime });
	const tTrunk1 = performance.now();

	// --- snapshot ------------------------------------------------------------
	// Copy all of linear memory. Restoring it puts every engine byte (map,
	// sprites, timers, malloc state) back exactly as it was at the branch tick.
	const heapSnapshot = new Uint8Array(memory.buffer).slice();
	const snapshotCityTime = micropolis.cityTime;

	// --- continuations -------------------------------------------------------
	ctx.muted = false;
	const tCont0 = performance.now();
	let nRuns = 0;
	let nMissed = 0;
	const restore = (rngSeed) => {
		if (memory.buffer.byteLength !== heapSnapshot.byteLength) {
			// Heap grew since the snapshot: a prefix restore would leave malloc's
			// idea of the heap end inconsistent with the actual memory size.
			throw new Error(`WASM heap size changed (${heapSnapshot.byteLength} -> ${memory.buffer.byteLength}); cannot restore snapshot`);
		}
		new Uint8Array(memory.buffer).set(heapSnapshot);
		if (micropolis.cityTime !== snapshotCityTime) throw new Error('heap restore failed: cityTime mismatch');
		micropolis.seedRandom(rngSeed);
	};
	// Applies every intervention; returns the lines they emitted, or throws InterventionError.
	const intervene = () => {
		const lines = [];
		capture = lines;
		try {
			for (const iv of interventions) {
				ctx.tick = interventionEventTick(iv.turn);
				const entry = { ...eventEntry(micropolis, ctx, 'intervention'), turn: iv.turn, action: iv.action };
				const result = applyIntervention(micropolis, iv, sentMessages);
				if (result !== undefined) entry.result = result;
				emit(entry);
			}
		} finally {
			capture = null;
		}
		return lines;
	};
	for (let seed = seedRange.lo; seed <= seedRange.hi; seed++) {
		ctx.seed = seed;
		const begin = { kind: 'begin', seed, run: 'continuation', fromTick: branchTick, toTick: endTick };
		let lines = [];
		restore(seed);
		if (interventions.length) {
			const missed = [];
			for (let rngSeed = seed; ; rngSeed = seed + missed.length * RETRY_STRIDE) {
				if (rngSeed !== seed) restore(rngSeed);
				try {
					lines = intervene();
					begin.interventionSeed = rngSeed;
					break;
				} catch (err) {
					if (!(err instanceof InterventionError)) throw err;
					missed.push(rngSeed);
					if (missed.length >= MAX_ATTEMPTS) {
						flush();
						micropolis.delete();
						console.error(`Error: ${err.message}; branch seed ${seed}: ${MAX_ATTEMPTS} attempts`);
						process.exitCode = 1;
						return;
					}
				}
			}
			begin.missedSeeds = missed;
			nMissed += missed.length;
		}

		ctx.tick = branchTick;
		emit(begin);
		for (const line of lines) emit(line);
		const rows = runTicks(branchTick, endTick);
		emit({ kind: 'end', seed, run: 'continuation', rows, cityTime: micropolis.cityTime });
		nRuns++;
		if (argv.progress) console.error(`seed ${seed}: ${rows} rows, cityTime ${micropolis.cityTime}`);
	}
	flush();
	const tCont1 = performance.now();

	micropolis.delete();

	const contMs = tCont1 - tCont0;
	console.error(
		[
			`city=${argv.city} trunkSeed=${argv.seed} S=${argv.branchAt} H=${argv.horizon} seeds=${seedRange.lo}-${seedRange.hi}`,
			`startup ${(tTrunk0 - t0).toFixed(0)} ms, trunk (${branchTick} ticks) ${(tTrunk1 - tTrunk0).toFixed(0)} ms,`,
			`${nRuns} continuations (${argv.horizon * TICKS_PER_TURN} ticks each) ${contMs.toFixed(0)} ms total,`,
			...(interventions.length ? [`${nMissed} intervention attempts missed,`] : []),
			`${(contMs / Math.max(nRuns, 1)).toFixed(1)} ms each, ${((nRuns * argv.horizon * TICKS_PER_TURN) / (contMs / 1000) / 1000).toFixed(0)}k ticks/s`
		].join(' ')
	);
}

await main();
