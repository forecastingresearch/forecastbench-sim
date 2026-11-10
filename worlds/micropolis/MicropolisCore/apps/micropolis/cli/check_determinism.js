#!/usr/bin/env tsx

/**
 * Determinism checker for the headless simulation.
 *
 * Runs the same city, seed and tick count several times and diffs the
 * resulting per-tick stat time-series against each other. If seeding via
 * seedRandom() were the only source of randomness, every run would produce
 * byte-identical series; any reported divergence is residual non-determinism
 * in the engine.
 *
 * Mirrors run_sim.js's setup (init -> loadCity -> seedRandom -> simTick loop)
 * but keeps everything in memory and writes only a diff report.
 *
 * SEEDING ORDER MATTERS
 *
 * seedRandom() is always called *before* loadCity(), never after. This is the
 * only ordering that yields a reproducible run:
 *
 *   Micropolis::init()      -> simInit() -> randomlySeedRandom()
 *   Micropolis::loadCity()  -> loadFile() -> initWillStuff()
 *                                        -> doSimInit() -> mapScan(0, WORLD_W)
 *
 * init() deliberately seeds from the wall clock, so that an unseeded game is
 * random. loadCity()'s map scans then run the zone simulation over the freshly
 * loaded city -- doResidential/doCommercial/doIndustrial etc. draw getRandom()
 * and *mutate map tiles* (e.g. `map[x][y] = ... + getRandom(2)` in zone.cpp).
 * So a seed applied after the load arrives too late: the world it was supposed
 * to determine has already been built from the clock-derived seed. Seeding
 * before the load is what makes the load itself reproducible.
 *
 * HISTORY
 *
 * This used to be the other way round. initWillStuff() called
 * randomlySeedRandom() on every path, so a pre-load seed was always clobbered
 * and the script seeded after the load instead, with a --seed-before-load flag
 * kept only to demonstrate that seeding early did not help. The reseed has
 * since moved out of initWillStuff() to simInit() (i.e. to init() alone), which
 * inverts that conclusion: seeding before the load now works, seeding after it
 * does not, and the flag is gone because there is no longer a reason to choose.
 *
 * A related fix removed a redundant doSimInit() call from loadCity(), which had
 * been running a stray uninitialized mapScan and advancing the city one
 * simulation step past the file on disk.
 *
 * Usage:
 *   pnpm tsx apps/micropolis/cli/check_determinism.js --city haight --ticks 320 --runs 3
 *   pnpm tsx apps/micropolis/cli/check_determinism.js --fresh-module   # reload WASM per run
 */

import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import { loadMicropolisMainModule } from '../src/lib/wasm/node.ts';
import { callbackMethodNames } from '../src/lib/wasm/callbacks.ts';
import { createMapMopViews } from '../src/lib/wasm/views.ts';

const argv = yargs(hideBin(process.argv))
	.scriptName('check_determinism')
	.usage('$0 --city <name> --ticks <n> --runs <n>')
	.option('city', { alias: 'c', type: 'string', default: 'haight', describe: 'Builtin city to load' })
	.option('ticks', { alias: 't', type: 'number', default: 320, describe: 'simTick() calls per run' })
	.option('runs', { alias: 'r', type: 'number', default: 3, describe: 'Number of repeat runs to compare' })
	.option('seed', { type: 'number', default: 42, describe: 'Seed passed to seedRandom() before loadCity()' })
	.option('disasters', { type: 'boolean', default: true, describe: 'Pass --no-disasters to disable random disasters' })
	.option('fresh-module', {
		type: 'boolean',
		default: false,
		describe: 'Load a brand-new WASM module instance for each run (default: reuse one module, new Micropolis object)'
	})
	.option('log-every-tick', { type: 'boolean', default: false, describe: 'Snapshot every tick instead of phaseCycle===15' })
	.option('max-diffs', { type: 'number', default: 20, describe: 'Max individual field diffs to print per run pair' })
	.option('freeze-clock', {
		type: 'boolean',
		default: false,
		describe:
			'Stub Date.now() to a constant for the duration of each run. The engine reaches the clock only through ' +
			'emscripten_date_now (= Date.now), so this makes every gettimeofday() inside the WASM engine — including ' +
			'randomlySeedRandom() — return the same value in every run. If the runs become identical under this flag, ' +
			'the clock is the sole source of non-determinism.'
	})
	.option('freeze-clock-until-load', {
		type: 'boolean',
		default: false,
		describe:
			'Freeze Date.now() only for init()+loadCity(), then restore it before ticking. Isolates the setup-phase ' +
			'clock reads (randomlySeedRandom) from the per-tick one (tickCount -> blinkFlag).'
	})
	.option('trace-rng', {
		type: 'boolean',
		default: false,
		describe: 'After the run, draw N getRandom16() values and compare them across runs too'
	})
	.help()
	.alias('help', 'h')
	.strict()
	.parseSync();

// Same field set run_sim.js records, so a divergence here is a divergence
// there too.
const FIELDS = [
	'cityTime',
	'cityYear',
	'cityMonth',
	'cityClass',
	'cityScore',
	'cityScoreDelta',
	'cityPop',
	'cityPopDelta',
	'totalFunds',
	'cashFlow',
	'cityTax',
	'cityTaxAverage',
	'gameLevel',
	'trafficAverage',
	'pollutionAverage',
	'crimeAverage',
	'landValueAverage',
	'resPop',
	'comPop',
	'indPop',
	'roadTotal',
	'railTotal',
	'policeStationPop',
	'fireStationPop',
	'hospitalPop',
	'stadiumPop',
	'seaportPop',
	'airportPop',
	'coalPowerPop',
	'nuclearPowerPop',
	'poweredZoneCount',
	'unpoweredZoneCount',
	'externalMarket',
	'roadEffect',
	'policeEffect',
	'fireEffect'
];

// mapHash isn't an engine field (it's computed from the tile array), so it
// lives only on the diff side.
const DIFF_FIELDS = ['mapHash', ...FIELDS];

function unwrap(value) {
	if (value && typeof value === 'object' && 'value' in value) return value.value;
	return value;
}

// FNV-1a over the raw tile array — a cheap fingerprint of the whole world
// state, so we can tell whether two runs' maps already differ before the
// first simTick().
function hashU16(view) {
	let h = 0x811c9dc5;
	for (let i = 0; i < view.length; i++) {
		h ^= view[i];
		h = Math.imul(h, 0x01000193) >>> 0;
	}
	return h.toString(16).padStart(8, '0');
}

function snapshot(micropolis, tick, views) {
	const row = { tick };
	for (const field of FIELDS) row[field] = unwrap(micropolis[field]);
	if (views) row.mapHash = hashU16(views.mapData);
	return row;
}

// Silent callback: the engine requires one, and we don't want the event
// stream itself to perturb anything.
function makeSilentCallback(engine, counters) {
	const handlers = {};
	for (const name of callbackMethodNames) {
		handlers[name] = () => {
			counters[name] = (counters[name] ?? 0) + 1;
		};
	}
	return new engine.JSCallback(handlers);
}

// The engine's only route to wall-clock time is emscripten_date_now, which
// the generated glue defines as `() => Date.now()`. Freezing Date.now
// therefore freezes gettimeofday() inside the WASM module.
const FROZEN_NOW = 1_700_000_000_000;

async function runOnce(engine, runIndex) {
	const realDateNow = Date.now;
	if (argv.freezeClock) Date.now = () => FROZEN_NOW;
	try {
		return await runOnceInner(engine, runIndex, realDateNow);
	} finally {
		Date.now = realDateNow;
	}
}

async function runOnceInner(engine, runIndex, realDateNow) {
	const counters = {};
	const micropolis = new engine.Micropolis();
	const callback = makeSilentCallback(engine, counters);
	micropolis.setCallback(callback, {});

	if (argv.freezeClockUntilLoad) Date.now = () => FROZEN_NOW;
	micropolis.init();

	if (!argv.disasters) micropolis.enableDisasters = false;

	// Must be before loadCity(): the load's map scans consume the RNG and write
	// the results into map tiles, so a seed applied afterwards is too late to
	// determine the loaded world. See SEEDING ORDER MATTERS above.
	micropolis.seedRandom(argv.seed);

	const loaded = micropolis.loadCity(`/cities/${argv.city}.cty`);
	if (argv.freezeClockUntilLoad) Date.now = realDateNow;
	if (!loaded) throw new Error(`Failed to load builtin city "${argv.city}"`);

	// Fingerprint the world the instant loading finished. The seed was applied
	// before the load, so these hashes must match across runs; if they differ,
	// something in the load path is still consuming unseeded randomness.
	const views = createMapMopViews(engine, micropolis);
	const postLoadHash = views ? hashU16(views.mapData) : null;

	const shouldLog = (m) => argv.logEveryTick || m.phaseCycle === 15;
	const rows = [];
	if (shouldLog(micropolis)) rows.push(snapshot(micropolis, 0, views));
	for (let tick = 1; tick <= argv.ticks; tick++) {
		micropolis.simTick();
		if (shouldLog(micropolis)) rows.push(snapshot(micropolis, tick, views));
	}

	// Optional post-run RNG probe: if the raw stream has drifted, the engine
	// consumed a different number of draws along the way.
	let rngTail = null;
	if (argv.traceRng) {
		rngTail = [];
		for (let i = 0; i < 16; i++) rngTail.push(micropolis.getRandom16());
	}

	// Only safe to delete when the whole module is being thrown away with it:
	// Micropolis::~Micropolis() -> setCallback(NULL, ...) does
	// `if (callback != NULL) delete callback;`, and `callback` is never
	// initialized by the constructor or by init(). On a fresh module the heap
	// is zero-filled so the check happens to pass, but a second Micropolis
	// allocated over freed memory sees the stale pointer and deletes it again
	// ("RuntimeError: table index is out of bounds"). Leaking here keeps
	// --runs > 1 usable on a shared module.
	if (argv.freshModule) micropolis.delete();
	return { runIndex, rows, counters, rngTail, postLoadHash };
}

// Returns the list of (row, field) mismatches between two runs, plus the
// earliest diverging row index.
function diffRuns(a, b) {
	const diffs = [];
	const n = Math.min(a.rows.length, b.rows.length);
	for (let i = 0; i < n; i++) {
		for (const field of DIFF_FIELDS) {
			if (a.rows[i][field] !== b.rows[i][field]) {
				diffs.push({ rowIndex: i, tick: a.rows[i].tick, field, a: a.rows[i][field], b: b.rows[i][field] });
			}
		}
	}
	if (a.rows.length !== b.rows.length) {
		diffs.push({ rowIndex: n, tick: null, field: '<row count>', a: a.rows.length, b: b.rows.length });
	}
	return diffs;
}

function diffCounters(a, b) {
	const names = new Set([...Object.keys(a.counters), ...Object.keys(b.counters)]);
	const out = [];
	for (const name of [...names].sort()) {
		const ca = a.counters[name] ?? 0;
		const cb = b.counters[name] ?? 0;
		if (ca !== cb) out.push({ name, a: ca, b: cb });
	}
	return out;
}

async function main() {
	console.log(
		`city=${argv.city} seed=${argv.seed} ticks=${argv.ticks} runs=${argv.runs} ` +
			`disasters=${argv.disasters} freshModule=${argv.freshModule}`
	);

	let sharedEngine = argv.freshModule ? null : await loadMicropolisMainModule();

	const results = [];
	for (let i = 0; i < argv.runs; i++) {
		const engine = argv.freshModule ? await loadMicropolisMainModule() : sharedEngine;
		const result = await runOnce(engine, i);
		results.push(result);
		console.log(
			`  run ${i}: ${result.rows.length} rows, post-load mapHash=${result.postLoadHash}, ` +
				`last cityPop=${result.rows.at(-1)?.cityPop}`
		);
	}

	const loadHashes = new Set(results.map((r) => r.postLoadHash));
	console.log(
		loadHashes.size === 1
			? `\npost-loadCity() map state is identical across runs (${[...loadHashes][0]})`
			: `\npost-loadCity() map state DIFFERS across runs: ${[...loadHashes].join(', ')}` +
					'\n  => the load path consumed randomness the pre-load seedRandom() did not determine'
	);

	let anyDiff = false;
	for (let i = 1; i < results.length; i++) {
		const diffs = diffRuns(results[0], results[i]);
		const counterDiffs = diffCounters(results[0], results[i]);
		const rngDiff =
			results[0].rngTail && JSON.stringify(results[0].rngTail) !== JSON.stringify(results[i].rngTail);

		if (diffs.length === 0 && counterDiffs.length === 0 && !rngDiff) {
			console.log(`run 0 vs run ${i}: IDENTICAL`);
			continue;
		}
		anyDiff = true;
		console.log(`\nrun 0 vs run ${i}: ${diffs.length} field diffs across the series`);
		if (diffs.length > 0) {
			const first = diffs[0];
			console.log(
				`  first divergence: row ${first.rowIndex} (tick ${first.tick}) field ${first.field}: ${first.a} vs ${first.b}`
			);
			const fieldsSeen = [...new Set(diffs.map((d) => d.field))];
			console.log(`  diverging fields (${fieldsSeen.length}): ${fieldsSeen.join(', ')}`);
			for (const d of diffs.slice(0, argv.maxDiffs)) {
				console.log(`    row ${d.rowIndex} tick ${d.tick} ${d.field}: ${d.a} vs ${d.b}`);
			}
			if (diffs.length > argv.maxDiffs) console.log(`    ... ${diffs.length - argv.maxDiffs} more`);
		}
		if (counterDiffs.length > 0) {
			console.log('  callback count diffs:');
			for (const c of counterDiffs) console.log(`    ${c.name}: ${c.a} vs ${c.b}`);
		}
		if (rngDiff) {
			console.log(`  post-run RNG draws differ:\n    ${results[0].rngTail}\n    ${results[i].rngTail}`);
		}
	}

	if (anyDiff) {
		console.log('\nRESULT: NON-DETERMINISTIC');
		process.exitCode = 1;
	} else {
		console.log('\nRESULT: deterministic across all runs');
	}
}

await main();
