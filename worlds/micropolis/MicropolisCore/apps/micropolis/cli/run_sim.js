#!/usr/bin/env tsx

/**
 * Standalone headless simulation runner.
 *
 * Loads one of the cities packaged inside the WASM engine, ticks it forward
 * a fixed number of times, and records per-tick city stats as JSONL. Every
 * engine callback (messages, budget/date/fund updates, tool results,
 * disasters, etc.) is also logged as JSONL to a companion .log file next to
 * the stats output. Pass --verbose-log to also include the high-frequency,
 * low-signal events (simulateRobots, simulateChurch, updateMap,
 * updateHistory, updateDate), which are skipped by default.
 *
 * Seeding: seedRandom() is called before loadCity(), which is the only ordering
 * that makes a run reproducible — the load's map scans consume the RNG and bake
 * the result into map tiles, so a later seed cannot determine the loaded world.
 *
 * Usage:
 *   pnpm run run-sim --city haight --ticks 200
 *   pnpm run run-sim --help   # lists all builtin city choices
 *
 * Output goes to <output-base-dir>/<city>/log-seed<seed>[-nodisasters].jsonl,
 * with the event log next to it as events-seed<seed>[-nodisasters].jsonl.
 *
 * --at TURN:ACTION (repeatable) triggers a disaster or demolishes buildings
 * just after turn TURN's row is logged (timing in lib/interventions.js). Each one
 * logs an "intervention" event, and the output names gain -at<TURN>_<ACTION>.
 * One that has no effect (e.g. a flood on a map without river edges) is an
 * error: the run stops and exits nonzero.
 */

import fs from 'node:fs';
import path from 'node:path';
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
	interventionsTag,
	parseInterventions
} from './lib/interventions.js';

// Events that fire every tick (or nearly so) and rarely carry information
// useful for offline analysis. Skipped unless --verbose-log is passed.
const LOW_SIGNAL_EVENTS = new Set(['simulateRobots', 'simulateChurch', 'updateMap', 'updateHistory', 'updateDate']);

// updateEvaluation/updateOptions carry no args of their own — they're a
// signal that fields on the Micropolis object were just refreshed, so we
// read the relevant fields straight off the engine when they fire.
const EVALUATION_FIELDS = ['cityPop', 'cityPopDelta', 'cityAssessedValue', 'cityClass', 'cityScore', 'cityScoreDelta', 'cityYes', 'gameLevel'];
const OPTIONS_FIELDS = ['autoGoto', 'autoBudget', 'autoBulldoze', 'enableSound', 'enableDisasters', 'doAnimation', 'doMessages', 'doNotices'];

// Mirrors MicropolisSimulator.ts's cityFileNames — the cities preloaded into
// the WASM module's virtual filesystem (the only ones loadCity() can reach).
const BUILTIN_CITIES = [
	'about',
	'badnews',
	'bluebird',
	'bruce',
	'deadwood',
	'finnigan',
	'freds',
	'haight',
	'happisle',
	'joffburg',
	'kamakura',
	'kobe',
	'kowloon',
	'kyoto',
	'linecity',
	'med_isle',
	'ndulls',
	'neatmap',
	'radial',
	'scenario_bern',
	'scenario_boston',
	'scenario_detroit',
	'scenario_dullsville',
	'scenario_hamburg',
	'scenario_rio_de_janeiro',
	'scenario_san_francisco',
	'scenario_tokyo',
	'senri',
	'southpac',
	'splats',
	'wetcity',
	'yokohama'
];

const argv = yargs(hideBin(process.argv))
	.scriptName('run_sim')
	.usage('$0 --city <name> --ticks <n> --output-base-dir <dir>')
	.option('city', {
		alias: 'c',
		type: 'string',
		choices: BUILTIN_CITIES,
		default: 'haight',
		describe: 'Builtin city to load.'
	})
	.option('ticks', {
		alias: 't',
		type: 'number',
		describe: 'Number of simTick() calls to run (default: 100 if --turns is not given)'
	})
	.option('turns', {
		type: 'number',
		describe:
			'Number of in-game turns to run instead of --ticks (1 turn = 16 ticks = 1 cityTime increment — ' +
			'see simulate.cpp). Without --log-every-tick this logs exactly N rows. Mutually exclusive with --ticks.'
	})
	.conflicts('ticks', 'turns')
	.option('output-base-dir', {
		alias: 'o',
		type: 'string',
		default: './sim-runs/',
		describe:
			'Base directory to write output under. Stats go to ' +
			'<output-base-dir>/<city>/log-seed<seed>-[no]disasters.jsonl, with the event log next to it ' +
			'as events-seed<seed>-[no]disasters.jsonl.'
	})
	.option('log-every-tick', {
		type: 'boolean',
		default: false,
		describe:
			'Log every simTick(), instead of just the last tick before cityTime increments (cityTime only advances once per 16-tick phaseCycle — see simulate.cpp).'
	})
	.option('verbose-log', {
		type: 'boolean',
		default: false,
		describe:
			'Include high-frequency, low-signal events in the event log (simulateRobots, simulateChurch, updateMap, updateHistory, updateDate). Off by default.'
	})
	.option('disasters', {
		type: 'boolean',
		default: true,
		describe: 'Random disasters (fires, floods, monster, etc.). Pass --no-disasters to turn them off.'
	})
	.option('seed', {
		type: 'number',
		default: 42,
		describe:
			'Seed the RNG just before loadCity() (applied every run — a fixed default keeps output paths and results reproducible). Runs with the same city, seed and tick count are reproducible; verify with cli/check_determinism.js.'
	})
	.option('at', {
		type: 'array',
		string: true,
		default: [],
		describe:
			'Intervention "TURN:ACTION", applied just after turn TURN\'s row is logged. Repeatable. ' +
			`ACTION is one of: ${ACTIONS.join(', ')}. An intervention that has no effect is an error.`
	})
	.example('$0 --city haight --turns 200 --at 96:demolishPowerPlants', 'Demolish every power plant after turn 96')
	.example('$0 --city haight --ticks 500', 'Run haight for 500 ticks')
	.example('$0 --city haight --turns 500', 'Run haight for 500 turns (8000 ticks), logging exactly 500 rows')
	.help()
	.alias('help', 'h')
	.strict()
	.parseSync();

// A turn is one cityTime increment — 16 ticks (see simulate.cpp's 16-phase
// phaseCycle) — so --turns N logs exactly N rows without --log-every-tick.
const TICKS_PER_TURN = 16;
const totalTicks = argv.turns !== undefined ? argv.turns * TICKS_PER_TURN : argv.ticks ?? 100;

// Unwraps embind enum-style values ({value: N}) down to a plain number;
// passes plain numbers/strings/booleans through untouched.
function unwrap(value) {
	if (value && typeof value === 'object' && 'value' in value) return value.value;
	return value;
}

// Tile-census categories over the LOMASK tile number (tool.h enums). One
// linear pass over the live 120x100 map view per snapshot; cheap relative to
// the 16 simTick()s per cityTime.
const LOMASK = 0x03ff;
const ZONEBIT = 0x0400;
function tileCensus(mapData) {
	const c = { water: 0, shore: 0, tree: 0, rubble: 0, flood: 0, radioactive: 0, fire: 0, road: 0, wire: 0, rail: 0, zoneCenters: 0 };
	for (let i = 0; i < mapData.length; i++) {
		const v = mapData[i];
		if (v & ZONEBIT) c.zoneCenters++;
		const t = v & LOMASK;
		// shore = river-edge tiles (FIRSTRIVEDGE..LASTRIVEDGE). Flood exposure
		// scales with these, not with open water: makeFlood() only starts a
		// flood from a river-edge tile (disasters.cpp), so maps whose water is
		// all RIVER/CHANNEL tiles (happisle, kowloon, badnews) can never flood.
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

function snapshot(micropolis, tick, mapData) {
	return {
		tick, // number of simTick() calls made so far (script-side counter, not an engine field)
		cityName: String(micropolis.cityName ?? ''), // name of the loaded city
		cityTime: micropolis.cityTime, // in-game clock, advances once per 16-tick phase cycle
		cityYear: micropolis.cityYear, // current in-game year
		cityMonth: micropolis.cityMonth, // current in-game month
		cityClass: unwrap(micropolis.cityClass), // city size classification (village/town/city/capital/metropolis/megalopolis)
		cityScore: micropolis.cityScore, // overall city evaluation score
		cityScoreDelta: micropolis.cityScoreDelta, // change in city score since last evaluation
		cityPop: micropolis.cityPop, // total city population
		cityPopDelta: micropolis.cityPopDelta, // change in population since last evaluation
		totalFunds: micropolis.totalFunds, // city treasury balance
		cashFlow: micropolis.cashFlow, // net income/expense this budget cycle
		cityTax: micropolis.cityTax, // current property tax rate (0-20)
		cityTaxAverage: micropolis.cityTaxAverage, // time-weighted average tax rate over the current budget cycle
		gameLevel: unwrap(micropolis.gameLevel), // difficulty level (0=easy, 1=medium, 2=hard)
		autoBudget: micropolis.autoBudget, // whether auto-budget (vs. manual funding) is enabled
		trafficAverage: micropolis.trafficAverage, // city-wide average traffic density
		pollutionAverage: micropolis.pollutionAverage, // city-wide average pollution level
		crimeAverage: micropolis.crimeAverage, // city-wide average crime level
		landValueAverage: micropolis.landValueAverage, // city-wide average land value
		resPop: micropolis.resPop, // residential zone population
		comPop: micropolis.comPop, // commercial zone population
		indPop: micropolis.indPop, // industrial zone population
		roadTotal: micropolis.roadTotal, // total count of road tiles
		railTotal: micropolis.railTotal, // total count of rail tiles
		policeStationPop: micropolis.policeStationPop, // number of police stations
		fireStationPop: micropolis.fireStationPop, // number of fire stations
		hospitalPop: micropolis.hospitalPop, // number of hospitals
		stadiumPop: micropolis.stadiumPop, // number of stadiums
		seaportPop: micropolis.seaportPop, // number of seaports
		airportPop: micropolis.airportPop, // number of airports
		coalPowerPop: micropolis.coalPowerPop, // number of coal power plants
		nuclearPowerPop: micropolis.nuclearPowerPop, // number of nuclear power plants
		poweredZoneCount: micropolis.poweredZoneCount, // count of zone tiles currently receiving power
		unpoweredZoneCount: micropolis.unpoweredZoneCount, // count of zone tiles currently lacking power
		externalMarket: micropolis.externalMarket, // external market demand multiplier for industry
		roadEffect: micropolis.roadEffect, // road funding effectiveness multiplier (funding level -> service quality)
		policeEffect: micropolis.policeEffect, // police funding effectiveness multiplier
		fireEffect: micropolis.fireEffect, // fire funding effectiveness multiplier
		totalPop: micropolis.totalPop, // census total (resPop/8 + comPop + indPop), the tax base
		churchPop: micropolis.churchPop, // number of churches (self-built, tracks resPop like hospitals)
		cityAssessedValue: micropolis.cityAssessedValue, // infrastructure book value (updated at each evaluation)
		resValve: micropolis.resValve, // residential demand valve (-2000..2000, drives zone growth/decay)
		comValve: micropolis.comValve, // commercial demand valve (-1500..1500)
		indValve: micropolis.indValve, // industrial demand valve (-1500..1500)
		resCap: micropolis.resCap, // residential growth capped pending a stadium (resPop > 500, none built)
		comCap: micropolis.comCap, // commercial growth capped pending an airport (comPop > 100)
		indCap: micropolis.indCap, // industrial growth capped pending a seaport (indPop > 70)
		census: tileCensus(mapData) // tile counts: water/tree/rubble/flood/radioactive/fire/road/wire/rail/zoneCenters
	};
}

// Base fields every logged event carries, so the event log can be joined
// against the stats JSONL on tick/cityTime.
function eventBase(micropolis, tick, name) {
	return {
		tick,
		event: name,
		cityTime: micropolis.cityTime,
		cityYear: micropolis.cityYear,
		cityMonth: micropolis.cityMonth
	};
}

function fieldSnapshot(micropolis, fields) {
	const out = {};
	for (const field of fields) {
		out[field] = unwrap(micropolis[field]);
	}
	return out;
}

// Names for each callback's extra args, in order, per the C++ call sites
// (packages/micropolis-engine/src/*.cpp — see callback.h for the canonical
// parameter names). Callbacks not listed here take no extra args.
const ARG_NAMES = {
	// doAutoGoto(x, y, message) — message.cpp: tile coords + the message that triggered the goto.
	autoGoto: ['x', 'y', 'message'],
	// generateSomeCity(seed) — generate.cpp: RNG seed used to generate the map.
	didGenerateMap: ['seed'],
	// fileio.cpp: path of the city file just loaded.
	didLoadCity: ['fileName'],
	// loadScenario(s, name, fname) — fileio.cpp: scenario city name + file path loaded.
	didLoadScenario: ['scenarioName', 'fileName'],
	// fileio.cpp: path of the city file just saved.
	didSaveCity: ['fileName'],
	// tool.cpp: tool name + tile coords where it was applied.
	didTool: ['toolName', 'x', 'y'],
	// fileio.cpp: path of the city file that failed to load.
	didntLoadCity: ['fileName'],
	// fileio.cpp: path of the city file that failed to save.
	didntSaveCity: ['fileName'],
	// micropolis.cpp: sound channel ("city"/"interface"), sound effect name, tile coords (-1,-1 = non-positional).
	makeSound: ['channel', 'sound', 'x', 'y'],
	// fileio.cpp: path the city is being saved as.
	saveCityAs: ['fileName'],
	// sendMessage(messageNum, x, y, pictureFlag, important) — text.h has the message enum;
	// messageText is derived from messageNum, not a positional arg (see namedArgs below).
	sendMessage: ['messageNum', 'x', 'y', 'pictureFlag', 'important'],
	// tool.cpp: zone status popup — string-table indices per category, plus tile coords.
	showZoneStatus: [
		'tileCategoryIndex',
		'populationDensityIndex',
		'landValueIndex',
		'crimeRateIndex',
		'pollutionIndex',
		'growthRateIndex',
		'x',
		'y'
	],
	// zone.cpp: church zone tile coords + which church tile/sprite variant (0-7).
	simulateChurch: ['x', 'y', 'churchNumber'],
	// micropolis.cpp doEarthquake(strength): earthquake magnitude.
	startEarthquake: ['strength'],
	// micropolis.cpp doStartScenario(scenario): Scenario enum id being started.
	startScenario: ['scenario'],
	// utilities.cpp: city name.
	updateCityName: ['cityName'],
	// updateDemand(r, c, i) — update.cpp setDemand: residential/commercial/industrial demand valve values.
	updateDemand: ['residentialDemand', 'commercialDemand', 'industrialDemand'],
	// update.cpp: current totalFunds.
	updateFunds: ['totalFunds'],
	// utilities.cpp: GameLevel enum value.
	updateGameLevel: ['gameLevel'],
	// utilities.cpp setPasses: simPasses, sub-passes run per tick.
	updatePasses: ['passes'],
	// utilities.cpp pause()/resume(): simPaused.
	updatePaused: ['paused'],
	// utilities.cpp setSpeed: simSpeed level (0-3).
	updateSpeed: ['speed'],
	// updateTaxRate(cityTax): current tax rate.
	updateTaxRate: ['taxRate'],
	// update.cpp: cityYear/cityMonth as of this date update (redundant with the
	// base event fields, named for clarity when --verbose-log is on).
	updateDate: ['cityYear', 'cityMonth']
};

function namedArgs(name, args) {
	const names = ARG_NAMES[name];
	const out = {};
	args.forEach((value, i) => {
		const key = names?.[i] ?? `arg${i}`;
		out[key] = unwrap(value);
	});
	if (name === 'sendMessage') {
		out.messageText = messageText(out.messageNum);
	}
	return out;
}

function makeLoggingCallback(engine, micropolis, getTick, logStream, verbose, onMessage) {
	const handlers = {};
	for (const name of callbackMethodNames) {
		if (name === 'sendMessage') {
			handlers[name] = (...allArgs) => {
				onMessage(unwrap(allArgs[2]));
				logStream.write(JSON.stringify({ ...eventBase(micropolis, getTick(), name), ...namedArgs(name, allArgs.slice(2)) }) + '\n');
			};
			continue;
		}
		if (!verbose && LOW_SIGNAL_EVENTS.has(name)) {
			handlers[name] = () => { };
			continue;
		}
		handlers[name] = (...allArgs) => {
			// Every callback's first two args are the Micropolis* and an opaque
			// user-data handle — not useful to log, so they're dropped here.
			const args = allArgs.slice(2);
			const entry = eventBase(micropolis, getTick(), name);
			if (name === 'updateEvaluation') {
				Object.assign(entry, fieldSnapshot(micropolis, EVALUATION_FIELDS));
			} else if (name === 'updateOptions') {
				Object.assign(entry, fieldSnapshot(micropolis, OPTIONS_FIELDS));
			} else {
				Object.assign(entry, namedArgs(name, args));
			}
			logStream.write(JSON.stringify(entry) + '\n');
		};
	}
	return new engine.JSCallback(handlers);
}

// Builds the stats and event log paths for this run:
//   <outputBaseDir>/<city>/log-seed<seed>[-no]disasters.jsonl
//   <outputBaseDir>/<city>/events-seed<seed>[-no]disasters.jsonl
function outputPathsFor(outputBaseDir, city, seed, disastersEnabled, interventions) {
	const suffix = (disastersEnabled ? '-disasters' : '-nodisasters') + interventionsTag(interventions);
	const dir = path.join(outputBaseDir, city);
	return {
		dir,
		statsPath: path.join(dir, `log-seed${seed}${suffix}.jsonl`),
		logPath: path.join(dir, `events-seed${seed}${suffix}.jsonl`)
	};
}

async function main() {
	const cityPath = `/cities/${argv.city}.cty`;
	const engine = await loadMicropolisMainModule();
	const micropolis = new engine.Micropolis();

	let interventions;
	try {
		interventions = parseInterventions(argv.at);
	} catch (err) {
		console.error(err.message);
		process.exitCode = 1;
		return;
	}
	const { dir, statsPath, logPath } = outputPathsFor(argv.outputBaseDir, argv.city, argv.seed, argv.disasters, interventions);
	fs.mkdirSync(dir, { recursive: true });

	const logStream = fs.createWriteStream(logPath, { flags: 'w' });
	const tickRef = { current: 0 };
	let messages = [];
	const sentMessages = () => {
		const sent = messages;
		messages = [];
		return sent;
	};
	const callback = makeLoggingCallback(engine, micropolis, () => tickRef.current, logStream, argv.verboseLog, (num) => messages.push(num));
	micropolis.setCallback(callback, {});
	micropolis.init();

	// init() unconditionally sets enableDisasters = true (micropolis.cpp), so
	// this must run after it; loadCity() doesn't touch the flag, so it's safe
	// to set before or after that call.
	if (!argv.disasters) {
		micropolis.enableDisasters = false;
	}

	// Must be before loadCity(): the load's map scans draw from the RNG and
	// write the results into map tiles (zone.cpp), so a seed applied afterwards
	// arrives too late to determine the loaded world. init() seeds from the wall
	// clock so that an unseeded game is random, which is why this has to come
	// after init() but before the load.
	micropolis.seedRandom(argv.seed);

	const loaded = micropolis.loadCity(cityPath);
	if (!loaded) {
		console.error(`Failed to load builtin city "${argv.city}" (${cityPath})`);
		micropolis.delete();
		process.exitCode = 1;
		return;
	}

	console.log(`Loaded ${argv.city} — running ${totalTicks} ticks, writing stats to ${statsPath} and events to ${logPath}`);
	if (!argv.logEveryTick) {
		console.log('Logging one row per cityTime value (pass --log-every-tick to log every tick instead).');
	}

	// phaseCycle wraps 0..15 (simulate.cpp). Phase 0 increments cityTime and
	// runs cityEvaluation() (cityPop, cityScore, cityClass) on the *first*
	// cycle after load; phases 1..14 rebuild the census/map scans. Logging at
	// phaseCycle === 0 catches the tick before any of that ran for this
	// cityTime value (cityPop still 0). phaseCycle === 15 is the last phase
	// of the cycle, after every scan for the current cityTime has completed.
	const shouldLog = (m) => argv.logEveryTick || m.phaseCycle === 15;

	// Live Uint16 view over the engine's map for the per-snapshot tile census.
	// Re-derived per snapshot: the WASM heap can grow and detach old views.
	const mapView = () => {
		const heap = heapU16FromEmscriptenModule(engine);
		const start = micropolis.getMapAddress() / 2;
		return heap.subarray(start, start + micropolis.getMapSize() / 2);
	};

	const out = fs.createWriteStream(statsPath, { flags: 'w' });
	let rowsWritten = 0;
	if (shouldLog(micropolis)) {
		out.write(JSON.stringify(snapshot(micropolis, 0, mapView())) + '\n');
		rowsWritten++;
	}

	// Interventions at turn T run after tick interventionTick(T), the branch tick
	// of run_continuations.js --branch-at T+1; their events carry tick 16T+15.
	let nextIv = 0;
	const intervene = (tick) => {
		while (nextIv < interventions.length && interventionTick(interventions[nextIv].turn) === tick) {
			const iv = interventions[nextIv++];
			tickRef.current = interventionEventTick(iv.turn);
			const entry = { ...eventBase(micropolis, tickRef.current, 'intervention'), turn: iv.turn, action: iv.action };
			const result = applyIntervention(micropolis, iv, sentMessages);
			if (result !== undefined) entry.result = result;
			logStream.write(JSON.stringify(entry) + '\n');
		}
	};

	let failure = null;
	try {
		for (let tick = 1; tick <= totalTicks; tick++) {
			tickRef.current = tick;
			micropolis.simTick();
			if (shouldLog(micropolis)) {
				out.write(JSON.stringify(snapshot(micropolis, tick, mapView())) + '\n');
				rowsWritten++;
			}
			intervene(tick);
		}
	} catch (err) {
		if (!(err instanceof InterventionError)) throw err;
		failure = err;
	}
	if (!failure && nextIv < interventions.length) {
		failure = new InterventionError(`${interventions.length - nextIv} intervention(s) fall after the last turn and were not applied`);
	}

	await Promise.all([
		new Promise((resolve, reject) => out.end((err) => (err ? reject(err) : resolve()))),
		new Promise((resolve, reject) => logStream.end((err) => (err ? reject(err) : resolve())))
	]);

	micropolis.delete();
	if (failure) {
		// Leave no partial output that a cache could mistake for a finished run.
		fs.rmSync(statsPath, { force: true });
		fs.rmSync(logPath, { force: true });
		console.error(`Error: ${failure.message}`);
		process.exitCode = 1;
		return;
	}
	console.log(`Done. Wrote ${rowsWritten} rows to ${statsPath}`);
}

await main();
