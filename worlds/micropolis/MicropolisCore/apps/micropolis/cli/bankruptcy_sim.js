#!/usr/bin/env tsx

/**
 * Bankruptcy sweep: how often, and how soon, does each builtin city go broke?
 *
 * The engine's own bankruptcy signal is MESSAGE_NO_MONEY ("YOUR CITY HAS GONE
 * BROKE!"), sent from doBudgetNow() when autoBudget is on and
 * taxFund + totalFunds <= the requested police+fire+road spending
 * (budget.cpp). That same branch calls setAutoBudget(false), so the message
 * can only fire once per run: afterwards the budget takes the manual path,
 * which silently spends whatever is affordable.
 *
 * Because of that, this runner also tracks the treasury directly (min funds,
 * first turn at/below zero, time spent at zero), which is the definition that
 * survives autoBudget being flipped off.
 *
 * Seeding follows run_sim.js/sweep_sim.js: seedRandom() before loadCity().
 *
 * Message numbers come from src/lib/engineMessages.ts (MESSAGE_NO_MONEY and
 * DISASTER_KINDS) rather than a local map — see engineMessages.test.ts, which
 * checks them against the enum in the engine's text.h.
 *
 * Usage:
 *   pnpm tsx cli/bankruptcy_sim.js --cities all --seeds 30 --years 100 \
 *       --out /tmp/bankruptcy.jsonl
 */

import fs from 'node:fs';
import path from 'node:path';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import { loadMicropolisMainModule } from '../src/lib/wasm/node.ts';
import { callbackMethodNames } from '../src/lib/wasm/callbacks.ts';
import { DISASTER_KINDS, MESSAGE_NO_MONEY } from '../src/lib/engineMessages.ts';

const TICKS_PER_TURN = 16; // one cityTime increment (simulate.cpp phaseCycle)
const TURNS_PER_YEAR = 48; // CITYTIMES_PER_YEAR

// The populated, non-scenario builtin cities plus the empty/decorative ones,
// so --cities all stays honest; callers pick the subset they want.
const BUILTIN_CITIES = [
	'about', 'badnews', 'bluebird', 'bruce', 'deadwood', 'finnigan', 'freds',
	'haight', 'happisle', 'joffburg', 'kamakura', 'kobe', 'kowloon', 'kyoto',
	'linecity', 'med_isle', 'ndulls', 'neatmap', 'radial', 'senri', 'southpac',
	'splats', 'wetcity', 'yokohama'
];

const argv = yargs(hideBin(process.argv))
	.scriptName('bankruptcy_sim')
	.usage('$0 --cities <a,b|all> --seeds <n> --years <n> --out <file>')
	.option('cities', { type: 'string', default: 'all' })
	.option('seeds', { type: 'number', default: 30 })
	.option('seed-base', { type: 'number', default: 0 })
	.option('years', { type: 'number', default: 100 })
	.option('disasters', { type: 'boolean', default: true })
	.option('game-level', { type: 'number', describe: 'Force level 0..2, applied AFTER loadCity().' })
	.option('out', { type: 'string', demandOption: true })
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

function unwrap(v) {
	return v && typeof v === 'object' && 'value' in v ? v.value : v;
}

async function main() {
	const engine = await loadMicropolisMainModule();
	const GAME_LEVELS = [engine.GameLevel.LEVEL_EASY, engine.GameLevel.LEVEL_MEDIUM, engine.GameLevel.LEVEL_HARD];
	if (argv.out !== '-') fs.mkdirSync(path.dirname(path.resolve(argv.out)), { recursive: true });
	const out = argv.out === '-' ? process.stdout : fs.createWriteStream(argv.out, { flags: 'w' });

	const totalTurns = Math.round(argv.years * TURNS_PER_YEAR);
	let runCount = 0;
	const t0 = Date.now();

	for (const city of cities) {
		for (let s = 1; s <= argv.seeds; s++) {
			const seed = argv.seedBase + s;
			const micropolis = new engine.Micropolis();

			let turnRef = 0;
			const brokeTurns = [];
			const disasterCounts = {};
			// [turn, kind] per incident, so cumulative counts can be cut at any
			// horizon downstream instead of only at the end of the run.
			const disasterEvents = [];
			const handlers = {};
			for (const name of callbackMethodNames) handlers[name] = () => {};
			const logDisaster = (kind) => {
				disasterCounts[kind] = (disasterCounts[kind] || 0) + 1;
				disasterEvents.push([turnRef, kind]);
			};
			handlers.sendMessage = (_m, _d, messageNum) => {
				const num = unwrap(messageNum);
				if (num === MESSAGE_NO_MONEY) brokeTurns.push(turnRef);
				const kind = DISASTER_KINDS[num];
				if (kind) logDisaster(kind);
			};
			handlers.startEarthquake = () => logDisaster('quakeStart');
			const callback = new engine.JSCallback(handlers);

			micropolis.setCallback(callback, {});
			micropolis.init();
			if (!argv.disasters) micropolis.enableDisasters = false;
			micropolis.seedRandom(seed);
			if (!micropolis.loadCity(`/cities/${city}.cty`)) {
				console.error(`Failed to load ${city}`);
				process.exit(1);
			}
			if (argv.gameLevel !== undefined) micropolis.setGameLevel(GAME_LEVELS[argv.gameLevel]);

			const start = {
				funds: micropolis.totalFunds,
				tax: micropolis.cityTax,
				autoBudget: micropolis.autoBudget,
				gameLevel: unwrap(micropolis.gameLevel),
				cityTime: micropolis.cityTime,
				cityYear: micropolis.cityYear,
				cityPop: micropolis.cityPop,
				totalPop: micropolis.totalPop,
				cityScore: micropolis.cityScore,
				cityClass: unwrap(micropolis.cityClass)
			};

			let minFunds = start.funds;
			let minFundsTurn = 0;
			let firstZeroTurn = null;      // first turn with totalFunds <= 0
			let turnsAtZero = 0;
			let peakFunds = start.funds;
			let peakFundsTurn = 0;
			let peakPop = start.cityPop;
			let peakPopTurn = 0;
			const yearly = [];             // compact yearly trace

			for (let turn = 1; turn <= totalTurns; turn++) {
				turnRef = turn;
				for (let t = 0; t < TICKS_PER_TURN; t++) micropolis.simTick();

				const funds = micropolis.totalFunds;
				if (funds < minFunds) { minFunds = funds; minFundsTurn = turn; }
				if (funds > peakFunds) { peakFunds = funds; peakFundsTurn = turn; }
				if (funds <= 0) {
					if (firstZeroTurn === null) firstZeroTurn = turn;
					turnsAtZero++;
				}
				const pop = micropolis.cityPop;
				if (pop > peakPop) { peakPop = pop; peakPopTurn = turn; }

				if (turn % TURNS_PER_YEAR === 0 || turn === totalTurns) {
					yearly.push([
						Math.round(turn / TURNS_PER_YEAR),
						funds,
						pop,
						micropolis.cashFlow,
						micropolis.cityTax,
						micropolis.autoBudget ? 1 : 0,
						micropolis.cityScore,
						micropolis.totalPop
					]);
				}
			}

			out.write(JSON.stringify({
				city, seed, disasters: argv.disasters, years: argv.years,
				start,
				end: {
					funds: micropolis.totalFunds,
					cityPop: micropolis.cityPop,
					totalPop: micropolis.totalPop,
					cityScore: micropolis.cityScore,
					cityClass: unwrap(micropolis.cityClass),
					tax: micropolis.cityTax,
					autoBudget: micropolis.autoBudget
				},
				brokeTurns, brokeCount: brokeTurns.length,
				firstBrokeTurn: brokeTurns.length ? brokeTurns[0] : null,
				minFunds, minFundsTurn, firstZeroTurn, turnsAtZero,
				peakFunds, peakFundsTurn, peakPop, peakPopTurn,
				disasterCounts, disasterEvents,
				yearlyFields: ['year', 'funds', 'cityPop', 'cashFlow', 'tax', 'autoBudget', 'cityScore', 'totalPop'],
				yearly
			}) + '\n');

			micropolis.delete();
			try { callback.delete(); } catch { /* fine to leak in a CLI */ }
			runCount++;
			if (argv.out !== '-') {
				const rate = (Date.now() - t0) / 1000 / runCount;
				process.stderr.write(`\r${runCount} runs done (${city} seed ${seed}) ${rate.toFixed(1)}s/run   `);
			}
		}
	}

	if (argv.out !== '-') {
		process.stderr.write('\n');
		await new Promise((resolve, reject) => out.end((err) => (err ? reject(err) : resolve())));
	}
	console.error(`Wrote ${runCount} runs in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}

await main();
