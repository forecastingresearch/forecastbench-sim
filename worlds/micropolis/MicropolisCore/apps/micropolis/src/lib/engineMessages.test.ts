import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
	DISASTER_ENUM_NAMES,
	DISASTER_KINDS,
	ENGINE_MESSAGES,
	MESSAGE_NO_MONEY,
	messageText
} from './engineMessages';

/**
 * Parse the MessageNumber enum out of the engine source.
 *
 * text.h carries only two explicit `=` anchors, so every other value is
 * implied by position. That is exactly why a hand-written number -> name map
 * drifts: mis-count once and meltdowns get filed as floods. Reading the enum
 * here makes the C++ the single source of truth for these tests.
 */
function parseMessageEnum(): Map<string, number> {
	const source = readFileSync(
		new URL('../../../../packages/micropolis-engine/src/text.h', import.meta.url),
		'utf8'
	);
	const start = source.indexOf('MESSAGE_NEED_MORE_RESIDENTIAL');
	expect(start, 'MessageNumber enum not found in text.h').toBeGreaterThan(-1);

	const values = new Map<string, number>();
	let next = 0;
	for (const line of source.slice(start).split('\n')) {
		const match = /^\s*(MESSAGE_[A-Z0-9_]+)\s*(?:=\s*(-?\d+))?/.exec(line);
		if (!match) {
			// The enum's closing brace ends the run of value lines.
			if (/^\s*\}/.test(line)) break;
			continue;
		}
		const [, name, explicit] = match;
		next = explicit === undefined ? next + 1 : Number(explicit);
		values.set(name, next);
	}
	return values;
}

const ENUM = parseMessageEnum();

describe('MessageNumber enum parsing', () => {
	it('reads the anchors and the last value from text.h', () => {
		expect(ENUM.get('MESSAGE_NEED_MORE_RESIDENTIAL')).toBe(1);
		expect(ENUM.get('MESSAGE_LAST')).toBe(57);
		// MESSAGE_LAST is an alias for the final real message, not a value of its own.
		expect(ENUM.get('MESSAGE_SCENARIO_RIO_DE_JANEIRO')).toBe(57);
	});
});

describe('DISASTER_KINDS', () => {
	it('names every slug it maps', () => {
		for (const slug of Object.values(DISASTER_KINDS)) {
			expect(DISASTER_ENUM_NAMES, `no enum name recorded for "${slug}"`).toHaveProperty(slug);
		}
		expect(Object.keys(DISASTER_ENUM_NAMES).sort()).toEqual(
			Object.values(DISASTER_KINDS).sort()
		);
	});

	it('sits on the message numbers text.h actually assigns', () => {
		for (const [number, slug] of Object.entries(DISASTER_KINDS)) {
			const enumName = DISASTER_ENUM_NAMES[slug];
			expect(ENUM.has(enumName), `${enumName} missing from text.h`).toBe(true);
			expect(ENUM.get(enumName), `${slug} should be ${enumName}`).toBe(Number(number));
		}
	});

	it('pins the three values that are easiest to shift by one', () => {
		expect(DISASTER_KINDS[42]).toBe('flood');
		expect(DISASTER_KINDS[43]).toBe('meltdown');
		expect(DISASTER_KINDS[44]).toBe('riots');
	});
});

describe('MESSAGE_NO_MONEY', () => {
	it('matches the enum and the advisory text', () => {
		expect(MESSAGE_NO_MONEY).toBe(ENUM.get('MESSAGE_NO_MONEY'));
		expect(messageText(MESSAGE_NO_MONEY).toLowerCase()).toContain('broke');
	});
});

describe('ENGINE_MESSAGES', () => {
	it('covers 1..MESSAGE_LAST with no gaps', () => {
		const last = ENUM.get('MESSAGE_LAST')!;
		const indices = Object.keys(ENGINE_MESSAGES).map(Number).sort((a, b) => a - b);
		expect(indices[0]).toBe(1);
		expect(indices[indices.length - 1]).toBe(last);
		expect(indices).toHaveLength(last);
	});

	it('describes the same event the enum name does, at every disaster index', () => {
		// The advisory wording is not the enum name, so match on the word the two
		// share. A one-off shift breaks every pair below at once.
		const keyword: Record<string, string> = {
			fire: 'fire',
			monster: 'monster',
			tornado: 'tornado',
			earthquake: 'earthquake',
			planeCrash: 'plane',
			shipCrash: 'shipwreck',
			trainCrash: 'train',
			helicopterCrash: 'helicopter',
			firebombing: 'firebombing',
			explosion: 'explosion',
			flood: 'flooding',
			meltdown: 'meltdown',
			riots: 'rioting'
		};
		for (const [number, slug] of Object.entries(DISASTER_KINDS)) {
			expect(messageText(Number(number)).toLowerCase(), `index ${number} (${slug})`).toContain(
				keyword[slug]
			);
		}
	});
});

/**
 * The Python analyzers keep their own number -> label maps (they cannot import
 * this module). They are only three entries deep in risk, so rather than
 * duplicating the whole table, pin the trio that actually drifts. A shift in
 * either file fails here.
 */
describe('the Python analyzers agree on the flood/meltdown/riots trio', () => {
	const FILES = ['../../cli/analyze_sweep.py', '../../cli/plot_sim_run.py'];
	const EXPECTED: Array<[number, string]> = [
		[42, 'flood'],
		[43, 'meltdown'],
		[44, 'riot']
	];

	for (const file of FILES) {
		it(file.replace('../../', ''), () => {
			const source = readFileSync(new URL(file, import.meta.url), 'utf8');
			// Matches both `42: "flood",` and `42: ('Flooding', 'tab:blue'),`.
			const labels = new Map<number, string>();
			for (const [, num, label] of source.matchAll(/(\d+):\s*\(?['"]([^'"]+)['"]/g)) {
				labels.set(Number(num), label.toLowerCase());
			}
			for (const [number, keyword] of EXPECTED) {
				expect(labels.has(number), `no label parsed at index ${number}`).toBe(true);
				expect(labels.get(number), `index ${number}`).toContain(keyword);
			}
		});
	}
});
