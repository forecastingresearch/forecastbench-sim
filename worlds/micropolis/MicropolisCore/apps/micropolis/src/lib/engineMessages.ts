/** Advisory strings mirrored from micropolis-engine message indices (see packages/micropolis-engine/src/text.h). */
export const ENGINE_MESSAGES: Record<number, string> = {
	1: 'More residential zones needed.',
	2: 'More commercial zones needed.',
	3: 'More industrial zones needed.',
	4: 'More roads required.',
	5: 'Inadequate rail system.',
	6: 'Build a power plant.',
	7: 'Residents demand a Stadium.',
	8: 'Industry requires a Sea Port.',
	9: 'Commerce requires an Airport.',
	10: 'Pollution very high.',
	11: 'Crime very high.',
	12: 'Frequent traffic jams reported.',
	13: 'Citizens demand a Fire Department.',
	14: 'Citizens demand a Police Department.',
	15: 'Blackouts reported. Check power map.',
	16: 'Citizens upset. The tax rate is too high.',
	17: 'Roads deteriorating, due to lack of funds.',
	18: 'Fire departments need funding.',
	19: 'Police departments need funding.',
	20: 'Fire reported!',
	21: 'A monster has been sighted!',
	22: 'Tornado reported!',
	23: 'Major earthquake reported!',
	24: 'A plane has crashed!',
	25: 'Shipwreck reported!',
	26: 'A train crashed!',
	27: 'A helicopter crashed!',
	28: 'Unemployment rate is high.',
	29: 'Your city has gone broke!',
	30: 'Firebombing reported!',
	31: 'Need more parks.',
	32: 'Explosion detected!',
	33: 'Insufficient funds to build that.',
	34: 'Area must be bulldozed first.',
	35: 'Population has reached 2,000.',
	36: 'Population has reached 10,000.',
	37: 'Population has reached 50,000.',
	38: 'Population has reached 100,000.',
	39: 'Population has reached 500,000.',
	40: 'Brownouts, build another Power Plant.',
	41: 'Heavy Traffic reported.',
	42: 'Flooding reported!',
	43: 'A Nuclear Meltdown has occurred!',
	44: "They're rioting in the streets!",
	45: 'Started a New City.',
	46: 'Restored a Saved City.',
	47: 'You won the scenario.',
	48: 'You lost the scenario.',
	49: 'About Micropolis.',
	50: 'Dullsville scenario.',
	51: 'San Francisco scenario.',
	52: 'Hamburg scenario.',
	53: 'Bern scenario.',
	54: 'Tokyo scenario.',
	55: 'Detroit scenario.',
	56: 'Boston scenario.',
	57: 'Rio de Janeiro scenario.'
};

/**
 * The engine's bankruptcy signal — "YOUR CITY HAS GONE BROKE!", raised by
 * doBudgetNow() when autoBudget is on and the treasury plus this year's tax
 * take cannot cover the requested police/fire/road spending (budget.cpp).
 */
export const MESSAGE_NO_MONEY = 29;

/**
 * Disaster and incident messages, mapped to the short stable slugs the CLI
 * sweep tooling uses as record keys. Import this instead of writing a local
 * number -> name map.
 *
 * The numbers are the `MessageNumber` enum in
 * `packages/micropolis-engine/src/text.h`, which carries only two explicit `=`
 * anchors (`MESSAGE_NEED_MORE_RESIDENTIAL = 1` and `MESSAGE_LAST = 57`), so
 * every other value has to be counted — 42/43/44 are especially easy to shift
 * by one, which silently files meltdowns as floods. `emscripten.cpp` annotates
 * every value with its number and is the cross-check; `engineMessages.test.ts`
 * parses the enum out of text.h and asserts these values still agree with it.
 *
 * This covers only the messages that mean something damaged the city — the
 * advisories, funding warnings, population milestones and scenario messages
 * are deliberately left out.
 */
export const DISASTER_KINDS: Record<number, string> = {
	20: 'fire',
	21: 'monster',
	22: 'tornado',
	23: 'earthquake',
	24: 'planeCrash',
	25: 'shipCrash',
	26: 'trainCrash',
	27: 'helicopterCrash',
	30: 'firebombing',
	32: 'explosion',
	42: 'flood',
	43: 'meltdown',
	44: 'riots'
};

/**
 * The `MESSAGE_*` enum name each DISASTER_KINDS slug is expected to sit on.
 * Kept beside the map so the test can check both halves against text.h.
 */
export const DISASTER_ENUM_NAMES: Record<string, string> = {
	fire: 'MESSAGE_FIRE_REPORTED',
	monster: 'MESSAGE_MONSTER_SIGHTED',
	tornado: 'MESSAGE_TORNADO_SIGHTED',
	earthquake: 'MESSAGE_EARTHQUAKE',
	planeCrash: 'MESSAGE_PLANE_CRASHED',
	shipCrash: 'MESSAGE_SHIP_CRASHED',
	trainCrash: 'MESSAGE_TRAIN_CRASHED',
	helicopterCrash: 'MESSAGE_HELICOPTER_CRASHED',
	firebombing: 'MESSAGE_FIREBOMBING',
	explosion: 'MESSAGE_EXPLOSION_REPORTED',
	flood: 'MESSAGE_FLOODING_REPORTED',
	meltdown: 'MESSAGE_NUCLEAR_MELTDOWN',
	riots: 'MESSAGE_RIOTS_REPORTED'
};

export function messageText(index: number): string {
	if (index < 0) return '';
	return ENGINE_MESSAGES[index] ?? `City message #${index}`;
}
