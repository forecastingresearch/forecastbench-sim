# Native cached interfaces

No API key, simulator, parameter recipe or private data is needed to import these interfaces. They consume already acquired text or trajectories. They do not reproduce the full live benchmark.

`fbsim_benchmark.parsing.registry.parse_response(world, format, text, **options)` selects these native routines:

| World | Format | Options | Return |
|---|---|---|---|
| FreeCiv | binary-single | optional mode | native probability, optionally parse mode |
| FreeCiv | continuous-single | optional mode | p5/p25/p50/p75/p95 list, optionally mode |
| FreeCiv | batch | positions, kind (`binary` or `cont`) | position map and native mode |
| Micropolis | binary-single | optional quiet | probability or None |
| Micropolis | binary-batch | labels; optional quiet | one probability/None per label |
| Micropolis | continuous-single | optional quiet | p10/p25/p50/p75/p90 dict or None |
| Micropolis | continuous-batch | labels; optional quiet | one dict/None per label |
| Micropolis | continuous-semantic | labels, tags; optional quiet | dict/None in requested tag order |
| Starsim | binary | keys | strict key/probability dict; raises ValueError on invalid input |
| Starsim | binary-historical | keys | historical key/probability dict or None; may coerce invalid tokens |
| Starsim | continuous | keys | native key/quantile dict or None |

These parsers deliberately retain historical differences. FreeCiv accepts bare percentages greater than one; Micropolis requires an explicit percent sign. Micropolis rejects nonmonotone quantiles. The administered Starsim continuous parser sorts nonnegative values. Parsing therefore does not imply a common forecast-repair policy. Missing/unparseable values retain native conventions; selection and missingness treatment belong in paper analysis. FreeCiv's five-quantile grid differs from the other two worlds.

CLI example from the repository root, using the included synthetic response:

```sh
uv run --offline --no-project --python .venv-offline/bin/python -m fbsim_benchmark parse --world micropolis --format binary-single --input packages/fbsim-benchmark/fixtures/micropolis-response.txt --options '{"quiet":true}'
```

`fbsim_benchmark.worlds.starsim.to_world(region_series, names)` converts cached metric arrays to the core turn-major schema. `worlds.micropolis.to_world(sims, metrics=..., ticks_per_turn=...)` accepts objects exposing `log_data`; schema constants are explicit arguments instead of production globals. FreeCiv already emits the turn-major representation consumed by `fbsim_core.questions.resolver`; no replacement FreeCiv engine serializer is claimed. Existing core question and conditional schemas/loaders remain available.

Scoring is separate: `CachedForecast` validates the version-1 record and dispatches to coauthor numerical routines. Paper normalization, exclusions, repeated-elicitation aggregation and nine-cell fitting are not benchmark defaults. Synthetic tests exercise parsing boundaries and conversion; no private prompt text is included.

Strict Starsim binary parsing reads the last flat JSON object, requires all requested keys and finite numeric values in [0,1], and rejects booleans, strings, duplicate keys and malformed numeric tokens. Valid scientific notation is accepted as a whole number, not a prefix. It does not fall back to an earlier object or unbraced answers.

`binary-historical` and the original `starsim_binary.parse` preserve the administered implementation exactly. Its fallback regex can parse 1.5 or 10 as 1.0, 0.8e2 as 0.8, and its JSON path can coerce true to 1.0. These are documented compatibility defects, not valid probability validation. The compatibility mode exists to inspect historical parsing; frozen inputs and scores are not recomputed.

Other native parsers can return nonfinite values. The CLI refuses to serialize them and exits unsuccessfully. CachedForecast validates input finiteness but extreme finite inputs may overflow during arithmetic; the CLI likewise rejects a nonfinite score. No numerical scorer is changed by this output check. Installed copies of these documents can be read with `python -m fbsim_benchmark docs --name NATIVE_INTERFACES.md`.
