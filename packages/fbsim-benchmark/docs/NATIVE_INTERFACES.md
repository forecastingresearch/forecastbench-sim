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
| Starsim | binary | keys | native key/probability dict or None |
| Starsim | continuous | keys | native key/quantile dict or None |

These parsers deliberately retain historical differences. FreeCiv accepts bare percentages greater than one; Micropolis requires an explicit percent sign. Micropolis rejects nonmonotone quantiles. The administered Starsim continuous parser sorts nonnegative values. Parsing therefore does not imply a common forecast-repair policy. Missing/unparseable values retain native conventions; selection and missingness treatment belong in paper analysis. FreeCiv's five-quantile grid differs from the other two worlds.

CLI example from the repository root, using the included synthetic response:

```sh
uv run --offline --no-project --python .venv-offline/bin/python -m fbsim_benchmark parse --world micropolis --format binary-single --input packages/fbsim-benchmark/fixtures/micropolis-response.txt --options '{"quiet":true}'
```

`fbsim_benchmark.worlds.starsim.to_world(region_series, names)` converts cached metric arrays to the core turn-major schema. `worlds.micropolis.to_world(sims, metrics=..., ticks_per_turn=...)` accepts objects exposing `log_data`; schema constants are explicit arguments instead of production globals. FreeCiv already emits the turn-major representation consumed by `fbsim_core.questions.resolver`; no replacement FreeCiv engine serializer is claimed. Existing core question and conditional schemas/loaders remain available.

Scoring is separate: `CachedForecast` validates the version-1 record and dispatches to coauthor numerical routines. Paper normalization, exclusions, repeated-elicitation aggregation and nine-cell fitting are not benchmark defaults. Synthetic tests exercise parsing boundaries and conversion; no private prompt text is included.
