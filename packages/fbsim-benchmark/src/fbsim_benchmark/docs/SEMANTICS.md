# Contracts and compatibility

These interfaces process cached data. They do not replace world engines, live evaluation scripts, or their parsers. `fbsim-core` remains the sole provider of question/conditional schemas, resolvers and historical realized-outcome metrics. The new distribution owns only `fbsim_benchmark`.

`CachedForecast.parse` accepts exactly schema_version, world, metric, forecast and truth. Version is the string `"1"`; worlds are freeciv, micropolis and starsim. Binary metrics require scalar probabilities in [0,1]. quantile_crps requires five finite nondecreasing forecast quantiles and a nonempty array of finite replay outcomes. Booleans, missing/extra fields and nonfinite values are rejected. JSON Schema enforces metric-specific shapes; runtime additionally checks ordering and finiteness. The record is a scoring primitive, not a complete experiment manifest: callers must separately retain question/model identity, truth provenance, selection and replay alignment.

| Metric | Meaning | Not equivalent to |
|---|---|---|
| excess_brier | Squared distance between forecast and reference probability | Core Brier score against a realized binary outcome |
| excess_bits | Bernoulli KL divergence in bits with the inherited 0.001 forecast-probability clipping convention (the reference probability is not clipped) | Unclipped divergence or realized log loss |
| quantile_crps | Raw five-quantile approximation averaged over supplied replay outcomes | Paper excess nCRPS, normalized/combined scores, or full-distribution CRPS |

FreeCiv uses quantiles .05/.25/.50/.75/.95; Micropolis and Starsim use .10/.25/.50/.75/.90. Matching these quantiles need not identify a full distribution. The preserved native functions have different signatures and conventions; they are not substitutes for one another merely because their names resemble each other.

See NATIVE_INTERFACES.md for each parser's arguments and missing-value convention. In particular, strict Starsim binary parsing and historical Starsim JSON parsing are not the last-number/clamping parser in worlds/pandemic/pandemic_world/scale_eval.py. No existing public parser is replaced. FreeCiv accepts some bare percentages, Micropolis requires the percent sign, and historical Starsim continuous parsing sorts nonnegative values. These are retained behaviors, not a newly standardized repair policy. Paper missingness and eligibility rules remain paper-specific.

The fixtures contain invented values and synthetic response snippets, not administered prompts or real forecasts. Cached trajectory conversion tests do not run a simulator. The package imports no model provider or world engine. Core's existing dependency metadata still installs provider libraries; changing that layering is outside this additive change.

Starsim binary parsing defaults to the new strict contract. `binary-historical` is the explicit administered compatibility mode; see NATIVE_INTERFACES.md for its numeric-prefix/coercion defects. A successful CLI command always emits standard finite JSON. Runtime input finiteness does not guarantee finite arithmetic for extreme values; output serialization checks that boundary without changing the scorers.

Native parser diagnostic messages go to stderr in the CLI, leaving stdout as a single JSON value (including null for a native missing-value result).
