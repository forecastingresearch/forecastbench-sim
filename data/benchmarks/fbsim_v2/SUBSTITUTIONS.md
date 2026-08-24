# Deterministic-crasher seed substitutions (2026-08-24)
These seeds hung or failed >=3 times each across the campaign; each is replaced
by an out-of-range seed. Seeds are arbitrary RNG inputs, so a substitute is
statistically identical -- same anchor, same t60 state, same t210 horizon.
| anchor | failed seed | substitute | evidence |
|---|---|---|---|
| s0   | 5076 | 6002 | hung 88min after 4 prior failures |
| s1   | 5449 | 6003 | hung 88min |
| s1   | 5757 | 6004 | hung 88min |
| s345 | 5669 | 6005 | hung 85min; failed repeatedly since 07:42 |
| s355 | 5747 | 6006 | 4 recorded FAILED markers |
