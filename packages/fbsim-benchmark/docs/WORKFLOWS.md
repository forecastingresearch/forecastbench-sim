# Workflow map

| Workflow | Location and status | Inputs / effects |
|---|---|---|
| Core question resolution and realized-outcome scoring | packages/fbsim-core, retained byte-for-byte | Cached world/question records; original public tests |
| FreeCiv engine, service CLI, benchmark preparation, mining and provider evaluation | worlds/freeciv, retained byte-for-byte | Live engines/providers or production data; not run by offline tests |
| Pandemic runners, templates, corpus/probe exporters, null-conditional and scale evaluation | worlds/pandemic, retained byte-for-byte | Live simulations and potentially paid model calls; not an offline smoke path |
| Historical human-subject survey/generation/power materials | human_subjects, retained in tree and history | Existing research artifacts; not migrated, deleted or newly validated |
| Documentation deployment | Existing .github/workflows/doc.yml, retained | Existing deployment policy; separate from new test CI |
| Cached native parsing, scoring, trajectory conversion | packages/fbsim-benchmark, migrated and tested on synthetic fixtures | Local JSON/text/arrays; no network, engines or private inputs |
| Paper repaired continuous Starsim and nine-cell aggregation; cached FreeCiv/Micropolis tables and figures | Separate private paper home, retained executable routes | Versioned private frozen inputs; not bundled here |
| Exploratory/legacy plotting, raw-cache gathering, historical comparison and production orchestration | Preserved original private sources; not implemented as maintained new commands | Source exists; implementation gaps are not missing-evidence claims |

New CI runs installed core and benchmark fixture tests only. It does not validate every retained live workflow. Existing public scripts keep their names, paths and behavior. In particular, pandemic scale_eval --dry-run can still simulate and --eval can call models on cache misses; neither is an offline test command.

Paper baseline remains f214afe3. Historical Starsim truth is distinct from repaired continuous truth; binary results remain appendix-only. Jaeho's aligned delivery has been validated but shared tags remain private. Fabio's delivered raw texts/quantiles exist privately; complete raw-to-selected-paper linkage, a producer dirty patch and aligned binary replay vectors are not established by this migration. Full exact reproduction is not claimed for public fixtures. No private supplement is exported.
