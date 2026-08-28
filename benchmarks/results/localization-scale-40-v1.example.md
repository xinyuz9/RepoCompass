# RepoCompass localization evaluation

Report `32c6ba817e5e9d2d0f3ef9d10070e3a37ff9d95f38a45a3abf2ba125e162aa99` covers 40 dataset cases and 200 observations.

Scope: localization-only, SWE-bench-derived evidence. This is not a SWE-bench resolved rate because no patch was generated or executed by the official harness.

| Variant | Runs | File Hit@1 | File Hit@3 | File Hit@5 | Recall@5 macro | Recall@5 micro | MRR | Symbol Hit@5 | Evidence validity | Unsupported relations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tool_wrapper | 40 | 25.0000% | 30.0000% | 30.0000% | 28.7500% | 27.0833% | 27.5000% | 0.0000% | 18.4873% | n/a (no_eligible_observations) |
| evaluator_loop | 40 | 42.5000% | 55.0000% | 55.0000% | 52.5000% | 50.0000% | 48.7500% | 27.5000% | 63.5220% | 16.6666% |
| evaluator_loop_symbol_tools | 40 | 60.0000% | 75.0000% | 75.0000% | 75.0000% | 75.0000% | 67.5000% | 75.0000% | 94.9748% | 4.0000% |
| evaluator_loop_context_builder | 40 | 70.0000% | 87.5000% | 87.5000% | 87.5000% | 87.5000% | 78.7500% | 87.5000% | 97.9079% | 0.0000% |
| evaluator_loop_context_builder_memory | 40 | 72.5000% | 92.5000% | 92.5000% | 92.5000% | 91.6666% | 82.5000% | 92.5000% | 98.9247% | 0.0000% |
