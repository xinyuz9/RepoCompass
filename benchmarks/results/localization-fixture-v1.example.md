# RepoCompass localization evaluation

Report `01dd3fbd5ee267186c01313fa0c67450e85b3564449d4c8c4d91f0eb22ce55ef` covers 2 dataset cases and 10 observations.

Scope: localization-only, SWE-bench-derived evidence. This is not a SWE-bench resolved rate because no patch was generated or executed by the official harness.

| Variant | Runs | File Hit@1 | File Hit@3 | File Hit@5 | Recall@5 macro | Recall@5 micro | MRR | Symbol Hit@5 | Evidence validity | Unsupported relations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tool_wrapper | 2 | 0.0000% | 50.0000% | 50.0000% | 25.0000% | 33.3333% | 25.0000% | 0.0000% | 33.3333% | n/a (no_eligible_observations) |
| evaluator_loop | 2 | 50.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 75.0000% | 50.0000% | 60.0000% | 66.6666% |
| evaluator_loop_symbol_tools | 2 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 0.0000% |
| evaluator_loop_context_builder | 2 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 0.0000% |
| evaluator_loop_context_builder_memory | 2 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 0.0000% |
