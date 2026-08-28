# RepoCompass localization analysis

Analysis 79566159b7f411223df25a8af365436fb17bb8134308c02c0f22e42b411ee054 of report 01dd3fbd5ee267186c01313fa0c67450e85b3564449d4c8c4d91f0eb22ce55ef.

Synthetic fixture: validates harness behavior, not model quality.

Scope: localization-only, SWE-bench-derived evidence. This is not a SWE-bench resolved rate.

Intervals: 95.0000% case-resampling percentile bootstrap, 10000 resamples, seed 20260828.

| Variant | File Hit@1 | File Hit@5 | Recall@5 | MRR | Symbol Hit@5 |
|---|---:|---:|---:|---:|---:|
| tool_wrapper | 0.0000% [0.0000%, 0.0000%] | 50.0000% [0.0000%, 100.0000%] | 25.0000% [0.0000%, 50.0000%] | 25.0000% [0.0000%, 50.0000%] | 0.0000% [0.0000%, 0.0000%] |
| evaluator_loop | 50.0000% [0.0000%, 100.0000%] | 100.0000% [100.0000%, 100.0000%] | 100.0000% [100.0000%, 100.0000%] | 75.0000% [50.0000%, 100.0000%] | 50.0000% [0.0000%, 100.0000%] |
| evaluator_loop_symbol_tools | 100.0000% [100.0000%, 100.0000%] | 100.0000% [100.0000%, 100.0000%] | 100.0000% [100.0000%, 100.0000%] | 100.0000% [100.0000%, 100.0000%] | 100.0000% [100.0000%, 100.0000%] |
| evaluator_loop_context_builder | 100.0000% [100.0000%, 100.0000%] | 100.0000% [100.0000%, 100.0000%] | 100.0000% [100.0000%, 100.0000%] | 100.0000% [100.0000%, 100.0000%] | 100.0000% [100.0000%, 100.0000%] |
| evaluator_loop_context_builder_memory | 100.0000% [100.0000%, 100.0000%] | 100.0000% [100.0000%, 100.0000%] | 100.0000% [100.0000%, 100.0000%] | 100.0000% [100.0000%, 100.0000%] | 100.0000% [100.0000%, 100.0000%] |

| Variant | Context tokens | Tool calls | Latency ms | Input tokens | Output tokens | Estimated cost USD |
|---|---:|---:|---:|---:|---:|---:|
| tool_wrapper | 7500.00 (2/2) | 12.00 (2/2) | 900.00 (2/2) | 8000.00 (2/2) | 700.00 (2/2) | n/a |
| evaluator_loop | 6900.00 (2/2) | 11.00 (2/2) | 1150.00 (2/2) | 9100.00 (2/2) | 1000.00 (2/2) | n/a |
| evaluator_loop_symbol_tools | 6100.00 (2/2) | 9.00 (2/2) | 1080.00 (2/2) | 8400.00 (2/2) | 900.00 (2/2) | n/a |
| evaluator_loop_context_builder | 3200.00 (2/2) | 8.00 (2/2) | 980.00 (2/2) | 5000.00 (2/2) | 750.00 (2/2) | n/a |
| evaluator_loop_context_builder_memory | 3650.00 (2/2) | 6.00 (2/2) | 920.00 (2/2) | 5350.00 (2/2) | 700.00 (2/2) | n/a |

Deterministic error taxonomy flags 4/10 logical observations.

Estimated costs appear only when provider usage and matching pricing were available; missing cost is never treated as zero.
