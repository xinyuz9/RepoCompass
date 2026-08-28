# RepoCompass localization analysis

Analysis a477e4ac7dc9ed436baaf326136b072bdf35c2f5529069d6e0f84962620eee9c of report 32c6ba817e5e9d2d0f3ef9d10070e3a37ff9d95f38a45a3abf2ba125e162aa99.

Synthetic fixture: validates harness behavior, not model quality.

Scope: localization-only, SWE-bench-derived evidence. This is not a SWE-bench resolved rate.

Intervals: 95.0000% case-resampling percentile bootstrap, 2000 resamples, seed 20260828.

| Variant | File Hit@1 | File Hit@5 | Recall@5 | MRR | Symbol Hit@5 |
|---|---:|---:|---:|---:|---:|
| tool_wrapper | 25.0000% [12.5000%, 40.0000%] | 30.0000% [15.0000%, 45.0000%] | 28.7500% [15.0000%, 42.5000%] | 27.5000% [15.0000%, 42.5000%] | 0.0000% [0.0000%, 0.0000%] |
| evaluator_loop | 42.5000% [27.5000%, 57.5000%] | 55.0000% [40.0000%, 70.0000%] | 52.5000% [37.5000%, 67.5000%] | 48.7500% [35.0000%, 63.7500%] | 27.5000% [12.5000%, 40.0000%] |
| evaluator_loop_symbol_tools | 60.0000% [45.0000%, 75.0000%] | 75.0000% [60.0000%, 87.5000%] | 75.0000% [60.0000%, 87.5000%] | 67.5000% [55.0000%, 80.0000%] | 75.0000% [62.5000%, 87.5000%] |
| evaluator_loop_context_builder | 70.0000% [55.0000%, 85.0000%] | 87.5000% [77.5000%, 97.5000%] | 87.5000% [77.5000%, 97.5000%] | 78.7500% [67.5000%, 88.7500%] | 87.5000% [75.0000%, 97.5000%] |
| evaluator_loop_context_builder_memory | 72.5000% [60.0000%, 85.0000%] | 92.5000% [82.5000%, 100.0000%] | 92.5000% [85.0000%, 100.0000%] | 82.5000% [72.5000%, 91.2500%] | 92.5000% [82.5000%, 100.0000%] |

| Variant | Context tokens | Tool calls | Latency ms | Input tokens | Output tokens | Estimated cost USD |
|---|---:|---:|---:|---:|---:|---:|
| tool_wrapper | 7914.50 (40/40) | 12.50 (40/40) | 978.50 (40/40) | 8236.50 (40/40) | 727.18 (40/40) | 0.387634 |
| evaluator_loop | 7264.50 (40/40) | 11.50 (40/40) | 1228.50 (40/40) | 9336.50 (40/40) | 1027.18 (40/40) | 0.455634 |
| evaluator_loop_symbol_tools | 6364.50 (40/40) | 9.50 (40/40) | 1148.50 (40/40) | 8636.50 (40/40) | 927.18 (40/40) | 0.419634 |
| evaluator_loop_context_builder | 3664.50 (40/40) | 8.50 (40/40) | 1048.50 (40/40) | 5286.50 (40/40) | 777.18 (40/40) | 0.273634 |
| evaluator_loop_context_builder_memory | 3914.50 (40/40) | 6.50 (40/40) | 988.50 (40/40) | 5586.50 (40/40) | 727.18 (40/40) | 0.281634 |

Deterministic error taxonomy flags 102/200 logical observations.

Estimated costs appear only when provider usage and matching pricing were available; missing cost is never treated as zero.
