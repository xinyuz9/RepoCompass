# RepoCompass localization regression gate

Decision: PASS

Rule: the upper confidence bound of case-paired regression must not exceed the configured non-inferiority threshold.

| Variant | Metric | Cases | Regression | Upper bound | Threshold | Result |
|---|---|---:|---:|---:|---:|---|
| tool_wrapper | file_hit_at_1 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| tool_wrapper | file_hit_at_5 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| tool_wrapper | file_recall_at_5 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| tool_wrapper | mean_reciprocal_rank | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| tool_wrapper | symbol_hit_at_5 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| tool_wrapper | evidence_validity | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop | file_hit_at_1 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop | file_hit_at_5 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop | file_recall_at_5 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop | mean_reciprocal_rank | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop | symbol_hit_at_5 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop | evidence_validity | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_symbol_tools | file_hit_at_1 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_symbol_tools | file_hit_at_5 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_symbol_tools | file_recall_at_5 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_symbol_tools | mean_reciprocal_rank | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_symbol_tools | symbol_hit_at_5 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_symbol_tools | evidence_validity | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_context_builder | file_hit_at_1 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_context_builder | file_hit_at_5 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_context_builder | file_recall_at_5 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_context_builder | mean_reciprocal_rank | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_context_builder | symbol_hit_at_5 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_context_builder | evidence_validity | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_context_builder_memory | file_hit_at_1 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_context_builder_memory | file_hit_at_5 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_context_builder_memory | file_recall_at_5 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_context_builder_memory | mean_reciprocal_rank | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_context_builder_memory | symbol_hit_at_5 | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |
| evaluator_loop_context_builder_memory | evidence_validity | 40 | 0.0000% | 0.0000% | 0.0000% | PASS |

Positive regression values are worse. Missing or insufficient paired cases fail closed.
