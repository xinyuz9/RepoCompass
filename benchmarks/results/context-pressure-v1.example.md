# Context pressure v1 — deterministic example result

This report compares RepoCompass's production Active Context Builder with an
unbounded baseline that forwards every candidate. It is a deterministic,
model-free ablation: no provider API was called and latency was not measured.
The pricing snapshot is deliberately illustrative and is not provider pricing.

## Result

| Metric | Active Context | All-context baseline |
| --- | ---: | ---: |
| Selected candidates | 3 / 6 | 6 / 6 |
| Conservative selected-token bound | 2,330 | 17,615 |
| Budget overflow | 0 | 15,285 |
| Required evidence retained | 2 / 2 (100%) | 2 / 2 (100%) |
| Evidence provenance coverage | 2 / 2 (100%) | 2 / 3 (66.6666%) |
| Estimated input cost | USD 0.002330 | USD 0.017615 |
| Latency | Not measured | Not measured |

The production builder reduced the selected-token upper bound by 15,285
(86.77%) while retaining all required evidence. Under the bundled example
price of USD 1 per million input tokens, the estimated input-cost reduction is
USD 0.015285. These values are estimates, not a provider invoice.

## Reproduce

From the repository root:

```bash
uv run repocompass-benchmark \
  benchmarks/cases/context-pressure-v1.json \
  --pricing-file examples/pricing.example.json \
  --provider openai \
  --model example-openai-model \
  --output /tmp/context-pressure-v1.json
```

The command refuses to overwrite an existing output. The resulting report ID
must be:

```text
d64e740822df166dd568ce132508a0d2b320ddcc2a7b471ad13ae6ce022ae745
```

The committed machine-readable artifact is
`benchmarks/results/context-pressure-v1.example.json`. Its identity binds the
case, both variant aggregates, model identity, and the exact content-addressed
pricing table. The artifact contains aggregate IDs and metrics, not candidate
content.
