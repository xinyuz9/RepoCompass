# Memory recall on/off ablation — `memory-recall-v1`

This is a deterministic local benchmark. It makes no model or network call and uses the
production Active Context Builder plus a strict, body-free `MemoryRecallReport`.

| Metric | Memory off | Memory on | Delta |
|---|---:|---:|---:|
| Context candidates | 3 | 5 | +2 |
| Selected context items | 3 | 5 | +2 |
| Conservative context tokens | 2,330 | 4,953 | +2,623 |
| Retrieval results | 0 | 3 | +3 |
| Usable memory candidates | 0 | 2 | +2 |
| Selected memory candidates | 0 | 2 | +2 |
| Stale results excluded | 0 | 1 | +1 |
| Required evidence retained | 100% | 100% | 0 ppm |
| Example estimated input cost | USD 0.002330 | USD 0.004953 | +USD 0.002623 |

The three recall results comprise two repository results and one episodic result. One
repository result is stale and cannot become a context item; both usable candidates are
selected. Enabling memory does not evict either required verified source excerpt in this
case, but it has an explicit context and cost overhead.

The price is from `examples/pricing.example.json` at an illustrative USD 1 per million
input tokens. It is an engineering estimate, not a provider invoice. This fixture proves
accounting, provenance, stale exclusion, and evidence-retention behavior; it is not a
claim that one synthetic case measures end-task quality. M6 evaluates localization
quality on a broader labeled task set.

Reproduce the immutable JSON report:

```bash
uv run repocompass-memory-benchmark \
  benchmarks/cases/memory-recall-v1.json \
  --pricing-file examples/pricing.example.json \
  --provider openai \
  --model example-openai-model \
  --output /tmp/memory-recall-v1.json
```

- Case ID: `0c817884151f5d7e4586f71e3e3dee7dce6556a60ee434df003c4127009aa43e`
- Report ID: `fdad7ec73736c30cfd77fb63b59ad16bba896a3414d1322216e8f9ffdcd88820`
