# Configuration and usage

Use Linux/WSL, Python 3.12+, Git, ripgrep, and uv. Run `uv sync --locked` from the checkout.
Node.js 24 is needed only to rebuild the UI. Docker packages the runtime dependencies.

## Provider and pricing

Copy `.env.example` to the Git-ignored `.env`. Set the provider, a model available to your
account, and its credential. For example:

```dotenv
REPOCOMPASS_MODEL_PROVIDER=openai
REPOCOMPASS_MODEL=your-openai-model
REPOCOMPASS_API_KEY=your-api-key
REPOCOMPASS_MODEL_TIMEOUT_SECONDS=60
REPOCOMPASS_ATTEMPT_TIMEOUT_SECONDS=600
LANGSMITH_TRACING=false
```

DeepSeek uses provider `deepseek`, an appropriate model name, and
`REPOCOMPASS_BASE_URL=https://api.deepseek.com`. Existing environment variables take precedence.
Provider adapters keep provider-specific settings separate. SDK retries are disabled; requests
and execution attempts have bounded deadlines. Cancellation cannot undo work already billed.

Copy `examples/pricing.example.json` to a private configuration directory. Replace the version,
effective date, exact provider/model names, and prices with reviewed current values. Example
prices are placeholders; dated snapshots are regression fixtures, not current price guidance.
Each entry defines USD per million input/output tokens plus cached/reasoning-token treatment.
A matching pricing snapshot is required for a monetary cap. Missing usage is not treated as free.

## Interactive workspace

```bash
uv run repocompass-dashboard \
  --repo-path /absolute/path/to/repository \
  --pricing-file /absolute/path/to/pricing.json \
  --enable-lifecycle-mutations
```

Open <http://127.0.0.1:8766/>. Enter a goal and optional repository-relative focus path.
Task settings control recall, review, and per-attempt cost. Choose a cap before submitting
paid work; follow-ups and resumed attempts can incur additional charges.

- **Quick lookup** skips model planning/semantic review, retaining source validation and shared
  tool, cost, memory, trace, checkpoint, and cancellation controls.
- **Detailed investigation** adds planning, structural tools, context selection, semantic review,
  and bounded revision. It supports human review before finalization.
- Mode is fixed for a thread. Start another goal to change it.
- Memory recall is opt-in; recalled material must be checked against current evidence.

Omit `--enable-lifecycle-mutations` for read-only inspection. Memory deletion has a separate
`--enable-memory-mutations` flag and confirmation. Use `--port 8879` if 8766 is occupied.
Restart the server after backend changes.

## Storage and recovery

The launch directory's `.repocompass/` holds checkpoints, raw calls, memory, operations,
cancellation records, and traces. Reuse that directory when reopening a workspace. Back up
stores together; do not mix repository identities or native and container histories.

| State | Action |
|---|---|
| Unfinished, same snapshot | Resume the checkpoint |
| Completed | Send a follow-up to create a new turn |
| Pending review | Approve, revise, or cancel that interrupt |
| Commit changed | Inspect old results and start a fresh goal |

Exact lifecycle retries are idempotent. Reusing an ID with a different request is rejected.
Refresh and inspection do not replay model calls.

## CLI

```bash
uv run repocompass /absolute/path/to/repository \
  --objective "Trace how proposal verification gates application" \
  --pricing-file /absolute/path/to/pricing.json \
  --max-attempt-cost-usd 0.10
```

The cap is an example, not a quoted task price. `repocompass --help` lists resume, follow-up,
human review, usage reports, memory management, and tool/context budgets.
