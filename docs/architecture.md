# Architecture

All interfaces share a repository-bound service. The model proposes; deterministic code
owns execution authority and validates the evidence it can publish.

## Runtime and evidence

`service/` owns lifecycle requests, idempotency, and execution ownership. `agent/` contains
plans, candidates, graph nodes, the controller, context selection, and orchestration.
`models/` contains provider adapters, request deadlines, and cooperative cancellation.

Detailed investigations explore one active plan step, build a candidate, validate source
references, and obtain a separate semantic assessment. The controller advances, revises,
stops, or requests review. Quick lookup reuses the tool/evidence gates without model planning
or semantic evaluation. An accepted workflow is not an independent correctness judgment.

`tools/` provides bounded listing, search, reads, and structural queries. The gateway validates
paths and arguments, reserves budgets, detects duplicate calls, and caches successful results
against the snapshot. `source_reader.py` bounds reads and protects against symlink traversal.
`structure/` builds content-addressed Python AST relationships and traceable reading paths;
verified syntax and inferred relationships remain distinct.

## Memory

`memory/` uses SQLite and FTS5/BM25 for repository facts and terminal episodes. Each new turn
can perform bounded recall; resuming an existing turn does not repeat it. Repository identity,
source freshness, and the context budget govern selection.

Eligible accepted results can promote bounded static Python facts after re-indexing current
source and checking snapshot identity. Model prose is not automatically promoted to verified
knowledge. Episodes remain distinct from source-backed facts. Payload-free tombstones and
a deletion audit prevent automatic re-creation of deleted facts.

## Persistent state

| Store | Responsibility |
|---|---|
| Checkpoint | Graph state, pending work, thread/turn identity |
| Operation ledger | Idempotent requests and worker ownership |
| Cancellation ledger | Durable control of active attempts |
| Raw run store | Invocations, selected context, artifacts, usage |
| Trace journal / JSONL | Ordered events and offline inspection |
| Memory | Facts, episodes, and deletion records |

Costs bind usage to a versioned pricing snapshot. Unknown usage/reservations remain explicit.
The FastAPI dashboard and React UI read these stores instead of starting a second agent.

## Interfaces and scope

`dashboard/` serves the API and packaged SPA; `ui/` contains React/TypeScript source.
Chat is the main entry, Path and Plan explain the result, and Inspect opens Trace, Context,
Memory, and Cost. Streaming events update the UI; refresh remains read-only.

`mcp_server.py` provides seven high-level lifecycle tools. `improve/` is a separate opt-in,
proposal-ID-bound workflow for small verified changes. Static HTML inspection uses an isolated
browser with an operator-selected page list. See [Advanced tools](advanced.md).

The workbench is local and single-user; structural analysis is Python-specific. Arbitrary
network browsing, general shell execution, distributed workers, and autonomous cross-repository
editing are outside the default workflow. Evaluation utilities are developer tools, not a
claim of general model-quality improvement.
