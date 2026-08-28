import { AlertCircle, CheckCircle2, CircleDot, Clock3, Search } from "lucide-react";

import type { DashboardThreadSummary, ExecutionState } from "./types";

interface ThreadListProps {
  threads: DashboardThreadSummary[];
  selectedId: string | null;
  query: string;
  onQueryChange: (value: string) => void;
  onSelect: (threadId: string) => void;
}

export function ThreadList({
  threads,
  selectedId,
  query,
  onQueryChange,
  onSelect,
}: ThreadListProps) {
  return (
    <aside className="thread-panel" aria-label="Investigation threads">
      <div className="thread-panel__header">
        <div>
          <p className="eyebrow">workspace</p>
          <h2>Investigations</h2>
        </div>
        <span className="count-pill">{threads.length}</span>
      </div>
      <label className="thread-search">
        <Search size={15} aria-hidden="true" />
        <span className="sr-only">Filter investigations</span>
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Filter objective or ID"
        />
      </label>
      <div className="thread-list">
        {threads.map((thread) => (
          <button
            key={thread.thread_id}
            type="button"
            className={`thread-card${thread.thread_id === selectedId ? " is-active" : ""}`}
            onClick={() => onSelect(thread.thread_id)}
          >
            <div className="thread-card__topline">
              <StateIcon state={thread.execution_state} />
              <span>{stateLabel(thread.execution_state)}</span>
              <code>{shortId(thread.thread_id)}</code>
            </div>
            <strong>{thread.objective ?? "Checkpoint unavailable"}</strong>
            <div className="thread-card__meta">
              <span><Clock3 size={12} /> {formatTimestamp(thread.latest_attempt.started_at_ms)}</span>
              <span>{thread.attempt_count} run{thread.attempt_count === 1 ? "" : "s"}</span>
            </div>
          </button>
        ))}
        {threads.length === 0 ? (
          <p className="thread-list__empty">No matching investigations.</p>
        ) : null}
      </div>
    </aside>
  );
}

function StateIcon({ state }: { state: ExecutionState | null }) {
  if (state === "completed") return <CheckCircle2 size={13} aria-hidden="true" />;
  if (state === "interrupted") return <AlertCircle size={13} aria-hidden="true" />;
  return <CircleDot size={13} aria-hidden="true" />;
}

function stateLabel(state: ExecutionState | null): string {
  return state ?? "unavailable";
}

function shortId(value: string): string {
  return value.length <= 10 ? value : value.slice(0, 8);
}

function formatTimestamp(value: number): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}
