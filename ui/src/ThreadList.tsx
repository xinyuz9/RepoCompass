import {
  AlertCircle,
  CheckCircle2,
  CircleDot,
  Clock3,
  Plus,
  Search,
} from "lucide-react";

import type { DashboardThreadSummary, ExecutionState } from "./types";

interface ThreadListProps {
  hidden?: boolean;
  threads: DashboardThreadSummary[];
  selectedId: string | null;
  query: string;
  currentRepositoryId: string | null;
  currentCommitSha: string | null;
  onQueryChange: (value: string) => void;
  onNew: () => void;
  onSelect: (threadId: string) => void;
}

export function ThreadList({
  hidden = false,
  threads,
  selectedId,
  query,
  currentRepositoryId,
  currentCommitSha,
  onQueryChange,
  onNew,
  onSelect,
}: ThreadListProps) {
  return (
    <aside className="thread-panel" id="investigation-threads" aria-label="Investigation threads" hidden={hidden}>
      <div className="thread-panel__header">
        <div>
          <p className="eyebrow">workspace</p>
          <h2>Investigations</h2>
        </div>
        <div className="thread-panel__actions">
          <span className="count-pill">{threads.length}</span>
          <button type="button" className="new-thread-button" onClick={onNew} aria-label="New investigation" title="Declare a new goal">
            <Plus size={16} />
          </button>
        </div>
      </div>
      <label className="thread-search">
        <Search size={15} aria-hidden="true" />
        <span className="sr-only">Filter investigations</span>
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Filter goal or ID"
        />
      </label>
      <div className="thread-list">
        {threads.map((thread) => {
          const stale = currentRepositoryId !== null && (
            thread.latest_attempt.repository_id !== currentRepositoryId
            || thread.latest_attempt.commit_sha !== currentCommitSha
          );
          return (
            <button
              key={thread.thread_id}
              type="button"
              className={`thread-card${thread.thread_id === selectedId ? " is-active" : ""}${stale ? " is-stale" : ""}`}
              aria-current={thread.thread_id === selectedId ? "true" : undefined}
              onClick={() => onSelect(thread.thread_id)}
            >
              <div className="thread-card__topline">
                <StateIcon state={thread.execution_state} stale={stale} />
                <span>{stale ? "stale snapshot" : stateLabel(thread.execution_state)}</span>
                <code>{shortId(thread.thread_id)}</code>
              </div>
              <strong>{thread.objective ?? "Checkpoint unavailable"}</strong>
              <div className="thread-card__meta">
                <span><Clock3 size={12} /> {formatTimestamp(thread.latest_attempt.started_at_ms)}</span>
                <span>{thread.latest_attempt.status} · {thread.attempt_count} run{thread.attempt_count === 1 ? "" : "s"}</span>
              </div>
            </button>
          );
        })}
        {threads.length === 0 ? (
          <p className="thread-list__empty">{query.trim() ? "No matching investigations." : "Your investigations will appear here. Start with a question about your repository."}</p>
        ) : null}
      </div>
    </aside>
  );
}

function StateIcon({ state, stale }: { state: ExecutionState | null; stale: boolean }) {
  if (stale) return <AlertCircle size={13} aria-hidden="true" />;
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
