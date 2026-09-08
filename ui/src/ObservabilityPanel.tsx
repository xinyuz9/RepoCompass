import { useQuery } from "@tanstack/react-query";
import { Activity, AlertTriangle, Clock3 } from "lucide-react";
import { useEffect } from "react";

import { dashboardQueries } from "./api";
import { activeRefreshInterval } from "./active-refresh";
import { ContextLens } from "./ContextLens";
import { CostLens } from "./CostLens";
import { compactId, formatDuration, humanize } from "./format";
import { MemoryLens } from "./MemoryLens";
import { PanelEmpty, PlanLens } from "./PlanLens";
import { TraceLens } from "./TraceLens";
import type { DashboardThreadDetail, RawRunAttempt } from "./types";

export type ObservabilityLens = "plan" | "trace" | "context" | "memory" | "cost";

interface ObservabilityPanelProps {
  lens: ObservabilityLens;
  detail: DashboardThreadDetail;
  memoryMutationsEnabled: boolean;
  requestedRunId: string | null;
  onRunSelected: (runId: string, replaceHistory: boolean) => void;
}

export function ObservabilityPanel({
  lens,
  detail,
  memoryMutationsEnabled,
  requestedRunId,
  onRunSelected,
}: ObservabilityPanelProps) {
  if (lens === "plan") return <PlanLens detail={detail} />;
  return (
    <RunObservability
      lens={lens}
      detail={detail}
      memoryMutationsEnabled={memoryMutationsEnabled}
      requestedRunId={requestedRunId}
      onRunSelected={onRunSelected}
    />
  );
}

function RunObservability({
  lens,
  detail,
  memoryMutationsEnabled,
  requestedRunId,
  onRunSelected,
}: Omit<ObservabilityPanelProps, "lens"> & {
  lens: Exclude<ObservabilityLens, "plan">;
}) {
  const activeRunId = detail.attempts.some(
    (item) => item.attempt.run_id === requestedRunId,
  )
    ? requestedRunId
    : (detail.attempts[0]?.attempt.run_id ?? null);
  const run = useQuery({
    ...dashboardQueries.run(activeRunId ?? ""),
    enabled: activeRunId !== null,
    staleTime: 30_000,
    retry: false,
    refetchInterval: (query) => activeRefreshInterval(
      query.state.data?.attempt.status,
      query.state.status === "error",
      query.state.dataUpdateCount,
    ),
  });

  useEffect(() => {
    if (activeRunId && activeRunId !== requestedRunId) {
      onRunSelected(activeRunId, true);
    }
  }, [activeRunId, onRunSelected, requestedRunId]);

  if (!activeRunId) {
    return (
      <section className="panel panel--standalone">
        <PanelEmpty
          icon={<AlertTriangle size={20} />}
          title="No persisted attempts"
          detail="The checkpoint exists without a repository-bound run attempt."
        />
      </section>
    );
  }

  return (
    <div className="run-observability">
      <RunSelector
        attempts={detail.attempts.map((item) => item.attempt)}
        activeRunId={activeRunId}
        onSelect={(runId) => onRunSelected(runId, false)}
      />
      {run.isPending ? (
        <div className="run-loading" aria-label="Loading persisted run">
          <span /><span /><span /><p>Replaying persisted run</p>
        </div>
      ) : null}
      {run.isError ? (
        <section className="panel panel--standalone">
          <div className="run-error" role="alert">
            <AlertTriangle size={20} />
            <div><strong>Run replay unavailable</strong><p>{run.error.message}</p></div>
          </div>
        </section>
      ) : null}
      {run.data && lens === "trace" ? <TraceLens run={run.data} /> : null}
      {run.data?.attempt.status === "running" ? <p className="muted">Read-only updates every 1.5s, up to 120 snapshots. Updates pause in the background or on error. Refresh manually if the attempt remains unfinished.</p> : null}
      {run.data && lens === "context" ? <ContextLens run={run.data} /> : null}
      {run.data && lens === "memory" ? (
        <MemoryLens run={run.data} mutationsEnabled={memoryMutationsEnabled} />
      ) : null}
      {run.data && lens === "cost" ? <CostLens run={run.data} /> : null}
    </div>
  );
}

function RunSelector({
  attempts,
  activeRunId,
  onSelect,
}: {
  attempts: RawRunAttempt[];
  activeRunId: string;
  onSelect: (runId: string) => void;
}) {
  const active = attempts.find((item) => item.run_id === activeRunId) ?? attempts[0];
  if (!active) return null;
  return (
    <div className="run-selector">
      <div className="run-selector__label">
        <Activity size={15} />
        <span>Attempt replay</span>
        <strong>{humanize(active.execution_mode)} · turn {active.turn_number}</strong>
      </div>
      <label>
        <span className="sr-only">Select persisted run</span>
        <select value={activeRunId} onChange={(event) => onSelect(event.target.value)}>
          {attempts.map((attempt) => (
            <option key={attempt.run_id} value={attempt.run_id}>
              turn {attempt.turn_number} · {attempt.execution_mode} · {compactId(attempt.run_id)}
            </option>
          ))}
        </select>
      </label>
      <span className={`attempt-status attempt-status--${active.status}`}>
        {active.status}
      </span>
      <span className="attempt-duration">
        <Clock3 size={14} />
        {active.finished_at_ms === null
          ? "running"
          : formatDuration(active.finished_at_ms - active.started_at_ms)}
      </span>
    </div>
  );
}
