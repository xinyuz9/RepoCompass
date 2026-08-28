import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  BrainCircuit,
  CircleDollarSign,
  Compass,
  GitCommitHorizontal,
  Layers3,
  ListChecks,
  Network,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import { useMemo, useState } from "react";

import { DashboardApiError, dashboardQueries } from "./api";
import { TaskGraph } from "./TaskGraph";
import { ThreadList } from "./ThreadList";
import type { DashboardThreadDetail } from "./types";

type Lens = "topology" | "plan" | "trace" | "context" | "memory" | "cost";

const LENSES: { id: Lens; label: string; icon: typeof Network; stage?: string }[] = [
  { id: "topology", label: "Topology", icon: Network },
  { id: "plan", label: "Plan", icon: ListChecks, stage: "M8.3" },
  { id: "trace", label: "Run trace", icon: Activity, stage: "M8.3" },
  { id: "context", label: "Context", icon: Layers3, stage: "M8.3" },
  { id: "memory", label: "Memory", icon: BrainCircuit, stage: "M8.3" },
  { id: "cost", label: "Cost", icon: CircleDollarSign, stage: "M8.3" },
];

export function App() {
  const queryClient = useQueryClient();
  const system = useQuery({ ...dashboardQueries.system, staleTime: 60_000 });
  const catalog = useQuery({ ...dashboardQueries.threads, staleTime: 20_000 });
  const [selectedThreadId, setSelectedThreadId] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [lens, setLens] = useState<Lens>("topology");

  const selectedExists = catalog.data?.threads.some(
    (item) => item.thread_id === selectedThreadId,
  );
  const activeThreadId = selectedExists
    ? selectedThreadId
    : (catalog.data?.threads[0]?.thread_id ?? null);

  const thread = useQuery({
    ...dashboardQueries.thread(activeThreadId ?? ""),
    enabled: activeThreadId !== null,
    staleTime: 15_000,
  });
  const visibleThreads = useMemo(() => {
    const normalized = filter.trim().toLocaleLowerCase();
    if (!normalized) return catalog.data?.threads ?? [];
    return (catalog.data?.threads ?? []).filter((item) =>
      `${item.objective ?? ""} ${item.thread_id}`.toLocaleLowerCase().includes(normalized),
    );
  }, [catalog.data, filter]);

  const refresh = () => queryClient.invalidateQueries({ queryKey: ["dashboard"] });
  const rootError = system.error ?? catalog.error;

  return (
    <div className="app-shell">
      <aside className="lens-rail" aria-label="Investigation lenses">
        <div className="brand-mark" title="RepoCompass">
          <Compass size={23} strokeWidth={1.7} />
        </div>
        <nav>
          {LENSES.map(({ id, label, icon: Icon, stage }) => (
            <button
              key={id}
              type="button"
              className={lens === id ? "is-active" : ""}
              onClick={() => setLens(id)}
              aria-label={label}
              title={stage ? `${label} · ${stage}` : label}
            >
              <Icon size={19} />
              <span>{label}</span>
              {stage ? <i /> : null}
            </button>
          ))}
        </nav>
        <div className="rail-trust" title="Local-only, repository-bound">
          <ShieldCheck size={18} />
        </div>
      </aside>

      <ThreadList
        threads={visibleThreads}
        selectedId={activeThreadId}
        query={filter}
        onQueryChange={setFilter}
        onSelect={(threadId) => {
          setSelectedThreadId(threadId);
          setLens("topology");
        }}
      />

      <main className="main-stage">
        <header className="topbar">
          <div className="topbar__brand">
            <span className="wordmark">RepoCompass</span>
            <span>investigation observatory</span>
          </div>
          <div className="topbar__status">
            {system.data ? (
              <>
                <span className="live-indicator"><i /> local</span>
                <span className="snapshot-chip">
                  <GitCommitHorizontal size={13} />
                  {shortHash(system.data.commit_sha ?? "unversioned")}
                </span>
                <code>{shortHash(system.data.repository_id)}</code>
              </>
            ) : null}
            <button type="button" className="icon-button" onClick={refresh} title="Refresh data">
              <RefreshCw size={16} />
              <span className="sr-only">Refresh dashboard data</span>
            </button>
          </div>
        </header>

        {rootError ? <ErrorState error={rootError} onRetry={refresh} /> : null}
        {!rootError && (catalog.isPending || system.isPending) ? <LoadingState /> : null}
        {!rootError && !catalog.isPending && catalog.data?.threads.length === 0 ? (
          <EmptyWorkspace />
        ) : null}
        {!rootError && activeThreadId && thread.isError ? (
          <ErrorState error={thread.error} onRetry={refresh} />
        ) : null}
        {!rootError && activeThreadId && thread.isPending ? <LoadingState compact /> : null}
        {!rootError && thread.data ? (
          <InvestigationStage lens={lens} detail={thread.data} />
        ) : null}
      </main>
    </div>
  );
}

function InvestigationStage({ lens, detail }: { lens: Lens; detail: DashboardThreadDetail }) {
  const { thread, attempts } = detail;
  const slice = thread.structure?.reading_path.graph_slice;
  const verifiedEdges = slice?.edges.filter((edge) => edge.verification_status === "verified").length ?? 0;

  return (
    <div className="investigation-stage">
      <section className="investigation-hero">
        <div>
          <div className="hero-kicker">
            <span className={`state-chip state-chip--${thread.execution_state}`}>
              {thread.execution_state}
            </span>
            <code>thread/{shortHash(thread.thread_id)}</code>
          </div>
          <h1>{thread.objective}</h1>
          <p>
            Task-bounded repository evidence · turn {thread.turn_number} · {attempts.length} persisted attempt{attempts.length === 1 ? "" : "s"}
          </p>
        </div>
        <div className="hero-trust">
          <ShieldCheck size={20} />
          <div><strong>Evidence boundary active</strong><span>Task slice · no raw artifacts</span></div>
        </div>
      </section>

      <section className="metric-strip" aria-label="Investigation summary">
        <Metric label="Plan steps" value={thread.plan?.steps.length ?? 0} detail={`${thread.plan_revision_count} revisions`} />
        <Metric label="Indexed files" value={thread.structure?.indexed_paths.length ?? 0} detail="candidate scoped" />
        <Metric label="Graph entities" value={slice?.nodes.length ?? 0} detail={slice?.truncated ? "bounded · truncated" : "bounded slice"} />
        <Metric label="Verified links" value={verifiedEdges} detail={`${slice?.edges.length ?? 0} total relations`} />
      </section>

      {lens === "topology" ? (
        thread.structure ? <TaskGraph structure={thread.structure} /> : <StructureEmpty />
      ) : (
        <LensPreview lens={lens} />
      )}
    </div>
  );
}

function Metric({ label, value, detail }: { label: string; value: number; detail: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

function LensPreview({ lens }: { lens: Exclude<Lens, "topology"> }) {
  const descriptor = LENSES.find((item) => item.id === lens);
  const Icon = descriptor?.icon ?? Activity;
  return (
    <section className="lens-preview">
      <div className="lens-preview__orb"><Icon size={28} /></div>
      <p className="eyebrow">observability lens</p>
      <h2>{descriptor?.label}</h2>
      <p>The shell and data boundary are ready. This evidence lens is populated in M8.3.</p>
      <span>M8.3 · contract already available</span>
    </section>
  );
}

function EmptyWorkspace() {
  return (
    <section className="empty-workspace">
      <div className="empty-compass"><Compass size={36} /></div>
      <p className="eyebrow">local flight recorder</p>
      <h1>No investigations recorded</h1>
      <p>Run RepoCompass against this repository, then return here to replay the evidence path.</p>
      <code>uv run repocompass . --objective "Trace the authentication flow"</code>
      <div className="empty-workspace__features">
        <span><Network size={14} /> task graph</span>
        <span><ListChecks size={14} /> plan replay</span>
        <span><ShieldCheck size={14} /> source provenance</span>
      </div>
    </section>
  );
}

function StructureEmpty() {
  return (
    <section className="structure-empty">
      <Network size={24} />
      <div><h2>No structural slice yet</h2><p>The agent has not produced a validated Python evidence graph for this checkpoint.</p></div>
    </section>
  );
}

function LoadingState({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`loading-state${compact ? " is-compact" : ""}`} aria-label="Loading dashboard data">
      <span /><span /><span />
    </div>
  );
}

function ErrorState({ error, onRetry }: { error: Error; onRetry: () => void }) {
  const label = error instanceof DashboardApiError ? error.problem?.title ?? "Dashboard unavailable" : "Dashboard unavailable";
  return (
    <section className="error-state" role="alert">
      <Activity size={22} />
      <div><h2>{label}</h2><p>{error.message}</p></div>
      <button type="button" onClick={onRetry}>Try again</button>
    </section>
  );
}

function shortHash(value: string): string {
  return value.length <= 10 ? value : value.slice(0, 8);
}
