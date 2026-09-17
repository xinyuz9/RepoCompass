import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  BrainCircuit,
  CircleDollarSign,
  Compass,
  FolderGit2,
  GitCommitHorizontal,
  Layers3,
  ListChecks,
  MessageSquareText,
  Network,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Columns2,
} from "lucide-react";
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { DashboardApiError, dashboardQueries } from "./api";
import { activeRefreshInterval } from "./active-refresh";
import { useRunEvents } from "./use-run-events";
import {
  ConversationWorkspace,
  type InspectorLens,
} from "./ConversationWorkspace";
import { HumanReviewPanel } from "./HumanReviewPanel";
import { DriftPanel } from "./DriftPanel";
import { GoalTitle } from "./GoalTitle";
import { LifecycleRecoveryCenter } from "./lifecycle-recovery";
import { ObservabilityPanel } from "./ObservabilityPanel";
import { ThreadList } from "./ThreadList";
import { investigationStateLabel } from "./format";
import type { DashboardThreadDetail } from "./types";
import {
  isRunLens,
  readDashboardNavigation,
  writeDashboardNavigation,
  type DashboardLens,
  type DashboardNavigationMode,
  type DashboardNavigationState,
} from "./url-state";
import "./workspace-review.css";

const LENSES: { id: DashboardLens; label: string; icon: typeof Network }[] = [
  { id: "chat", label: "Chat", icon: MessageSquareText },
  { id: "topology", label: "Path", icon: Network },
  { id: "plan", label: "Plan", icon: ListChecks },
  { id: "trace", label: "Run trace", icon: Activity },
  { id: "context", label: "Context", icon: Layers3 },
  { id: "memory", label: "Memory", icon: BrainCircuit },
  { id: "cost", label: "Cost", icon: CircleDollarSign },
];

const TaskGraph = lazy(() => import("./TaskGraph").then((module) => ({ default: module.TaskGraph })));

export function App() {
  const queryClient = useQueryClient();
  const system = useQuery({ ...dashboardQueries.system, staleTime: 60_000 });
  const catalog = useQuery({ ...dashboardQueries.threads, staleTime: 20_000 });
  const [navigation, setNavigation] = useState(readDashboardNavigation);
  const [filter, setFilter] = useState("");
  const [refreshEpoch, setRefreshEpoch] = useState(0);
  const canFollow = system.data?.event_follow_enabled === true && typeof EventSource !== "undefined";
  const [sidebarOpen, setSidebarOpen] = useState(() =>
    !readDashboardNavigation().splitView && !window.matchMedia?.("(max-width: 930px)").matches);
  const sidebarToggle = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const onEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && sidebarOpen
        && document.getElementById("investigation-threads")?.contains(document.activeElement)) {
        setSidebarOpen(false);
        sidebarToggle.current?.focus();
      }
    };
    window.addEventListener("keydown", onEscape);
    return () => window.removeEventListener("keydown", onEscape);
  }, [sidebarOpen]);
  const {
    threadId: selectedThreadId,
    lens,
    runId: requestedRunId,
    composingNew,
  } = navigation;

  const defaultThreadId = selectedThreadId
    ?? catalog.data?.threads[0]?.thread_id
    ?? null;
  const activeThreadId = composingNew
    ? null
    : defaultThreadId;
  const thread = useQuery({
    ...dashboardQueries.thread(activeThreadId ?? ""),
    enabled: activeThreadId !== null,
    staleTime: 15_000,
    retry: false,
    refetchInterval: (query) => canFollow ? false : activeRefreshInterval(
      query.state.data?.attempts[0]?.attempt.status,
      query.state.status === "error",
      query.state.dataUpdateCount,
    ),
  });
  const latestAttempt = thread.data?.attempts[0]?.attempt;
  const streamEnabled = canFollow && latestAttempt?.status === "running" && !thread.isError;
  const streamState = useRunEvents(latestAttempt?.run_id ?? null, activeThreadId, streamEnabled, refreshEpoch);
  const visibleThreads = useMemo(() => {
    const normalized = filter.trim().toLocaleLowerCase();
    if (!normalized) return catalog.data?.threads ?? [];
    return (catalog.data?.threads ?? []).filter((item) =>
      `${item.objective ?? ""} ${item.thread_id}`
        .toLocaleLowerCase()
        .includes(normalized),
    );
  }, [catalog.data, filter]);

  const navigate = useCallback((
    next: DashboardNavigationState,
    mode: DashboardNavigationMode = "push",
  ) => {
    setNavigation(next);
    writeDashboardNavigation(next, mode);
  }, []);

  useEffect(() => {
    const restoreNavigation = () => setNavigation(readDashboardNavigation());
    window.addEventListener("popstate", restoreNavigation);
    return () => window.removeEventListener("popstate", restoreNavigation);
  }, []);

  const refresh = () => {
    setRefreshEpoch((value) => value + 1);
    return queryClient.invalidateQueries({ queryKey: ["dashboard"] });
  };
  const openNewGoal = useCallback(() => navigate({
    threadId: null,
    lens: "chat",
    runId: null,
    composingNew: true,
  }), [navigate]);
  const activateThread = useCallback((threadId: string) => navigate({
    threadId,
    lens: "chat",
    runId: null,
    composingNew: false,
  }), [navigate]);
  const selectThread = useCallback((threadId: string) => {
    if (window.matchMedia?.("(max-width: 930px)").matches) {
      setSidebarOpen(false);
      sidebarToggle.current?.focus();
    }
    navigate({ threadId, lens: "chat", runId: null, composingNew: false });
  }, [navigate]);
  const openInspector = useCallback((nextLens: InspectorLens) => navigate({
    threadId: activeThreadId,
    lens: nextLens,
    runId: null,
    composingNew: false,
    splitView: navigation.splitView,
  }), [activeThreadId, navigate, navigation.splitView]);
  const selectLens = useCallback((nextLens: DashboardLens) => {
    const keepComposing = nextLens === "chat" && composingNew;
    const threadId = keepComposing ? null : defaultThreadId;
    navigate({
      threadId,
      lens: nextLens,
      runId: isRunLens(nextLens) && threadId === selectedThreadId
        ? requestedRunId
        : null,
      composingNew: keepComposing,
      splitView: nextLens !== "chat" && !keepComposing ? navigation.splitView : undefined,
      nodeId: nextLens === "topology" ? navigation.nodeId : undefined,
    });
  }, [composingNew, defaultThreadId, navigate, requestedRunId, selectedThreadId, navigation.splitView, navigation.nodeId]);
  const selectRun = useCallback((runId: string, replaceHistory: boolean) => {
    if (!defaultThreadId) return;
    navigate({
      threadId: defaultThreadId,
      lens,
      runId,
      composingNew: false,
      splitView: navigation.splitView,
    }, replaceHistory ? "replace" : "push");
  }, [defaultThreadId, lens, navigate, navigation.splitView]);
  const selectEntity = (nodeId: string) => navigate({ ...navigation, threadId: activeThreadId, nodeId });
  const toggleSplit = () => {
    if (!navigation.splitView) setSidebarOpen(false);
    navigate({ ...navigation, threadId: activeThreadId, splitView: !navigation.splitView });
  };
  const rootError = system.error ?? catalog.error;

  return (
    <div className="app-shell" data-sidebar-open={sidebarOpen}>
      <a className="skip-link" href="#workspace-content">Skip to workspace</a>
      <aside className="lens-rail" aria-label="Investigation lenses">
        <div className="brand-mark" title="RepoCompass">
          <Compass size={23} strokeWidth={1.7} />
        </div>
        <nav aria-label="Primary lenses">
          {LENSES.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              className={lens === id ? "is-active" : ""}
              onClick={() => selectLens(id)}
              aria-label={label}
              aria-current={lens === id ? "page" : undefined}
              title={label}
            >
              <Icon size={19} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="rail-trust" title="Local-only, repository-bound">
          <ShieldCheck size={18} />
        </div>
      </aside>

      <ThreadList
        hidden={!sidebarOpen}
        threads={visibleThreads}
        selectedId={activeThreadId}
        query={filter}
        currentRepositoryId={system.data?.repository_id ?? null}
        currentCommitSha={system.data?.commit_sha ?? null}
        onQueryChange={setFilter}
        onNew={openNewGoal}
        onSelect={selectThread}
      />

      <main className="main-stage" id="workspace-content" tabIndex={-1}>
        <header className="topbar">
          <div className="topbar__brand">
            <button
              ref={sidebarToggle}
              type="button"
              className="icon-button sidebar-toggle"
              aria-label={sidebarOpen ? "Hide investigations" : "Show investigations"}
              aria-expanded={sidebarOpen}
              aria-controls="investigation-threads"
              onClick={() => setSidebarOpen((open) => !open)}
            >
              {sidebarOpen ? <PanelLeftClose size={19} /> : <PanelLeftOpen size={19} />}
            </button>
            <span className="wordmark">RepoCompass</span>
            <span>repository investigation</span>
          </div>
          <div className="topbar__status">
            {system.data ? (
              <>
                <span className="workspace-chip" title="Repository root is selected by the dashboard process">
                  <FolderGit2 size={13} />
                  {system.data.workspace_name}<code>/{system.data.workspace_path}</code>
                </span>
                {system.data.demo_mode ? (
                  <span className="demo-indicator">
                    <Sparkles size={12} />sanitized demo
                  </span>
                ) : null}
                <span className="live-indicator"><i /> local</span>
                <span className="snapshot-chip">
                  <GitCommitHorizontal size={13} />
                  {shortHash(system.data.commit_sha ?? "unversioned")}
                </span>
              </>
            ) : null}
            <button type="button" className="icon-button" onClick={refresh} title="Refresh data">
              <RefreshCw size={16} />
              <span className="sr-only">Refresh dashboard data</span>
            </button>
          </div>
        </header>

        {streamEnabled ? <p className="event-stream-status" role="status">Live events: {streamState}. Read-only, at most 6 bounded sessions. Errors pause updates; Refresh reconnects without rerunning the goal.</p> : null}

        <LifecycleRecoveryCenter system={system.data} onThreadActivated={activateThread}
          visibleThreadId={lens === "chat" ? thread.data?.thread.thread_id : null} />

        {rootError ? <ErrorState error={rootError} onRetry={refresh} /> : null}
        {!rootError && (catalog.isPending || system.isPending) ? <LoadingState /> : null}
        {!rootError && activeThreadId && thread.isError ? (
          <ErrorState error={thread.error} onRetry={refresh} />
        ) : null}
        {!rootError && activeThreadId && thread.isPending ? <LoadingState compact /> : null}

        {!rootError
        && system.data
        && !catalog.isPending
        && lens === "chat"
        && (activeThreadId === null || thread.data !== undefined) ? (
          <ConversationWorkspace
            key={thread.data?.thread.thread_id ?? "new-goal"}
            system={system.data}
            detail={thread.data ?? null}
            lifecycleMutationsEnabled={system.data.lifecycle_mutations_enabled}
            onThreadActivated={activateThread}
            onOpenLens={openInspector}
            onNewGoal={openNewGoal}
          />
        ) : null}

        {!rootError && lens !== "chat" && thread.data ? (
          <>
          <div className="workspace-tools">
            <button type="button" aria-pressed={navigation.splitView === true} onClick={toggleSplit}><Columns2 size={15} />Chat + inspector</button>
            <TerminologyGuide />
          </div>
          <div className={navigation.splitView ? "review-layout review-layout--split" : "review-layout"}>
          {navigation.splitView && system.data ? <section className="split-conversation" aria-label="Conversation beside evidence">
            <ConversationWorkspace key={thread.data.thread.thread_id} system={system.data} detail={thread.data}
              lifecycleMutationsEnabled={system.data.lifecycle_mutations_enabled} onThreadActivated={activateThread}
              onOpenLens={openInspector} onNewGoal={openNewGoal} />
          </section> : null}
          <InvestigationStage
            lens={lens}
            detail={thread.data}
            lifecycleMutationsEnabled={system.data?.lifecycle_mutations_enabled ?? false}
            memoryMutationsEnabled={system.data?.memory_mutations_enabled ?? false}
            requestedRunId={requestedRunId}
            eventFollowEnabled={canFollow}
            onRunSelected={selectRun}
            requestedEntityId={navigation.nodeId}
            onEntitySelected={selectEntity}
            showReview={!navigation.splitView}
          />
          </div>
          </>
        ) : null}

        {!rootError && lens !== "chat" && !activeThreadId && !catalog.isPending ? (
          <section className="inspector-empty">
            <Network size={25} />
            <h1>Start a goal before opening inspectors</h1>
            <p>Plan, Path, Trace, Context, Memory, and Cost are populated from persisted attempts.</p>
            <button type="button" onClick={openNewGoal}>Declare current goal</button>
          </section>
        ) : null}
      </main>
    </div>
  );
}

function InvestigationStage({
  lens,
  detail,
  lifecycleMutationsEnabled,
  memoryMutationsEnabled,
  requestedRunId,
  eventFollowEnabled,
  onRunSelected,
  requestedEntityId,
  onEntitySelected,
  showReview,
}: {
  lens: InspectorLens;
  detail: DashboardThreadDetail;
  lifecycleMutationsEnabled: boolean;
  memoryMutationsEnabled: boolean;
  requestedRunId: string | null;
  eventFollowEnabled: boolean;
  onRunSelected: (runId: string, replaceHistory: boolean) => void;
  requestedEntityId?: string;
  onEntitySelected: (id: string) => void;
  showReview: boolean;
}) {
  const { thread, attempts } = detail;
  const slice = thread.structure?.reading_path.graph_slice;
  const verifiedEdges = slice?.edges.filter(
    (edge) => edge.verification_status === "verified",
  ).length ?? 0;

  return (
    <div className="investigation-stage">
      <section className="investigation-hero">
        <div>
          <div className="hero-kicker">
            <span className={`state-chip state-chip--${investigationStateLabel(thread)}`}>
              {investigationStateLabel(thread)}
            </span>
            <code>thread/{shortHash(thread.thread_id)}</code>
          </div>
          <GoalTitle objective={thread.objective} />
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

      {thread.pending_review && showReview ? (
        <HumanReviewPanel
          key={thread.interrupt_id ?? "pending-review"}
          detail={detail}
          enabled={lifecycleMutationsEnabled}
        />
      ) : null}

      {lens === "topology" ? (
        thread.structure ? (
          <>
          <DriftPanel key={thread.structure.artifact_id} threadId={thread.thread_id} artifactId={thread.structure.artifact_id} />
          <Suspense fallback={<LoadingState compact />}>
            <TaskGraph key={thread.structure.artifact_id} threadId={thread.thread_id} structure={thread.structure} requestedEntityId={requestedEntityId} onEntitySelected={onEntitySelected} />
          </Suspense>
          </>
        ) : <StructureEmpty />
      ) : (
        <ObservabilityPanel
          eventFollowEnabled={eventFollowEnabled}
          lens={lens}
          detail={detail}
          memoryMutationsEnabled={memoryMutationsEnabled}
          requestedRunId={requestedRunId}
          onRunSelected={onRunSelected}
        />
      )}
    </div>
  );
}

function Metric({ label, value, detail }: { label: string; value: number; detail: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

function TerminologyGuide() {
  return <details className="terminology-guide"><summary>中文 / English · terminology</summary>
    <dl lang="zh-CN" tabIndex={0} aria-label="Chinese and English terminology">
      <dt>Goal · 当前目标</dt><dd>用户声明的问题。任务绑定启动服务时选择的仓库；Focus path 仅缩小仓库内范围。</dd>
      <dt>Plan · 执行计划</dt><dd>步骤由控制器推进。Accepted 是评估通过，Completed 是运行结束；不代表人工批准。</dd>
      <dt>Path / Evidence · 证据路径</dt><dd>基于源码快照的阅读路线。Verified 表示确定性校验，Inferred 仍是推断，不能等同真实调用。</dd>
      <dt>Trace / Context · 轨迹与上下文</dt><dd>Trace 记录发生的动作；Context 显示每次模型调用实际入选的信息及被排除的原因。</dd>
      <dt>Memory · 记忆</dt><dd>持久化事实与历史经验仍需范围匹配和重检。被召回不代表进入上下文，也不代表提升效果。</dd>
      <dt>Cost · 费用分析</dt><dd>按用量与定价估算，不是供应商账单。未知用量不是免费；预算预留不等于已结算支出。</dd>
      <dt>Drift · 证据漂移</dt><dd>对比已保存来源与当前 HEAD；指纹相同不证明行为相同，不会自动重跑或修改记忆。</dd>
    </dl>
  </details>;
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
    <div className={`loading-state${compact ? " is-compact" : ""}`} role="status" aria-label="Loading dashboard data">
      <span /><span /><span />
    </div>
  );
}

function ErrorState({ error, onRetry }: { error: Error; onRetry: () => void }) {
  const label = error instanceof DashboardApiError
    ? error.problem?.title ?? "Dashboard unavailable"
    : "Dashboard unavailable";
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
