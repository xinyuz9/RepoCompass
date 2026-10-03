import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowUp,
  Bot,
  CheckCircle2,
  CirclePlay,
  Clock3,
  FileCode2,
  FolderGit2,
  GitCommitHorizontal,
  LoaderCircle,
  MessageSquareText,
  RotateCcw,
  ShieldAlert,
  Settings2,
  Sparkles,
  UserRound,
} from "lucide-react";
import { useMemo, useState } from "react";

import {
  continueThread,
  DashboardApiError,
  resumeThread,
  startInvestigation,
} from "./api";
import { HumanReviewPanel } from "./HumanReviewPanel";
import { ConversationHistory } from "./ConversationHistory";
import { CurrentReviewHistory } from "./ReviewHistory";
import { RunCancellationControl } from "./RunCancellationControl";
import { GoalTitle } from "./GoalTitle";
import { InvestigationResult } from "./InvestigationResult";
import { investigationStateLabel } from "./format";
import {
  forgetLifecycleRequest,
  rememberLifecycleRequest,
} from "./lifecycle-request-store";
import type {
  DashboardContinueCommand,
  DashboardResumeCommand,
  DashboardStartCommand,
  DashboardSystemInfo,
  DashboardThreadDetail,
  InvestigationOperationResult,
} from "./types";

export type InspectorLens =
  | "topology"
  | "plan"
  | "trace"
  | "context"
  | "memory"
  | "cost";

interface ConversationWorkspaceProps {
  system: DashboardSystemInfo;
  detail: DashboardThreadDetail | null;
  lifecycleMutationsEnabled: boolean;
  onThreadActivated: (threadId: string) => void;
  onOpenLens: (lens: InspectorLens, entityId?: string) => void;
  onNewGoal: () => void;
}

const INSPECTORS: { id: InspectorLens; label: string }[] = [
  { id: "topology", label: "Path" },
  { id: "plan", label: "Plan" },
  { id: "trace", label: "Trace" },
  { id: "context", label: "Context" },
  { id: "memory", label: "Memory" },
  { id: "cost", label: "Cost" },
];

let fallbackRequestSequence = 0;

export function ConversationWorkspace({
  system,
  detail,
  lifecycleMutationsEnabled,
  onThreadActivated,
  onOpenLens,
  onNewGoal,
}: ConversationWorkspaceProps) {
  if (!detail) {
    return (
      <NewGoalWorkspace
        system={system}
        enabled={lifecycleMutationsEnabled}
        onThreadActivated={onThreadActivated}
      />
    );
  }
  return (
    <ThreadConversation
      system={system}
      detail={detail}
      enabled={lifecycleMutationsEnabled}
      onThreadActivated={onThreadActivated}
      onOpenLens={onOpenLens}
      onNewGoal={onNewGoal}
    />
  );
}

function NewGoalWorkspace({
  system,
  enabled,
  onThreadActivated,
}: {
  system: DashboardSystemInfo;
  enabled: boolean;
  onThreadActivated: (threadId: string) => void;
}) {
  const queryClient = useQueryClient();
  const [goal, setGoal] = useState("");
  const [focusPath, setFocusPath] = useState(".");
  const [memoryEnabled, setMemoryEnabled] = useState(system.memory_available);
  const [reviewEnabled, setReviewEnabled] = useState(true);
  const [costBudget, setCostBudget] = useState("");
  const [lockedCommand, setLockedCommand] =
    useState<DashboardStartCommand | null>(null);
  const mutation = useMutation({
    mutationFn: startInvestigation,
    retry: false,
    onSuccess: (result) => {
      forgetLifecycleRequest(result.request_id);
      void queryClient.invalidateQueries({ queryKey: ["dashboard"] });
      onThreadActivated(result.thread_id);
    },
  });

  const resetCommand = () => {
    if (lockedCommand) forgetLifecycleRequest(lockedCommand.request_id);
    setLockedCommand(null);
    mutation.reset();
  };
  const submit = () => {
    if (!enabled || !goal.trim() || mutation.isPending || mutation.isSuccess) return;
    const command: DashboardStartCommand = {
      schema_version: 1,
      request_id: createRequestId("start"),
      expected_repository_id: system.repository_id,
      expected_commit_sha: system.commit_sha,
      goal: goal.trim(),
      focus_path: focusPath.trim() || ".",
      human_review_policy: reviewEnabled ? "before_finalize" : "never",
      memory_retrieval_enabled: system.memory_available && memoryEnabled,
      max_attempt_cost_usd: costBudget.trim() || null,
    };
    setLockedCommand(command);
    rememberLifecycleRequest({
      kind: "start",
      thread_id: null,
      command,
    });
    mutation.mutate(command);
  };

  return (
    <div className="conversation-stage conversation-stage--new">
      <section className="goal-welcome" aria-labelledby="new-goal-title">
        <span className="goal-welcome__mark"><Sparkles size={22} /></span>
        <p className="eyebrow">repository agent workspace</p>
        <h1 id="new-goal-title">Declare the current goal</h1>
        <p>
          Ask a question about this repository. Follow the answer back to source,
          then decide what to investigate next.
        </p>
        <WorkspaceBoundary system={system} focusPath={focusPath} />
      </section>

      <section className="goal-composer" aria-label="Declare current goal">
        <div className="goal-composer__heading">
          <div><MessageSquareText size={17} /><span>Current goal</span></div>
          <small>new investigation</small>
        </div>
        <label className="goal-field">
          <span>What should RepoCompass investigate?</span>
          <textarea
            value={goal}
            rows={4}
            maxLength={16_384}
            disabled={lockedCommand !== null || mutation.isPending || mutation.isSuccess}
            placeholder="Trace how focused-improve verifies and applies a proposal"
            onChange={(event) => setGoal(event.target.value)}
          />
        </label>
        <div className="goal-suggestions" aria-label="Example investigation goals">
          {[
            ["Trace a request", "Trace the main request path from its entry point and cite the functions involved."],
            ["Find regression coverage", "Find the tests that protect the main workflow and identify missing coverage."],
            ["Map dependencies", "Explain the main module dependencies with source evidence."],
          ].map(([label, objective]) => (
            <button key={label} type="button"
              disabled={!enabled || lockedCommand !== null || mutation.isPending || mutation.isSuccess}
              onClick={() => setGoal(objective ?? "")}>{label}</button>
          ))}
        </div>
        <details className="task-settings">
          <summary>
            <Settings2 size={15} /><strong>Task settings</strong>
            <span className="task-settings__preview">
              {reviewEnabled ? "Review on" : "Review off"} · Memory {memoryEnabled && system.memory_available ? "on" : "off"} · {costBudget.trim() ? `USD ${costBudget.trim()} cap` : "No task cost cap"}
            </span>
          </summary>
          <p>Optional for this new goal. Submitted requests keep their original settings.</p>
          <ComposerOptions
          system={system}
          focusPath={focusPath}
          onFocusPathChange={setFocusPath}
          memoryEnabled={memoryEnabled}
          onMemoryEnabledChange={setMemoryEnabled}
          reviewEnabled={reviewEnabled}
          onReviewEnabledChange={setReviewEnabled}
          costBudget={costBudget}
          onCostBudgetChange={setCostBudget}
          disabled={lockedCommand !== null || mutation.isPending || mutation.isSuccess}
          />
          <BrowserBoundary system={system} />
        </details>
        <MutationFooter
          enabled={enabled}
          pending={mutation.isPending}
          result={mutation.data ?? null}
          error={mutation.error}
          actionLabel="Start investigation"
          disabled={!goal.trim() || lockedCommand !== null}
          onSubmit={submit}
          onRetry={() => lockedCommand && mutation.mutate(lockedCommand)}
          onReset={resetCommand}
        />
      </section>
    </div>
  );
}

function ThreadConversation({
  system,
  detail,
  enabled,
  onThreadActivated,
  onOpenLens,
  onNewGoal,
}: {
  system: DashboardSystemInfo;
  detail: DashboardThreadDetail;
  enabled: boolean;
  onThreadActivated: (threadId: string) => void;
  onOpenLens: (lens: InspectorLens, entityId?: string) => void;
  onNewGoal: () => void;
}) {
  const queryClient = useQueryClient();
  const { thread } = detail;
  const attempts = useMemo(
    () => [...detail.attempts].sort((left, right) =>
      left.attempt.started_at_ms - right.attempt.started_at_ms),
    [detail.attempts],
  );
  const latestAttempt = attempts.at(-1)?.attempt ?? null;
  const unfinishedAttempt = latestAttempt?.status === "running";
  const stale = latestAttempt !== null && (
    latestAttempt.repository_id !== system.repository_id
    || latestAttempt.commit_sha !== system.commit_sha
  );
  const [prompt, setPrompt] = useState("");
  const [focusPath, setFocusPath] = useState(".");
  const [continueCommand, setContinueCommand] =
    useState<DashboardContinueCommand | null>(null);
  const [resumeCommand, setResumeCommand] =
    useState<DashboardResumeCommand | null>(null);

  const afterLifecycle = (result: InvestigationOperationResult) => {
    forgetLifecycleRequest(result.request_id);
    void queryClient.invalidateQueries({ queryKey: ["dashboard"] });
    onThreadActivated(result.thread_id);
  };
  const continueMutation = useMutation({
    mutationFn: (command: DashboardContinueCommand) =>
      continueThread(thread.thread_id, command),
    retry: false,
    onSuccess: afterLifecycle,
  });
  const resumeMutation = useMutation({
    mutationFn: (command: DashboardResumeCommand) =>
      resumeThread(thread.thread_id, command),
    retry: false,
    onSuccess: afterLifecycle,
  });

  const submitFollowUp = () => {
    if (!enabled || stale || unfinishedAttempt || !prompt.trim() || continueMutation.isPending) return;
    const command: DashboardContinueCommand = {
      schema_version: 1,
      request_id: createRequestId("continue"),
      prompt: prompt.trim(),
      focus_path: focusPath.trim() || ".",
      expected_repository_id: system.repository_id,
      expected_commit_sha: system.commit_sha,
    };
    setContinueCommand(command);
    rememberLifecycleRequest({
      kind: "continue",
      thread_id: thread.thread_id,
      command,
    });
    continueMutation.mutate(command);
  };
  const resume = () => {
    if (!enabled || stale || unfinishedAttempt || resumeMutation.isPending) return;
    const command: DashboardResumeCommand = {
      schema_version: 1,
      request_id: createRequestId("resume"),
      expected_repository_id: system.repository_id,
      expected_commit_sha: system.commit_sha,
    };
    setResumeCommand(command);
    rememberLifecycleRequest({
      kind: "resume",
      thread_id: thread.thread_id,
      command,
    });
    resumeMutation.mutate(command);
  };

  return (
    <div className="conversation-stage">
      <header className="conversation-goal">
        <div>
          <p className="eyebrow">current goal</p>
          <GoalTitle objective={thread.objective} />
          <span>turn {thread.turn_number} · thread/{shortId(thread.thread_id)}</span>
        </div>
        <button type="button" className="new-goal-link" onClick={onNewGoal}>
          <Sparkles size={14} />New goal
        </button>
      </header>

      <div className="conversation-feed" aria-label="Investigation conversation">
        {latestAttempt ? <RunCancellationControl attempt={latestAttempt} enabled={system.cancellation_enabled === true} /> : null}
        <ConversationHistory key={thread.thread_id} threadId={thread.thread_id}
          turnNumber={thread.turn_number} repositoryId={system.repository_id} commitSha={system.commit_sha} />
        <article className="chat-message chat-message--user">
          <span className="chat-avatar"><UserRound size={16} /></span>
          <div>
            <header><strong>You</strong><small>recorded objective</small></header>
            <p>{thread.objective}</p>
          </div>
        </article>

        <CurrentReviewHistory threadId={thread.thread_id} turnNumber={thread.turn_number}
          repositoryId={system.repository_id} commitSha={system.commit_sha}
          checkpointKey={`${latestAttempt?.run_id}:${latestAttempt?.status}:${thread.interrupt_id}`} />

        <article className="chat-message chat-message--agent">
          <span className="chat-avatar"><Bot size={17} /></span>
          <div>
            <header>
              <strong>RepoCompass</strong>
              <span className={`state-chip state-chip--${investigationStateLabel(thread)}`}>
                {unfinishedAttempt ? "attempt open" : investigationStateLabel(thread)}
              </span>
            </header>
            <InvestigationResult thread={thread} onOpenLens={onOpenLens} />
            <details className="conversation-inspection">
              <summary>Inspect this investigation</summary>
              <div className="inspector-shortcuts" aria-label="Open investigation inspector">
                {INSPECTORS.map((item) => (
                  <button key={item.id} type="button" onClick={() => onOpenLens(item.id)}>
                    {item.label}
                  </button>
                ))}
              </div>
              <AttemptRail attempts={attempts.map((item) => item.attempt)} />
              <BrowserBoundary system={system} />
            </details>
          </div>
        </article>

        {stale ? (
          <div className="snapshot-warning" role="alert">
            <GitCommitHorizontal size={18} />
            <div>
              <strong>Repository snapshot changed</strong>
              <span>This checkpoint targets {shortId(latestAttempt?.commit_sha ?? "unversioned")}; current HEAD is {shortId(system.commit_sha ?? "unversioned")}.</span>
            </div>
            <button type="button" onClick={onNewGoal}>Start fresh</button>
          </div>
        ) : null}

        {thread.pending_review ? (
          <HumanReviewPanel detail={detail} enabled={enabled && !stale && !unfinishedAttempt} />
        ) : null}
      </div>

      <footer className="conversation-command">
        {thread.execution_state === "completed" && enabled && !unfinishedAttempt ? (
          <>
            <div className="follow-up-box">
              <label>
                <span className="sr-only">Ask a follow-up</span>
                <textarea
                  rows={2}
                  value={prompt}
                  maxLength={16_384}
                  disabled={!enabled || stale || continueCommand !== null || continueMutation.isPending}
                  placeholder="Ask a follow-up about this repository…"
                  onChange={(event) => setPrompt(event.target.value)}
                />
              </label>
              <div className="follow-up-box__bar">
                <details className="follow-up-scope">
                  <summary>Follow-up scope · <code>{focusPath.trim() || "."}</code></summary>
                  <label className="compact-path"><FileCode2 size={13} /><span>Focus</span><input value={focusPath} maxLength={2048} disabled={!enabled || stale || continueCommand !== null} onChange={(event) => setFocusPath(event.target.value)} /></label>
                  <small>Repository-relative path; applies to the next turn.</small>
                </details>
                <button type="button" aria-label="Send follow-up" disabled={!prompt.trim() || !enabled || stale || continueCommand !== null || continueMutation.isPending} onClick={submitFollowUp}>
                  {continueMutation.isPending ? <LoaderCircle className="is-spinning" size={16} /> : <ArrowUp size={16} />}
                </button>
              </div>
            </div>
            <InlineMutationResult
              pending={continueMutation.isPending}
              result={continueMutation.data ?? null}
              error={continueMutation.error}
              onRetry={() => continueCommand && continueMutation.mutate(continueCommand)}
              onReset={() => {
                if (continueCommand) {
                  forgetLifecycleRequest(continueCommand.request_id);
                }
                setContinueCommand(null);
                continueMutation.reset();
              }}
            />
          </>
        ) : null}

        {unfinishedAttempt ? <p className="command-hint">Attempt not finalized. A resumable checkpoint does not mean the worker has stopped. Inspect events or request cancellation; no second resume is sent.</p> : null}

        {thread.execution_state === "resumable" && !unfinishedAttempt ? (
          <div className="resume-command">
            <div><RotateCcw size={18} /><span><strong>Checkpoint can resume</strong><small>Continue pending graph nodes with the same budgets and evidence boundary.</small></span></div>
            <button type="button" disabled={!enabled || stale || resumeCommand !== null || resumeMutation.isPending} onClick={resume}>
              {resumeMutation.isPending ? <LoaderCircle className="is-spinning" size={15} /> : <CirclePlay size={15} />}Resume
            </button>
            <InlineMutationResult pending={resumeMutation.isPending} result={resumeMutation.data ?? null} error={resumeMutation.error} onRetry={() => resumeCommand && resumeMutation.mutate(resumeCommand)} />
          </div>
        ) : null}

        {thread.execution_state === "interrupted" && !thread.pending_review ? (
          <p className="command-hint"><ShieldAlert size={15} />This thread is interrupted but no public review payload is available.</p>
        ) : null}

        {!enabled ? (
          <div className="read-only-notice">
            <ShieldAlert size={18} />
            <div>
              <strong>{system.demo_mode ? "Explore a completed example" : "Read-only workspace"}</strong>
              <p>{system.demo_mode
                ? "This example uses scripted model responses with real tools and storage. Explore the evidence, memory and cost; it does not measure model quality."
                : "You can inspect saved investigations. To run new goals, enable lifecycle controls when starting the dashboard."}</p>
            </div>
          </div>
        ) : null}
      </footer>
    </div>
  );
}

function ComposerOptions({
  system,
  focusPath,
  onFocusPathChange,
  memoryEnabled,
  onMemoryEnabledChange,
  reviewEnabled,
  onReviewEnabledChange,
  costBudget,
  onCostBudgetChange,
  disabled,
}: {
  system: DashboardSystemInfo;
  focusPath: string;
  onFocusPathChange: (value: string) => void;
  memoryEnabled: boolean;
  onMemoryEnabledChange: (value: boolean) => void;
  reviewEnabled: boolean;
  onReviewEnabledChange: (value: boolean) => void;
  costBudget: string;
  onCostBudgetChange: (value: string) => void;
  disabled: boolean;
}) {
  return (
    <div className="composer-options">
      <label className="path-option">
        <span><FileCode2 size={13} />Focus path</span>
        <input value={focusPath} maxLength={2048} disabled={disabled} placeholder="." onChange={(event) => onFocusPathChange(event.target.value)} />
        <small>repository-relative; validated by the server</small>
      </label>
      <div className="toggle-options">
        <label><input type="checkbox" checked={memoryEnabled && system.memory_available} disabled={disabled || !system.memory_available} onChange={(event) => onMemoryEnabledChange(event.target.checked)} /><span>Recall memory</span></label>
        <label><input type="checkbox" checked={reviewEnabled} disabled={disabled} onChange={(event) => onReviewEnabledChange(event.target.checked)} /><span>Review before finalize</span></label>
      </div>
      <label className="cost-option">
        <span>Attempt cap</span>
        <input value={costBudget} disabled={disabled || !system.pricing_configured} inputMode="decimal" placeholder={system.pricing_configured ? "USD, optional" : "pricing unavailable"} onChange={(event) => onCostBudgetChange(event.target.value)} />
      </label>
    </div>
  );
}

export function BrowserBoundary({ system }: { system: DashboardSystemInfo }) {
  const pages = system.static_browser_page_ids ?? [];
  return (
    <details className="browser-boundary">
      <summary><ShieldAlert size={14} />{pages.length
        ? `Static browser · ${pages.length} operator-approved pages`
        : "Agent browser disabled"}</summary>
      <p>{pages.length
        ? "只读页面检查 · Static HTML only. No site scripts, login state, forms, uploads or downloads. Runtime isolation is checked on use."
        : "浏览器默认关闭 · Only the local operator can bind self-owned test pages at server startup. Chat messages cannot grant access."}</p>
      {pages.length ? <ul aria-label="Operator-approved page IDs">{pages.map((id) => <li key={id}><code>{id}</code></li>)}</ul> : null}
      <p>Page text is untrusted, not verified repository evidence. Tool admissions and refusals appear in Run trace.</p>
    </details>
  );
}

function WorkspaceBoundary({ system, focusPath }: { system: DashboardSystemInfo; focusPath: string }) {
  return (
    <div className="workspace-boundary">
      <FolderGit2 size={16} />
      <span><strong>{system.workspace_name}</strong><code>/{focusPath.trim() || "."}</code></span>
      <small>server-bound</small>
    </div>
  );
}

function MutationFooter({
  enabled,
  pending,
  result,
  error,
  actionLabel,
  disabled,
  onSubmit,
  onRetry,
  onReset,
}: {
  enabled: boolean;
  pending: boolean;
  result: InvestigationOperationResult | null;
  error: Error | null;
  actionLabel: string;
  disabled: boolean;
  onSubmit: () => void;
  onRetry: () => void;
  onReset: () => void;
}) {
  return (
    <div className="composer-submit-row">
      <span><ShieldAlert size={13} />Read-only investigation · repository fixed by server</span>
      <button type="button" disabled={!enabled || disabled || pending || result !== null} onClick={onSubmit}>
        {pending ? <LoaderCircle className="is-spinning" size={15} /> : <CirclePlay size={15} />}{pending ? "Running…" : actionLabel}
      </button>
      {!enabled ? <p className="mutation-inline is-error">Lifecycle mutations are disabled.</p> : null}
      <InlineMutationResult pending={pending} result={result} error={error} onRetry={onRetry} onReset={onReset} />
    </div>
  );
}

function InlineMutationResult({ pending, result, error, onRetry, onReset }: {
  pending: boolean;
  result: InvestigationOperationResult | null;
  error: Error | null;
  onRetry: () => void;
  onReset?: () => void;
}) {
  if (pending) return <p className="mutation-inline" role="status"><LoaderCircle className="is-spinning" size={13} />The agent is running. This request may take a few minutes.</p>;
  if (error) return (
    <p className="mutation-inline is-error" role="alert">
      <ShieldAlert size={13} />{mutationMessage(error)}
      <button type="button" onClick={onRetry}>Retry same request</button>
      {onReset ? <button type="button" onClick={onReset}>Edit request</button> : null}
    </p>
  );
  if (result) return <p className="mutation-inline is-success" role="status"><CheckCircle2 size={13} />Checkpoint {result.execution_state} · {shortId(result.request_id)}</p>;
  return null;
}

function AttemptRail({ attempts }: { attempts: DashboardThreadDetail["attempts"][number]["attempt"][] }) {
  if (attempts.length === 0) return null;
  return (
    <ol className="attempt-rail" aria-label="Persisted attempts">
      {attempts.map((attempt) => (
        <li key={attempt.run_id} className={`attempt-rail__item attempt-rail__item--${attempt.status}`}>
          <span>{attempt.turn_number}</span>
          <div><strong>{attempt.execution_mode}</strong><small><Clock3 size={11} />{attempt.status}</small></div>
        </li>
      ))}
    </ol>
  );
}

function createRequestId(kind: "start" | "resume" | "continue"): string {
  const randomUuid = globalThis.crypto?.randomUUID?.();
  if (randomUuid) return `dashboard-${kind}:${randomUuid}`;
  fallbackRequestSequence += 1;
  return `dashboard-${kind}:${Date.now().toString(36)}:${fallbackRequestSequence.toString(36)}`;
}

function mutationMessage(error: Error): string {
  return error instanceof DashboardApiError
    ? error.problem?.detail ?? error.message
    : "The lifecycle request ended without a confirmed result.";
}

function shortId(value: string): string {
  return value.length <= 10 ? value : value.slice(0, 8);
}
