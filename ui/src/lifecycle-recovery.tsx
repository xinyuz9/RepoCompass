import { useIsMutating, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  CheckCircle2,
  LoaderCircle,
  Radio,
  RefreshCw,
  RotateCcw,
  X,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import {
  continueThread,
  DashboardApiError,
  dashboardQueries,
  respondToReview,
  resumeThread,
  startInvestigation,
  reconcileOperation,
} from "./api";
import { compactId } from "./format";
import { RunCancellationControl } from "./RunCancellationControl";
import {
  forgetLifecycleRequest,
  lifecycleRequestChangeEvent,
  readLifecycleRequests,
  type StoredLifecycleRequest,
} from "./lifecycle-request-store";
import type {
  DashboardLifecycleOperationStatus,
  DashboardSystemInfo,
  InvestigationOperationResult,
} from "./types";

type RecoveryWorkspace = Pick<DashboardSystemInfo,
  "repository_id" | "commit_sha" | "lifecycle_mutations_enabled" | "cancellation_enabled">;

export function LifecycleRecoveryCenter({
  system,
  onThreadActivated,
  visibleThreadId,
}: {
  system: RecoveryWorkspace | undefined;
  onThreadActivated: (threadId: string) => void;
  visibleThreadId?: string | null;
}) {
  const [requests, setRequests] = useState(readLifecycleRequests);

  useEffect(() => {
    const refresh = () => setRequests(readLifecycleRequests());
    const changeEvent = lifecycleRequestChangeEvent();
    window.addEventListener(changeEvent, refresh);
    window.addEventListener("storage", refresh);
    return () => {
      window.removeEventListener(changeEvent, refresh);
      window.removeEventListener("storage", refresh);
    };
  }, []);

  if (requests.length === 0) return null;
  return (
    <section className="recovery-center" aria-label="Lifecycle request recovery">
      {requests.map((request) => (
        <LifecycleRecoveryItem
          key={request.command.request_id}
          request={request}
          system={system}
          onThreadActivated={onThreadActivated}
          visibleThreadId={visibleThreadId}
        />
      ))}
    </section>
  );
}

function LifecycleRecoveryItem({
  request,
  system,
  onThreadActivated,
  visibleThreadId,
}: {
  request: StoredLifecycleRequest;
  system: RecoveryWorkspace | undefined;
  onThreadActivated: (threadId: string) => void;
  visibleThreadId?: string | null;
}) {
  const queryClient = useQueryClient();
  const blockReason = recoveryBlockReason(request, system);
  const canObserve = blockReason === null;
  const locallySubmitting = useIsMutating({
    predicate: (mutation) => {
      const variables = mutation.state.variables;
      return typeof variables === "object" && variables !== null
        && "request_id" in variables && variables.request_id === request.command.request_id;
    },
  }) > 0;
  const operationQuery = dashboardQueries.operation(request.command.request_id);
  const operation = useQuery({
    ...operationQuery,
    queryKey: [...operationQuery.queryKey, system?.repository_id, system?.commit_sha],
    enabled: canObserve,
    retry: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchInterval: (query) => {
      if (!canObserve || query.state.dataUpdateCount >= 120
        || query.state.data?.recovery_status === "reconcilable") return false;
      if (query.state.status === "error") {
        // A locally pending POST may lose the race to its first status GET.
        // Recovery after reload has no local mutation: its 404 still stops.
        return locallySubmitting && query.state.error instanceof DashboardApiError
          && query.state.error.status === 404 && query.state.errorUpdateCount < 3 ? 500 : false;
      }
      return query.state.data?.state === "completed"
        || query.state.data?.latest_attempt?.status === "failed" ? false : 1_500;
    },
  });
  const completeRecovery = useCallback(
    (result: InvestigationOperationResult) => {
      forgetLifecycleRequest(result.request_id);
      void queryClient.invalidateQueries({ queryKey: ["dashboard"] });
      onThreadActivated(result.thread_id);
    },
    [onThreadActivated, queryClient],
  );
  const replay = useMutation({
    mutationFn: () => {
      if (!canObserve) return Promise.reject(new Error(blockReason ?? "Recovery paused"));
      return replayLifecycleRequest(request);
    },
    retry: false,
    onSuccess: (result) => completeRecovery(result),
    onError: () => { void operation.refetch(); },
  });
  const reconcile = useMutation({
    mutationFn: () => {
      const runId = operation.data?.bound_run_id;
      if (!canObserve || !runId || !system || operation.data?.recovery_status !== "reconcilable") {
        return Promise.reject(new Error("An exact inactive worker binding is required."));
      }
      return reconcileOperation({ request_id: request.command.request_id, expected_run_id: runId,
        expected_repository_id: system.repository_id, expected_commit_sha: system.commit_sha });
    },
    retry: false,
    onSuccess: () => { void queryClient.invalidateQueries({ queryKey: ["dashboard"] }); },
    onError: () => { void operation.refetch(); },
  });

  useEffect(() => {
    const result = operation.data?.result;
    if (canObserve && operation.data?.state === "completed" && result && !result.recovery_note) completeRecovery(result);
  }, [canObserve, completeRecovery, operation.data]);

  const notFound = operation.error instanceof DashboardApiError
    && operation.error.status === 404;
  const attemptFailed = operation.data?.latest_attempt?.status === "failed";
  const canReplay = canObserve && !locallySubmitting && notFound;
  const canReconcile = canObserve && operation.data?.state === "pending"
    && operation.data.recovery_status === "reconcilable" && !!operation.data.bound_run_id;
  const phase = blockReason ?? (notFound && locallySubmitting ? "Waiting for request reservation"
    : operationPhase(operation.data, operation.error, operation.isPending));
  const running = operation.data?.latest_attempt?.status === "running" && !canReconcile;
  const threadId = operation.data?.thread_id;

  return (
    <article className={`recovery-item${blockReason || attemptFailed || operation.isError ? " is-alert" : ""}`}>
      <span className="recovery-item__signal">
        {blockReason ? <AlertTriangle size={16} />
          : running || replay.isPending || operation.isPending
          ? <LoaderCircle className="is-spinning" size={16} />
          : operation.data?.state === "completed"
            ? <CheckCircle2 size={16} />
            : attemptFailed || operation.isError
              ? <AlertTriangle size={16} />
              : <Radio size={16} />}
      </span>
      <div className="recovery-item__copy">
        <strong>{phase}</strong>
        <span>
          {request.kind} · request/{compactId(request.command.request_id, 12)}
          {operation.data?.latest_attempt
            ? ` · run/${compactId(operation.data.latest_attempt.run_id, 10)}`
            : ""}
        </span>
      </div>
      <div className="recovery-item__actions">
        {canObserve && threadId ? (
          <button type="button" onClick={() => operation.data?.result?.recovery_note ? completeRecovery(operation.data.result) : onThreadActivated(threadId)}>
            {operation.data?.result?.recovery_note ? "Open reconciled thread" : "Open thread"}
          </button>
        ) : null}
        {canReplay ? (
          <button
            type="button"
            disabled={replay.isPending}
            onClick={() => replay.mutate()}
          >
            <RotateCcw size={12} />Replay exact request
          </button>
        ) : null}
        {canReconcile ? <button type="button" disabled={reconcile.isPending} onClick={() => reconcile.mutate()}>
          <RefreshCw size={12} />Reconcile saved run
        </button> : null}
        {canObserve && (operation.isError || attemptFailed || operation.data?.state === "pending") ? (
          <button type="button" disabled={operation.isFetching || replay.isPending} onClick={() => void operation.refetch()}>
            <RefreshCw size={12} />Retry status
          </button>
        ) : null}
        <button
          type="button"
          title="Hide this tracker only; the backend operation is not cancelled."
          aria-label={`Hide tracker for ${request.command.request_id}`}
          onClick={() => forgetLifecycleRequest(request.command.request_id)}
        >
          <X size={13} />
        </button>
      </div>
      {replay.isError ? (
        <p role="alert">Exact replay was not confirmed. The same request remains recoverable.</p>
      ) : null}
      {canReconcile ? <p>No worker holds this request lock. Reconciliation checks the exact checkpoint and repairs local records; it does not rerun models or release unknown cost reservations.</p> : null}
      {operation.data?.result?.recovery_note ? <p className="recovery-note">Local records reconciled without model replay. The interrupted attempt remains failed; checkpoint acceptance is a separate result. Open the thread to inspect or explicitly resume pending work.</p> : null}
      {reconcile.isError ? <p role="alert">Reconciliation was not confirmed. Inspect the persisted state or retry status; no automatic rerun occurs.</p> : null}
      {canObserve && operation.data?.latest_attempt && threadId !== visibleThreadId && !canReconcile ? <RunCancellationControl
        attempt={operation.data.latest_attempt} enabled={system?.cancellation_enabled === true} /> : null}
    </article>
  );
}

function replayLifecycleRequest(
  request: StoredLifecycleRequest,
): Promise<InvestigationOperationResult> {
  switch (request.kind) {
    case "start": return startInvestigation(request.command);
    case "resume": return resumeThread(request.thread_id, request.command);
    case "continue": return continueThread(request.thread_id, request.command);
    case "respond": return respondToReview(request.thread_id, request.command);
  }
}

function recoveryBlockReason(
  request: StoredLifecycleRequest,
  system: RecoveryWorkspace | undefined,
): string | null {
  if (!system) return "Waiting for workspace identity";
  if (!system.lifecycle_mutations_enabled) return "Read-only workspace · recovery paused";
  if (request.kind !== "respond" && (
    request.command.expected_repository_id !== system.repository_id
    || request.command.expected_commit_sha !== system.commit_sha
  )) {
    return "Repository snapshot changed · open the original workspace or start a new goal";
  }
  return null;
}

function operationPhase(
  operation: DashboardLifecycleOperationStatus | undefined,
  error: Error | null,
  pending: boolean,
): string {
  if (pending) return "Locating durable request";
  if (error instanceof DashboardApiError && error.status === 404) {
    return "Request was not reserved";
  }
  if (error) return "Operation status disconnected";
  if (!operation) return "Waiting for operation status";
  if (operation.state === "completed") return operation.result?.recovery_note ? "Local records reconciled" : "Checkpoint confirmed";
  if (operation.recovery_status === "reconcilable") return "Worker inactive · reconciliation available";
  if (operation.latest_attempt?.status === "running") return "Agent attempt running";
  if (operation.latest_attempt?.status === "failed") return "Attempt failed · inspect saved state";
  if (operation.latest_attempt) return "Finalizing durable checkpoint";
  return "Request reserved";
}
