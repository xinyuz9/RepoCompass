import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
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
} from "./api";
import { compactId } from "./format";
import {
  forgetLifecycleRequest,
  lifecycleRequestChangeEvent,
  readLifecycleRequests,
  type StoredLifecycleRequest,
} from "./lifecycle-request-store";
import type {
  DashboardLifecycleOperationStatus,
  InvestigationOperationResult,
} from "./types";

export function LifecycleRecoveryCenter({
  onThreadActivated,
}: {
  onThreadActivated: (threadId: string) => void;
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
          onThreadActivated={onThreadActivated}
        />
      ))}
    </section>
  );
}

function LifecycleRecoveryItem({
  request,
  onThreadActivated,
}: {
  request: StoredLifecycleRequest;
  onThreadActivated: (threadId: string) => void;
}) {
  const queryClient = useQueryClient();
  const operation = useQuery({
    ...dashboardQueries.operation(request.command.request_id),
    retry: false,
    refetchInterval: (query) =>
      query.state.data?.state === "completed" ? false : 1_500,
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
    mutationFn: () => replayLifecycleRequest(request),
    retry: false,
    onSuccess: (result) => completeRecovery(result),
  });

  useEffect(() => {
    const result = operation.data?.result;
    if (operation.data?.state === "completed" && result) completeRecovery(result);
  }, [completeRecovery, operation.data]);

  const notFound = operation.error instanceof DashboardApiError
    && operation.error.status === 404;
  const attemptFailed = operation.data?.latest_attempt?.status === "failed";
  const canReplay = notFound || attemptFailed;
  const phase = operationPhase(operation.data, operation.error, operation.isPending);
  const running = operation.data?.latest_attempt?.status === "running";
  const threadId = operation.data?.thread_id;

  return (
    <article className={`recovery-item${attemptFailed || operation.isError ? " is-alert" : ""}`}>
      <span className="recovery-item__signal">
        {running || replay.isPending || operation.isPending
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
        {threadId ? (
          <button type="button" onClick={() => onThreadActivated(threadId)}>
            Open thread
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
        {operation.isError && !notFound ? (
          <button type="button" onClick={() => void operation.refetch()}>
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
    </article>
  );
}

function replayLifecycleRequest(
  request: StoredLifecycleRequest,
): Promise<InvestigationOperationResult> {
  switch (request.kind) {
    case "start":
      return startInvestigation(request.command);
    case "resume":
      return resumeThread(request.thread_id, request.command);
    case "continue":
      return continueThread(request.thread_id, request.command);
    case "respond":
      return respondToReview(request.thread_id, request.command);
  }
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
  if (operation.state === "completed") return "Checkpoint confirmed";
  if (operation.latest_attempt?.status === "running") return "Agent attempt running";
  if (operation.latest_attempt?.status === "failed") return "Attempt failed · replay available";
  if (operation.latest_attempt) return "Finalizing durable checkpoint";
  return "Request reserved";
}
