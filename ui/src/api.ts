import type {
  DashboardContinueCommand,
  DashboardHumanResponseCommand,
  DashboardLifecycleOperationStatus,
  DashboardMemoryAudit,
  DashboardMemoryCatalog,
  DashboardMemoryDeleteCommand,
  DashboardMemoryDeleteResult,
  DashboardProblemDetail,
  DashboardRunDetail,
  DashboardSourceSnippet,
  DashboardResumeCommand,
  DashboardStartCommand,
  DashboardSystemInfo,
  DashboardThreadCatalog,
  DashboardThreadDetail,
  InvestigationOperationResult,
  PublicConversationPage,
} from "./types";

const API_ROOT = "/api/v1";

export function getConversation(threadId: string, beforeTurn: number, signal?: AbortSignal) {
  return getJson<PublicConversationPage>(
    "/threads/" + encodeURIComponent(threadId) + "/conversation?before_turn=" + beforeTurn + "&limit=10",
    signal,
  );
}

export class DashboardApiError extends Error {
  readonly status: number;
  readonly problem: DashboardProblemDetail | null;

  constructor(status: number, problem: DashboardProblemDetail | null) {
    super(problem?.detail ?? "The local dashboard request failed.");
    this.name = "DashboardApiError";
    this.status = status;
    this.problem = problem;
  }
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    method: "GET",
    headers: { Accept: "application/json" },
    credentials: "same-origin",
    signal,
  });
  if (!response.ok) {
    throw new DashboardApiError(response.status, await readProblem(response));
  }
  return (await response.json()) as T;
}

async function mutateJson<Command extends { request_id: string }, Result>(
  path: string,
  method: "POST" | "DELETE",
  command: Command,
): Promise<Result> {
  const response = await fetch(`${API_ROOT}${path}`, {
    method,
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      "Idempotency-Key": command.request_id,
    },
    credentials: "same-origin",
    body: JSON.stringify(command),
  });
  if (!response.ok) {
    throw new DashboardApiError(response.status, await readProblem(response));
  }
  return (await response.json()) as Result;
}

async function readProblem(response: Response): Promise<DashboardProblemDetail | null> {
  if (!response.headers.get("content-type")?.includes("application/problem+json")) {
    return null;
  }
  return (await response.json()) as DashboardProblemDetail;
}

export function startInvestigation(
  command: DashboardStartCommand,
): Promise<InvestigationOperationResult> {
  return mutateJson("/investigations", "POST", command);
}

export function resumeThread(
  threadId: string,
  command: DashboardResumeCommand,
): Promise<InvestigationOperationResult> {
  return mutateJson(
    `/threads/${encodeURIComponent(threadId)}/resume`,
    "POST",
    command,
  );
}

export function continueThread(
  threadId: string,
  command: DashboardContinueCommand,
): Promise<InvestigationOperationResult> {
  return mutateJson(
    `/threads/${encodeURIComponent(threadId)}/turns`,
    "POST",
    command,
  );
}

export function respondToReview(
  threadId: string,
  command: DashboardHumanResponseCommand,
): Promise<InvestigationOperationResult> {
  return mutateJson(
    `/threads/${encodeURIComponent(threadId)}/responses`,
    "POST",
    command,
  );
}

export function deleteMemory(
  memoryId: string,
  command: DashboardMemoryDeleteCommand,
): Promise<DashboardMemoryDeleteResult> {
  return mutateJson(`/memory/${encodeURIComponent(memoryId)}`, "DELETE", command);
}

export const dashboardQueries = {
  system: {
    queryKey: ["dashboard", "system"] as const,
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getJson<DashboardSystemInfo>("/system", signal),
  },
  threads: {
    queryKey: ["dashboard", "threads"] as const,
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getJson<DashboardThreadCatalog>("/threads", signal),
  },
  thread: (threadId: string) => ({
    queryKey: ["dashboard", "threads", threadId] as const,
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getJson<DashboardThreadDetail>(
        `/threads/${encodeURIComponent(threadId)}`,
        signal,
      ),
  }),
  source: (threadId: string, entityId: string) => ({
    queryKey: ["dashboard", "threads", threadId, "source", entityId] as const,
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getJson<DashboardSourceSnippet>(
        `/threads/${encodeURIComponent(threadId)}/source/${encodeURIComponent(entityId)}`,
        signal,
      ),
  }),
  run: (runId: string) => ({
    queryKey: ["dashboard", "runs", runId] as const,
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getJson<DashboardRunDetail>(`/runs/${encodeURIComponent(runId)}`, signal),
  }),
  operation: (requestId: string) => ({
    queryKey: ["dashboard", "operations", requestId] as const,
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getJson<DashboardLifecycleOperationStatus>(
        `/operations/${encodeURIComponent(requestId)}`,
        signal,
      ),
  }),
  memory: {
    queryKey: ["dashboard", "memory"] as const,
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getJson<DashboardMemoryCatalog>("/memory", signal),
  },
  memoryAudit: (memoryId: string) => ({
    queryKey: ["dashboard", "memory", memoryId, "audit"] as const,
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getJson<DashboardMemoryAudit>(
        `/memory/${encodeURIComponent(memoryId)}/audit`,
        signal,
      ),
  }),
};
