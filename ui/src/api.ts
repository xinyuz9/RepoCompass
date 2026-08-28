import type {
  DashboardHumanResponseCommand,
  DashboardMemoryAudit,
  DashboardMemoryCatalog,
  DashboardMemoryDeleteCommand,
  DashboardMemoryDeleteResult,
  DashboardProblemDetail,
  DashboardRunDetail,
  DashboardSystemInfo,
  DashboardThreadCatalog,
  DashboardThreadDetail,
  InvestigationOperationResult,
} from "./types";

const API_ROOT = "/api/v1";

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
  run: (runId: string) => ({
    queryKey: ["dashboard", "runs", runId] as const,
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getJson<DashboardRunDetail>(`/runs/${encodeURIComponent(runId)}`, signal),
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
