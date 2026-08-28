import type {
  DashboardProblemDetail,
  DashboardSystemInfo,
  DashboardThreadCatalog,
  DashboardThreadDetail,
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
    let problem: DashboardProblemDetail | null = null;
    if (response.headers.get("content-type")?.includes("application/problem+json")) {
      problem = (await response.json()) as DashboardProblemDetail;
    }
    throw new DashboardApiError(response.status, problem);
  }
  return (await response.json()) as T;
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
};
