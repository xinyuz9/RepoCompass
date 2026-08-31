import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

const repositoryId = "a".repeat(64);
const commitSha = "b".repeat(40);
const attempt = {
  schema_version: 1,
  run_id: "run-1",
  thread_id: "thread-1",
  turn_number: 1,
  execution_mode: "start",
  repository_id: repositoryId,
  commit_sha: commitSha,
  objective_digest: "c".repeat(64),
  status: "completed",
  started_at_ms: 1_000,
  finished_at_ms: 1_250,
};
const system = {
  schema_version: 1,
  product: "RepoCompass",
  product_version: "0.1.0",
  repository_id: repositoryId,
  commit_sha: commitSha,
  workspace_name: "RepoCompass",
  workspace_path: ".",
  memory_available: true,
  pricing_configured: false,
  lifecycle_mutations_enabled: false,
  memory_mutations_enabled: false,
};
const catalog = {
  schema_version: 1,
  threads: [{
    schema_version: 1,
    thread_id: "thread-1",
    checkpoint_availability: "available",
    objective: "Trace the authentication flow.",
    execution_state: "completed",
    turn_number: 1,
    termination_status: "accepted",
    latest_attempt: attempt,
    attempt_count: 1,
  }],
  scanned_attempt_count: 1,
  truncated: false,
};
const thread = {
  schema_version: 1,
  thread: {
    schema_version: 1,
    thread_id: "thread-1",
    execution_state: "completed",
    turn_number: 1,
    objective: "Trace the authentication flow.",
    plan: { steps: [] },
    plan_revision_count: 0,
    max_plan_revisions: 2,
    next_nodes: [],
    termination_status: "accepted",
    step_stop_reason: null,
    structure: null,
    interrupt_id: null,
  },
  attempts: [{
    schema_version: 1,
    attempt,
    trace_availability: "available",
    cost_availability: "not_configured",
  }],
  plan_timeline: [],
  evaluator_timeline: [],
};
const run = {
  schema_version: 1,
  attempt,
  trace_availability: "available",
  trace_events: [],
  plan_timeline: [],
  evaluator_timeline: [],
  context_packs: [],
  model_invocations: [],
  cost_availability: "not_configured",
  usage_cost_report: null,
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("App URL navigation", () => {
  it("opens a shared thread/lens/run link and restores composition on popstate", async () => {
    window.history.replaceState(
      null,
      "",
      "/?thread=thread-1&lens=context&run=run-1",
    );
    const requestedPaths: string[] = [];
    installFetch(requestedPaths);

    renderApp();

    expect(await screen.findByRole("heading", { name: "No ContextPack persisted" })).toBeVisible();
    expect(requestedPaths).toContain("/api/v1/runs/run-1");
    expect(within(screen.getByRole("navigation", { name: "Primary lenses" }))
      .getByRole("button", { name: "Context" })).toHaveClass("is-active");

    act(() => {
      window.history.pushState(null, "", "/?compose=1");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(await screen.findByRole("heading", { name: "Declare the current goal" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Declare current goal" })).toBeVisible();
  });

  it("replaces an unknown persisted run before issuing any useless run request", async () => {
    window.history.replaceState(
      null,
      "",
      "/?thread=thread-1&lens=trace&run=run-missing",
    );
    const requestedPaths: string[] = [];
    installFetch(requestedPaths);

    renderApp();

    expect(await screen.findByRole("heading", { name: "Run trace" })).toBeVisible();
    await waitFor(() => expect(window.location.search).toBe(
      "?thread=thread-1&lens=trace&run=run-1",
    ));
    expect(requestedPaths).toContain("/api/v1/runs/run-1");
    expect(requestedPaths).not.toContain("/api/v1/runs/run-missing");
  });
});

function renderApp() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 60_000 } },
  });
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  );
}

function installFetch(requestedPaths: string[]) {
  const fetchMock = vi.fn((input: string | URL | Request) => {
    const path = new URL(String(input), "http://dashboard.local").pathname;
    requestedPaths.push(path);
    const body = path.endsWith("/system")
      ? system
      : path.endsWith("/threads")
        ? catalog
        : path.endsWith("/threads/thread-1")
          ? thread
          : run;
    return Promise.resolve(new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
