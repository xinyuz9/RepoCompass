import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

const system = {
  schema_version: 1,
  product: "RepoCompass",
  product_version: "0.1.0",
  repository_id: "a".repeat(64),
  commit_sha: "b".repeat(40),
  workspace_name: "RepoCompass",
  workspace_path: ".",
  demo_mode: false,
  memory_available: true,
  pricing_configured: false,
  lifecycle_mutations_enabled: false,
  memory_mutations_enabled: false,
};

const catalog = {
  schema_version: 1,
  threads: [],
  scanned_attempt_count: 0,
  truncated: false,
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("App query shell", () => {
  it("does not mount a disposable goal composer while the thread catalog is pending", async () => {
    window.history.replaceState({}, "", "/?lens=chat");
    let release!: (response: Response) => void;
    const pendingCatalog = new Promise<Response>((resolve) => { release = resolve; });
    vi.stubGlobal("fetch", vi.fn((input: string | URL | Request) => String(input).endsWith("/system")
      ? Promise.resolve(new Response(JSON.stringify(system), { headers: { "content-type": "application/json" } }))
      : pendingCatalog));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
    await waitFor(() => expect(client.getQueriesData({ queryKey: ["dashboard"] }).some(([, data]) => Boolean(data))).toBe(true));
    expect(screen.queryByRole("heading", { name: "Declare the current goal" })).not.toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "What should RepoCompass investigate?" })).not.toBeInTheDocument();
    await act(async () => { release(new Response(JSON.stringify(catalog), { headers: { "content-type": "application/json" } })); });
    expect(await screen.findByRole("heading", { name: "Declare the current goal" })).toBeVisible();
  });

  it("deduplicates cached bootstrap reads and renders the first-run empty state", async () => {
    const fetchMock = vi.fn((input: string | URL | Request) => {
      const url = String(input);
      const body = url.endsWith("/system") ? system : catalog;
      return Promise.resolve(
        new Response(JSON.stringify(body), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );
    });
    vi.stubGlobal("fetch", fetchMock);
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false, staleTime: 60_000 },
      },
    });

    const view = render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    expect(await screen.findByRole("heading", { name: "Declare the current goal" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Declare current goal" })).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Hide investigations" }));
    expect(screen.queryByRole("complementary", { name: "Investigation threads" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show investigations" }));
    screen.getByRole("textbox", { name: "Filter investigations" }).focus();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.getByRole("button", { name: "Show investigations" })).toHaveFocus();
    expect(screen.getByRole("button", { name: "Show investigations" })).toHaveAttribute("aria-expanded", "false");
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

    view.rerender(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });

  it("lazy-loads observability resources and reuses the cached run across lenses", async () => {
    const attempt = {
      schema_version: 1,
      run_id: "run-1",
      thread_id: "thread-1",
      turn_number: 1,
      execution_mode: "start",
      repository_id: "a".repeat(64),
      commit_sha: "b".repeat(40),
      objective_digest: "c".repeat(64),
      status: "completed",
      started_at_ms: 1_000,
      finished_at_ms: 1_250,
    };
    const populatedCatalog = {
      ...catalog,
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
      attempts: [{ schema_version: 1, attempt, trace_availability: "available", cost_availability: "not_configured" }],
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
    const memory = { schema_version: 1, availability: "not_configured", entries: [], tombstones: [] };
    const requestedPaths: string[] = [];
    const fetchMock = vi.fn((input: string | URL | Request) => {
      const path = new URL(String(input), "http://dashboard.local").pathname;
      requestedPaths.push(path);
      const body = path.endsWith("/system")
        ? system
        : path.endsWith("/threads")
          ? populatedCatalog
          : path.endsWith("/threads/thread-1")
            ? thread
            : path.endsWith("/runs/run-1")
              ? run
              : memory;
      return Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } }));
    });
    vi.stubGlobal("fetch", fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: 60_000 } } });

    render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
    expect(await screen.findByRole("heading", { name: "Trace the authentication flow." })).toBeVisible();
    expect(requestedPaths.filter((path) => path.includes("/runs/")).length).toBe(0);
    const lensNavigation = screen.getByRole("navigation", { name: "Primary lenses" });

    expect(within(lensNavigation).queryByRole("button", { name: "Memory" })).not.toBeInTheDocument();
    fireEvent.click(within(lensNavigation).getByRole("button", { name: "Inspect" }));
    expect(requestedPaths.filter((path) => path.includes("/runs/") || path.endsWith("/memory"))).toEqual([]);
    within(lensNavigation).getByRole("button", { name: "Memory" }).focus();
    fireEvent.keyDown(within(lensNavigation).getByRole("button", { name: "Memory" }), { key: "Escape" });
    expect(within(lensNavigation).getByRole("button", { name: "Inspect" })).toHaveFocus();
    expect(within(lensNavigation).queryByRole("button", { name: "Memory" })).not.toBeInTheDocument();
    fireEvent.click(within(lensNavigation).getByRole("button", { name: "Inspect" }));

    fireEvent.click(within(lensNavigation).getByRole("button", { name: "Context" }));
    expect(await screen.findByRole("heading", { name: "No ContextPack persisted" })).toBeVisible();
    expect(requestedPaths.filter((path) => path.endsWith("/runs/run-1")).length).toBe(1);

    fireEvent.click(within(lensNavigation).getByRole("button", { name: "Cost" }));
    expect(await screen.findByRole("heading", { name: "Pricing not configured" })).toBeVisible();
    expect(requestedPaths.filter((path) => path.endsWith("/runs/run-1")).length).toBe(1);

    fireEvent.click(within(lensNavigation).getByRole("button", { name: "Memory" }));
    expect(await screen.findByRole("heading", { name: "Memory store not configured" })).toBeVisible();
    expect(requestedPaths.filter((path) => path.endsWith("/memory")).length).toBe(1);
  });
});
