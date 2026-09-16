import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { LifecycleRecoveryCenter } from "./lifecycle-recovery";
import {
  readLifecycleRequests,
  rememberLifecycleRequest,
} from "./lifecycle-request-store";
import type { DashboardStartCommand } from "./types";

const workspace = {
  repository_id: "a".repeat(64),
  commit_sha: "b".repeat(40),
  lifecycle_mutations_enabled: true,
};

afterEach(() => {
  cleanup();
  window.sessionStorage.clear();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("LifecycleRecoveryCenter", () => {
  it("reconciles an inactive exact run only after a click and keeps the recovery explanation", async () => {
    rememberLifecycleRequest(startDraft());
    const onThreadActivated = vi.fn();
    let reconciled = false;
    const result = { ...operationResult(), run_id: "run-1", recovery_note: "orphan_reconciled_without_model_replay" };
    const fetch = vi.fn((_input: string | URL | Request, init?: RequestInit) => {
      if (init?.method === "POST") { reconciled = true; return Promise.resolve(jsonResponse(result)); }
      return Promise.resolve(jsonResponse({ schema_version: 1, request_id: "request-start", kind: "start", thread_id: "thread-1",
        state: reconciled ? "completed" : "pending", reserved_at_ms: 1000,
        bound_run_id: "run-1", recovery_status: reconciled ? "not_needed" : "reconcilable",
        latest_attempt: null, result: reconciled ? result : null }));
    });
    vi.stubGlobal("fetch", fetch);
    renderRecovery(onThreadActivated);
    const button = await screen.findByRole("button", { name: "Reconcile saved run" });
    expect(fetch.mock.calls.filter((call) => call[1]?.method === "POST")).toHaveLength(0);
    expect(screen.queryByRole("button", { name: "Replay exact request" })).not.toBeInTheDocument();
    fireEvent.click(button);
    expect(await screen.findByText("Local records reconciled")).toBeVisible();
    expect(onThreadActivated).not.toHaveBeenCalled();
    const posts = fetch.mock.calls.filter((call) => call[1]?.method === "POST");
    expect(posts).toHaveLength(1);
    expect(posts[0]?.[0]).toBe("/api/v1/operations/request-start/reconcile");
    expect(JSON.parse(String(posts[0]?.[1]?.body))).toEqual({ request_id: "request-start", expected_run_id: "run-1",
      expected_repository_id: workspace.repository_id, expected_commit_sha: workspace.commit_sha });
    fireEvent.click(screen.getByRole("button", { name: "Open reconciled thread" }));
    expect(onThreadActivated).toHaveBeenCalledWith("thread-1");
    expect(readLifecycleRequests()).toEqual([]);
  });

  it("projects a running backend attempt without offering a duplicate replay", async () => {
    rememberLifecycleRequest(startDraft());
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(jsonResponse({
      schema_version: 1,
      request_id: "request-start",
      kind: "start",
      thread_id: "thread-1",
      state: "pending",
      reserved_at_ms: 1_000,
      latest_attempt: {
        schema_version: 1,
        run_id: "run-1",
        thread_id: "thread-1",
        turn_number: 1,
        execution_mode: "start",
        repository_id: "a".repeat(64),
        commit_sha: "b".repeat(40),
        objective_digest: "c".repeat(64),
        status: "running",
        started_at_ms: 1_100,
        finished_at_ms: null,
      },
      result: null,
    }))));

    renderRecovery();

    expect(await screen.findByText("Agent attempt running")).toBeVisible();
    expect(screen.getByText(/run\/run-1/)).toBeVisible();
    expect(screen.queryByRole("button", { name: "Replay exact request" })).not.toBeInTheDocument();
  });

  it("clears a tracker and opens its thread when the durable result is confirmed", async () => {
    rememberLifecycleRequest(startDraft());
    const onThreadActivated = vi.fn();
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(jsonResponse({
      schema_version: 1,
      request_id: "request-start",
      kind: "start",
      thread_id: "thread-1",
      state: "completed",
      reserved_at_ms: 1_000,
      latest_attempt: null,
      result: operationResult(),
    }))));

    renderRecovery(onThreadActivated);

    await waitFor(() => expect(onThreadActivated).toHaveBeenCalledWith("thread-1"));
    expect(readLifecycleRequests()).toEqual([]);
    expect(screen.queryByRole("region", { name: "Lifecycle request recovery" })).not.toBeInTheDocument();
  });

  it("replays the exact bounded request only after a missing reservation is confirmed", async () => {
    const draft = startDraft();
    rememberLifecycleRequest(draft);
    const onThreadActivated = vi.fn();
    const fetchMock = vi.fn((_input: string | URL | Request, init?: RequestInit) => {
      if ((init?.method ?? "GET") === "GET") {
        return Promise.resolve(new Response(JSON.stringify({
          schema_version: 1,
          type: "urn:repocompass:problem:operation-not-found",
          title: "Operation not found",
          status: 404,
          detail: "No durable reservation exists for this request.",
          instance: "/api/v1/operations/request-start",
        }), {
          status: 404,
          headers: { "content-type": "application/problem+json" },
        }));
      }
      return Promise.resolve(jsonResponse(operationResult()));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderRecovery(onThreadActivated);
    fireEvent.click(await screen.findByRole("button", { name: "Replay exact request" }));

    await waitFor(() => expect(onThreadActivated).toHaveBeenCalledWith("thread-1"));
    const postCall = fetchMock.mock.calls.find((call) => call[1]?.method === "POST");
    expect(postCall?.[0]).toBe("/api/v1/investigations");
    expect(postCall?.[1]?.body).toBe(JSON.stringify(draft.command));
    expect((postCall?.[1]?.headers as Record<string, string>)["Idempotency-Key"]).toBe(
      draft.command.request_id,
    );
    expect(readLifecycleRequests()).toEqual([]);
  });
});

function renderRecovery(onThreadActivated = vi.fn(), system = workspace) {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={client}>
      <LifecycleRecoveryCenter system={system} onThreadActivated={onThreadActivated} />
    </QueryClientProvider>,
  );
}

describe("recovery safety and bounded polling", () => {
  it("allows only three reservation reads for an exact locally pending submission", async () => {
    vi.useFakeTimers();
    const draft = startDraft();
    rememberLifecycleRequest(draft);
    const fetcher = vi.fn().mockImplementation(() => Promise.resolve(new Response("", { status: 404 })));
    vi.stubGlobal("fetch", fetcher);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    let release!: () => void;
    const submission = client.getMutationCache().build<void, Error, DashboardStartCommand, unknown>(client, {
      mutationFn: () => new Promise<void>((resolve) => { release = resolve; }),
    }).execute(draft.command);
    render(<QueryClientProvider client={client}><LifecycleRecoveryCenter system={workspace} onThreadActivated={vi.fn()} /></QueryClientProvider>);
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    expect(fetcher).toHaveBeenCalledTimes(3);
    expect(screen.getByText("Waiting for request reservation")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Replay exact request" })).not.toBeInTheDocument();
    await act(async () => { release(); await submission; await vi.advanceTimersByTimeAsync(10); });
    expect(screen.getByRole("button", { name: "Replay exact request" })).toBeVisible();
    client.clear();
  });

  it.each([403, 404, 503])("stops automatic polling after HTTP %i and allows manual status retry", async (status) => {
    vi.useFakeTimers();
    rememberLifecycleRequest(startDraft());
    const fetchMock = vi.fn(() => Promise.resolve(new Response(JSON.stringify({
      status, detail: "Request unavailable", title: "Unavailable",
      type: "urn:repocompass:problem:unavailable", instance: "/api/v1/operations/request-start",
    }), { status, headers: { "content-type": "application/problem+json" } })));
    vi.stubGlobal("fetch", fetchMock);
    renderRecovery();
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Retry status" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Retry status" }));
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it.each([
    { ...workspace, lifecycle_mutations_enabled: false },
    { ...workspace, repository_id: "c".repeat(64) },
    { ...workspace, commit_sha: "c".repeat(40) },
  ])("never polls or replays a disabled or mismatched workspace", async (system) => {
    rememberLifecycleRequest(startDraft());
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const onThreadActivated = vi.fn();
    renderRecovery(onThreadActivated, system);
    await act(async () => {});
    expect(fetchMock).not.toHaveBeenCalled();
    expect(onThreadActivated).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Replay exact request" })).not.toBeInTheDocument();
    expect(readLifecycleRequests()).toHaveLength(1);
  });
});

function startDraft() {
  const command: DashboardStartCommand = {
    schema_version: 1,
    request_id: "request-start",
    goal: "Trace the proposal verifier.",
    expected_repository_id: "a".repeat(64),
    expected_commit_sha: "b".repeat(40),
    focus_path: "src/repocompass",
    human_review_policy: "before_finalize",
    memory_retrieval_enabled: true,
    max_attempt_cost_usd: null,
  };
  return { kind: "start" as const, thread_id: null, command };
}

function operationResult() {
  return {
    schema_version: 1,
    request_id: "request-start",
    thread_id: "thread-1",
    execution_state: "completed",
    run_id: "run-1",
    idempotent_replay: true,
    recovered_from_checkpoint: false,
  };
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}
