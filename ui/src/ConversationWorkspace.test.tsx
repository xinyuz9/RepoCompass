import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { BrowserBoundary, ConversationWorkspace, InvestigationResult } from "./ConversationWorkspace";
import type {
  DashboardSystemInfo,
  DashboardThreadDetail,
} from "./types";

const repositoryId = "a".repeat(64);
const commitSha = "b".repeat(40);
const system: DashboardSystemInfo = {
  schema_version: 1,
  product: "RepoCompass",
  product_version: "0.1.0",
  repository_id: repositoryId,
  commit_sha: commitSha,
  workspace_name: "RepoCompass",
  workspace_path: ".",
  demo_mode: false,
  memory_available: true,
  pricing_configured: true,
  lifecycle_mutations_enabled: true,
  memory_mutations_enabled: true,
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ConversationWorkspace lifecycle commands", () => {
  it("keeps settings optional and preserves edits when the panel is collapsed", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    renderWorkspace({ detail: null });
    const focus = screen.getByLabelText(/Focus path/);
    expect(focus).not.toBeVisible();
    expect(screen.getByText(/Review on · Memory on · No task cost cap/)).toBeVisible();
    fireEvent.click(screen.getByText("Task settings"));
    expect(focus).toBeVisible();
    fireEvent.change(focus, { target: { value: "src" } });
    fireEvent.click(screen.getByLabelText("Recall memory"));
    fireEvent.click(screen.getByLabelText("Review before finalize"));
    fireEvent.change(screen.getByLabelText(/Attempt cap/), { target: { value: "0.25" } });
    fireEvent.click(screen.getByText("Task settings"));
    expect(focus).not.toBeVisible();
    expect(screen.getByText(/Review off · Memory off · USD 0.25 cap/)).toBeVisible();
    fireEvent.click(screen.getByText("Task settings"));
    expect(focus).toHaveValue("src");
    expect(screen.getByLabelText("Recall memory")).not.toBeChecked();
    expect(screen.getByLabelText("Review before finalize")).not.toBeChecked();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("starts with safe existing defaults without requiring settings to be opened", async () => {
    const onThreadActivated = vi.fn();
    const fetchMock = vi.fn((_input: string | URL | Request, init?: RequestInit) => {
      const command = JSON.parse(String(init?.body)) as { request_id: string };
      return Promise.resolve(operationResponse(command.request_id, "thread-new", "start-run"));
    });
    vi.stubGlobal("fetch", fetchMock);
    renderWorkspace({ detail: null, onThreadActivated });
    fireEvent.change(screen.getByLabelText("What should RepoCompass investigate?"), { target: { value: "Trace the entry point." } });
    fireEvent.click(screen.getByRole("button", { name: "Start investigation" }));
    await waitFor(() => expect(onThreadActivated).toHaveBeenCalledWith("thread-new"));
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toMatchObject({
      focus_path: ".", human_review_policy: "before_finalize",
      memory_retrieval_enabled: true, max_attempt_cost_usd: null,
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("reports browser default-off without an editable URL or permission switch", () => {
    render(<BrowserBoundary system={system} />);
    expect(screen.getByText("Agent browser disabled")).toBeVisible();
    fireEvent.click(screen.getByText("Agent browser disabled"));
    expect(screen.getByText(/Chat messages cannot grant access/)).toBeVisible();
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  });

  it("shows only the operator-approved static page catalog and its limits", () => {
    render(<BrowserBoundary system={{ ...system, static_browser_page_ids: ["docs", "guide"] }} />);
    fireEvent.click(screen.getByText("Static browser · 2 operator-approved pages"));
    expect(screen.getByRole("list", { name: "Operator-approved page IDs" })).toBeVisible();
    expect(screen.getByText("docs")).toBeVisible();
    expect(screen.getByText(/No site scripts, login state/)).toBeVisible();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
  it("distinguishes exhausted completion from an accepted result and retains open questions", () => {
    const detail = threadDetail({ executionState: "completed", attemptCommitSha: commitSha });
    detail.thread.termination_status = "exhausted";
    detail.thread.final_result = {
      summary: "The request handler was located, but the caller is unresolved.",
      relevant_files: ["src/handler.py"], relevant_symbols: [],
      evidence: ["handler validates the request."],
      unresolved_questions: ["Which caller starts the request?"],
    };
    render(<InvestigationResult thread={detail.thread} onOpenLens={vi.fn()} />);
    expect(screen.getByText("Best effort")).toBeVisible();
    expect(screen.queryByText("Accepted")).not.toBeInTheDocument();
    expect(screen.getByText("Which caller starts the request?")).toBeVisible();
    expect(screen.getByText("src/handler.py")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Explore source evidence" })).not.toBeInTheDocument();
  });

  it("keeps a candidate explicitly provisional and exposes its remaining evidence on demand", () => {
    const detail = threadDetail({ executionState: "resumable", attemptCommitSha: commitSha });
    detail.thread.candidate_result = {
      summary: "A provisional finding.", relations: [], unresolved_questions: [],
      evidence: ["First", "Second", "Third", "Fourth"].map((claim) => ({
        claim, path: "src/handler.py", symbol: null, line_start: 1, line_end: 2,
      })),
    };
    render(<InvestigationResult thread={detail.thread} onOpenLens={vi.fn()} />);
    expect(screen.getByText("Working draft")).toBeVisible();
    expect(screen.getByText("This candidate has not been finalized.")).toBeVisible();
    expect(screen.getByText("Fourth")).not.toBeVisible();
    fireEvent.click(screen.getByText("Show 1 more evidence items"));
    expect(screen.getByText("Fourth")).toBeVisible();
  });

  it("offers trace inspection when a checkpoint has no result", () => {
    const onOpenLens = vi.fn();
    const detail = threadDetail({ executionState: "resumable", attemptCommitSha: commitSha });
    render(<InvestigationResult thread={detail.thread} onOpenLens={onOpenLens} />);
    fireEvent.click(screen.getByRole("button", { name: "Inspect last attempt" }));
    expect(onOpenLens).toHaveBeenCalledWith("trace");
    expect(screen.queryByText("Accepted")).not.toBeInTheDocument();
  });

  it("starts a goal with an idempotency key and never accepts a browser repository root", async () => {
    const onThreadActivated = vi.fn();
    const fetchMock = vi.fn((_input: string | URL | Request, init?: RequestInit) => {
      const command = JSON.parse(String(init?.body)) as { request_id: string };
      return Promise.resolve(operationResponse(command.request_id, "thread-new", "start-run"));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWorkspace({ detail: null, onThreadActivated });
    fireEvent.click(screen.getByText("Task settings"));
    fireEvent.change(screen.getByLabelText("What should RepoCompass investigate?"), {
      target: { value: " Trace the proposal verifier. " },
    });
    fireEvent.change(screen.getByLabelText(/Focus path/), {
      target: { value: " src/repocompass " },
    });
    fireEvent.change(screen.getByLabelText(/Attempt cap/), {
      target: { value: "1.25" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start investigation" }));

    await waitFor(() => expect(onThreadActivated).toHaveBeenCalledWith("thread-new"));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [input, init] = fetchMock.mock.calls[0] ?? [];
    const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    expect(String(input)).toBe("/api/v1/investigations");
    expect(init?.method).toBe("POST");
    expect((init?.headers as Record<string, string>)["Idempotency-Key"]).toBe(body.request_id);
    expect(body).toMatchObject({
      schema_version: 1,
      goal: "Trace the proposal verifier.",
      expected_repository_id: repositoryId,
      expected_commit_sha: commitSha,
      focus_path: "src/repocompass",
      human_review_policy: "before_finalize",
      memory_retrieval_enabled: true,
      max_attempt_cost_usd: "1.25",
    });
    expect(body).not.toHaveProperty("repo_path");
  });
  it("locks a failed command for exact retry until the user explicitly edits it", async () => {
    const fetchMock = vi.fn(() => Promise.resolve(new Response(JSON.stringify({
      schema_version: 1,
      type: "urn:repocompass:problem:lifecycle-failed",
      title: "Lifecycle request failed",
      status: 503,
      detail: "The provider was temporarily unavailable.",
      instance: "/api/v1/investigations",
    }), {
      status: 503,
      headers: { "content-type": "application/problem+json" },
    })));
    vi.stubGlobal("fetch", fetchMock);

    renderWorkspace({ detail: null });
    const goal = screen.getByLabelText("What should RepoCompass investigate?");
    fireEvent.change(goal, { target: { value: "Trace the retry path." } });
    fireEvent.click(screen.getByRole("button", { name: "Start investigation" }));

    expect(await screen.findByRole("button", { name: "Retry same request" })).toBeVisible();
    expect(goal).toBeDisabled();
    fireEvent.click(screen.getByText("Task settings"));
    expect(screen.getByLabelText(/Focus path/)).toBeDisabled();
    expect(screen.getByLabelText("Recall memory")).toBeDisabled();
    expect(screen.getByLabelText(/Attempt cap/)).toBeDisabled();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "Edit request" }));
    expect(goal).toBeEnabled();
    fireEvent.change(goal, { target: { value: "Trace a different path." } });
    expect(screen.getByRole("button", { name: "Start investigation" })).toBeEnabled();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not offer another resume while a resumable checkpoint has an open attempt", () => {
    const detail = threadDetail({ executionState: "resumable", attemptCommitSha: commitSha });
    detail.attempts[0]!.attempt.status = "running";
    detail.attempts[0]!.attempt.finished_at_ms = null;
    renderWorkspace({ detail });
    expect(screen.queryByRole("button", { name: "Resume" })).not.toBeInTheDocument();
    expect(screen.getByText(/Attempt not finalized/)).toBeVisible();
    expect(screen.getByText("attempt open")).toBeVisible();
  });

  it("blocks resume when the persisted attempt targets a stale repository snapshot", () => {
    const fetchMock = vi.fn();
    const onNewGoal = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    renderWorkspace({
      detail: threadDetail({
        executionState: "resumable",
        attemptCommitSha: "c".repeat(40),
      }),
      onNewGoal,
    });

    expect(screen.getByRole("alert")).toHaveTextContent("Repository snapshot changed");
    expect(screen.getByRole("button", { name: "Resume" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Start fresh" }));
    expect(onNewGoal).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("continues a completed thread against the visible focus path and snapshot", async () => {
    const onThreadActivated = vi.fn();
    const fetchMock = vi.fn((_input: string | URL | Request, init?: RequestInit) => {
      const command = JSON.parse(String(init?.body)) as { request_id: string };
      return Promise.resolve(operationResponse(command.request_id, "thread-1", "continue-run"));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWorkspace({
      detail: threadDetail({ executionState: "completed", attemptCommitSha: commitSha }),
      onThreadActivated,
    });
    fireEvent.change(screen.getByPlaceholderText("Ask a follow-up about this repository…"), {
      target: { value: " Show the exact verification call path. " },
    });
    expect(screen.getByLabelText("Focus")).not.toBeVisible();
    fireEvent.click(screen.getByText(/Follow-up scope/));
    fireEvent.change(screen.getByLabelText("Focus"), {
      target: { value: "src/repocompass/agent" },
    });
    fireEvent.click(screen.getByText(/Follow-up scope/));
    expect(screen.getByLabelText("Focus")).not.toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Send follow-up" }));

    await waitFor(() => expect(onThreadActivated).toHaveBeenCalledWith("thread-1"));
    const [input, init] = fetchMock.mock.calls[0] ?? [];
    const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    expect(String(input)).toBe("/api/v1/threads/thread-1/turns");
    expect((init?.headers as Record<string, string>)["Idempotency-Key"]).toBe(body.request_id);
    expect(body).toMatchObject({
      schema_version: 1,
      prompt: "Show the exact verification call path.",
      focus_path: "src/repocompass/agent",
      expected_repository_id: repositoryId,
      expected_commit_sha: commitSha,
    });
    expect(body).not.toHaveProperty("repo_path");
  });
});

function renderWorkspace({
  detail,
  onThreadActivated = vi.fn(),
  onNewGoal = vi.fn(),
}: {
  detail: DashboardThreadDetail | null;
  onThreadActivated?: (threadId: string) => void;
  onNewGoal?: () => void;
}) {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={client}>
      <ConversationWorkspace
        system={system}
        detail={detail}
        lifecycleMutationsEnabled
        onThreadActivated={onThreadActivated}
        onOpenLens={vi.fn()}
        onNewGoal={onNewGoal}
      />
    </QueryClientProvider>,
  );
}

function threadDetail({
  executionState,
  attemptCommitSha,
}: {
  executionState: "completed" | "resumable";
  attemptCommitSha: string;
}): DashboardThreadDetail {
  return {
    schema_version: 1,
    thread: {
      schema_version: 1,
      thread_id: "thread-1",
      execution_state: executionState,
      turn_number: 1,
      objective: "Trace the proposal verifier.",
      plan: { steps: [] },
      plan_revision_count: 0,
      max_plan_revisions: 2,
      next_nodes: executionState === "resumable" ? ["build_candidate"] : [],
      termination_status: executionState === "completed" ? "accepted" : null,
      step_stop_reason: null,
      candidate_result: null,
      evaluation_report: null,
      structure: null,
      final_result: null,
      pending_review: null,
      interrupt_id: null,
    },
    attempts: [{
      schema_version: 1,
      attempt: {
        schema_version: 1,
        run_id: "run-1",
        thread_id: "thread-1",
        turn_number: 1,
        execution_mode: "start",
        repository_id: repositoryId,
        commit_sha: attemptCommitSha,
        objective_digest: "d".repeat(64),
        status: executionState === "completed" ? "completed" : "interrupted",
        started_at_ms: 1_000,
        finished_at_ms: 2_000,
      },
      trace_availability: "available",
      cost_availability: "available",
    }],
    plan_timeline: [],
    evaluator_timeline: [],
  };
}

function operationResponse(requestId: string, threadId: string, runId: string): Response {
  return new Response(JSON.stringify({
    schema_version: 1,
    request_id: requestId,
    thread_id: threadId,
    execution_state: "completed",
    run_id: runId,
    idempotent_replay: false,
    recovered_from_checkpoint: false,
  }), { status: 200, headers: { "content-type": "application/json" } });
}
