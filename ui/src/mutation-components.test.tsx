import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { HumanReviewPanel } from "./HumanReviewPanel";
import { MemoryDeleteControl } from "./MemoryDeleteControl";
import type { DashboardThreadDetail } from "./types";

const detail = {
  thread: {
    thread_id: "thread-1",
    interrupt_id: "interrupt-1",
    pending_review: {
      turn_number: 1,
      candidate_result: {
        evidence: [{
          path: "src/auth.py",
          line_start: 10,
          line_end: 14,
          symbol: "authenticate",
          claim: "authenticate validates credentials.",
        }],
        relations: [],
        summary: "Authentication is owned by src/auth.py.",
        unresolved_questions: [],
      },
      evaluation_report: {
        passed: true,
        deterministic_issues: [],
        semantic_assessment: {
          active_step_completed: true,
          answers_objective: true,
          evidence_supports_summary: true,
          has_critical_omissions: false,
          missing_evidence: [],
          recommended_focus: [],
          summary: "Ready for review.",
        },
      },
      allowed_actions: ["approve", "revise", "cancel"],
    },
  },
} as unknown as DashboardThreadDetail;

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("explicit mutation controls", () => {
  it("never auto-retries and manually replays the exact Human-review request", async () => {
    let attempt = 0;
    const fetchMock = vi.fn((_input: string | URL | Request, init?: RequestInit) => {
      attempt += 1;
      const command = JSON.parse(String(init?.body)) as { request_id: string };
      if (attempt === 1) {
        return Promise.resolve(new Response(JSON.stringify({
          schema_version: 1,
          type: "urn:repocompass:problem:mutation-unavailable",
          title: "Mutation unavailable",
          status: 503,
          detail: "The local mutation could not be completed safely.",
          instance: "/api/v1/threads/thread-1/responses",
        }), { status: 503, headers: { "content-type": "application/problem+json" } }));
      }
      return Promise.resolve(new Response(JSON.stringify({
        schema_version: 1,
        request_id: command.request_id,
        thread_id: "thread-1",
        execution_state: "completed",
        run_id: "respond-run",
        idempotent_replay: false,
        recovered_from_checkpoint: false,
      }), { status: 200, headers: { "content-type": "application/json" } }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithClient(<HumanReviewPanel detail={detail} enabled />);
    fireEvent.click(screen.getByRole("button", { name: "Submit approve" }));

    expect(await screen.findByText("Response not confirmed")).toBeVisible();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const firstInit = fetchMock.mock.calls[0]?.[1];
    expect(firstInit?.method).toBe("POST");
    expect((firstInit?.headers as Record<string, string>)["Idempotency-Key"]).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Retry same request" }));
    expect(await screen.findByText("Response accepted")).toBeVisible();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const secondInit = fetchMock.mock.calls[1]?.[1];
    expect(secondInit?.body).toBe(firstInit?.body);
    expect((secondInit?.headers as Record<string, string>)["Idempotency-Key"]).toBe(
      (firstInit?.headers as Record<string, string>)["Idempotency-Key"],
    );
  });

  it("requires focused feedback for revise and stays read-only when disabled", () => {
    const view = renderWithClient(<HumanReviewPanel detail={detail} enabled />);

    fireEvent.click(screen.getByRole("button", { name: "Revise" }));
    const submit = screen.getByRole("button", { name: "Submit revise" });
    expect(submit).toBeDisabled();
    fireEvent.change(screen.getByPlaceholderText(/Name the missing evidence/), {
      target: { value: "Inspect the router registration call site." },
    });
    expect(submit).toBeEnabled();

    view.unmount();
    renderWithClient(<HumanReviewPanel detail={detail} enabled={false} />);
    expect(screen.getByText("Review controls are read-only")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Submit approve" })).not.toBeInTheDocument();
  });

  it("requires an exact suffix and binds memory deletion to a deterministic key", async () => {
    const memoryId = `${"d".repeat(56)}deadbeef`;
    const fetchMock = vi.fn((_input: string | URL | Request, init?: RequestInit) => {
      const command = JSON.parse(String(init?.body)) as { request_id: string };
      return Promise.resolve(new Response(JSON.stringify({
        schema_version: 1,
        request_id: command.request_id,
        memory_id: memoryId,
        tombstone: {
          schema_version: 1,
          tombstone_id: "e".repeat(64),
          memory_id: memoryId,
          repository_key: "a".repeat(64),
          memory_scope: "repository",
          deletion_event_id: "f".repeat(64),
          reason: "user_request",
          deleted_at_ms: 2_000,
        },
        idempotent_replay: false,
      }), { status: 200, headers: { "content-type": "application/json" } }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithClient(<MemoryDeleteControl memoryId={memoryId} enabled />);
    const input = screen.getByLabelText("Type deadbeef to confirm memory deletion");
    const remove = screen.getByRole("button", { name: "Delete memory" });
    expect(remove).toBeDisabled();
    fireEvent.change(input, { target: { value: "deadbeee" } });
    expect(remove).toBeDisabled();
    fireEvent.change(input, { target: { value: "DEADBEEF" } });
    expect(remove).toBeEnabled();
    fireEvent.click(remove);

    expect(await screen.findByText("Memory deleted")).toBeVisible();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const init = fetchMock.mock.calls[0]?.[1];
    const requestId = `memory-delete:${memoryId}`;
    expect(init?.method).toBe("DELETE");
    expect((init?.headers as Record<string, string>)["Idempotency-Key"]).toBe(requestId);
    expect(JSON.parse(String(init?.body))).toEqual({
      schema_version: 1,
      request_id: requestId,
      confirm_memory_id: memoryId,
    });
  });
});

function renderWithClient(element: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={client}>
      {element}
    </QueryClientProvider>,
  );
}
