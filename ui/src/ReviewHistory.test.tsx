import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { CurrentReviewHistory, ReviewHistory } from "./ReviewHistory";
import type { HumanReviewHistory, PublicConversationPage } from "./types";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const history: HumanReviewHistory = {
  omitted_count: 0, prior_history_unavailable: false,
  records: ["revise", "approve", "cancel", null].map((action, index) => ({
    review_number: index + 1, request_digest: "a".repeat(64),
    candidate_summary: "Candidate " + index, evaluation_passed: true,
    deterministic_issue_count: 0, allowed_actions: ["approve", "revise", "cancel"],
    action: action as "revise" | "approve" | "cancel" | null,
    feedback: index === 0 ? "<script>alert('not executable')</script>\nInspect boundary" : null,
    text_truncated: false,
  })),
};

it("renders recorded decisions and feedback as text with no mutation controls", () => {
  const { container } = render(<ReviewHistory history={history} />);
  for (const label of ["Revision requested", "Approved", "Cancelled", "No decision recorded"]) {
    expect(screen.getByText(label)).toBeVisible();
  }
  fireEvent.click(screen.getByText("Revision requested"));
  const feedback = screen.getByText(/<script>alert/);
  expect(feedback).toBeVisible();
  expect(feedback.textContent).toBe(history.records[0]?.feedback);
  expect(container.querySelector("script")).toBeNull();
  expect(screen.queryByRole("button")).not.toBeInTheDocument();
});

it("distinguishes unavailable legacy history, bounded loss and no recorded reviews", () => {
  const { rerender } = render(<ReviewHistory history={undefined} />);
  expect(screen.getByText(/Earlier review history is unavailable/)).toBeVisible();
  rerender(<ReviewHistory history={{ ...history, omitted_count: 5, prior_history_unavailable: true }} />);
  expect(screen.getByText(/5 older reviews omitted/)).toBeVisible();
  rerender(<ReviewHistory history={{ records: [], omitted_count: 0, prior_history_unavailable: false }} />);
  expect(screen.getByText(/Automatic completion is not human approval/)).toBeVisible();
});

function page(threadId = "thread-1"): PublicConversationPage {
  return { schema_version: 1, thread_id: threadId, latest_turn_number: 1,
    provenance: "checkpoint_public_turns", next_before_turn: null, turns: [{
      turn_number: 1, objective: "Goal", summary: null, termination_status: null,
      text_truncated: false, review_history: history,
    }] };
}

it("reads only on demand and refetches when the checkpoint identity changes", async () => {
  const fetcher = vi.fn().mockImplementation(() => Promise.resolve(Response.json(page())));
  vi.stubGlobal("fetch", fetcher);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const show = (checkpointKey: string) => <QueryClientProvider client={client}>
    <CurrentReviewHistory threadId="thread-1" turnNumber={1} repositoryId="repo" commitSha={null} checkpointKey={checkpointKey} />
  </QueryClientProvider>;
  const { rerender } = render(show("run-1:interrupted"));
  expect(fetcher).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Review history" }));
  expect(await screen.findByText("Revision requested")).toBeVisible();
  expect(fetcher).toHaveBeenCalledTimes(1);
  expect(fetcher.mock.calls[0]?.[0]).toBe("/api/v1/threads/thread-1/conversation?before_turn=2&limit=1");
  rerender(show("run-2:completed"));
  expect(await screen.findByText("Revision requested")).toBeVisible();
  expect(fetcher).toHaveBeenCalledTimes(2);
  expect(fetcher.mock.calls.every((call) => call[1].method === "GET")).toBe(true);
});

it("shows read errors without retry storms or disabling the rest of the conversation", async () => {
  const fetcher = vi.fn().mockResolvedValue(new Response("", { status: 503 }));
  vi.stubGlobal("fetch", fetcher);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><CurrentReviewHistory threadId="thread-1" turnNumber={1} repositoryId="repo" commitSha={null} checkpointKey="run" /></QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: "Review history" }));
  const alert = await screen.findByRole("alert");
  expect(within(alert).getByText("Review history could not be loaded.")).toBeVisible();
  expect(fetcher).toHaveBeenCalledTimes(1);
});
