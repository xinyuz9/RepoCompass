import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ConversationHistory } from "./ConversationHistory";
import type { PublicConversationPage } from "./types";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

function show(turnNumber = 3) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>
    <ConversationHistory threadId="thread-1" turnNumber={turnNumber} repositoryId={"a".repeat(64)} commitSha={null} />
  </QueryClientProvider>);
}
function page(turn: number, next: number | null): PublicConversationPage {
  return { schema_version: 1, thread_id: "thread-1", latest_turn_number: 3,
    provenance: "checkpoint_public_turns", next_before_turn: next, turns: [{
      turn_number: turn, objective: "Goal " + turn, summary: "Answer " + turn,
      termination_status: turn === 2 ? "exhausted" : "accepted", text_truncated: false,
    }] };
}

it("loads only on demand and prepends earlier pages without writes or duplicate calls", async () => {
  const fetcher = vi.fn()
    .mockResolvedValueOnce(Response.json(page(2, 2)))
    .mockResolvedValueOnce(Response.json(page(1, null)));
  vi.stubGlobal("fetch", fetcher);
  show();
  expect(fetcher).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Earlier turns · 2" }));
  expect(await screen.findByText("Answer 2")).toBeVisible();
  expect(screen.getByText("Stopped without acceptance.")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Load older turns" }));
  expect(await screen.findByText("Answer 1")).toBeVisible();
  const turns = screen.getAllByRole("region", { name: /^Turn/ });
  expect(turns.map((item) => item.getAttribute("aria-label"))).toEqual(["Turn 1", "Turn 2"]);
  expect(fetcher.mock.calls.map((call) => call[0])).toEqual([
    "/api/v1/threads/thread-1/conversation?before_turn=3&limit=10",
    "/api/v1/threads/thread-1/conversation?before_turn=2&limit=10",
  ]);
  expect(fetcher.mock.calls.every((call) => call[1].method === "GET")).toBe(true);
  fireEvent.click(screen.getByRole("button", { name: "Hide earlier turns" }));
  fireEvent.click(screen.getByRole("button", { name: "Earlier turns · 2" }));
  await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2));
});

it("keeps errors visible and never retries automatically", async () => {
  const fetcher = vi.fn().mockResolvedValue(new Response("", { status: 503 }));
  vi.stubGlobal("fetch", fetcher);
  show();
  fireEvent.click(screen.getByRole("button", { name: "Earlier turns · 2" }));
  expect(await screen.findByRole("alert")).toBeVisible();
  expect(fetcher).toHaveBeenCalledTimes(1);
});

it("does not request history for the first turn", () => {
  const fetcher = vi.fn();
  vi.stubGlobal("fetch", fetcher);
  show(1);
  expect(screen.queryByRole("button")).not.toBeInTheDocument();
  expect(fetcher).not.toHaveBeenCalled();
});
