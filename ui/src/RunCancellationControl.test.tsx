import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { RunCancellationControl } from "./RunCancellationControl";
import { cancellationRefreshInterval } from "./active-refresh";
import type { CancellationRecord, RawRunAttempt } from "./types";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
const attempt: RawRunAttempt = {
  schema_version: 1, run_id: "run-1", thread_id: "thread-1", turn_number: 1,
  execution_mode: "start", repository_id: "a".repeat(64), commit_sha: null,
  objective_digest: "b".repeat(64), status: "running", started_at_ms: 1, finished_at_ms: null,
};
function record(state: CancellationRecord["state"]): CancellationRecord {
  return { schema_version: 1, run_id: attempt.run_id, thread_id: attempt.thread_id,
    repository_id: attempt.repository_id, commit_sha: null, state,
    request_id: state === "running" ? null : "cancel-1",
    requested_at_ms: state === "running" ? null : 10,
    finished_at_ms: state === "cancelled" || state === "finished" ? 20 : null };
}
function show(enabled = true) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><RunCancellationControl attempt={attempt} enabled={enabled} /></QueryClientProvider>);
}

it("is inert in read-only mode", () => {
  const fetcher = vi.fn(); vi.stubGlobal("fetch", fetcher);
  show(false);
  expect(fetcher).not.toHaveBeenCalled();
  expect(screen.queryByRole("button")).not.toBeInTheDocument();
});

it("separates a durable request from confirmation and sends only one POST", async () => {
  const fetcher = vi.fn<typeof globalThis.fetch>()
    .mockResolvedValueOnce(Response.json(record("running")))
    .mockResolvedValueOnce(Response.json(record("cancel_requested")))
    .mockResolvedValueOnce(Response.json(record("cancelled")));
  vi.stubGlobal("fetch", fetcher);
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Request cancellation" }));
  expect(await screen.findByText("Cancellation requested · awaiting confirmation")).toBeVisible();
  expect(screen.queryByText("Cancellation confirmed")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Request cancellation" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Check cancellation status" }));
  expect(await screen.findByText("Cancellation confirmed")).toBeVisible();
  const posts = fetcher.mock.calls.filter((call) => call[1]?.method === "POST");
  expect(posts).toHaveLength(1);
  const body = JSON.parse(String(posts[0]?.[1]?.body));
  expect(body).toMatchObject({ run_id: "run-1", thread_id: "thread-1", repository_id: attempt.repository_id });
  expect(posts[0]?.[1]?.headers).toMatchObject({ "Idempotency-Key": body.request_id });
});

it("does not automatically retry POST and reuses the exact identity on manual retry", async () => {
  const posts: string[] = [];
  const fetcher = vi.fn<typeof globalThis.fetch>(async (_, init) => {
    if (init?.method === "POST") {
      posts.push(String(init.body));
      return new Response("", { status: 503 });
    }
    return Response.json(record("running"));
  });
  vi.stubGlobal("fetch", fetcher);
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Request cancellation" }));
  expect(await screen.findByRole("alert")).toBeVisible();
  expect(posts).toHaveLength(1);
  fireEvent.click(screen.getByRole("button", { name: "Retry same cancellation" }));
  await waitFor(() => expect(posts).toHaveLength(2));
  expect(posts[0]).toBe(posts[1]);
});

it("bounds status polling and never interprets missing status as cancellation", async () => {
  expect(cancellationRefreshInterval("cancel_requested", false, 119)).toBe(1500);
  expect(cancellationRefreshInterval("cancel_requested", false, 120)).toBe(false);
  expect(cancellationRefreshInterval("cancel_requested", true, 1)).toBe(false);
  for (const state of ["running", "finished", "cancelled", undefined] as const) {
    expect(cancellationRefreshInterval(state, false, 1)).toBe(false);
  }
  const fetcher = vi.fn().mockResolvedValue(new Response("", { status: 404 }));
  vi.stubGlobal("fetch", fetcher);
  show();
  expect(await screen.findByText("Cancellation status unavailable")).toBeVisible();
  expect(screen.queryByText("Cancellation confirmed")).not.toBeInTheDocument();
  expect(fetcher).toHaveBeenCalledTimes(1);
});
