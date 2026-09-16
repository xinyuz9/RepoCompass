import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { DriftPanel } from "./DriftPanel";
import type { EvidenceDriftReport } from "./drift-types";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

function fixture(): EvidenceDriftReport {
  return { schema_version: 1, repository_id: "a".repeat(64), baseline_commit: "b".repeat(40), current_commit: "c".repeat(40),
    worktree_clean: true, snapshot_stable: true, changes_outside_scope: false, scope_truncated: false,
    sources: [{ path: "pkg/handler.py", baseline_digests: ["d".repeat(64)], current_digest: "e".repeat(64), status: "changed", reason: null, bytes_read: 40 }],
    subjects: Array.from({ length: 45 }, (_, index) => ({ id: "subject-" + index, kind: "memory" as const, paths: ["pkg/handler.py"], status: "evidence_invalidated", reasons: ["source_digest_changed"] })),
    recheck_paths: ["pkg/handler.py"], full_recheck_paths: ["pkg/handler.py"], bytes_read: 40, read_budget_used_bytes: 40, file_reads: 1, provider_calls: 0, disclaimer: "provenance_only_not_semantic_revalidation" };
}

function mount() { return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><DriftPanel threadId="thread-drift" /></QueryClientProvider>); }

it("only compares on click, bounds subjects, and sends GET without replaying the agent", async () => {
  const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(fixture()), { status: 200 }));
  vi.stubGlobal("fetch", fetch);
  mount();
  expect(fetch).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Compare saved evidence with HEAD" }));
  expect(await screen.findByText(/unchanged bytes do not prove unchanged behavior/)).toBeVisible();
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(fetch.mock.calls[0]![0]).toBe("/api/v1/threads/thread-drift/drift");
  expect(fetch.mock.calls[0]![1].method).toBe("GET");
  const table = screen.getByRole("table", { name: "Evidence invalidation and revalidation reasons", hidden: true });
  expect(within(table).getAllByRole("row", { hidden: true })).toHaveLength(21);
  fireEvent.click(screen.getByRole("button", { name: "Next subjects", hidden: true }));
  expect(within(table).getAllByRole("row", { hidden: true })).toHaveLength(21);
  expect(fetch).toHaveBeenCalledTimes(1);
});

it("stops after a failed read and labels saved data as unrefreshed", async () => {
  const fetch = vi.fn().mockResolvedValue(new Response("failed", { status: 503 }));
  vi.stubGlobal("fetch", fetch);
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Compare saved evidence with HEAD" }));
  await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("has not been refreshed"));
  expect(fetch).toHaveBeenCalledTimes(1);
});
