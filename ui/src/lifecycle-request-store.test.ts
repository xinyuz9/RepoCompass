import { afterEach, describe, expect, it } from "vitest";

import {
  forgetLifecycleRequest,
  readLifecycleRequests,
  rememberLifecycleRequest,
} from "./lifecycle-request-store";
import type { DashboardStartCommand } from "./types";

const STORAGE_KEY = "repocompass.pending-lifecycle.v1";

afterEach(() => window.sessionStorage.clear());

describe("lifecycle request recovery store", () => {
  it("deduplicates by idempotency key and removes only an explicitly named request", () => {
    rememberLifecycleRequest(startDraft("request-1", "Trace the first path."));
    rememberLifecycleRequest(startDraft("request-1", "Trace the revised path."));
    rememberLifecycleRequest(startDraft("request-2", "Trace the second path."));

    expect(readLifecycleRequests().map((item) => item.command.request_id)).toEqual([
      "request-1",
      "request-2",
    ]);
    expect(readLifecycleRequests()[0]?.command).toMatchObject({
      goal: "Trace the revised path.",
    });

    forgetLifecycleRequest("request-1");
    expect(readLifecycleRequests().map((item) => item.command.request_id)).toEqual([
      "request-2",
    ]);
  });

  it("keeps only the eight most recent bounded recovery envelopes", () => {
    for (let index = 0; index < 10; index += 1) {
      rememberLifecycleRequest(
        startDraft(`request-${index}`, `Trace bounded path ${index}.`),
      );
    }

    const records = readLifecycleRequests();
    expect(records).toHaveLength(8);
    expect(records[0]?.command.request_id).toBe("request-2");
    expect(records[7]?.command.request_id).toBe("request-9");
  });

  it("rejects malformed or oversized browser-controlled command envelopes", () => {
    window.sessionStorage.setItem(STORAGE_KEY, "not json");
    expect(readLifecycleRequests()).toEqual([]);
  });

  it.each(["a", "调", "🧭"])("retains full contract length using Unicode code points: %s", (character) => {
    const draft = startDraft("long-goal", character.repeat(16_384));
    draft.command.focus_path = "a".repeat(2_048);
    rememberLifecycleRequest(draft);
    expect(readLifecycleRequests()[0]?.command).toEqual(draft.command);
  });

  it("evicts oldest envelopes by UTF-8 bytes while retaining the newest request", () => {
    for (let index = 0; index < 8; index += 1) {
      rememberLifecycleRequest(startDraft(`large-${index}`, "调".repeat(16_384)));
    }
    const records = readLifecycleRequests();
    expect(records.length).toBeGreaterThan(0);
    expect(records.length).toBeLessThan(8);
    expect(records.at(-1)?.command.request_id).toBe("large-7");
    expect(new TextEncoder().encode(window.sessionStorage.getItem(STORAGE_KEY)!).length)
      .toBeLessThanOrEqual(131_072);
  });

  it("rejects invalid browser-controlled command envelopes", () => {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify([{
      schema_version: 1,
      recorded_at_ms: 1,
      kind: "start",
      thread_id: null,
      command: { schema_version: 1, request_id: "request-malformed" },
    }]));
    expect(readLifecycleRequests()).toEqual([]);

    window.sessionStorage.setItem(STORAGE_KEY, "x".repeat(131_073));
    expect(readLifecycleRequests()).toEqual([]);
    expect(window.sessionStorage.getItem(STORAGE_KEY)).toBeNull();
  });
});

function startDraft(requestId: string, goal: string) {
  const command: DashboardStartCommand = {
    schema_version: 1,
    request_id: requestId,
    expected_repository_id: "a".repeat(64),
    expected_commit_sha: "b".repeat(40),
    goal,
    focus_path: ".",
    human_review_policy: "before_finalize",
    memory_retrieval_enabled: true,
    max_attempt_cost_usd: null,
  };
  return { kind: "start" as const, thread_id: null, command };
}
