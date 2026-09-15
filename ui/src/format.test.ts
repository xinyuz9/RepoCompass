import { describe, expect, it } from "vitest";
import { investigationStateLabel } from "./format";
import type { ExecutionState, ThreadInvestigationView } from "./types";

describe("investigation outcome labels", () => {
  it.each<[ExecutionState | null, ThreadInvestigationView["termination_status"], string]>([
    ["completed", "cancelled", "cancelled"],
    ["completed", "exhausted", "exhausted"],
    ["completed", "accepted", "completed"],
    ["resumable", "cancelled", "resumable"],
    [null, null, "unavailable"],
  ])("keeps %s/%s distinct from success", (execution_state, termination_status, label) => {
    expect(investigationStateLabel({ execution_state, termination_status })).toBe(label);
  });
});
