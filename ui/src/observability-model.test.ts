import { describe, expect, it } from "vitest";

import { formatTimestamp } from "./format";
import { breakdownWidth, contextSectionTotals, latestPlanFrame } from "./observability-model";
import type { ContextPack, PlanReplayFrame, UsageCostBreakdown } from "./types";

describe("observability projections", () => {
  it("reconciles context section totals from selected items only", () => {
    const pack = {
      items: [
        { section: "evidence", estimated_tokens: 120 },
        { section: "evidence", estimated_tokens: 30 },
        { section: "memory", estimated_tokens: 50 },
      ],
    } as ContextPack;

    expect(contextSectionTotals(pack)).toEqual([
      { section: "evidence", tokens: 150, itemCount: 2 },
      { section: "memory", tokens: 50, itemCount: 1 },
    ]);
  });

  it("selects the latest plan frame deterministically across turns", () => {
    const frame = (runId: string, turn: number, sequence: number) => ({
      run_id: runId,
      turn_number: turn,
      sequence,
    }) as PlanReplayFrame;
    expect(latestPlanFrame([frame("b", 1, 7), frame("a", 2, 1), frame("c", 2, 3)])?.run_id).toBe("c");
  });

  it("normalizes cost bars without inventing unavailable amounts", () => {
    const item = (value: string | null) => ({ cost: { estimated_cost_usd: value } }) as UsageCostBreakdown;
    const low = item("0.25");
    const high = item("1");
    const unavailable = item(null);
    expect(breakdownWidth(low, [low, high, unavailable])).toBe(25);
    expect(breakdownWidth(high, [low, high, unavailable])).toBe(100);
    expect(breakdownWidth(unavailable, [low, high, unavailable])).toBe(0);
  });

  it("renders an explicit fallback for timestamps outside the JavaScript Date range", () => {
    expect(formatTimestamp(Number.MAX_SAFE_INTEGER)).toBe(`${Number.MAX_SAFE_INTEGER} ms since epoch`);
  });
});
