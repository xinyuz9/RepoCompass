import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { activeRefreshInterval } from "./active-refresh";
import { ObservabilityPanel } from "./ObservabilityPanel";
import type { DashboardThreadDetail, RawRunAttempt } from "./types";

afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

it("polls only active attempts, within the bounded snapshot count", () => {
  expect(activeRefreshInterval("running", false, 1)).toBe(1500);
  expect(activeRefreshInterval("running", false, 120)).toBe(false);
  expect(activeRefreshInterval("running", true, 1)).toBe(false);
  expect(activeRefreshInterval(undefined, false, 0)).toBe(false);
  for (const status of ["completed", "failed", "interrupted"] as const) {
    expect(activeRefreshInterval(status, false, 1)).toBe(false);
  }
});

it("refreshes a selected live trace then stops at failure without sending mutations", async () => {
  vi.useFakeTimers();
  const attempt = { run_id: "live", status: "running", execution_mode: "start", turn_number: 1, started_at_ms: 1000 } as RawRunAttempt;
  const detail = { attempts: [{ attempt }] } as DashboardThreadDetail;
  const initial = { attempt, trace_availability: "available", trace_events: [] };
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(["dashboard", "runs", "live"], initial);
  const fetch = vi.fn<typeof globalThis.fetch>(async () => new Response(JSON.stringify({
    ...initial, attempt: { ...attempt, status: "failed" },
    trace_events: [{ run_id: "live", sequence: 0, kind: "run_failed", payload: { error_code: "execution_failed" } }],
  }), { status: 200, headers: { "Content-Type": "application/json" } }));
  vi.stubGlobal("fetch", fetch);
  render(<QueryClientProvider client={client}><ObservabilityPanel lens="trace" detail={detail} memoryMutationsEnabled={false} requestedRunId="live" onRunSelected={vi.fn()} /></QueryClientProvider>);
  await act(async () => { await vi.advanceTimersByTimeAsync(1600); });
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(screen.getByText(/execution_failed/)).toBeVisible();
  await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(String(fetch.mock.calls[0]?.[0])).toContain("/runs/live");
  client.clear();
});
