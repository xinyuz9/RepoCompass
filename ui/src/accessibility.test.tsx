import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import axe from "axe-core";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { HumanReviewPanel } from "./HumanReviewPanel";
import { MemoryDeleteControl } from "./MemoryDeleteControl";
import type { DashboardThreadDetail } from "./types";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("dashboard accessibility", () => {
  it("has no detectable structural violations in the first-run shell", async () => {
    const fetchMock = vi.fn((input: string | URL | Request) => {
      const system = {
        schema_version: 1,
        product: "RepoCompass",
        product_version: "0.1.0",
        repository_id: "a".repeat(64),
        commit_sha: "b".repeat(40),
        pricing_configured: false,
        lifecycle_mutations_enabled: false,
        memory_mutations_enabled: false,
      };
      const catalog = {
        schema_version: 1,
        threads: [],
        scanned_attempt_count: 0,
        truncated: false,
      };
      const body = String(input).endsWith("/system") ? system : catalog;
      return Promise.resolve(new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      }));
    });
    vi.stubGlobal("fetch", fetchMock);
    const view = renderWithClient(<App />);

    expect(await screen.findByRole("heading", { name: "No investigations recorded" })).toBeVisible();
    const results = await axe.run(view.container, axeOptions);
    expect(results.violations).toEqual([]);
  });

  it("has labeled, keyboard-addressable Human and memory mutation controls", async () => {
    const view = renderWithClient(
      <main>
        <HumanReviewPanel detail={reviewDetail} enabled />
        <MemoryDeleteControl memoryId={`${"d".repeat(56)}deadbeef`} enabled />
      </main>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Revise" }));

    expect(screen.getByLabelText(/Focused revision guidance/)).toBeVisible();
    expect(screen.getByLabelText("Type deadbeef to confirm memory deletion")).toBeVisible();
    const results = await axe.run(view.container, axeOptions);
    expect(results.violations).toEqual([]);
  });
});

const axeOptions = {
  rules: {
    "color-contrast": { enabled: false },
  },
};

const reviewDetail = {
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
