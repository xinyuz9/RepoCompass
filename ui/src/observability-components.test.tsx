import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ContextLens } from "./ContextLens";
import { CostLens } from "./CostLens";
import { PlanLens } from "./PlanLens";
import { TraceLens } from "./TraceLens";
import type { DashboardRunDetail, DashboardThreadDetail } from "./types";

const attempt = {
  schema_version: 1,
  run_id: "run-1",
  thread_id: "thread-1",
  turn_number: 1,
  execution_mode: "start",
  repository_id: "a".repeat(64),
  commit_sha: "b".repeat(40),
  objective_digest: "c".repeat(64),
  status: "completed",
  started_at_ms: 1_000,
  finished_at_ms: 1_500,
} as const;

afterEach(cleanup);

describe("investigation observability lenses", () => {
  it("replays plan revisions beside evaluator decisions", () => {
    const detail = {
      thread: {
        plan: { steps: [{ id: "inspect-auth", objective: "Inspect authentication entry points.", expected_evidence: ["A source location."], status: "active" }] },
      },
      plan_timeline: [{ run_id: "run-1", turn_number: 1, sequence: 2, node: "plan_investigation", revision_count: 0, plan: { steps: [{ id: "inspect-auth", objective: "Inspect authentication entry points.", expected_evidence: ["A source location."], status: "active" }] } }],
      evaluator_timeline: [{ run_id: "run-1", turn_number: 1, sequence: 8, node: "evaluate_candidate", report: { passed: false, deterministic_issues: [], semantic_assessment: { active_step_completed: false, answers_objective: false, evidence_supports_summary: true, has_critical_omissions: true, missing_evidence: ["Authentication middleware registration."], recommended_focus: ["Inspect router setup."], summary: "One critical link is still missing." } } }],
    } as unknown as DashboardThreadDetail;

    render(<PlanLens detail={detail} />);
    expect(screen.getByRole("heading", { name: "Plan revisions" })).toBeVisible();
    expect(screen.getByText("Inspect authentication entry points.")).toBeVisible();
    expect(screen.getByText("One critical link is still missing.")).toBeVisible();
    expect(screen.getByText("Authentication middleware registration.")).toBeVisible();
  });

  it("shows a sanitized trace payload with its ordered sequence", () => {
    const run = {
      attempt,
      trace_availability: "available",
      trace_events: [{ schema_version: 1, run_id: "run-1", sequence: 3, kind: "node_completed", node: "explore_repository", payload: { tool_call_count: 2, stop_reason: "evidence_added" } }],
    } as unknown as DashboardRunDetail;

    render(<TraceLens run={run} />);
    expect(screen.getByRole("heading", { name: "Run trace" })).toBeVisible();
    expect(screen.getAllByText("explore_repository").length).toBeGreaterThan(0);
    expect(screen.getByText(/tool_call_count/)).toBeVisible();
    expect(screen.getByText(/evidence_added/)).toBeVisible();
    expect(screen.getByRole("region", { name: "Public event payload" })).toHaveAttribute("tabindex", "0");
  });

  it("separates model approval from deterministic validation without rewriting history", () => {
    const detail = {
      thread: { plan: null }, plan_timeline: [],
      evaluator_timeline: [{ run_id: "run-1", turn_number: 1, sequence: 8, report: {
        passed: false, deterministic_issues: [{ code: "path_not_found", message: "Rejected path." }],
        semantic_assessment: { active_step_completed: true, answers_objective: true, evidence_supports_summary: true, has_critical_omissions: false, missing_evidence: [], recommended_focus: [], summary: "Model approved." },
      } }],
    } as unknown as DashboardThreadDetail;
    render(<PlanLens detail={detail} />);
    expect(screen.getByText("✓ model step approval")).toBeVisible();
    expect(screen.getByText("× source validation")).toBeVisible();
    expect(screen.getByText(/Model approval alone does not establish step completion/)).toBeVisible();
    expect(screen.queryByText("✓ step done")).not.toBeInTheDocument();
  });

  it("bounds trace DOM to 50 rows and preserves history while new events arrive", () => {
    const makeRun = (count: number, id = "run-1") => ({
      attempt: { ...attempt, run_id: id }, trace_availability: "available",
      trace_events: Array.from({ length: count }, (_, sequence) => ({
        run_id: id, sequence, kind: "node_completed", node: sequence % 2 ? "agent" : "evaluate_candidate", payload: { private_search_marker: "not searched" },
      })),
    } as unknown as DashboardRunDetail);
    const view = render(<TraceLens run={makeRun(10000)} />);
    const events = () => within(screen.getByRole("list", { name: "Trace events" }));
    expect(events().getAllByRole("listitem")).toHaveLength(50);
    expect(events().getByText("#9999")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "First page" }));
    expect(events().getByText("#0")).toBeVisible();
    fireEvent.click(events().getByText("#12"));
    view.rerender(<TraceLens run={makeRun(10001)} />);
    expect(events().getByText("#0")).toBeVisible();
    expect(events().getByText("#12").closest("button")).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    expect(events().getByText("#50")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Previous page" }));
    expect(events().getByText("#0")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Follow latest" }));
    expect(events().getByText("#10000")).toBeVisible();
    expect(screen.getByRole("button", { name: "Next page" })).toBeDisabled();
    const search = screen.getByRole("searchbox", { name: "Find node, event kind or #sequence" });
    fireEvent.change(search, { target: { value: "#9876" } });
    expect(events().getAllByRole("listitem")).toHaveLength(1);
    expect(events().getByText("#9876")).toBeVisible();
    fireEvent.change(search, { target: { value: "private_search_marker" } });
    expect(screen.getByRole("heading", { name: "No matching events" })).toBeVisible();
    expect(screen.queryByRole("list", { name: "Trace events" })).not.toBeInTheDocument();
    // Changing the selected run clears search, filter, page and event identity.
    view.rerender(<TraceLens run={makeRun(1, "run-2")} />);
    expect(screen.getByRole("searchbox")).toHaveValue("");
    expect(events().getByText("#0")).toBeVisible();
  });

  it("explains ContextPack budget selection and memory freshness", () => {
    const itemId = "d".repeat(64);
    const run = {
      attempt,
      context_packs: [{
        pack_id: "e".repeat(64),
        estimated_tokens: 120,
        build_report: { budget_tokens: 400, candidate_count: 2, selected_count: 1, excluded_count: 1, decisions: [
          { item_id: itemId, outcome: "selected", reason: "verified_evidence", estimated_tokens: 120, selected_position: 0 },
          { item_id: "f".repeat(64), outcome: "excluded", reason: "budget_exceeded", estimated_tokens: 500, selected_position: null },
        ] },
        items: [{ item_id: itemId, section: "memory", verification_status: "human_approved", content: "Router middleware is registered in src/router.py.", estimated_tokens: 120, priority: 90, pinned: false, source: { source_id: "1".repeat(64), kind: "memory", location: null, provenance: [{ kind: "memory", id: "2".repeat(64) }], memory: { memory_id: "3".repeat(64), scope: "repository", freshness: "current" } } }],
      }],
      model_invocations: [],
    } as unknown as DashboardRunDetail;

    render(<ContextLens run={run} />);
    expect(screen.getByRole("heading", { name: "Context inspector" })).toBeVisible();
    expect(screen.getByText("Router middleware is registered in src/router.py.")).toBeVisible();
    expect(screen.getByText("verified evidence")).toBeVisible();
    expect(screen.getByText("budget exceeded")).toBeVisible();
    expect(screen.getByText("current")).toBeVisible();
  });

  it("filters persisted failures without depicting them as finished runs", () => {
    const run = {
      attempt: { ...attempt, status: "failed" },
      trace_availability: "available",
      trace_events: [
        { run_id: "run-1", sequence: 0, kind: "run_started", node: null, payload: {} },
        { run_id: "run-1", sequence: 1, kind: "run_failed", node: null, payload: { phase: "graph_execution", error_code: "execution_failed" } },
      ],
    } as unknown as DashboardRunDetail;
    render(<TraceLens run={run} />);
    fireEvent.click(screen.getByRole("button", { name: "run failed" }));
    expect(screen.getByText(/execution_failed/)).toBeVisible();
    expect(screen.getByText(/graph_execution/)).toBeVisible();
    expect(screen.queryByText("#0")).not.toBeInTheDocument();
  });

  it("renders exact priced usage without presenting it as an invoice", () => {
    const metric = { coverage: "complete", value: 36, contributing_invocations: 1, total_invocations: 1 };
    const invocationId = "4".repeat(64);
    const run = {
      attempt,
      cost_availability: "available",
      model_invocations: [{ invocation_id: invocationId, sequence: 0, node_name: "plan_investigation", role: "planner", provider: "openai", model: "gpt-test", status: "succeeded", latency_ms: 1230, usage: { availability: "reported", total_tokens: 36 }, plan_step_id: "inspect-auth" }],
      usage_cost_report: {
        pricing_version: "portfolio-2026-08",
        pricing_effective_from: "2026-08-01",
        invocation_costs: [{ invocation_id: invocationId, availability: "estimated", estimated_cost_usd: "0.000018705" }],
        breakdowns: [{ dimension: "attempt", key: "run-1", invocation_count: 1, succeeded_count: 1, failed_count: 0, latency_ms: 1230, input_tokens: metric, output_tokens: metric, total_tokens: metric, cached_input_tokens: { ...metric, value: 0 }, reasoning_output_tokens: { ...metric, value: 0 }, cost: { coverage: "complete", estimated_cost_usd: "0.000018705", estimated_invocations: 1, unavailable_invocations: 0, unavailable_reasons: [], disclaimer: "estimate_not_invoice" } }],
        budget: null,
      },
    } as unknown as DashboardRunDetail;

    render(<CostLens run={run} />);
    expect(screen.getByRole("heading", { name: "Cost analysis" })).toBeVisible();
    expect(screen.getAllByText("$0.000018705").length).toBeGreaterThan(0);
    expect(screen.getByText(/estimate, not invoice/)).toBeVisible();
    expect(screen.getByText("plan_investigation")).toBeVisible();
  });

  it("follows the latest invocation when it reuses an older snapshot", () => {
    const pack = (id: string) => ({
      pack_id: id, estimated_tokens: 0, items: [],
      build_report: { budget_tokens: 400, candidate_count: 1, selected_count: 0, excluded_count: 1, decisions: [] },
    });
    const run = {
      attempt, context_packs: [pack("pack-a"), pack("pack-b")],
      model_invocations: [
        { sequence: 7, context_pack_id: "pack-a", node_name: "build_candidate", role: "candidate_builder" },
        { sequence: 6, context_pack_id: "pack-b", node_name: "agent", role: "explorer" },
        { sequence: 4, context_pack_id: "pack-a", node_name: "build_candidate", role: "candidate_builder" },
      ],
    } as unknown as DashboardRunDetail;
    render(<ContextLens run={run} />);
    const selector = screen.getByRole("combobox", { name: "Context snapshot" });
    expect(selector).toHaveValue("pack-a");
    expect(screen.getByText("#7")).toBeVisible();
    expect(screen.getByText("used by 2 invocations · latest consumer")).toBeVisible();
    fireEvent.change(selector, { target: { value: "pack-b" } });
    expect(selector).toHaveValue("pack-b");
    expect(screen.getByText("#6")).toBeVisible();
    expect(screen.getByText("used by 1 invocation · latest consumer")).toBeVisible();
  });
});
