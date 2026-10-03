import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { InvestigationResult } from "./InvestigationResult";
import type { InvestigationStructure, ThreadInvestigationView } from "./types";

afterEach(cleanup);

function thread(): ThreadInvestigationView {
  return {
    schema_version: 1, thread_id: "thread-1", execution_state: "completed",
    turn_number: 1, objective: "Trace authentication", plan: null,
    plan_revision_count: 0, max_plan_revisions: 2, next_nodes: [],
    termination_status: "accepted", step_stop_reason: null,
    candidate_result: null, evaluation_report: null, structure: null,
    pending_review: null, interrupt_id: null,
    final_result: { summary: "Authentication checks a policy.", relevant_files: ["src/auth.py"],
      relevant_symbols: [], evidence: ["A policy is checked."], unresolved_questions: [] },
  };
}

function structure(): InvestigationStructure {
  return {
    artifact_id: "artifact", repository_id: "repo", commit_sha: "commit", indexed_paths: ["src/auth.py"],
    reading_path: {
      reading_path_id: "reading", objective: "Trace authentication", claim_lineages: [],
      graph_slice: {
        slice_id: "slice", graph_id: "graph", seed_entity_ids: ["entity"], traversal: "both",
        include_inferred: false, max_depth: 1, max_nodes: 10, max_edges: 10,
        truncated: false, truncation_reasons: [], edges: [], node_depths: [],
        nodes: [{ entity_id: "entity", node_id: "node", kind: "file", canonical_name: "src/auth.py",
          path: "src/auth.py", symbol_kind: null, is_test: false, evidence: [] }],
      },
      steps: [{ reading_step_id: "step", ordinal: 1, entity_id: "entity", node_id: "node",
        kind: "file", reason: "seed", canonical_name: "src/auth.py", path: "src/auth.py", depth: 0,
        path_verification_status: "verified", via_assertion_id: null,
        source_locations: [{ path: "src/auth.py", line_start: 2, line_end: 8 }] }],
    },
  };
}

describe("result interpretation and next actions", () => {
  it("does not describe workflow acceptance as independent correctness", () => {
    render(<InvestigationResult thread={thread()} onOpenLens={vi.fn()} />);
    expect(screen.getByText("Workflow accepted")).toBeVisible();
    expect(screen.getByText(/not independent proof/)).toBeVisible();
    expect(screen.queryByLabelText("Why this investigation stopped")).not.toBeInTheDocument();
  });

  it.each([
    ["step_no_progress", "No new evidence", "trace"],
    ["step_tool_call_budget_exhausted", "Tool-call limit reached", "trace"],
    ["step_explorer_budget_exhausted", "Exploration-call limit reached", "plan"],
    ["attempt_cost_budget_exhausted", "Spending limit reached", "cost"],
    ["attempt_cost_reservation_denied", "Next call exceeds the attempt allowance", "cost"],
    ["batch_cost_reservation_denied", "Shared spending allowance reached", "cost"],
    ["invalid_plan_revision", "Plan revision rejected", "trace"],
    ["future_reason", "Stopped before acceptance", "trace"],
    ["__proto__", "Stopped before acceptance", "trace"],
    ["constructor", "Stopped before acceptance", "trace"],
  ])("explains %s and only opens its inspector after a click", (reason, title, lens) => {
    const value = thread();
    value.termination_status = "exhausted";
    value.step_stop_reason = reason;
    const onOpenLens = vi.fn();
    render(<InvestigationResult thread={value} onOpenLens={onOpenLens} />);
    expect(screen.getByText(title)).toBeVisible();
    expect(onOpenLens).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: `Inspect ${lens}` }));
    expect(onOpenLens).toHaveBeenCalledExactlyOnceWith(lens);
    expect(screen.queryByRole("button", { name: /retry|resume|raise/i })).not.toBeInTheDocument();
  });

  it("does not claim a running step has terminated", () => {
    const value = thread();
    value.termination_status = null;
    value.execution_state = "resumable";
    value.final_result = null;
    value.step_stop_reason = "step_no_progress";
    render(<InvestigationResult thread={value} onOpenLens={vi.fn()} />);
    expect(screen.getByText("No result yet")).toBeVisible();
    expect(screen.queryByText("No new evidence")).not.toBeInTheDocument();
  });

  it("opens an existing source entity, never an arbitrary filename", () => {
    const value = thread();
    value.structure = structure();
    const onOpenLens = vi.fn();
    render(<InvestigationResult thread={value} onOpenLens={onOpenLens} />);
    fireEvent.click(screen.getByRole("button", { name: "View source: src/auth.py" }));
    expect(onOpenLens).toHaveBeenCalledExactlyOnceWith("topology", "entity");
  });

  it.each(["no structure", "unknown entity", "no location", "wrong first location"])(
    "does not offer a source read for %s", mode => {
      const value = thread();
      if (mode !== "no structure") {
        value.structure = structure();
        if (mode === "unknown entity") value.structure.reading_path.graph_slice.nodes = [];
        if (mode === "no location") value.structure.reading_path.steps[0]!.source_locations = [];
        if (mode === "wrong first location") value.structure.reading_path.steps[0]!.source_locations[0]!.path = "other.py";
      }
      render(<InvestigationResult thread={value} onOpenLens={vi.fn()} />);
      expect(screen.queryByRole("button", { name: /^View source:/ })).not.toBeInTheDocument();
      expect(screen.getByText(/Filenames alone do not authorize/)).toBeVisible();
    },
  );
});
