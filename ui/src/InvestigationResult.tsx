import { FileCode2 } from "lucide-react";

import type { InspectorLens } from "./ConversationWorkspace";
import type { ThreadInvestigationView } from "./types";

const STOP_ADVICE: Record<string, { title: string; next: string; lens: InspectorLens }> = {
  invalid_candidate_evidence: { title: "Source checks failed", next: "Inspect the rejected candidate in the trace, then refine the goal or start a detailed investigation.", lens: "trace" },
  step_no_progress: { title: "No new evidence", next: "Review the recent searches, then narrow the goal or focus path. Repeating the same request may not help.", lens: "trace" },
  step_tool_call_budget_exhausted: { title: "Tool-call limit reached", next: "Inspect which searches used the allowance, then choose a smaller follow-up scope.", lens: "trace" },
  step_explorer_budget_exhausted: { title: "Exploration-call limit reached", next: "Review the missing evidence before starting another investigation.", lens: "plan" },
  attempt_cost_budget_exhausted: { title: "Spending limit reached", next: "Review the cost ledger and narrow the next goal. No additional model call was admitted.", lens: "cost" },
  attempt_cost_reservation_denied: { title: "Next call exceeds the attempt allowance", next: "Review reserved and recorded costs before deciding on a new attempt.", lens: "cost" },
  batch_cost_reservation_denied: { title: "Shared spending allowance reached", next: "Review the shared budget with the local operator. Chat cannot raise this limit.", lens: "cost" },
  invalid_plan_revision: { title: "Plan revision rejected", next: "Inspect the failed revision in the trace. The rejected plan was not applied and is not automatically retried.", lens: "trace" },
};

export function InvestigationResult({ thread, onOpenLens }: {
  thread: ThreadInvestigationView;
  onOpenLens: (lens: InspectorLens, entityId?: string) => void;
}) {
  const result = thread.final_result;
  const candidate = thread.candidate_result;
  const evidence = result?.evidence ?? candidate?.evidence.map(item => item.claim) ?? [];
  const files = result?.relevant_files ?? [...new Set(candidate?.evidence.map(item => item.path) ?? [])];
  const questions = result?.unresolved_questions ?? candidate?.unresolved_questions ?? [];
  const status = thread.termination_status;
  const stopReason = thread.step_stop_reason ?? "";
  const knownAdvice = Object.hasOwn(STOP_ADVICE, stopReason) ? STOP_ADVICE[stopReason] : undefined;
  const label = thread.pending_review ? "Awaiting review"
    : status === "cancelled" ? "Cancelled"
    : status === "exhausted" ? "Best effort"
    : result && status === "accepted" ? thread.architecture_preset === "tool_wrapper" ? "Source checks passed" : "Workflow accepted" : "Working draft";
  const advice = status === "cancelled"
    ? { title: "Investigation cancelled", next: "Recorded findings are retained. Inspect the last attempt before deciding whether to start a new goal.", lens: "trace" as const }
    : status === "exhausted"
      ? knownAdvice ?? { title: "Stopped before acceptance", next: "The checkpoint has no recognized stopping reason. Inspect the last attempt; nothing will be retried automatically.", lens: "trace" as const }
      : null;
  const structure = thread.structure;
  const sourceStep = (path: string) => structure?.reading_path.steps.find(step =>
    step.path === path && step.source_locations[0]?.path === path
    && structure.reading_path.graph_slice.nodes.some(node => node.entity_id === step.entity_id));

  return (
    <section className={`result-card result-card--${status ?? "draft"}`} aria-label="Investigation result">
      <div className="result-card__heading">
        <h2>{result ? "Investigation result" : candidate ? "Candidate result" : "No result yet"}</h2>
        <span className="result-card__status">{label}</span>
      </div>
      <p>{result?.summary ?? candidate?.summary ?? "The latest checkpoint has no result to display. Inspect Run trace for the last completed step."}</p>
      {status === "exhausted" || status === "cancelled" ? (
        <p className="result-card__note">Stopped without acceptance. Review the evidence and open questions before relying on this result.</p>
      ) : result && status === "accepted" ? (
        <p className="result-card__note">{thread.architecture_preset === "tool_wrapper"
          ? "Source references passed deterministic checks. The answer has not had independent semantic review."
          : "Passed the configured workflow checks, not independent proof that every conclusion is correct."}</p>
      ) : candidate ? <p className="result-card__note">This candidate has not been finalized.</p> : null}
      {advice ? <aside className="result-stop" aria-label="Why this investigation stopped">
        <strong>{advice.title}</strong><p>{advice.next}</p>
        <button type="button" onClick={() => onOpenLens(advice.lens)}>Inspect {advice.lens}</button>
      </aside> : null}
      {!result && !candidate ? <button className="result-card__open" type="button" onClick={() => onOpenLens("trace")}>Inspect last attempt</button> : <>
        {files.length ? <>
          <h3>Relevant files · {files.length}</h3>
          <div className="result-files">{files.map(file => {
            const step = sourceStep(file);
            return step ? <button key={file} type="button" aria-label={`View source: ${file}`}
              onClick={() => onOpenLens("topology", step.entity_id)}><FileCode2 size={13} /><span>{file}</span></button>
              : <code key={file}>{file}</code>;
          })}</div>
          {!files.some(file => sourceStep(file)) ? <p className="result-card__note">No snapshot-bound source preview is available for these files. Filenames alone do not authorize a source read.</p> : null}
        </> : null}
        <h3>Evidence · {evidence.length}</h3>
        {evidence.length ? <ol>{evidence.slice(0, 3).map((claim, index) => <li key={index}>{claim}</li>)}</ol>
          : <p className="result-card__note">No evidence was reported.</p>}
        {evidence.length > 3 ? <details><summary>Show {evidence.length - 3} more evidence items</summary>
          <ol start={4}>{evidence.slice(3).map((claim, index) => <li key={index}>{claim}</li>)}</ol>
        </details> : null}
        <h3>Open questions · {questions.length}</h3>
        {questions.length ? <ul>{questions.map((question, index) => <li key={index}>{question}</li>)}</ul>
          : <p className="result-card__note">No unresolved questions were reported.</p>}
        {structure ? <button className="result-card__open" type="button" onClick={() => onOpenLens("topology")}>
          <FileCode2 size={15} />Explore source evidence
        </button> : null}
      </>}
    </section>
  );
}
