import { AlertTriangle, CheckCircle2, Circle, GitBranch, RotateCcw } from "lucide-react";
import { useState } from "react";

import { compactId, humanize } from "./format";
import { latestPlanFrame } from "./observability-model";
import type { DashboardThreadDetail, EvaluatorReplayFrame, PlanReplayFrame } from "./types";

export function PlanLens({ detail }: { detail: DashboardThreadDetail }) {
  const fallback = latestPlanFrame(detail.plan_timeline);
  const [requestedFrame, setRequestedFrame] = useState<string | null>(null);
  const selected =
    detail.plan_timeline.find((frame) => frameKey(frame) === requestedFrame) ?? fallback;
  const plan = selected?.plan ?? detail.thread.plan;

  return (
    <section className="observability-grid observability-grid--plan">
      <div className="panel panel--timeline">
        <PanelHeading
          eyebrow="checkpoint replay"
          title="Plan revisions"
          detail={`${detail.plan_timeline.length} persisted frame${detail.plan_timeline.length === 1 ? "" : "s"}`}
        />
        {detail.plan_timeline.length ? (
          <div className="revision-list" aria-label="Plan revision timeline">
            {detail.plan_timeline.map((frame, index) => (
              <button
                key={frameKey(frame)}
                type="button"
                className={frameKey(frame) === frameKey(selected) ? "is-selected" : ""}
                onClick={() => setRequestedFrame(frameKey(frame))}
              >
                <span className="revision-index">{String(index + 1).padStart(2, "0")}</span>
                <span><strong>Revision {frame.revision_count}</strong><small>turn {frame.turn_number} · {frame.node}</small></span>
                <code>{compactId(frame.run_id)}</code>
              </button>
            ))}
          </div>
        ) : (
          <PanelEmpty icon={<RotateCcw size={20} />} title="No replay frames" detail="The current checkpoint plan is shown without a persisted revision trace." />
        )}
      </div>

      <div className="panel panel--plan-board">
        <PanelHeading
          eyebrow="active contract"
          title={selected ? `Turn ${selected.turn_number} · revision ${selected.revision_count}` : "Current plan"}
          detail={selected ? `sequence ${selected.sequence} · ${selected.node}` : "checkpoint snapshot"}
        />
        {selected ? <p className="plan-frame-note">Historical checkpoint: active marks the step in focus at this frame, not a currently running task.</p> : null}
        {plan ? (
          <ol className="plan-step-list">
            {plan.steps.map((step, index) => (
              <li key={step.id} className={`plan-step plan-step--${step.status}`}>
                <div className="plan-step__rail">
                  <span>{step.status === "completed" ? <CheckCircle2 /> : <Circle />}</span>
                  {index < plan.steps.length - 1 ? <i /> : null}
                </div>
                <div className="plan-step__body">
                  <div><code>{step.id}</code><span>{humanize(step.status)}</span></div>
                  <h3>{step.objective}</h3>
                  <ul>{step.expected_evidence.map((item) => <li key={item}>{item}</li>)}</ul>
                </div>
              </li>
            ))}
          </ol>
        ) : (
          <PanelEmpty icon={<GitBranch size={20} />} title="Plan not created" detail="The planner has not persisted a valid investigation plan for this checkpoint." />
        )}
      </div>

      <EvaluatorTimeline frames={detail.evaluator_timeline} />
    </section>
  );
}

function EvaluatorTimeline({ frames }: { frames: EvaluatorReplayFrame[] }) {
  return (
    <div className="panel panel--evaluator">
      <PanelHeading eyebrow="acceptance gate" title="Evaluator timeline" detail={`${frames.length} decision${frames.length === 1 ? "" : "s"}`} />
      {frames.length ? (
        <div className="evaluation-list">
          {frames.map((frame) => {
            const assessment = frame.report.semantic_assessment;
            return (
              <article key={`${frame.run_id}:${frame.sequence}`} className={frame.report.passed ? "is-pass" : "is-review"}>
                <div className="evaluation-card__head">
                  <span>{frame.report.passed ? <CheckCircle2 size={17} /> : <AlertTriangle size={17} />}{frame.report.passed ? "accepted" : "more evidence"}</span>
                  <code>t{frame.turn_number} · #{frame.sequence}</code>
                </div>
                <p>{assessment.summary}</p>
                <div className="evaluation-checks">
                  <Check label="objective" value={assessment.answers_objective} />
                  <Check label="supported" value={assessment.evidence_supports_summary} />
                  <Check label="model step approval" value={assessment.active_step_completed} />
                  <Check label="source validation" value={frame.report.deterministic_issues.length === 0} />
                  <Check label="no omissions" value={!assessment.has_critical_omissions} />
                </div>
                {frame.report.deterministic_issues.length ? (
                  <>
                    <p className="validation-gate-note">Source validation failed. Model approval alone does not establish step completion. Historical plan frames are preserved as recorded.</p>
                    <ul className="evaluation-notes">
                      {frame.report.deterministic_issues.map((issue, index) => <li key={`${issue.code}:${index}`}><strong>{issue.code}</strong>{issue.message}</li>)}
                    </ul>
                  </>
                ) : null}
                {assessment.missing_evidence.length ? (
                  <div className="focus-box"><span>Missing evidence</span>{assessment.missing_evidence.map((item) => <p key={item}>{item}</p>)}</div>
                ) : null}
                {assessment.recommended_focus.length ? (
                  <div className="focus-box focus-box--next"><span>Next focus</span>{assessment.recommended_focus.map((item) => <p key={item}>{item}</p>)}</div>
                ) : null}
              </article>
            );
          })}
        </div>
      ) : (
        <PanelEmpty icon={<AlertTriangle size={20} />} title="No evaluator frames" detail="No persisted evaluator decision is available for this thread." />
      )}
    </div>
  );
}

function Check({ label, value }: { label: string; value: boolean }) {
  return <span className={value ? "is-true" : "is-false"}>{value ? "✓" : "×"} {label}</span>;
}

function frameKey(frame: PlanReplayFrame | null): string {
  return frame ? `${frame.run_id}:${frame.sequence}` : "";
}

export function PanelHeading({ eyebrow, title, detail }: { eyebrow: string; title: string; detail: string }) {
  return <header className="panel-heading"><div><span>{eyebrow}</span><h2>{title}</h2></div><small>{detail}</small></header>;
}

export function PanelEmpty({ icon, title, detail }: { icon: React.ReactNode; title: string; detail: string }) {
  return <div className="panel-empty">{icon}<div><h3>{title}</h3><p>{detail}</p></div></div>;
}
