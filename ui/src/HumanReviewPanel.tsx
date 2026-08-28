import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  FileCheck2,
  MessageSquareMore,
  RotateCcw,
  Send,
  ShieldAlert,
  XCircle,
} from "lucide-react";
import { useState } from "react";

import { DashboardApiError, respondToReview } from "./api";
import { compactId } from "./format";
import type {
  DashboardHumanResponseCommand,
  DashboardThreadDetail,
  HumanDecisionAction,
} from "./types";

const FEEDBACK_LIMIT = 16_384;
let fallbackRequestSequence = 0;

const ACTIONS: Record<
  HumanDecisionAction,
  { label: string; detail: string; icon: typeof CheckCircle2 }
> = {
  approve: {
    label: "Approve",
    detail: "Accept this evidence-backed result.",
    icon: CheckCircle2,
  },
  revise: {
    label: "Revise",
    detail: "Return focused guidance to the agent.",
    icon: RotateCcw,
  },
  cancel: {
    label: "Cancel",
    detail: "Stop this investigation without publishing.",
    icon: XCircle,
  },
};

export function HumanReviewPanel({
  detail,
  enabled,
}: {
  detail: DashboardThreadDetail;
  enabled: boolean;
}) {
  const queryClient = useQueryClient();
  const { thread } = detail;
  const review = thread.pending_review;
  const [action, setAction] = useState<HumanDecisionAction>(
    review?.allowed_actions[0] ?? "approve",
  );
  const [feedback, setFeedback] = useState("");
  const [submittedCommand, setSubmittedCommand] =
    useState<DashboardHumanResponseCommand | null>(null);
  const mutation = useMutation({
    mutationFn: (command: DashboardHumanResponseCommand) =>
      respondToReview(thread.thread_id, command),
    retry: false,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["dashboard", "threads"] });
    },
  });

  if (!review || !thread.interrupt_id) return null;

  const revisionFeedbackMissing = action === "revise" && feedback.trim().length === 0;
  const commandLocked = submittedCommand !== null;
  const submit = () => {
    if (!enabled || mutation.isPending || mutation.isSuccess || revisionFeedbackMissing) {
      return;
    }
    const command: DashboardHumanResponseCommand = {
      schema_version: 1,
      request_id: createReviewRequestId(),
      interrupt_id: thread.interrupt_id as string,
      decision: {
        action,
        feedback: action === "revise" ? feedback.trim() : null,
      },
    };
    setSubmittedCommand(command);
    mutation.mutate(command);
  };
  const retry = () => {
    if (submittedCommand && !mutation.isPending) mutation.mutate(submittedCommand);
  };
  const unlock = () => {
    mutation.reset();
    setSubmittedCommand(null);
  };

  return (
    <section className="review-gate" aria-labelledby="review-gate-title">
      <div className="review-gate__summary">
        <div className="review-gate__heading">
          <span className="review-gate__icon"><ShieldAlert size={18} /></span>
          <div>
            <p className="eyebrow">interrupt-bound decision</p>
            <h2 id="review-gate-title">Human review required</h2>
          </div>
          <code>{compactId(thread.interrupt_id, 12)}</code>
        </div>
        <p className="review-candidate">{review.candidate_result.summary}</p>
        <div className="review-evidence-row">
          <span><FileCheck2 size={14} />{review.candidate_result.evidence.length} evidence anchors</span>
          <span className={review.evaluation_report.passed ? "is-pass" : "is-review"}>
            {review.evaluation_report.passed ? "evaluator passed" : "evaluator flagged review"}
          </span>
          <span>{review.candidate_result.unresolved_questions.length} unresolved</span>
        </div>
        {review.candidate_result.evidence.length > 0 ? (
          <div className="review-source-strip" aria-label="Candidate evidence preview">
            {review.candidate_result.evidence.slice(0, 3).map((evidence, index) => (
              <span key={`${evidence.path}:${evidence.line_start ?? 0}:${index}`}>
                <code>{formatEvidenceLocation(evidence.path, evidence.line_start, evidence.line_end)}</code>
                <small>{evidence.claim}</small>
              </span>
            ))}
          </div>
        ) : null}
      </div>

      <div className="review-gate__controls">
        {!enabled ? (
          <div className="mutation-disabled" role="status">
            <ShieldAlert size={17} />
            <div><strong>Review controls are read-only</strong><span>Restart with <code>--enable-lifecycle-mutations</code> to respond.</span></div>
          </div>
        ) : (
          <>
            <div className="review-actions" role="group" aria-label="Review decision">
              {review.allowed_actions.map((allowedAction) => {
                const meta = ACTIONS[allowedAction];
                const Icon = meta.icon;
                return (
                  <button
                    key={allowedAction}
                    type="button"
                    className={`${action === allowedAction ? "is-selected" : ""} review-action--${allowedAction}`}
                    aria-pressed={action === allowedAction}
                    disabled={commandLocked || mutation.isPending || mutation.isSuccess}
                    onClick={() => setAction(allowedAction)}
                    title={meta.detail}
                  >
                    <Icon size={15} /><span>{meta.label}</span>
                  </button>
                );
              })}
            </div>
            {action === "revise" ? (
              <label className="review-feedback">
                <span><MessageSquareMore size={13} />Focused revision guidance</span>
                <textarea
                  value={feedback}
                  maxLength={FEEDBACK_LIMIT}
                  rows={3}
                  disabled={commandLocked || mutation.isPending || mutation.isSuccess}
                  placeholder="Name the missing evidence, path, symbol, or relationship to inspect."
                  onChange={(event) => setFeedback(event.target.value)}
                />
                <small>{feedback.length.toLocaleString()} / {FEEDBACK_LIMIT.toLocaleString()}</small>
              </label>
            ) : null}
            {!commandLocked ? (
              <button
                type="button"
                className={`review-submit review-submit--${action}`}
                disabled={mutation.isPending || revisionFeedbackMissing}
                onClick={submit}
              >
                <Send size={14} />Submit {ACTIONS[action].label.toLocaleLowerCase()}
              </button>
            ) : null}
            {mutation.isPending ? <p className="mutation-status" role="status">Applying one idempotent response…</p> : null}
            {mutation.isError ? (
              <div className="mutation-result is-error" role="alert">
                <strong>Response not confirmed</strong>
                <span>{mutationErrorMessage(mutation.error)}</span>
                <div><button type="button" onClick={retry}>Retry same request</button><button type="button" onClick={unlock}>Change decision</button></div>
              </div>
            ) : null}
            {mutation.data ? (
              <div className="mutation-result is-success" role="status" aria-live="polite">
                <strong>{mutation.data.idempotent_replay ? "Existing response recovered" : "Response accepted"}</strong>
                <span>Checkpoint is now {mutation.data.execution_state} · request {compactId(mutation.data.request_id, 10)}</span>
              </div>
            ) : null}
          </>
        )}
      </div>
    </section>
  );
}

function createReviewRequestId(): string {
  const randomUuid = globalThis.crypto?.randomUUID?.();
  if (randomUuid) return `dashboard-review:${randomUuid}`;
  fallbackRequestSequence += 1;
  return `dashboard-review:${Date.now().toString(36)}:${fallbackRequestSequence.toString(36)}`;
}

function formatEvidenceLocation(
  path: string,
  lineStart: number | null,
  lineEnd: number | null,
): string {
  if (lineStart === null) return path;
  return `${path}:${lineStart}${lineEnd !== null && lineEnd !== lineStart ? `-${lineEnd}` : ""}`;
}

function mutationErrorMessage(error: Error): string {
  return error instanceof DashboardApiError
    ? error.problem?.detail ?? error.message
    : "The local request ended without a confirmed response.";
}
