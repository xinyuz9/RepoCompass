import { useQuery } from "@tanstack/react-query";
import { History } from "lucide-react";
import { useState } from "react";
import { getConversation } from "./api";
import type { HumanReviewHistory } from "./types";

export function ReviewHistory({ history }: { history: HumanReviewHistory | null | undefined }) {
  return (
    <section className="review-history" aria-label="Recorded human reviews">
      {history == null || history.prior_history_unavailable ? (
        <p className="history-boundary">Earlier review history is unavailable in this checkpoint. No decisions have been inferred.</p>
      ) : null}
      {history?.omitted_count ? (
        <p className="history-boundary">{history.omitted_count} older reviews omitted from this bounded history.</p>
      ) : null}
      {history && history.records.length === 0 && !history.prior_history_unavailable ? (
        <p className="history-boundary">No human review requests recorded for this turn. Automatic completion is not human approval.</p>
      ) : null}
      {history?.records.map((record) => (
        <details className="review-exchange" key={record.review_number}>
          <summary>
            <span>Review {record.review_number}</span>
            <strong>{record.action === null ? "No decision recorded" : ({ approve: "Approved", revise: "Revision requested", cancel: "Cancelled" })[record.action]}</strong>
          </summary>
          <div className="review-exchange__body">
            <p className="history-boundary">Recorded candidate · request/{record.request_digest.slice(0, 12)}</p>
            <p>{record.candidate_summary}</p>
            <p className="history-boundary">Evaluation gate: {record.evaluation_passed ? "passed" : "not passed"} · {record.deterministic_issue_count} source validation issues</p>
            <p className="history-boundary">Actions offered: {record.allowed_actions.join(" / ")}</p>
            {record.feedback !== null ? (
              <div className="review-exchange__feedback"><strong>You · review feedback</strong><p>{record.feedback}</p></div>
            ) : null}
            {record.text_truncated ? <p className="history-boundary">Long text is truncated in this public preview.</p> : null}
          </div>
        </details>
      ))}
    </section>
  );
}

export function CurrentReviewHistory({ threadId, turnNumber, repositoryId, commitSha, checkpointKey }: {
  threadId: string; turnNumber: number; repositoryId: string; commitSha: string | null; checkpointKey: string;
}) {
  const [opened, setOpened] = useState(false);
  const query = useQuery({
    queryKey: ["dashboard", "current-reviews", repositoryId, commitSha, threadId, turnNumber, checkpointKey],
    queryFn: ({ signal }) => getConversation(threadId, turnNumber + 1, signal, 1),
    enabled: opened,
    retry: false, staleTime: 30_000, refetchOnWindowFocus: false, refetchOnReconnect: false,
  });
  const turn = query.data?.turns.find((item) => item.turn_number === turnNumber);
  return (
    <section className="conversation-history" aria-label="Current turn review history">
      <button className="history-toggle" type="button" aria-expanded={opened} onClick={() => setOpened(!opened)}>
        <History size={16} />{opened ? "Hide review history" : "Review history"}
      </button>
      {opened ? <div className="history-content">
        <p className="history-boundary">Read-only checkpoint records, not live approval controls. Model messages are not included.</p>
        {query.isPending ? <p role="status">Loading review history…</p> : null}
        {query.isError ? <div role="alert"><p>Review history could not be loaded.</p><button type="button" onClick={() => void query.refetch()}>Retry review history</button></div> : null}
        {turn ? <ReviewHistory history={turn.review_history} /> : null}
        {query.isSuccess && !turn ? <p role="status">This turn is not available in the checkpoint history.</p> : null}
      </div> : null}
    </section>
  );
}
