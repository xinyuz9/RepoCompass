import { useInfiniteQuery } from "@tanstack/react-query";
import { Bot, History, UserRound } from "lucide-react";
import { useState } from "react";
import { getConversation } from "./api";
import { ReviewHistory } from "./ReviewHistory";

export function ConversationHistory({
  threadId, turnNumber, repositoryId, commitSha,
}: {
  threadId: string; turnNumber: number; repositoryId: string; commitSha: string | null;
}) {
  const [opened, setOpened] = useState(false);
  const history = useInfiniteQuery({
    queryKey: ["dashboard", "conversation", repositoryId, commitSha, threadId, turnNumber],
    initialPageParam: turnNumber,
    queryFn: ({ pageParam, signal }) => getConversation(threadId, pageParam, signal),
    getNextPageParam: (page) => page.next_before_turn ?? undefined,
    enabled: opened && turnNumber > 1,
    retry: false,
    staleTime: 30_000,
  });
  if (turnNumber <= 1) return null;
  const turns = [...(history.data?.pages ?? [])].reverse().flatMap((page) => page.turns);
  return (
    <section className="conversation-history" aria-label="Earlier conversation">
      <button className="history-toggle" type="button" aria-expanded={opened}
        onClick={() => setOpened(!opened)}>
        <History size={16} />{opened ? "Hide earlier turns" : "Earlier turns · " + (turnNumber - 1)}
      </button>
      {opened ? (
        <div className="history-content">
          <p className="history-boundary">Saved goals, human reviews and final summaries. Inspectors below describe the current turn, not these earlier answers.</p>
          {history.isPending ? <p role="status">Loading saved conversation…</p> : null}
          {history.isError ? (
            <div role="alert">
              <p>Earlier conversation could not be loaded. The current turn remains available.</p>
              <button type="button" onClick={() => {
                if (history.data) void history.fetchNextPage();
                else void history.refetch();
              }}>Retry history read</button>
            </div>
          ) : null}
          {history.hasNextPage ? (
            <button type="button" disabled={history.isFetchingNextPage}
              onClick={() => void history.fetchNextPage()}>
              {history.isFetchingNextPage ? "Loading…" : "Load older turns"}
            </button>
          ) : null}
          {turns.map((turn) => (
            <section className="history-turn" key={turn.turn_number} aria-label={"Turn " + turn.turn_number}>
              <article className="chat-message chat-message--user">
                <span className="chat-avatar"><UserRound size={16} /></span>
                <div><header><strong>You</strong><small>turn {turn.turn_number}</small></header><p>{turn.objective}</p></div>
              </article>
              <ReviewHistory history={turn.review_history} />
              {turn.summary !== null ? (
                <article className="chat-message chat-message--agent">
                  <span className="chat-avatar"><Bot size={17} /></span>
                  <div>
                    <header><strong>RepoCompass</strong><small>{turn.termination_status}</small></header>
                    <p>{turn.summary}</p>
                    {turn.termination_status !== "accepted" ? <p className="history-boundary">Stopped without acceptance.</p> : null}
                  </div>
                </article>
              ) : <p className="history-boundary">No final answer was persisted.</p>}
              {turn.text_truncated ? <p className="history-boundary">Long text is truncated in this public preview.</p> : null}
            </section>
          ))}
        </div>
      ) : null}
    </section>
  );
}
