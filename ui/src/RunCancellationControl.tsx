import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Square } from "lucide-react";
import { useEffect, useState } from "react";
import { getCancellation, requestCancellation } from "./api";
import { cancellationRefreshInterval } from "./active-refresh";
import type { CancellationRequest, RawRunAttempt } from "./types";

export function RunCancellationControl({ attempt, enabled }: { attempt: RawRunAttempt; enabled: boolean }) {
  if (!enabled) return null;
  return <EnabledCancellation key={attempt.run_id} attempt={attempt} />;
}

function EnabledCancellation({ attempt }: { attempt: RawRunAttempt }) {
  const client = useQueryClient();
  const [command, setCommand] = useState<CancellationRequest | null>(null);
  const key = ["dashboard", "cancellation", attempt.repository_id, attempt.commit_sha, attempt.run_id, attempt.status];
  const query = useQuery({
    queryKey: key,
    queryFn: ({ signal }) => getCancellation(attempt.run_id, signal),
    retry: false, staleTime: 0, refetchOnWindowFocus: false, refetchOnReconnect: false,
    refetchInterval: (query) => cancellationRefreshInterval(query.state.data?.state, query.state.status === "error", query.state.dataUpdateCount),
  });
  const mutation = useMutation({
    mutationFn: requestCancellation,
    retry: false,
    onSuccess: (record) => client.setQueryData(key, record),
    onError: () => { void query.refetch(); },
  });
  const state = query.data?.state;
  useEffect(() => {
    if (state === "cancelled" || state === "finished") {
      void client.invalidateQueries({ queryKey: ["dashboard", "threads"] });
      void client.invalidateQueries({ queryKey: ["dashboard", "runs", attempt.run_id] });
      void client.invalidateQueries({ queryKey: ["dashboard", "operations"] });
    }
  }, [state, client, attempt.run_id]);

  const submit = () => {
    const exact = command ?? {
      schema_version: 1 as const, request_id: "cancel:" + globalThis.crypto.randomUUID(),
      run_id: attempt.run_id, thread_id: attempt.thread_id,
      repository_id: attempt.repository_id, commit_sha: attempt.commit_sha,
    };
    setCommand(exact);
    mutation.mutate(exact);
  };
  if (state === "finished" && query.data?.request_id === null) return null;
  const canRequest = state === "running" && attempt.status === "running" && !query.isError;
  return (
    <section className="run-cancellation" aria-label="Run cancellation">
      <div>
        <strong>{state === "cancelled" ? "Cancellation confirmed"
          : state === "cancel_requested" ? "Cancellation requested · awaiting confirmation"
          : state === "finished" ? "Run ended without cancellation confirmation"
          : query.isError ? "Cancellation status unavailable"
          : "Run control"}</strong>
        <p>{state === "cancelled" ? "The cancelled checkpoint and terminal trace are persisted. Recorded calls and cost remain available."
          : "The current node may finish and incur cost."}</p>
      </div>
      {query.isPending ? <p role="status">Reading run control…</p> : null}
      {canRequest ? <button type="button" disabled={mutation.isPending} onClick={submit}>
        <Square size={13} />{mutation.isPending ? "Sending cancellation…" : command ? "Retry same cancellation" : "Request cancellation"}
      </button> : null}
      {state === "cancel_requested" || query.isError || mutation.isError ? <button type="button" disabled={query.isFetching} onClick={() => void query.refetch()}>Check cancellation status</button> : null}
      {mutation.isError && state !== "cancel_requested" && state !== "cancelled" ? <p role="alert">Cancellation was not confirmed. Check status or retry the same request; do not assume the agent has stopped.</p> : null}
      {state !== "cancelled" ? <details className="run-cancellation__limits">
        <summary>Cancellation limits</summary>
        <p>Cooperative stop at the next graph node; this does not kill a provider request or sandbox process.</p>
        <p>Automatic checks are bounded. If confirmation does not arrive, check status manually; a crashed worker is not automatically recovered.</p>
      </details> : null}
    </section>
  );
}
