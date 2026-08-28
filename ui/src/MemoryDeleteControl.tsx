import { useMutation, useQueryClient } from "@tanstack/react-query";
import { LockKeyhole, RotateCcw, ShieldAlert, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";

import { DashboardApiError, deleteMemory } from "./api";
import { compactId } from "./format";
import type { DashboardMemoryDeleteCommand } from "./types";

export function MemoryDeleteControl({
  memoryId,
  enabled,
}: {
  memoryId: string;
  enabled: boolean;
}) {
  const queryClient = useQueryClient();
  const [confirmation, setConfirmation] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const suffix = memoryId.slice(-8);
  const command = useMemo<DashboardMemoryDeleteCommand>(
    () => ({
      schema_version: 1,
      request_id: `memory-delete:${memoryId}`,
      confirm_memory_id: memoryId,
    }),
    [memoryId],
  );
  const mutation = useMutation({
    mutationFn: () => deleteMemory(memoryId, command),
    retry: false,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["dashboard", "memory"] });
    },
  });

  if (!enabled) {
    return (
      <div className="mutation-disabled memory-delete-boundary" role="status">
        <LockKeyhole size={16} />
        <div><strong>Deletion is disabled</strong><span>Audit remains available. Restart with <code>--enable-memory-mutations</code> to delete.</span></div>
      </div>
    );
  }

  const confirmed = confirmation === suffix;
  const beginDelete = () => {
    if (!confirmed || mutation.isPending || mutation.isSuccess) return;
    setSubmitted(true);
    mutation.mutate();
  };
  const unlock = () => {
    mutation.reset();
    setSubmitted(false);
    setConfirmation("");
  };

  return (
    <div className="memory-delete-control">
      <div className="memory-delete-control__heading">
        <span><ShieldAlert size={15} /></span>
        <div><strong>Delete this memory</strong><small>Creates an append-only tombstone; content cannot be restored here.</small></div>
      </div>
      {!submitted ? (
        <>
          <label>
            <span>Type <code>{suffix}</code> to confirm the exact identity</span>
            <input
              value={confirmation}
              maxLength={8}
              spellCheck={false}
              autoComplete="off"
              onChange={(event) => setConfirmation(event.target.value.trim().toLocaleLowerCase())}
              aria-label={`Type ${suffix} to confirm memory deletion`}
            />
          </label>
          <button type="button" disabled={!confirmed} onClick={beginDelete}>
            <Trash2 size={14} />Delete memory
          </button>
        </>
      ) : null}
      {mutation.isPending ? <p className="mutation-status" role="status">Writing tombstone and audit event…</p> : null}
      {mutation.isError ? (
        <div className="mutation-result is-error" role="alert">
          <strong>Deletion not confirmed</strong>
          <span>{mutationErrorMessage(mutation.error)}</span>
          <div><button type="button" onClick={() => mutation.mutate()}><RotateCcw size={12} />Retry exact request</button><button type="button" onClick={unlock}>Cancel</button></div>
        </div>
      ) : null}
      {mutation.data ? (
        <div className="mutation-result is-success" role="status" aria-live="polite">
          <strong>{mutation.data.idempotent_replay ? "Existing tombstone recovered" : "Memory deleted"}</strong>
          <span>Tombstone {compactId(mutation.data.tombstone.tombstone_id, 10)} is now the durable record.</span>
        </div>
      ) : null}
    </div>
  );
}

function mutationErrorMessage(error: Error): string {
  return error instanceof DashboardApiError
    ? error.problem?.detail ?? error.message
    : "The local request ended without a confirmed tombstone.";
}
