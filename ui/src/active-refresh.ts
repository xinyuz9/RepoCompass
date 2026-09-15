import type { CancellationRecord, RawRunAttempt } from "./types";

export function cancellationRefreshInterval(state: CancellationRecord["state"] | undefined, failed: boolean, snapshots: number) {
  return state === "cancel_requested" && !failed && snapshots < 120 ? 1_500 : false;
}

/** Read-only polling pauses on terminal/error/120 snapshots, never resubmits work. */
export function activeRefreshInterval(
  status: RawRunAttempt["status"] | undefined,
  failed: boolean,
  snapshots: number,
): number | false {
  return !failed && status === "running" && snapshots < 120 ? 1_500 : false;
}
