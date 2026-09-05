import type {
  DashboardContinueCommand,
  DashboardHumanResponseCommand,
  DashboardResumeCommand,
  DashboardStartCommand,
} from "./types";

const STORAGE_KEY = "repocompass.pending-lifecycle.v1";
const CHANGE_EVENT = "repocompass:lifecycle-change";
const MAX_PENDING_REQUESTS = 8;
const MAX_STORAGE_BYTES = 131_072;
const REQUEST_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const THREAD_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/;
const REPOSITORY_ID = /^[0-9a-f]{64}$/;
const COMMIT_SHA = /^[0-9a-f]{40}$/;
const MAX_PROMPT_LENGTH = 16_384;
const MAX_PATH_LENGTH = 2_048;

export type LifecycleRequestDraft =
  | { kind: "start"; thread_id: null; command: DashboardStartCommand }
  | { kind: "resume"; thread_id: string; command: DashboardResumeCommand }
  | { kind: "continue"; thread_id: string; command: DashboardContinueCommand }
  | { kind: "respond"; thread_id: string; command: DashboardHumanResponseCommand };

export type StoredLifecycleRequest = LifecycleRequestDraft & {
  schema_version: 1;
  recorded_at_ms: number;
};

export function rememberLifecycleRequest(
  draft: LifecycleRequestDraft,
): StoredLifecycleRequest {
  const record = {
    ...draft,
    schema_version: 1 as const,
    recorded_at_ms: Date.now(),
  };
  const existing = readLifecycleRequests().filter(
    (item) => item.command.request_id !== record.command.request_id,
  );
  writeLifecycleRequests([...existing, record].slice(-MAX_PENDING_REQUESTS));
  return record;
}

export function forgetLifecycleRequest(requestId: string): void {
  if (!REQUEST_ID.test(requestId)) return;
  writeLifecycleRequests(
    readLifecycleRequests().filter(
      (item) => item.command.request_id !== requestId,
    ),
  );
}

export function readLifecycleRequests(): StoredLifecycleRequest[] {
  if (typeof window === "undefined") return [];
  let encoded: string | null;
  try {
    encoded = window.sessionStorage.getItem(STORAGE_KEY);
  } catch {
    return [];
  }
  if (!encoded) return [];
  if (new TextEncoder().encode(encoded).length > MAX_STORAGE_BYTES) {
    clearInvalidStorage();
    return [];
  }
  try {
    const value: unknown = JSON.parse(encoded);
    if (!Array.isArray(value)) {
      clearInvalidStorage();
      return [];
    }
    return value.filter(isStoredLifecycleRequest).slice(-MAX_PENDING_REQUESTS);
  } catch {
    clearInvalidStorage();
    return [];
  }
}

export function lifecycleRequestChangeEvent(): string {
  return CHANGE_EVENT;
}

function writeLifecycleRequests(records: StoredLifecycleRequest[]): void {
  if (typeof window === "undefined") return;
  try {
    if (records.length === 0) {
      window.sessionStorage.removeItem(STORAGE_KEY);
    } else {
      const bounded = records.slice(-MAX_PENDING_REQUESTS);
      let encoded = JSON.stringify(bounded);
      // Keep the newest recoverable request even when multibyte goals fill the cap.
      while (new TextEncoder().encode(encoded).length > MAX_STORAGE_BYTES) {
        bounded.shift();
        encoded = JSON.stringify(bounded);
      }
      window.sessionStorage.setItem(STORAGE_KEY, encoded);
    }
    window.dispatchEvent(new Event(CHANGE_EVENT));
  } catch {
    // Recovery tracking is best-effort; backend idempotency remains authoritative.
  }
}

function clearInvalidStorage(): void {
  try {
    window.sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // Ignore unavailable browser storage.
  }
}

function isStoredLifecycleRequest(value: unknown): value is StoredLifecycleRequest {
  if (!isRecord(value) || value.schema_version !== 1) return false;
  if (
    typeof value.recorded_at_ms !== "number"
    || !Number.isSafeInteger(value.recorded_at_ms)
    || value.recorded_at_ms < 0
    || !isRecord(value.command)
    || value.command.schema_version !== 1
    || typeof value.command.request_id !== "string"
    || !REQUEST_ID.test(value.command.request_id)
  ) {
    return false;
  }
  if (value.kind === "start") {
    return value.thread_id === null && isStartCommand(value.command);
  }
  if (typeof value.thread_id !== "string" || !THREAD_ID.test(value.thread_id)) {
    return false;
  }
  if (value.kind === "resume") return isResumeCommand(value.command);
  if (value.kind === "continue") return isContinueCommand(value.command);
  if (value.kind === "respond") return isHumanResponseCommand(value.command);
  return false;
}

function isStartCommand(command: Record<string, unknown>): boolean {
  return (
    isBoundedString(command.goal, 1, MAX_PROMPT_LENGTH)
    && isExpectedSnapshot(command)
    && isBoundedString(command.focus_path, 1, MAX_PATH_LENGTH)
    && (command.human_review_policy === "never"
      || command.human_review_policy === "before_finalize")
    && typeof command.memory_retrieval_enabled === "boolean"
    && (command.max_attempt_cost_usd === null
      || isBoundedString(command.max_attempt_cost_usd, 1, 64))
  );
}

function isResumeCommand(command: Record<string, unknown>): boolean {
  return isExpectedSnapshot(command);
}

function isContinueCommand(command: Record<string, unknown>): boolean {
  return (
    isBoundedString(command.prompt, 1, MAX_PROMPT_LENGTH)
    && isBoundedString(command.focus_path, 1, MAX_PATH_LENGTH)
    && isExpectedSnapshot(command)
  );
}

function isHumanResponseCommand(command: Record<string, unknown>): boolean {
  if (
    typeof command.interrupt_id !== "string"
    || !THREAD_ID.test(command.interrupt_id)
    || !isRecord(command.decision)
  ) {
    return false;
  }
  const action = command.decision.action;
  const feedback = command.decision.feedback;
  return (
    (action === "approve" || action === "revise" || action === "cancel")
    && (feedback === null || isBoundedString(feedback, 0, MAX_PROMPT_LENGTH))
  );
}

function isExpectedSnapshot(command: Record<string, unknown>): boolean {
  return (
    typeof command.expected_repository_id === "string"
    && REPOSITORY_ID.test(command.expected_repository_id)
    && (command.expected_commit_sha === null
      || (typeof command.expected_commit_sha === "string"
        && COMMIT_SHA.test(command.expected_commit_sha)))
  );
}

function isBoundedString(
  value: unknown,
  minimumLength: number,
  maximumLength: number,
): value is string {
  return (
    typeof value === "string"
    && Array.from(value).length >= minimumLength
    && Array.from(value).length <= maximumLength
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
