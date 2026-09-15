import type { ExecutionState, ThreadInvestigationView } from "./types";

export function investigationStateLabel(thread: {
  execution_state: ExecutionState | null;
  termination_status: ThreadInvestigationView["termination_status"];
}) {
  if (thread.execution_state === "completed"
    && (thread.termination_status === "cancelled" || thread.termination_status === "exhausted")) {
    return thread.termination_status;
  }
  return thread.execution_state ?? "unavailable";
}

export function compactId(value: string, width = 8): string {
  return value.length <= width ? value : value.slice(0, width);
}

export function formatDuration(milliseconds: number): string {
  if (milliseconds < 1_000) return `${milliseconds} ms`;
  if (milliseconds < 60_000) return `${(milliseconds / 1_000).toFixed(2)} s`;
  const minutes = Math.floor(milliseconds / 60_000);
  const seconds = Math.round((milliseconds % 60_000) / 1_000);
  return `${minutes}m ${seconds}s`;
}

export function formatTokens(value: number | null | undefined): string {
  if (value === null || value === undefined) return "unknown";
  return new Intl.NumberFormat("en", { notation: value >= 10_000 ? "compact" : "standard" }).format(value);
}

export function formatTimestamp(milliseconds: number): string {
  const timestamp = new Date(milliseconds);
  if (Number.isNaN(timestamp.getTime())) return `${milliseconds} ms since epoch`;
  return new Intl.DateTimeFormat("en-AU", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(timestamp);
}

export function formatUsd(value: string | null | undefined): string {
  if (value === null || value === undefined) return "unavailable";
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0) return "unavailable";
  if (numeric === 0) return "$0";
  if (numeric < 0.01) return `$${value}`;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 6,
  }).format(numeric);
}

export function humanize(value: string): string {
  return value.replaceAll("_", " ");
}
