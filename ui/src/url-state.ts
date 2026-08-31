import type { InspectorLens } from "./ConversationWorkspace";

export type DashboardLens = "chat" | InspectorLens;

export interface DashboardNavigationState {
  threadId: string | null;
  lens: DashboardLens;
  runId: string | null;
  composingNew: boolean;
}

export type DashboardNavigationMode = "push" | "replace";

const LENSES: ReadonlySet<string> = new Set([
  "chat",
  "topology",
  "plan",
  "trace",
  "context",
  "memory",
  "cost",
]);
const RUN_LENSES: ReadonlySet<string> = new Set([
  "trace",
  "context",
  "memory",
  "cost",
]);
const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/;

export function readDashboardNavigation(
  search = typeof window === "undefined" ? "" : window.location.search,
): DashboardNavigationState {
  const params = new URLSearchParams(search);
  if (params.get("compose") === "1") {
    return { threadId: null, lens: "chat", runId: null, composingNew: true };
  }
  const threadId = safeId(params.get("thread"));
  const lensValue = params.get("lens");
  const lens = isDashboardLens(lensValue) ? lensValue : "chat";
  const runId = threadId !== null && isRunLens(lens)
    ? safeId(params.get("run"))
    : null;
  return { threadId, lens, runId, composingNew: false };
}

export function writeDashboardNavigation(
  state: DashboardNavigationState,
  mode: DashboardNavigationMode = "push",
): void {
  if (typeof window === "undefined") return;
  const next = dashboardNavigationHref(state, window.location.href);
  const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (next === current) return;
  if (mode === "replace") {
    window.history.replaceState(null, "", next);
  } else {
    window.history.pushState(null, "", next);
  }
}

export function dashboardNavigationHref(
  state: DashboardNavigationState,
  currentHref: string,
): string {
  const url = new URL(currentHref);
  url.searchParams.delete("thread");
  url.searchParams.delete("lens");
  url.searchParams.delete("run");
  url.searchParams.delete("compose");

  if (state.composingNew) {
    url.searchParams.set("compose", "1");
  } else {
    if (state.threadId && SAFE_ID.test(state.threadId)) {
      url.searchParams.set("thread", state.threadId);
    }
    if (state.lens !== "chat") url.searchParams.set("lens", state.lens);
    if (state.runId && isRunLens(state.lens) && SAFE_ID.test(state.runId)) {
      url.searchParams.set("run", state.runId);
    }
  }
  return `${url.pathname}${url.search}${url.hash}`;
}

export function isRunLens(lens: DashboardLens): boolean {
  return RUN_LENSES.has(lens);
}

function isDashboardLens(value: string | null): value is DashboardLens {
  return value !== null && LENSES.has(value);
}

function safeId(value: string | null): string | null {
  return value !== null && SAFE_ID.test(value) ? value : null;
}
