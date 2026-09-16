import type { InspectorLens } from "./ConversationWorkspace";

export type DashboardLens = "chat" | InspectorLens;

export interface DashboardNavigationState {
  threadId: string | null;
  lens: DashboardLens;
  runId: string | null;
  composingNew: boolean;
  nodeId?: string;
  splitView?: boolean;
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
  const nodeId = params.get("node");
  return {
    threadId, lens, runId, composingNew: false,
    ...(threadId && lens === "topology" && nodeId && /^[a-f0-9]{64}$/.test(nodeId) ? { nodeId } : {}),
    ...(threadId && lens !== "chat" && params.get("split") === "1" ? { splitView: true } : {}),
  };
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
  url.searchParams.delete("node");
  url.searchParams.delete("split");

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
    if (state.threadId && SAFE_ID.test(state.threadId) && state.lens !== "chat") {
      if (state.splitView) url.searchParams.set("split", "1");
      if (state.lens === "topology" && state.nodeId && /^[a-f0-9]{64}$/.test(state.nodeId)) {
        url.searchParams.set("node", state.nodeId);
      }
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
