import { afterEach, describe, expect, it } from "vitest";

import {
  dashboardNavigationHref,
  readDashboardNavigation,
  writeDashboardNavigation,
} from "./url-state";

afterEach(() => window.history.replaceState(null, "", "/"));

describe("dashboard URL state", () => {
  it("parses a safe thread, inspector, and persisted run deep link", () => {
    expect(readDashboardNavigation(
      "?thread=thread-1&lens=context&run=run-2",
    )).toEqual({
      threadId: "thread-1",
      lens: "context",
      runId: "run-2",
      composingNew: false,
    });
  });

  it("rejects path-like identifiers and ignores run ids outside run lenses", () => {
    expect(readDashboardNavigation(
      "?thread=../../etc/passwd&lens=trace&run=%2Ftmp%2Frun",
    )).toEqual({
      threadId: null,
      lens: "trace",
      runId: null,
      composingNew: false,
    });
    expect(readDashboardNavigation(
      "?thread=thread-1&lens=plan&run=run-2",
    ).runId).toBeNull();
  });

  it("makes goal composition exclusive of persisted thread state", () => {
    expect(readDashboardNavigation(
      "?compose=1&thread=thread-1&lens=cost&run=run-1",
    )).toEqual({
      threadId: null,
      lens: "chat",
      runId: null,
      composingNew: true,
    });
  });

  it("writes only bounded navigation state while preserving unrelated local flags", () => {
    const href = dashboardNavigationHref({
      threadId: "thread-1",
      lens: "trace",
      runId: "run-2",
      composingNew: false,
    }, "http://127.0.0.1:8766/?debug=1&compose=1#event-2");
    expect(href).toBe(
      "/?debug=1&thread=thread-1&lens=trace&run=run-2#event-2",
    );

    window.history.replaceState(null, "", "/?debug=1");
    writeDashboardNavigation({
      threadId: null,
      lens: "chat",
      runId: null,
      composingNew: true,
    });
    expect(window.location.search).toBe("?debug=1&compose=1");
  });
});
