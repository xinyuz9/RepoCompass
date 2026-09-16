/** Read-only SSE observer. Reconnect only at declared bounds, never on errors. */
export type StreamState = "connecting" | "live" | "paused" | "terminal" | "limited" | "error";
export interface EventConnection {
  addEventListener(name: string, listener: (event: MessageEvent) => void): void;
  close(): void;
}

export function observeRunEvents(options: {
  runId: string;
  create: (url: string) => EventConnection;
  changed: () => void;
  state: (state: StreamState) => void;
  maxSessions?: number;
}) {
  let cursor = -1;
  let sessions = 0;
  let connection: EventConnection | null = null;
  let stopped = false;
  let paused = false;
  const finish = (state: StreamState) => {
    connection?.close();
    connection = null;
    options.state(state);
  };
  const connect = () => {
    if (stopped || paused) return;
    if (sessions >= (options.maxSessions ?? 6)) { finish("limited"); stopped = true; return; }
    sessions += 1;
    options.state("connecting");
    let source: EventConnection;
    try {
      source = options.create(`/api/v1/runs/${encodeURIComponent(options.runId)}/events?follow=true&max_events=250&after_sequence=${cursor}`);
    } catch { stopped = true; finish("error"); return; }
    connection = source;
    const current = () => connection === source && !stopped && !paused;
    source.addEventListener("open", () => { if (current()) options.state("live"); });
    source.addEventListener("trace", (event) => {
      if (!current()) return;
      const sequence = /^\d+$/.test(event.lastEventId) ? Number(event.lastEventId) : NaN;
      if (!Number.isSafeInteger(sequence) || sequence !== cursor + 1) {
        stopped = true; finish("error"); return;
      }
      cursor = sequence;
      options.changed();
    });
    source.addEventListener("stream_end", (event) => {
      if (!current()) return;
      let end: { reason?: string; after_sequence?: number };
      try { end = JSON.parse(event.data) as typeof end; }
      catch { stopped = true; finish("error"); return; }
      if (!end || end.after_sequence !== cursor) { stopped = true; finish("error"); return; }
      source.close(); connection = null;
      if (end.reason === "deadline" || end.reason === "event_limit") { connect(); return; }
      stopped = true;
      finish(end.reason === "terminal" ? "terminal" : "limited");
      options.changed();
    });
    const error = () => { if (current()) { stopped = true; finish("error"); } };
    source.addEventListener("error", error);
    source.addEventListener("stream_error", error);
  };
  connect();
  return {
    close() { stopped = true; connection?.close(); connection = null; },
    pause() { if (!stopped) { paused = true; finish("paused"); } },
    resume() { if (paused && !stopped) { paused = false; connect(); } },
  };
}
