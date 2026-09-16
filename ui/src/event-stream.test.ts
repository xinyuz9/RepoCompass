import { expect, it, vi } from "vitest";
import { observeRunEvents, type EventConnection } from "./event-stream";

class FakeSource implements EventConnection {
  handlers = new Map<string, (event: MessageEvent) => void>();
  close = vi.fn();
  addEventListener(name: string, callback: (event: MessageEvent) => void) { this.handlers.set(name, callback); }
  emit(name: string, data = "", id = "") { this.handlers.get(name)?.({ data, lastEventId: id } as MessageEvent); }
}
function setup(maxSessions = 6) {
  const sources: FakeSource[] = [];
  const create = vi.fn(() => { const source = new FakeSource(); sources.push(source); return source; });
  const changed = vi.fn(); const state = vi.fn();
  const observer = observeRunEvents({ runId: "a/b", create, changed, state, maxSessions });
  return { sources, create, changed, state, observer };
}

it("resumes only after declared limits with the exact last received cursor", () => {
  const x = setup(2);
  expect(x.create.mock.calls[0]).toEqual([expect.stringContaining("a%2Fb/events?follow=true")]);
  x.sources[0]!.emit("trace", "{}", "0");
  x.sources[0]!.emit("stream_end", JSON.stringify({ reason: "deadline", after_sequence: 0 }));
  expect(x.sources[0]!.close).toHaveBeenCalledOnce();
  expect(x.create.mock.calls[1]).toEqual([expect.stringContaining("after_sequence=0")]);
  x.sources[1]!.emit("trace", "{}", "1");
  x.sources[1]!.emit("stream_end", JSON.stringify({ reason: "deadline", after_sequence: 1 }));
  expect(x.create).toHaveBeenCalledTimes(2);
  expect(x.state).toHaveBeenLastCalledWith("limited");
  expect(x.changed).toHaveBeenCalledTimes(2);
});

it.each(["error", "stream_error"])("closes %s without the browser's implicit retry", (event) => {
  const x = setup(); x.sources[0]!.emit(event);
  expect(x.sources[0]!.close).toHaveBeenCalledOnce();
  x.observer.pause(); x.observer.resume();
  expect(x.create).toHaveBeenCalledOnce();
  expect(x.state).toHaveBeenLastCalledWith("error");
});

it.each(["2", "-1", "nan", "9007199254740992"])("rejects invalid or gapped cursor %s", (id) => {
  const x = setup(); x.sources[0]!.emit("trace", "{}", id);
  expect(x.changed).not.toHaveBeenCalled();
  expect(x.state).toHaveBeenLastCalledWith("error");
});

it("stops after terminal, byte bounds or malformed completion without reconnect", () => {
  for (const data of ['{"reason":"terminal","after_sequence":-1}', '{"reason":"byte_limit","after_sequence":-1}', "null", '{"reason":"deadline","after_sequence":9}', "not-json"]) {
    const x = setup(); x.sources[0]!.emit("stream_end", data);
    expect(x.sources[0]!.close).toHaveBeenCalled();
    expect(x.create).toHaveBeenCalledOnce();
  }
});

it("pauses in the background and ignores stale events after close", () => {
  const x = setup(); x.sources[0]!.emit("trace", "{}", "0");
  x.observer.pause(); x.sources[0]!.emit("trace", "{}", "1");
  expect(x.changed).toHaveBeenCalledOnce();
  x.observer.resume();
  expect(x.create.mock.calls[1]).toEqual([expect.stringContaining("after_sequence=0")]);
  x.observer.close(); x.sources[1]!.emit("trace", "{}", "1");
  expect(x.changed).toHaveBeenCalledOnce();
});
