import { Activity, Braces, CheckCircle2, PauseCircle, PlayCircle, XCircle } from "lucide-react";
import { useLayoutEffect, useMemo, useRef, useState } from "react";

import { compactId, humanize } from "./format";
import { PanelEmpty, PanelHeading } from "./PlanLens";
import type { DashboardRunDetail, PublicTraceEvent } from "./types";

type TraceFilter = "all" | PublicTraceEvent["kind"];
const TRACE_PAGE_SIZE = 50;

export function TraceLens({ run }: { run: DashboardRunDetail }) {
  return <TraceContent key={run.attempt.run_id} run={run} />;
}

function TraceContent({ run }: { run: DashboardRunDetail }) {
  const [filter, setFilter] = useState<TraceFilter>("all");
  const [query, setQuery] = useState("");
  const [requestedPage, setRequestedPage] = useState<number | null>(null);
  const visible = useMemo(
    () => run.trace_events.filter((event) =>
      (filter === "all" || event.kind === filter) &&
      `${event.node ?? ""} ${event.kind} #${event.sequence}`.toLowerCase().includes(query.trim().toLowerCase()),
    ),
    [filter, query, run.trace_events],
  );
  const lastPage = Math.max(0, Math.ceil(visible.length / TRACE_PAGE_SIZE) - 1);
  const page = Math.min(requestedPage ?? lastPage, lastPage);
  const pageStart = page * TRACE_PAGE_SIZE;
  const pageEvents = visible.slice(pageStart, pageStart + TRACE_PAGE_SIZE);
  const [requestedSequence, setRequestedSequence] = useState<number | null>(null);
  const selected =
    pageEvents.find((event) => event.sequence === requestedSequence) ?? pageEvents.at(-1) ?? null;
  const eventsRef = useRef<HTMLOListElement>(null);
  useLayoutEffect(() => {
    const list = eventsRef.current;
    const button = list?.querySelector<HTMLButtonElement>('button[aria-pressed="true"]');
    if (!list || !button) return;
    const bounds = list.getBoundingClientRect();
    const row = button.getBoundingClientRect();
    // Scroll only the event list, never move the surrounding workspace.
    if (row.bottom > bounds.bottom) list.scrollTop += row.bottom - bounds.bottom;
    else if (row.top < bounds.top) list.scrollTop += row.top - bounds.top;
  }, [selected?.sequence, page, visible.length]);

  if (run.trace_availability !== "available") {
    return <PanelSurface><PanelEmpty icon={<Activity size={20} />} title="Trace not persisted" detail="This attempt has metadata but no committed event journal or JSONL export." /></PanelSurface>;
  }

  return (
    <section className="observability-grid observability-grid--trace">
      <div className="panel panel--trace-list">
        <PanelHeading eyebrow="append-only flight recorder" title="Run trace" detail={`${run.trace_events.length} events · ${compactId(run.attempt.run_id)}`} />
        <div className="segmented-control" aria-label="Trace event filter">
          {(["all", "run_started", "node_completed", "run_interrupted", "run_finished", "run_failed"] as const).map((kind) => (
            <button key={kind} type="button" aria-pressed={filter === kind} className={filter === kind ? "is-active" : ""} onClick={() => { setFilter(kind); setRequestedPage(null); setRequestedSequence(null); }}>{kind === "all" ? "all" : humanize(kind)}</button>
          ))}
        </div>
        <label className="trace-search">
          <span>Find node, event kind or #sequence</span>
          <input type="search" value={query} maxLength={120} placeholder="evaluate_candidate or #42" onChange={(event) => { setQuery(event.target.value); setRequestedPage(null); setRequestedSequence(null); }} />
        </label>
        <nav className="trace-pagination" aria-label="Trace pages">
          <button type="button" disabled={page === 0} onClick={() => { setRequestedPage(0); setRequestedSequence(null); }}>First page</button>
          <button type="button" disabled={page === 0} onClick={() => { setRequestedPage(page - 1); setRequestedSequence(null); }}>Previous page</button>
          <button type="button" disabled={page === lastPage} onClick={() => { setRequestedPage(page + 1); setRequestedSequence(null); }}>Next page</button>
          <button type="button" aria-pressed={requestedPage === null} onClick={() => { setRequestedPage(null); setRequestedSequence(null); }}>Follow latest</button>
          <span role="status">{visible.length ? `${pageStart + 1}–${pageStart + pageEvents.length}` : "0"} of {visible.length} matching events · {requestedPage === null ? "following latest" : "history page pinned"}</span>
        </nav>
        {visible.length ? (
          <ol ref={eventsRef} className="trace-list" aria-label="Trace events" start={pageStart + 1}>
            {pageEvents.map((event) => (
              <li key={event.sequence}>
                <button type="button" aria-pressed={selected?.sequence === event.sequence} className={selected?.sequence === event.sequence ? "is-selected" : ""} onClick={() => setRequestedSequence(event.sequence)}>
                  <span className={`trace-kind trace-kind--${event.kind}`}>{traceIcon(event.kind)}</span>
                  <span><strong>{event.node ?? humanize(event.kind)}</strong><small>{humanize(event.kind)} · {Object.keys(event.payload).length} public fields</small></span>
                  <code>#{event.sequence}</code>
                </button>
              </li>
            ))}
          </ol>
        ) : (
          <PanelEmpty icon={<Activity size={20} />} title="No matching events" detail="Change the event category or clear the search. Payload bodies are not searched." />
        )}
      </div>

      <div className="panel panel--trace-inspector">
        <PanelHeading eyebrow="sanitized public payload" title={selected?.node ?? (selected ? humanize(selected.kind) : "Event inspector")} detail={selected ? `sequence ${selected.sequence}` : "no selection"} />
        {selected ? (
          <>
            <div className="trace-inspector__meta">
              <span><i /> ordered</span><span><i /> repository-redacted</span><span><i /> secret-filtered</span>
            </div>
            <dl className="key-value-grid">
              <div><dt>event kind</dt><dd>{humanize(selected.kind)}</dd></div>
              <div><dt>run</dt><dd><code>{compactId(selected.run_id, 12)}</code></dd></div>
              <div><dt>node</dt><dd>{selected.node ?? "lifecycle"}</dd></div>
              <div><dt>sequence</dt><dd>{selected.sequence}</dd></div>
            </dl>
            <div className="payload-viewer">
              <header><Braces size={15} /><span>public event payload</span><small>{JSON.stringify(selected.payload).length} chars</small></header>
              <pre key={selected.sequence} tabIndex={0} role="region" aria-label="Public event payload">{JSON.stringify(selected.payload, null, 2)}</pre>
            </div>
          </>
        ) : <PanelEmpty icon={<Braces size={20} />} title="Select an event" detail="The bounded, redacted event payload will appear here." />}
      </div>
    </section>
  );
}

function PanelSurface({ children }: { children: React.ReactNode }) {
  return <section className="panel panel--standalone">{children}</section>;
}

function traceIcon(kind: PublicTraceEvent["kind"]) {
  if (kind === "run_failed") return <XCircle size={17} />;
  if (kind === "run_started") return <PlayCircle size={17} />;
  if (kind === "run_interrupted") return <PauseCircle size={17} />;
  if (kind === "run_finished") return <CheckCircle2 size={17} />;
  return <Activity size={17} />;
}
