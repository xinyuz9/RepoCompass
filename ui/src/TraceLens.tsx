import { Activity, Braces, CheckCircle2, PauseCircle, PlayCircle, XCircle } from "lucide-react";
import { useMemo, useState } from "react";

import { compactId, humanize } from "./format";
import { PanelEmpty, PanelHeading } from "./PlanLens";
import type { DashboardRunDetail, PublicTraceEvent } from "./types";

type TraceFilter = "all" | PublicTraceEvent["kind"];

export function TraceLens({ run }: { run: DashboardRunDetail }) {
  const [filter, setFilter] = useState<TraceFilter>("all");
  const visible = useMemo(
    () => run.trace_events.filter((event) => filter === "all" || event.kind === filter),
    [filter, run.trace_events],
  );
  const [requestedSequence, setRequestedSequence] = useState<number | null>(null);
  const selected =
    visible.find((event) => event.sequence === requestedSequence) ?? visible.at(-1) ?? null;

  if (run.trace_availability !== "available") {
    return <PanelSurface><PanelEmpty icon={<Activity size={20} />} title="Trace not persisted" detail="This attempt has metadata but no committed event journal or JSONL export." /></PanelSurface>;
  }

  return (
    <section className="observability-grid observability-grid--trace">
      <div className="panel panel--trace-list">
        <PanelHeading eyebrow="append-only flight recorder" title="Run trace" detail={`${run.trace_events.length} events · ${compactId(run.attempt.run_id)}`} />
        <div className="segmented-control" aria-label="Trace event filter">
          {(["all", "run_started", "node_completed", "run_interrupted", "run_finished", "run_failed"] as const).map((kind) => (
            <button key={kind} type="button" className={filter === kind ? "is-active" : ""} onClick={() => setFilter(kind)}>{kind === "all" ? "all" : humanize(kind)}</button>
          ))}
        </div>
        {visible.length ? (
          <ol className="trace-list">
            {visible.map((event) => (
              <li key={event.sequence}>
                <button type="button" className={selected?.sequence === event.sequence ? "is-selected" : ""} onClick={() => setRequestedSequence(event.sequence)}>
                  <span className={`trace-kind trace-kind--${event.kind}`}>{traceIcon(event.kind)}</span>
                  <span><strong>{event.node ?? humanize(event.kind)}</strong><small>{humanize(event.kind)} · {Object.keys(event.payload).length} public fields</small></span>
                  <code>#{event.sequence}</code>
                </button>
              </li>
            ))}
          </ol>
        ) : (
          <PanelEmpty icon={<Activity size={20} />} title="No matching events" detail="Choose another event category to continue the replay." />
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
              <pre>{JSON.stringify(selected.payload, null, 2)}</pre>
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
