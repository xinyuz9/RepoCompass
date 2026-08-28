import { useQuery } from "@tanstack/react-query";
import { BrainCircuit, Database, History, ShieldCheck, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";

import { dashboardQueries } from "./api";
import { compactId, formatTimestamp, humanize } from "./format";
import { PanelEmpty, PanelHeading } from "./PlanLens";
import type { DashboardRunDetail, MemoryEntryMetadata, MemoryTombstone } from "./types";

export function MemoryLens({ run }: { run: DashboardRunDetail }) {
  const catalog = useQuery({ ...dashboardQueries.memory, staleTime: 30_000 });
  const hitIds = useMemo(
    () => new Set(run.context_packs.flatMap((pack) => pack.items.map((item) => item.source.memory?.memory_id).filter((id): id is string => Boolean(id)))),
    [run.context_packs],
  );
  const records = catalog.data ? [...catalog.data.entries, ...catalog.data.tombstones] : [];
  const [requestedId, setRequestedId] = useState<string | null>(null);
  const activeId = records.some((item) => item.memory_id === requestedId) ? requestedId : (records[0]?.memory_id ?? null);
  const audit = useQuery({
    ...dashboardQueries.memoryAudit(activeId ?? ""),
    enabled: activeId !== null,
    staleTime: 30_000,
  });

  if (catalog.isPending) return <MemoryLoading />;
  if (catalog.isError) return <MemoryError detail={catalog.error.message} />;
  if (catalog.data.availability === "not_configured") {
    return <section className="panel panel--standalone"><PanelEmpty icon={<Database size={20} />} title="Memory store not configured" detail="The investigation remains usable; repository and episodic memory are explicitly unavailable for this runtime." /></section>;
  }

  return (
    <section className="observability-grid observability-grid--memory">
      <div className="panel panel--memory-catalog">
        <PanelHeading eyebrow="payload-free catalog" title="Repository memory" detail={`${catalog.data.entries.length} live · ${catalog.data.tombstones.length} deleted`} />
        {records.length ? (
          <div className="memory-list">
            {records.map((record) => {
              const deleted = isTombstone(record);
              return (
                <button key={record.memory_id} type="button" className={activeId === record.memory_id ? "is-selected" : ""} onClick={() => setRequestedId(record.memory_id)}>
                  <span className={deleted ? "memory-icon is-deleted" : "memory-icon"}>{deleted ? <Trash2 size={17} /> : <BrainCircuit size={17} />}</span>
                  <span><strong>{humanize(record.memory_scope)} {deleted ? "tombstone" : "memory"}</strong><small>{deleted ? humanize(record.reason) : humanize(record.verification_status)}</small></span>
                  <code>{compactId(record.memory_id)}</code>
                  {hitIds.has(record.memory_id) ? <i>this run</i> : null}
                </button>
              );
            })}
          </div>
        ) : <PanelEmpty icon={<BrainCircuit size={20} />} title="No memory records" detail="The store is configured, but this repository has no live entries or tombstones." />}
      </div>

      <div className="panel panel--memory-audit">
        <PanelHeading eyebrow="append-only lifecycle" title="Memory audit" detail={activeId ? compactId(activeId, 12) : "no selection"} />
        {activeId && audit.isPending ? <div className="inline-loading"><span /><span /><span /></div> : null}
        {audit.isError ? <MemoryError detail={audit.error.message} /> : null}
        {!activeId ? <PanelEmpty icon={<History size={20} />} title="No memory selected" detail="Choose a live entry or tombstone to replay its payload-free lifecycle." /> : null}
        {audit.data ? (
          <>
            <div className="memory-boundary"><ShieldCheck size={17} /><div><strong>Content stays private</strong><span>Identity, trust, source counts and lifecycle events only.</span></div></div>
            <dl className="key-value-grid">
              <div><dt>scope</dt><dd>{humanize((audit.data.metadata ?? audit.data.tombstone)?.memory_scope ?? "unknown")}</dd></div>
              <div><dt>state</dt><dd>{audit.data.metadata ? humanize(audit.data.metadata.verification_status) : "deleted"}</dd></div>
              <div><dt>events</dt><dd>{audit.data.events.length}</dd></div>
              <div><dt>run hit</dt><dd>{hitIds.has(audit.data.memory_id) ? "selected in context" : "not selected"}</dd></div>
            </dl>
            <div className="audit-timeline">
              {audit.data.events.map((event) => (
                <article key={event.event_id}>
                  <span><History size={15} /></span>
                  <div><strong>{humanize(event.action)}</strong><p>{humanize(event.reason)}</p><small>{formatTimestamp(event.created_at_ms)}{event.run_id ? ` · run ${compactId(event.run_id)}` : ""}</small></div>
                </article>
              ))}
            </div>
          </>
        ) : null}
      </div>
    </section>
  );
}

function isTombstone(record: MemoryEntryMetadata | MemoryTombstone): record is MemoryTombstone {
  return "tombstone_id" in record;
}

function MemoryLoading() {
  return <section className="panel panel--standalone"><div className="inline-loading" aria-label="Loading memory catalog"><span /><span /><span /></div></section>;
}

function MemoryError({ detail }: { detail: string }) {
  return <div className="memory-error" role="alert"><Database size={18} /><div><strong>Memory view unavailable</strong><p>{detail}</p></div></div>;
}
