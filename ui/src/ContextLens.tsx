import { Box, BrainCircuit, FileSearch, Layers3, ShieldCheck, XCircle } from "lucide-react";
import { useMemo, useState } from "react";

import { compactId, formatTokens, humanize } from "./format";
import { contextSectionTotals } from "./observability-model";
import { PanelEmpty, PanelHeading } from "./PlanLens";
import type { ContextPack, DashboardRunDetail, ModelInvocationRecord } from "./types";

function latestInvocation(invocations: ModelInvocationRecord[]) {
  return invocations.reduce<ModelInvocationRecord | null>(
    (latest, item) => latest === null || item.sequence > latest.sequence ? item : latest,
    null,
  );
}

export function ContextLens({ run }: { run: DashboardRunDetail }) {
  const [requestedPack, setRequestedPack] = useState<string | null>(null);
  const latest = latestInvocation(run.model_invocations);
  const pack = run.context_packs.find((item) => item.pack_id === requestedPack)
    ?? run.context_packs.find((item) => item.pack_id === latest?.context_pack_id)
    ?? run.context_packs.at(-1) ?? null;

  if (!pack) {
    return <section className="panel panel--standalone"><PanelEmpty icon={<Layers3 size={20} />} title="No ContextPack persisted" detail="The attempt has no active-context build artifact. Raw model inputs remain intentionally unavailable here." /></section>;
  }
  return <ContextPackView run={run} pack={pack} onPackChange={setRequestedPack} />;
}

function ContextPackView({ run, pack, onPackChange }: { run: DashboardRunDetail; pack: ContextPack; onPackChange: (id: string) => void }) {
  const decisions = pack.build_report.decisions;
  const [requestedItem, setRequestedItem] = useState<string | null>(null);
  const selectedDecision = decisions.find((item) => item.item_id === requestedItem) ?? decisions[0] ?? null;
  const selectedItem = pack.items.find((item) => item.item_id === selectedDecision?.item_id) ?? null;
  const sectionTotals = useMemo(() => contextSectionTotals(pack), [pack]);
  const utilization = pack.build_report.budget_tokens
    ? Math.min(100, (pack.estimated_tokens / pack.build_report.budget_tokens) * 100)
    : 0;
  const consumers = run.model_invocations.filter((item) => item.context_pack_id === pack.pack_id);
  const invocation = latestInvocation(consumers);
  const memoryItems = pack.items.filter((item) => item.source.memory !== null);

  return (
    <section className="observability-grid observability-grid--context">
      <div className="panel panel--context-summary">
        <PanelHeading eyebrow="hard-budget assembly" title="Context inspector" detail={`${pack.items.length} selected · ${pack.build_report.excluded_count} excluded`} />
        <label className="select-label">Context snapshot
          <select value={pack.pack_id} onChange={(event) => onPackChange(event.target.value)}>
            {run.context_packs.map((item, index) => <option key={item.pack_id} value={item.pack_id}>snapshot {index + 1} · {compactId(item.pack_id)}</option>)}
          </select>
        </label>
        <div className="budget-card">
          <div><span>active budget</span><strong>{formatTokens(pack.estimated_tokens)} <small>/ {formatTokens(pack.build_report.budget_tokens)} tokens</small></strong></div>
          <div className="budget-track" role="meter" aria-valuemin={0} aria-valuemax={pack.build_report.budget_tokens} aria-valuenow={pack.estimated_tokens}><i style={{ width: `${utilization}%` }} /></div>
          <footer><span>{utilization.toFixed(1)}% occupied</span><span>{pack.build_report.candidate_count} candidates audited</span></footer>
        </div>
        {invocation ? <div className="context-consumer"><Box size={16} /><div><span>used by {consumers.length} invocation{consumers.length === 1 ? "" : "s"} · latest consumer</span><strong>{invocation.node_name} · {invocation.role}</strong></div><code>#{invocation.sequence}</code></div> : null}
        <div className="section-ledger">
          {sectionTotals.map((section) => (
            <div key={section.section}><span>{humanize(section.section)}</span><strong>{formatTokens(section.tokens)}</strong><small>{section.itemCount} item{section.itemCount === 1 ? "" : "s"}</small></div>
          ))}
        </div>
        <div className="context-boundary"><ShieldCheck size={16} /><p><strong>Raw / active separation</strong>Only bounded selected content is visible. Artifact bodies and model messages are not exposed. Snapshots may be shared; each invocation is counted separately.</p></div>
      </div>

      <div className="panel panel--decision-ledger">
        <PanelHeading eyebrow="complete decision ledger" title="Selection reasons" detail={`${decisions.length} deterministic decisions`} />
        <div className="decision-list">
          {decisions.map((decision) => (
            <button key={decision.item_id} type="button" className={`${selectedDecision?.item_id === decision.item_id ? "is-selected" : ""} is-${decision.outcome}`} onClick={() => setRequestedItem(decision.item_id)}>
              <span>{decision.outcome === "selected" ? <FileSearch size={16} /> : <XCircle size={16} />}</span>
              <span><strong>{humanize(decision.reason)}</strong><small>{compactId(decision.item_id)} · {formatTokens(decision.estimated_tokens)} tokens</small></span>
              <i>{decision.selected_position === null ? "out" : `#${decision.selected_position + 1}`}</i>
            </button>
          ))}
        </div>
      </div>

      <div className="panel panel--context-item">
        <PanelHeading eyebrow="bounded active content" title={selectedItem ? humanize(selectedItem.section) : "Excluded candidate"} detail={selectedItem ? `${humanize(selectedItem.verification_status)} · priority ${selectedItem.priority}` : selectedDecision ? humanize(selectedDecision.reason) : "no selection"} />
        {selectedItem ? (
          <>
            <div className="item-provenance">
              <span>{humanize(selectedItem.source.kind)}</span>
              {selectedItem.source.location ? <code>{selectedItem.source.location.path}{selectedItem.source.location.line_start ? `:${selectedItem.source.location.line_start}` : ""}</code> : <code>{compactId(selectedItem.source.source_id, 12)}</code>}
            </div>
            <pre className="context-content">{selectedItem.content}</pre>
            <div className="provenance-links">{selectedItem.source.provenance.map((link) => <span key={`${link.kind}:${link.id}`}>{humanize(link.kind)} · {compactId(link.id)}</span>)}</div>
          </>
        ) : selectedDecision ? (
          <PanelEmpty icon={<XCircle size={20} />} title="Candidate excluded" detail={`No content body entered the active pack. Reason: ${humanize(selectedDecision.reason)}.`} />
        ) : null}
      </div>

      <div className="panel panel--memory-hits">
        <PanelHeading eyebrow="recalled knowledge" title="Memory hits" detail={`${memoryItems.length} selected hit${memoryItems.length === 1 ? "" : "s"}`} />
        {memoryItems.length ? memoryItems.map((item) => (
          <article key={item.item_id}>
            <BrainCircuit size={17} />
            <div><strong>{humanize(item.source.memory?.scope ?? "memory")}</strong><span>{humanize(item.source.memory?.freshness ?? "unknown")}</span><code>{compactId(item.source.memory?.memory_id ?? "")}</code></div>
          </article>
        )) : <PanelEmpty icon={<BrainCircuit size={20} />} title="No memory selected" detail="This ContextPack was built without repository or episodic memory." />}
      </div>
    </section>
  );
}
