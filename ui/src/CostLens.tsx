import { CircleDollarSign, Clock3, Cpu, Gauge, Zap } from "lucide-react";

import { compactId, formatDuration, formatTokens, formatUsd, humanize } from "./format";
import { breakdownWidth } from "./observability-model";
import { PanelEmpty, PanelHeading } from "./PlanLens";
import type { DashboardRunDetail, UsageCostBreakdown } from "./types";

export function CostLens({ run }: { run: DashboardRunDetail }) {
  const report = run.usage_cost_report;
  if (run.cost_availability !== "available" || !report) {
    return <section className="panel panel--standalone"><PanelEmpty icon={<CircleDollarSign size={20} />} title="Pricing not configured" detail="Token and latency telemetry remain visible in persisted invocations, but USD estimates require an operator-selected pricing snapshot." /><InvocationLedger run={run} /></section>;
  }
  const total = report.breakdowns.find((item) => item.dimension === "attempt") ?? report.breakdowns[0];
  if (!total) {
    return <section className="panel panel--standalone"><PanelEmpty icon={<CircleDollarSign size={20} />} title="Cost report is incomplete" detail="The persisted report contains no attempt breakdown and cannot be visualized safely." /></section>;
  }
  const roles = report.breakdowns.filter((item) => item.dimension === "role");
  const nodes = report.breakdowns.filter((item) => item.dimension === "node");

  return (
    <section className="observability-grid observability-grid--cost">
      <div className="panel panel--cost-summary">
        <PanelHeading eyebrow="usage ledger" title="Cost analysis" detail={`${report.pricing_version} · effective ${report.pricing_effective_from}`} />
        <div className="cost-hero">
          <span>estimated attempt cost</span>
          <strong>{formatUsd(total.cost.estimated_cost_usd)}</strong>
          <small>{humanize(total.cost.coverage)} coverage · estimate, not invoice</small>
        </div>
        <div className="cost-metrics">
          <Metric icon={<Cpu size={16} />} label="total tokens" value={formatTokens(total.total_tokens.value)} detail={humanize(total.total_tokens.coverage)} />
          <Metric icon={<Clock3 size={16} />} label="model latency" value={formatDuration(total.latency_ms)} detail={`${total.invocation_count} calls`} />
          <Metric icon={<Zap size={16} />} label="cached input" value={formatTokens(total.cached_input_tokens.value)} detail={humanize(total.cached_input_tokens.coverage)} />
        </div>
        {report.budget ? (
          <div className={`cost-budget cost-budget--${report.budget.status}`}>
            <Gauge size={18} />
            <div><span>attempt budget · {humanize(report.budget.status)}</span><strong>{formatUsd(report.budget.confirmed_cost_usd)} / {formatUsd(report.budget.limit_usd)}</strong><small>{humanize(report.budget.reason)} · next call {report.budget.next_invocation_allowed ? "allowed" : "blocked"}</small></div>
          </div>
        ) : <div className="cost-budget"><Gauge size={18} /><div><span>attempt budget</span><strong>not configured</strong><small>telemetry remains complete without a spend cap</small></div></div>}
      </div>

      <BreakdownChart title="Cost by role" eyebrow="workflow responsibility" items={roles} />
      <BreakdownChart title="Cost by node" eyebrow="execution hotspots" items={nodes} />
      <div className="panel panel--invocations"><PanelHeading eyebrow="per-call evidence" title="Model invocations" detail={`${run.model_invocations.length} persisted calls`} /><InvocationLedger run={run} /></div>
    </section>
  );
}

function BreakdownChart({ title, eyebrow, items }: { title: string; eyebrow: string; items: UsageCostBreakdown[] }) {
  return (
    <div className="panel panel--breakdown">
      <PanelHeading eyebrow={eyebrow} title={title} detail={`${items.length} group${items.length === 1 ? "" : "s"}`} />
      {items.length ? <div className="breakdown-list">{items.map((item) => (
        <article key={`${item.dimension}:${item.key}`}>
          <div><strong>{humanize(item.key)}</strong><span>{formatUsd(item.cost.estimated_cost_usd)}</span></div>
          <div className="breakdown-track"><i style={{ width: `${breakdownWidth(item, items)}%` }} /></div>
          <footer><span>{formatTokens(item.total_tokens.value)} tokens</span><span>{formatDuration(item.latency_ms)}</span><span>{item.invocation_count} calls</span></footer>
        </article>
      ))}</div> : <PanelEmpty icon={<CircleDollarSign size={20} />} title="No grouped cost" detail="No priced model calls contributed to this dimension." />}
    </div>
  );
}

function InvocationLedger({ run }: { run: DashboardRunDetail }) {
  const costs = new Map(run.usage_cost_report?.invocation_costs.map((item) => [item.invocation_id, item]));
  if (!run.model_invocations.length) return <PanelEmpty icon={<Cpu size={20} />} title="No model invocations" detail="This attempt contains no persisted provider-call ledger." />;
  return <div className="invocation-table" role="table" aria-label="Model invocation ledger">
    <div role="row" className="invocation-table__head"><span>node / role</span><span>model</span><span>tokens</span><span>latency</span><span>estimate</span></div>
    {run.model_invocations.map((item) => {
      const estimate = costs.get(item.invocation_id);
      return <div role="row" key={item.invocation_id}>
        <span><strong>{item.node_name}</strong><small>{item.role} · #{item.sequence}</small></span>
        <span><strong>{item.model}</strong><small>{item.provider} · {item.status}</small></span>
        <span>{formatTokens(item.usage.total_tokens)}<small>{humanize(item.usage.availability)}</small></span>
        <span>{formatDuration(item.latency_ms)}<small>{item.plan_step_id ?? "no step"}</small></span>
        <span>{formatUsd(estimate?.estimated_cost_usd)}<small>{estimate ? humanize(estimate.availability) : "not priced"}</small></span>
      </div>;
    })}
    <footer><span>run {compactId(run.attempt.run_id, 12)}</span><span>pricing is operator-configured and versioned</span></footer>
  </div>;
}

function Metric({ icon, label, value, detail }: { icon: React.ReactNode; label: string; value: string; detail: string }) {
  return <div>{icon}<span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}
