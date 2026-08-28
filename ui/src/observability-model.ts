import type {
  ContextPack,
  ContextSection,
  PlanReplayFrame,
  UsageCostBreakdown,
} from "./types";

export interface SectionTokenTotal {
  section: ContextSection;
  tokens: number;
  itemCount: number;
}

export function contextSectionTotals(pack: ContextPack): SectionTokenTotal[] {
  const totals = new Map<ContextSection, SectionTokenTotal>();
  for (const item of pack.items) {
    const current = totals.get(item.section) ?? {
      section: item.section,
      tokens: 0,
      itemCount: 0,
    };
    current.tokens += item.estimated_tokens;
    current.itemCount += 1;
    totals.set(item.section, current);
  }
  return [...totals.values()].sort(
    (left, right) => right.tokens - left.tokens || left.section.localeCompare(right.section),
  );
}

export function latestPlanFrame(frames: PlanReplayFrame[]): PlanReplayFrame | null {
  return [...frames].sort(
    (left, right) =>
      right.turn_number - left.turn_number ||
      right.sequence - left.sequence ||
      right.run_id.localeCompare(left.run_id),
  )[0] ?? null;
}

export function breakdownWidth(
  breakdown: UsageCostBreakdown,
  siblings: UsageCostBreakdown[],
): number {
  if (breakdown.cost.estimated_cost_usd === null) return 0;
  const value = Number(breakdown.cost.estimated_cost_usd);
  const maximum = Math.max(
    0,
    ...siblings.map((item) => Number(item.cost.estimated_cost_usd ?? 0)),
  );
  if (!Number.isFinite(value) || maximum <= 0) return 0;
  return Math.max(4, Math.min(100, (value / maximum) * 100));
}
