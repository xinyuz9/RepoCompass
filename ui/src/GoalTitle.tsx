/** Keep long issues discoverable without letting headings consume the workspace. */
export function GoalTitle({ objective }: { objective: string | null }) {
  const full = objective?.trim() || "Repository investigation";
  const first = full.split(/\r?\n/).find((line) => line.trim())?.trim() || full;
  const chars = Array.from(first);
  const title = chars.length > 180 ? chars.slice(0, 180).join("") + "…" : first;
  return <>
    <h1>{title}</h1>
    {title !== full ? <details className="full-goal"><summary>Read full goal · 完整目标</summary>
      <pre tabIndex={0} aria-label="Full investigation goal">{full}</pre>
    </details> : null}
  </>;
}
