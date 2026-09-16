import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { GitCompareArrows, RefreshCw } from "lucide-react";
import { getEvidenceDrift } from "./api";
import "./drift.css";

const PAGE_SIZE = 20;
const label = (value: string) => value.replaceAll("_", " ");

export function DriftPanel({ threadId, artifactId }: { threadId: string; artifactId?: string }) {
  const [page, setPage] = useState(0);
  const query = useQuery({
    queryKey: ["dashboard", "drift", threadId, artifactId ?? null],
    queryFn: ({ signal }) => getEvidenceDrift(threadId, signal, artifactId),
    enabled: false, retry: false,
  });
  const report = query.data;
  const subjects = report?.subjects ?? [];
  const pageCount = Math.max(1, Math.ceil(subjects.length / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount - 1);
  return <section className="drift-panel" aria-label="Evidence drift">
    <div className="drift-heading">
      <div><p className="eyebrow"><GitCompareArrows size={14} /> provenance check</p><h2>Evidence drift <span lang="zh">/ 证据变更</span></h2></div>
      <button type="button" disabled={query.isFetching} onClick={() => { setPage(0); void query.refetch(); }}>
        <RefreshCw size={14} /> {query.isFetching ? "Comparing evidence…" : "Compare saved evidence with HEAD"}
      </button>
    </div>
    <p>Read-only, on demand. Checks recorded source fingerprints and dependent claims; no model call, memory update, or automatic resume.</p>
    {query.isFetching ? <p role="status">Reading bounded repository evidence…</p> : null}
    {query.isError ? <p role="alert">Comparison unavailable. The saved report, if shown, has not been refreshed. Check the workspace and try again.</p> : null}
    {report ? <>
      <p className="drift-identity"><code>{report.baseline_commit?.slice(0, 8) ?? "unversioned"}</code><span>→</span><code>{report.current_commit?.slice(0, 8) ?? "unversioned"}</code><span>{report.worktree_clean ? "Clean worktree" : "Dirty / unavailable worktree"}</span></p>
      <div className="drift-metrics">
        <div><strong>{report.sources.filter((source) => source.status === "changed").length}</strong><span>changed sources</span></div>
        <div><strong>{report.recheck_paths.length} / {report.full_recheck_paths.length}</strong><span>suggested / full scoped recheck</span></div>
        <div><strong>0</strong><span>provider calls for this comparison</span></div>
      </div>
      <p className="drift-caution">Provenance only: unchanged bytes do not prove unchanged behavior. This is a saved comparison, not a live freshness guarantee. File counts are not token or dollar savings.</p>
      {report.scope_truncated || !report.snapshot_stable || report.changes_outside_scope !== false ? <p className="drift-caution">Conservative full scoped recheck: evidence is bounded, the checkout is unstable, or changes outside the recorded slice cannot be excluded.</p> : null}
      <details><summary>Source fingerprints ({report.sources.length})</summary><div className="drift-table-scroll" role="region" aria-label="Source fingerprint table scroll area" tabIndex={0}><table><caption>Recorded source files versus current checkout</caption><thead><tr><th scope="col">Source path</th><th scope="col">State</th><th scope="col">Reason</th></tr></thead><tbody>{report.sources.map((source) => <tr key={source.path}><th scope="row"><code>{source.path}</code></th><td>{label(source.status)}</td><td>{source.reason ? label(source.reason) : "Digest compared"}</td></tr>)}</tbody></table></div></details>
      <details><summary>Claims, entities &amp; same-snapshot memory ({subjects.length})</summary><p>Only facts whose sources are wholly inside this recorded slice are included. Other memories and new dependencies are not declared safe.</p><div className="drift-table-scroll" role="region" aria-label="Evidence subject table scroll area" tabIndex={0}><table><caption>Evidence invalidation and revalidation reasons</caption><thead><tr><th scope="col">Subject</th><th scope="col">State</th><th scope="col">Reasons / paths</th></tr></thead><tbody>{subjects.slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE).map((subject) => <tr key={subject.kind + subject.id}><th scope="row">{subject.kind}<code title={subject.id}>{subject.id.slice(0, 10)}</code></th><td>{label(subject.status)}</td><td>{subject.reasons.map(label).join(", ")}<small>{subject.paths.join(", ") || "No complete source anchors"}</small></td></tr>)}</tbody></table></div><nav aria-label="Drift subject pages"><button type="button" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>Previous subjects</button><span>{currentPage + 1} / {pageCount}</span><button type="button" disabled={currentPage + 1 >= pageCount} onClick={() => setPage(currentPage + 1)}>Next subjects</button></nav></details>
      <details><summary>Suggested recheck paths ({report.recheck_paths.length})</summary><p>These are starting points from known dependencies, not permission to reuse old conclusions on a new commit. Execute a new investigation separately.</p><ul>{report.recheck_paths.map((path) => <li key={path}><code>{path}</code></li>)}</ul>{report.recheck_paths.length === 0 ? <p>No recorded source fingerprint changed.</p> : null}</details>
      <p className="drift-footnote">{report.file_reads} bounded reads · {report.bytes_read.toLocaleString()} bytes successfully hashed · up to 64 files / 16 MiB. Historical evidence and memory are unchanged.</p>
    </> : null}
  </section>;
}
