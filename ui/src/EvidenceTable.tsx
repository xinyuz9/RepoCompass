import { useState } from "react";
import type { InvestigationStructure } from "./types";

const PAGE_SIZE = 20;

export function EvidenceTable({ structure, selectedId, onSelect }: {
  structure: InvestigationStructure;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const [page, setPage] = useState(0);
  const [edgePage, setEdgePage] = useState(0);
  const { nodes, edges, truncated } = structure.reading_path.graph_slice;
  const names = new Map(nodes.map((node) => [node.entity_id, node.canonical_name]));
  return <div className="evidence-tables" role="region" aria-label="Evidence tables" tabIndex={0}>
    <p>Keyboard-readable alternative to the graph. Recorded slice only; relations can be inferred, not verified.{truncated ? " This slice is truncated." : ""}</p>
    <table><caption>Evidence entities · {nodes.length}</caption>
      <thead><tr><th scope="col">Entity / source</th><th scope="col">Kind</th><th scope="col">Inspect</th></tr></thead>
      <tbody>{nodes.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).map((node) => <tr key={node.entity_id}>
        <th scope="row"><code>{node.canonical_name}</code><small>{node.path ?? "No repository source"}</small></th>
        <td>{node.kind.replaceAll("_", " ")}</td>
        <td><button type="button" aria-pressed={selectedId === node.entity_id} onClick={() => onSelect(node.entity_id)} aria-label={`Inspect ${node.canonical_name}`}>Inspect</button></td>
      </tr>)}</tbody>
    </table>
    <Pages name="entities" page={page} count={nodes.length} setPage={setPage} />
    <table><caption>Evidence relations · {edges.length}</caption>
      <thead><tr><th scope="col">From → to</th><th scope="col">Relation</th><th scope="col">Verification</th></tr></thead>
      <tbody>{edges.slice(edgePage * PAGE_SIZE, (edgePage + 1) * PAGE_SIZE).map((edge) => <tr key={edge.edge_id}>
        <th scope="row"><code>{names.get(edge.source_entity_id) ?? edge.source_entity_id} → {names.get(edge.target_entity_id) ?? edge.target_entity_id}</code></th>
        <td>{edge.relation.replaceAll("_", " ")}</td><td>{edge.verification_status}</td>
      </tr>)}</tbody>
    </table>
    <Pages name="relations" page={edgePage} count={edges.length} setPage={setEdgePage} />
  </div>;
}

function Pages({ name, page, count, setPage }: { name: string; page: number; count: number; setPage: (value: number) => void }) {
  const pages = Math.max(1, Math.ceil(count / PAGE_SIZE));
  return <nav aria-label={`${name} pages`}><button type="button" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous {name}</button><span>{page + 1} / {pages}</span><button type="button" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>Next {name}</button></nav>;
}
