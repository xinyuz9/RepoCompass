import {
  Boxes,
  Braces,
  CircleHelp,
  FileCode2,
  FlaskConical,
  GitBranch,
  Package,
} from "lucide-react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { memo } from "react";

import type { EvidenceNodeData } from "./graph-model";
import type { EvidenceNodeKind } from "./types";

const ICONS: Record<EvidenceNodeKind, typeof FileCode2> = {
  repository: GitBranch,
  file: FileCode2,
  class: Boxes,
  function: Braces,
  method: Braces,
  test: FlaskConical,
  external_module: Package,
  unresolved_symbol: CircleHelp,
};

function EvidenceNodeCard({
  data,
  selected,
}: NodeProps<Node<EvidenceNodeData, "evidence">>) {
  const { entity, isSeed, readingOrdinal } = data;
  const Icon = ICONS[entity.kind];
  const compactName = entity.canonical_name.split(".").at(-1) ?? entity.canonical_name;

  return (
    <div
      className={`evidence-node evidence-node--${entity.kind}${selected ? " is-selected" : ""}`}
      aria-label={`${entity.kind} ${entity.canonical_name}`}
    >
      <Handle type="target" position={Position.Left} className="evidence-handle" />
      <div className="evidence-node__topline">
        <span className="evidence-node__icon" aria-hidden="true">
          <Icon size={14} strokeWidth={1.8} />
        </span>
        <span className="evidence-node__kind">{entity.kind.replace("_", " ")}</span>
        {isSeed ? <span className="evidence-node__seed">seed</span> : null}
        {readingOrdinal ? (
          <span className="evidence-node__ordinal">{readingOrdinal}</span>
        ) : null}
      </div>
      <strong className="evidence-node__name" title={entity.canonical_name}>
        {compactName}
      </strong>
      <span className="evidence-node__path" title={entity.path ?? entity.canonical_name}>
        {entity.path ?? "external boundary"}
      </span>
      <div className="evidence-node__proof">
        <span className={entity.evidence.length ? "proof-dot" : "proof-dot is-muted"} />
        {entity.evidence.length ? `${entity.evidence.length} source proof` : "structural context"}
      </div>
      <Handle type="source" position={Position.Right} className="evidence-handle" />
    </div>
  );
}

export default memo(EvidenceNodeCard);
