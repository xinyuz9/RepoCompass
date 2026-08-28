import { MarkerType, type Edge, type Node } from "@xyflow/react";

import type {
  EvidenceEdge,
  EvidenceNode,
  InvestigationStructure,
  ReadingStep,
} from "./types";

export interface EvidenceNodeData extends Record<string, unknown> {
  entity: EvidenceNode;
  depth: number;
  isSeed: boolean;
  readingOrdinal: number | null;
}

export interface TaskGraphModel {
  nodes: Node<EvidenceNodeData, "evidence">[];
  edges: Edge[];
  readingSteps: ReadingStep[];
}

const KIND_ORDER: Record<EvidenceNode["kind"], number> = {
  repository: 0,
  file: 1,
  class: 2,
  function: 3,
  method: 4,
  test: 5,
  external_module: 6,
  unresolved_symbol: 7,
};

export function buildTaskGraph(structure: InvestigationStructure): TaskGraphModel {
  const { graph_slice: slice, steps } = structure.reading_path;
  const depthByEntity = new Map(
    slice.node_depths.map((item) => [item.entity_id, item]),
  );
  const readingOrder = new Map(steps.map((step) => [step.entity_id, step.ordinal]));
  const groups = new Map<number, EvidenceNode[]>();

  for (const entity of slice.nodes) {
    const depth = depthByEntity.get(entity.entity_id)?.depth ?? 0;
    const group = groups.get(depth) ?? [];
    group.push(entity);
    groups.set(depth, group);
  }
  for (const group of groups.values()) {
    group.sort((left, right) => {
      const kindDelta = KIND_ORDER[left.kind] - KIND_ORDER[right.kind];
      return kindDelta || left.canonical_name.localeCompare(right.canonical_name);
    });
  }

  const nodes: Node<EvidenceNodeData, "evidence">[] = [];
  for (const [depth, group] of [...groups.entries()].sort(([a], [b]) => a - b)) {
    group.forEach((entity, index) => {
      nodes.push({
        id: entity.entity_id,
        type: "evidence",
        position: {
          x: depth * 320,
          y: index * 154 - ((group.length - 1) * 154) / 2,
        },
        data: {
          entity,
          depth,
          isSeed: slice.seed_entity_ids.includes(entity.entity_id),
          readingOrdinal: readingOrder.get(entity.entity_id) ?? null,
        },
      });
    });
  }

  return {
    nodes,
    edges: slice.edges.map(toFlowEdge),
    readingSteps: [...steps].sort((left, right) => left.ordinal - right.ordinal),
  };
}

function toFlowEdge(edge: EvidenceEdge): Edge {
  const inferred = edge.verification_status === "inferred";
  return {
    id: edge.edge_id,
    source: edge.source_entity_id,
    target: edge.target_entity_id,
    label: edge.relation,
    type: "smoothstep",
    animated: inferred,
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: inferred ? "#e6ad5c" : "#5fc9bf",
      width: 16,
      height: 16,
    },
    style: {
      stroke: inferred ? "#e6ad5c" : "#5fc9bf",
      strokeDasharray: inferred ? "5 5" : undefined,
      strokeWidth: 1.35,
      opacity: 0.74,
    },
    labelStyle: {
      fill: inferred ? "#e8c690" : "#8ddbd4",
      fontSize: 10,
      fontFamily: "JetBrains Mono Variable, monospace",
      letterSpacing: "0.04em",
    },
    labelBgStyle: { fill: "#10171b", fillOpacity: 0.92 },
    labelBgPadding: [5, 3],
    labelBgBorderRadius: 5,
    data: { evidenceEdge: edge },
  };
}
