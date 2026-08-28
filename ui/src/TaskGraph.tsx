import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  type Node,
  type NodeMouseHandler,
} from "@xyflow/react";
import { CheckCircle2, ChevronRight, FileCode2, Route, ShieldCheck } from "lucide-react";
import { useMemo, useState } from "react";

import EvidenceNodeCard from "./EvidenceNode";
import { buildTaskGraph, type EvidenceNodeData } from "./graph-model";
import type { EvidenceNode, InvestigationStructure, ReadingStep } from "./types";

const NODE_TYPES = { evidence: EvidenceNodeCard };

interface TaskGraphProps {
  structure: InvestigationStructure;
}

export function TaskGraph({ structure }: TaskGraphProps) {
  const model = useMemo(() => buildTaskGraph(structure), [structure]);
  const firstEntity = model.readingSteps[0]?.entity_id ?? model.nodes[0]?.id ?? null;
  const [selectedId, setSelectedId] = useState<string | null>(firstEntity);
  const selectedExists = model.nodes.some((node) => node.id === selectedId);
  const activeSelectedId = selectedExists ? selectedId : firstEntity;

  const selectedNode = structure.reading_path.graph_slice.nodes.find(
    (node) => node.entity_id === activeSelectedId,
  );
  const selectedStep = model.readingSteps.find(
    (step) => step.entity_id === activeSelectedId,
  );
  const flowNodes = model.nodes.map((node) => ({
    ...node,
    selected: node.id === activeSelectedId,
  }));
  const onNodeClick: NodeMouseHandler = (_, node) => setSelectedId(node.id);

  return (
    <section className="graph-workspace" aria-label="Task-specific evidence graph">
      <div className="graph-canvas">
        <div className="graph-canvas__legend" aria-label="Graph legend">
          <span><i className="legend-line legend-line--verified" />verified</span>
          <span><i className="legend-line legend-line--inferred" />inferred</span>
          <span><i className="legend-seed" />seed</span>
        </div>
        <ReactFlow
          nodes={flowNodes}
          edges={model.edges}
          nodeTypes={NODE_TYPES}
          onNodeClick={onNodeClick}
          fitView
          fitViewOptions={{ padding: 0.22, maxZoom: 1.1 }}
          minZoom={0.32}
          maxZoom={1.65}
          nodesDraggable
          nodesConnectable={false}
          elementsSelectable
          proOptions={{ hideAttribution: true }}
          colorMode="dark"
          aria-label="Evidence graph canvas"
        >
          <Background
            variant={BackgroundVariant.Dots}
            gap={22}
            size={1}
            color="rgba(151, 177, 166, 0.13)"
          />
          <Controls showInteractive={false} position="bottom-left" />
          <MiniMap
            pannable
            zoomable
            position="bottom-right"
            nodeColor={miniMapColor}
            maskColor="rgba(8, 13, 16, 0.72)"
          />
        </ReactFlow>
      </div>

      <aside className="graph-inspector" aria-label="Reading path and provenance">
        <div className="inspector-section inspector-section--entity">
          <p className="eyebrow"><ShieldCheck size={13} /> evidence identity</p>
          {selectedNode ? (
            <EntityDetails entity={selectedNode} step={selectedStep} />
          ) : (
            <p className="muted-copy">Select a graph node to inspect provenance.</p>
          )}
        </div>

        <div className="inspector-section inspector-section--reading">
          <div className="section-heading">
            <div>
              <p className="eyebrow"><Route size={13} /> traceable route</p>
              <h3>Reading path</h3>
            </div>
            <span className="count-pill">{model.readingSteps.length}</span>
          </div>
          <ol className="reading-path">
            {model.readingSteps.map((step) => (
              <li key={step.reading_step_id}>
                <button
                  type="button"
                  className={step.entity_id === activeSelectedId ? "is-active" : ""}
                  onClick={() => setSelectedId(step.entity_id)}
                >
                  <span className="reading-path__ordinal">{step.ordinal}</span>
                  <span className="reading-path__copy">
                    <strong>{shortName(step.canonical_name)}</strong>
                    <small>{step.reason.replaceAll("_", " ")}</small>
                  </span>
                  <ChevronRight size={14} aria-hidden="true" />
                </button>
              </li>
            ))}
          </ol>
        </div>
      </aside>
    </section>
  );
}

function EntityDetails({ entity, step }: { entity: EvidenceNode; step?: ReadingStep }) {
  const location = entity.evidence[0]?.location ?? step?.source_locations[0];
  return (
    <div className="entity-details">
      <span className="entity-kind">{entity.kind.replace("_", " ")}</span>
      <h3>{shortName(entity.canonical_name)}</h3>
      <code>{entity.canonical_name}</code>
      <dl>
        <div>
          <dt><FileCode2 size={13} /> source</dt>
          <dd>{entity.path ?? "No repository source"}</dd>
        </div>
        <div>
          <dt><CheckCircle2 size={13} /> status</dt>
          <dd>{entity.evidence.length ? "Source verified" : "Structural context"}</dd>
        </div>
        {location?.line_start ? (
          <div>
            <dt>line range</dt>
            <dd>{location.line_start}–{location.line_end ?? location.line_start}</dd>
          </div>
        ) : null}
        {step ? (
          <div>
            <dt>reading reason</dt>
            <dd>{step.reason.replaceAll("_", " ")}</dd>
          </div>
        ) : null}
      </dl>
    </div>
  );
}

function miniMapColor(node: Node): string {
  const data = node.data as Partial<EvidenceNodeData>;
  const kind = data.entity?.kind;
  if (kind === "test") return "#d5a7f3";
  if (kind === "file") return "#67d0c6";
  if (kind === "external_module" || kind === "unresolved_symbol") return "#87938e";
  return "#b7ee78";
}

function shortName(value: string): string {
  return value.split(".").at(-1) ?? value;
}
