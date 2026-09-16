import { useQuery } from "@tanstack/react-query";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  type Node,
  type NodeMouseHandler,
} from "@xyflow/react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  Code2,
  FileCode2,
  LoaderCircle,
  RefreshCw,
  Route,
  ShieldCheck,
} from "lucide-react";
import { useMemo, useState } from "react";

import { DashboardApiError, dashboardQueries } from "./api";
import EvidenceNodeCard from "./EvidenceNode";
import { EvidenceTable } from "./EvidenceTable";
import "./workspace-review.css";
import { buildTaskGraph, type EvidenceNodeData } from "./graph-model";
import type {
  DashboardSourceSnippet,
  EvidenceNode,
  InvestigationStructure,
  ReadingStep,
} from "./types";

const NODE_TYPES = { evidence: EvidenceNodeCard };

interface TaskGraphProps {
  threadId: string;
  structure: InvestigationStructure;
  requestedEntityId?: string;
  onEntitySelected?: (id: string) => void;
}

export function TaskGraph({ threadId, structure, requestedEntityId, onEntitySelected }: TaskGraphProps) {
  const [readingPathOnly, setReadingPathOnly] = useState(true);
  const [tableView, setTableView] = useState(false);
  const showReadingRoute = readingPathOnly && (!requestedEntityId || structure.reading_path.steps.some((step) => step.entity_id === requestedEntityId));
  const model = useMemo(() => buildTaskGraph(structure, showReadingRoute), [structure, showReadingRoute]);
  const firstEntity = model.readingSteps[0]?.entity_id ?? model.nodes[0]?.id ?? null;
  const [selectedId, setSelectedId] = useState<string | null>(firstEntity);
  const desiredId = requestedEntityId ?? selectedId;
  const selectedExists = structure.reading_path.graph_slice.nodes.some((node) => node.entity_id === desiredId);
  const invalidLink = requestedEntityId !== undefined && !selectedExists;
  const activeSelectedId = selectedExists ? desiredId : invalidLink ? null : firstEntity;
  const selectEntity = (id: string) => {
    setSelectedId(id);
    onEntitySelected?.(id);
  };

  const selectedNode = structure.reading_path.graph_slice.nodes.find(
    (node) => node.entity_id === activeSelectedId,
  );
  const selectedStep = model.readingSteps.find(
    (step) => step.entity_id === activeSelectedId,
  );
  const sourceEnabled = Boolean(
    activeSelectedId && selectedStep?.source_locations.length,
  );
  const source = useQuery({
    ...dashboardQueries.source(threadId, activeSelectedId ?? ""),
    queryKey: [...dashboardQueries.source(threadId, activeSelectedId ?? "").queryKey, structure.artifact_id],
    enabled: sourceEnabled,
    retry: false,
    staleTime: Infinity,
  });
  const flowNodes = model.nodes.map((node) => ({
    ...node,
    selected: node.id === activeSelectedId,
  }));
  const onNodeClick: NodeMouseHandler = (_, node) => selectEntity(node.id);
  const snapshotMatches = !source.data || (source.data.repository_id === structure.repository_id && source.data.commit_sha === structure.commit_sha);

  return (
    <section className="graph-workspace" aria-label="Task-specific evidence graph">
      <div className="graph-canvas">
        <div className="graph-scope" role="group" aria-label="Graph scope">
          <button type="button" aria-pressed={showReadingRoute && !tableView} onClick={() => { setReadingPathOnly(true); setTableView(false); if (firstEntity) selectEntity(firstEntity); }}>Reading route</button>
          <button type="button" aria-pressed={!showReadingRoute && !tableView} onClick={() => { setReadingPathOnly(false); setTableView(false); }}>All evidence ({structure.reading_path.graph_slice.nodes.length})</button>
          <button type="button" aria-pressed={tableView} onClick={() => setTableView(true)}>Evidence table</button>
        </div>
        {tableView ? <EvidenceTable key={structure.artifact_id} structure={structure} selectedId={activeSelectedId} onSelect={selectEntity} /> : <>
        <div className="graph-canvas__legend" aria-label="Graph legend">
          <span><i className="legend-line legend-line--verified" />verified</span>
          <span><i className="legend-line legend-line--inferred" />inferred</span>
          <span><i className="legend-seed" />seed</span>
        </div>
        <ReactFlow
          key={showReadingRoute ? "reading-route" : "all-evidence"}
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
          {!showReadingRoute ? <MiniMap
            pannable
            zoomable
            position="bottom-right"
            nodeColor={miniMapColor}
            maskColor="rgba(8, 13, 16, 0.72)"
          /> : null}
        </ReactFlow>
        </>}
      </div>

      <aside className="graph-inspector" aria-label="Reading path and provenance">
        <div className="inspector-section inspector-section--entity">
          <p className="eyebrow"><ShieldCheck size={13} /> evidence identity</p>
          {invalidLink ? <p role="alert">The linked entity is not in this saved evidence slice. No source was requested for that ID.</p> : null}
          {selectedNode ? (
            <EntityDetails entity={selectedNode} step={selectedStep} />
          ) : (
            <p className="muted-copy">Select a graph node to inspect provenance.</p>
          )}
        </div>

        <div className="inspector-section inspector-section--source">
          <div className="section-heading section-heading--source">
            <div>
              <p className="eyebrow"><Code2 size={13} /> verified source</p>
              <h3>Evidence window</h3>
            </div>
            {source.data ? (
              <span className="count-pill">{source.data.lines.length} lines</span>
            ) : null}
          </div>
          {!snapshotMatches ? <p role="alert">The source snapshot changed. Refresh the investigation before inspecting this evidence.</p> : sourceEnabled ? (
            <SourcePreview
              snippet={source.data ?? null}
              pending={source.isPending}
              error={source.error}
              onRetry={() => void source.refetch()}
            />
          ) : (
            <p className="muted-copy">This structural node has no source-backed reading step.</p>
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
                  onClick={() => selectEntity(step.entity_id)}
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

function SourcePreview({
  snippet,
  pending,
  error,
  onRetry,
}: {
  snippet: DashboardSourceSnippet | null;
  pending: boolean;
  error: Error | null;
  onRetry: () => void;
}) {
  if (pending) {
    return (
      <div className="source-preview-state" role="status">
        <LoaderCircle className="is-spinning" size={15} />
        <span>Verifying source digest…</span>
      </div>
    );
  }
  if (error) {
    const title = error instanceof DashboardApiError
      ? error.problem?.title ?? "Source unavailable"
      : "Source unavailable";
    return (
      <div className="source-preview-state source-preview-state--error" role="alert">
        <AlertTriangle size={15} />
        <div><strong>{title}</strong><span>{error.message}</span></div>
        <button type="button" onClick={onRetry} title="Retry source verification">
          <RefreshCw size={13} /><span className="sr-only">Retry source verification</span>
        </button>
      </div>
    );
  }
  if (!snippet) return null;

  return (
    <div className="source-preview">
      <div className="source-preview__meta">
        <code title={snippet.path}>{snippet.path}</code>
        <span>L{snippet.evidence_line_start}–{snippet.evidence_line_end}</span>
      </div>
      <div className="source-code" role="region" tabIndex={0} aria-label={`Source preview for ${snippet.path}`}>
        {snippet.truncated_before ? <div className="source-omission">earlier lines omitted</div> : null}
        {snippet.lines.map((line) => {
          const evidence = line.number >= snippet.evidence_line_start
            && line.number <= snippet.evidence_line_end;
          return (
            <div key={line.number} className={`source-line${evidence ? " is-evidence" : ""}`}>
              <span>{line.number}</span>
              <code>{line.text || " "}</code>
            </div>
          );
        })}
        {snippet.truncated_after ? <div className="source-omission">later lines omitted</div> : null}
      </div>
      <div className="source-preview__proof">
        <ShieldCheck size={12} />digest {snippet.source_digest.slice(0, 10)} · snapshot matched
      </div>
    </div>
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
