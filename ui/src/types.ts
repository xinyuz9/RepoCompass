export type Availability =
  | "available"
  | "not_found"
  | "not_persisted"
  | "not_configured";

export type ExecutionState = "completed" | "interrupted" | "resumable";

export interface DashboardSystemInfo {
  schema_version: 1;
  product: "RepoCompass";
  product_version: string;
  repository_id: string;
  commit_sha: string | null;
  pricing_configured: boolean;
  lifecycle_mutations_enabled: boolean;
  memory_mutations_enabled: boolean;
}

export interface RawRunAttempt {
  schema_version: 1;
  run_id: string;
  thread_id: string;
  turn_number: number;
  execution_mode: "start" | "resume" | "continue" | "respond";
  repository_id: string;
  commit_sha: string | null;
  objective_digest: string;
  status: "running" | "completed" | "interrupted" | "failed";
  started_at_ms: number;
  finished_at_ms: number | null;
}

export interface DashboardThreadSummary {
  schema_version: 1;
  thread_id: string;
  checkpoint_availability: Availability;
  objective: string | null;
  execution_state: ExecutionState | null;
  turn_number: number | null;
  termination_status: "accepted" | "exhausted" | "cancelled" | null;
  latest_attempt: RawRunAttempt;
  attempt_count: number;
}

export interface DashboardThreadCatalog {
  schema_version: 1;
  threads: DashboardThreadSummary[];
  scanned_attempt_count: number;
  truncated: boolean;
}

export interface PlanStep {
  id: string;
  objective: string;
  expected_evidence: string[];
  status: "pending" | "active" | "completed";
}

export interface InvestigationPlan {
  steps: PlanStep[];
}

export interface SourceLocation {
  path: string;
  line_start?: number | null;
  line_end?: number | null;
  symbol?: string | null;
}

export interface GraphEvidence {
  evidence_id: string;
  kind: "python_symbol" | "python_import" | "python_reference";
  location: SourceLocation;
}

export type EvidenceNodeKind =
  | "repository"
  | "file"
  | "class"
  | "function"
  | "method"
  | "test"
  | "external_module"
  | "unresolved_symbol";

export interface EvidenceNode {
  entity_id: string;
  node_id: string;
  kind: EvidenceNodeKind;
  canonical_name: string;
  path: string | null;
  symbol_kind: string | null;
  is_test: boolean;
  evidence: GraphEvidence[];
}

export interface EvidenceEdge {
  edge_id: string;
  assertion_id: string;
  relation: "contains" | "imports" | "references";
  source_entity_id: string;
  target_entity_id: string;
  verification_status: "verified" | "inferred" | "rejected";
  evidence: GraphEvidence[];
  explanation: string | null;
}

export interface GraphNodeDepth {
  entity_id: string;
  depth: number;
  is_seed: boolean;
}

export interface EvidenceGraphSlice {
  slice_id: string;
  graph_id: string;
  seed_entity_ids: string[];
  traversal: "outbound" | "inbound" | "both";
  include_inferred: boolean;
  max_depth: number;
  max_nodes: number;
  max_edges: number;
  truncated: boolean;
  truncation_reasons: string[];
  nodes: EvidenceNode[];
  edges: EvidenceEdge[];
  node_depths: GraphNodeDepth[];
}

export interface ReadingStep {
  reading_step_id: string;
  ordinal: number;
  entity_id: string;
  node_id: string;
  kind: "file" | "symbol" | "test";
  reason:
    | "seed"
    | "structural_context"
    | "import_dependency"
    | "reference"
    | "related_test";
  canonical_name: string;
  path: string;
  depth: number;
  path_verification_status: "verified" | "inferred";
  via_assertion_id: string | null;
  source_locations: SourceLocation[];
}

export interface ClaimLineage {
  lineage_id: string;
  claim_id: string;
  text: string;
  verification_status: "verified" | "inferred";
  supporting_assertion_ids: string[];
  source_locations: SourceLocation[];
  provenance_complete: boolean;
  explanation: string | null;
}

export interface TraceableReadingPath {
  reading_path_id: string;
  objective: string;
  graph_slice: EvidenceGraphSlice;
  steps: ReadingStep[];
  claim_lineages: ClaimLineage[];
}

export interface InvestigationStructure {
  artifact_id: string;
  repository_id: string;
  commit_sha: string | null;
  indexed_paths: string[];
  reading_path: TraceableReadingPath;
}

export interface ThreadInvestigationView {
  schema_version: 1;
  thread_id: string;
  execution_state: ExecutionState;
  turn_number: number;
  objective: string;
  plan: InvestigationPlan | null;
  plan_revision_count: number;
  max_plan_revisions: number;
  next_nodes: string[];
  termination_status: "accepted" | "exhausted" | "cancelled" | null;
  step_stop_reason: string | null;
  structure: InvestigationStructure | null;
  interrupt_id: string | null;
}

export interface DashboardRunSummary {
  schema_version: 1;
  attempt: RawRunAttempt;
  trace_availability: Availability;
  cost_availability: Availability;
}

export interface DashboardThreadDetail {
  schema_version: 1;
  thread: ThreadInvestigationView;
  attempts: DashboardRunSummary[];
  plan_timeline: unknown[];
  evaluator_timeline: unknown[];
}

export interface DashboardProblemDetail {
  schema_version: 1;
  type: string;
  title: string;
  status: number;
  detail: string;
  instance: string;
}
