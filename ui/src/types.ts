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
  workspace_name: string;
  workspace_path: ".";
  demo_mode: boolean;
  memory_available: boolean;
  pricing_configured: boolean;
  lifecycle_mutations_enabled: boolean;
  memory_mutations_enabled: boolean;
  cancellation_enabled?: boolean;
  event_follow_enabled?: boolean;
}

export interface CancellationRequest {
  schema_version: 1;
  request_id: string;
  run_id: string;
  thread_id: string;
  repository_id: string;
  commit_sha: string | null;
}

export interface CancellationRecord extends Omit<CancellationRequest, "request_id"> {
  state: "running" | "cancel_requested" | "cancelled" | "finished";
  request_id: string | null;
  requested_at_ms: number | null;
  finished_at_ms: number | null;
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

export type HumanDecisionAction = "approve" | "revise" | "cancel";

export interface CandidateEvidenceItem {
  path: string;
  line_start: number | null;
  line_end: number | null;
  symbol: string | null;
  claim: string;
}

export interface CandidateRelation {
  source: string;
  relation: "contains" | "imports" | "references";
  target: string;
  verification_status: "inferred";
  supporting_evidence_indices: number[];
  explanation: string;
}

export interface CandidateResult {
  evidence: CandidateEvidenceItem[];
  relations: CandidateRelation[];
  summary: string;
  unresolved_questions: string[];
}

export interface DashboardFinalResult {
  relevant_files: string[];
  relevant_symbols: string[];
  evidence: string[];
  summary: string;
  unresolved_questions: string[];
}

export interface HumanReviewRequest {
  turn_number: number;
  candidate_result: CandidateResult;
  evaluation_report: EvaluationReport;
  allowed_actions: HumanDecisionAction[];
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
  candidate_result: CandidateResult | null;
  evaluation_report: EvaluationReport | null;
  structure: InvestigationStructure | null;
  final_result: DashboardFinalResult | null;
  pending_review: HumanReviewRequest | null;
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
  plan_timeline: PlanReplayFrame[];
  evaluator_timeline: EvaluatorReplayFrame[];
}

export interface DashboardSourceLine {
  number: number;
  text: string;
}

export interface DashboardSourceSnippet {
  schema_version: 1;
  thread_id: string;
  entity_id: string;
  repository_id: string;
  commit_sha: string | null;
  path: string;
  symbol: string | null;
  source_digest: string;
  evidence_line_start: number;
  evidence_line_end: number;
  snippet_line_start: number;
  snippet_line_end: number;
  total_lines: number;
  truncated_before: boolean;
  truncated_after: boolean;
  lines: DashboardSourceLine[];
}

export interface DashboardProblemDetail {
  schema_version: 1;
  type: string;
  title: string;
  status: number;
  detail: string;
  instance: string;
}

export interface HumanDecision {
  action: HumanDecisionAction;
  feedback: string | null;
}

export interface DashboardHumanResponseCommand {
  schema_version: 1;
  request_id: string;
  interrupt_id: string;
  decision: HumanDecision;
}

export interface DashboardStartCommand {
  schema_version: 1;
  request_id: string;
  expected_repository_id: string;
  expected_commit_sha: string | null;
  goal: string;
  focus_path: string;
  human_review_policy: "never" | "before_finalize";
  memory_retrieval_enabled: boolean;
  max_attempt_cost_usd: string | null;
}

export interface DashboardResumeCommand {
  schema_version: 1;
  request_id: string;
  expected_repository_id: string;
  expected_commit_sha: string | null;
}

export interface DashboardContinueCommand {
  schema_version: 1;
  request_id: string;
  prompt: string;
  focus_path: string;
  expected_repository_id: string;
  expected_commit_sha: string | null;
}

export interface InvestigationOperationResult {
  schema_version: 1;
  request_id: string;
  thread_id: string;
  execution_state: ExecutionState;
  run_id: string | null;
  idempotent_replay: boolean;
  recovered_from_checkpoint: boolean;
}

export type InvestigationOperationKind = "start" | "resume" | "continue" | "respond";
export type InvestigationOperationState = "pending" | "completed";

export interface DashboardLifecycleOperationStatus {
  schema_version: 1;
  request_id: string;
  kind: InvestigationOperationKind;
  thread_id: string;
  state: InvestigationOperationState;
  reserved_at_ms: number;
  latest_attempt: RawRunAttempt | null;
  result: InvestigationOperationResult | null;
}

export interface DashboardMemoryDeleteCommand {
  schema_version: 1;
  request_id: string;
  confirm_memory_id: string;
}

export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export interface PlanReplayFrame {
  schema_version: 1;
  run_id: string;
  turn_number: number;
  sequence: number;
  node: string;
  revision_count: number;
  plan: InvestigationPlan;
}

export interface ValidationIssue {
  code: string;
  message: string;
  path: string | null;
  evidence_index: number | null;
  relation_index: number | null;
}

export interface SemanticAssessment {
  remaining_step_coverage?: {
    step_id: string;
    disposition: "covered" | "unresolved";
    objective_evidence_indices: number[];
    requirements: { requirement_index: number; evidence_indices: number[]; explanation: string }[];
    explanation: string;
  }[];
  active_step_completed: boolean;
  answers_objective: boolean;
  evidence_supports_summary: boolean;
  has_critical_omissions: boolean;
  missing_evidence: string[];
  recommended_focus: string[];
  summary: string;
}

export interface EvaluationReport {
  plan_coverage?: {
    identity_digest: string;
    approved_step_ids: string[];
    reason: "all_pending_covered" | "not_requested" | "incomplete" | "invalid_references" | "evaluation_failed";
  } | null;
  deterministic_issues: ValidationIssue[];
  semantic_assessment: SemanticAssessment;
  passed: boolean;
}

export interface EvaluatorReplayFrame {
  schema_version: 1;
  run_id: string;
  turn_number: number;
  sequence: number;
  node: string;
  report: EvaluationReport;
}

export interface PublicTraceEvent {
  schema_version: 1;
  run_id: string;
  sequence: number;
  kind: "run_started" | "node_completed" | "run_interrupted" | "run_finished" | "run_failed";
  node: string | null;
  payload: Record<string, JsonValue>;
}

export type ContextSection = "objective" | "plan" | "feedback" | "evidence" | "memory" | "history";
export type ContextSelectionOutcome = "selected" | "excluded";

export interface ContextMemoryMetadata {
  schema_version: 1;
  memory_id: string;
  retrieval_id: string;
  query_digest: string;
  repository_key: string;
  source_repository_id: string;
  source_commit_sha: string | null;
  scope: "repository" | "episodic";
  freshness: "current" | "revalidation_required" | "stale" | "observed";
  staleness_reason: string | null;
}

export interface ContextSource {
  schema_version: 1;
  source_id: string;
  kind: "objective" | "plan_step" | "tool_result" | "source_excerpt" | "evaluator_request" | "human_feedback" | "candidate" | "memory" | "summary";
  raw_artifact_id: string | null;
  location: (SourceLocation & { repository_id: string; commit_sha: string | null }) | null;
  provenance: { schema_version: 1; kind: string; id: string }[];
  memory: ContextMemoryMetadata | null;
}

export interface ContextItem {
  schema_version: 1;
  item_id: string;
  source: ContextSource;
  section: ContextSection;
  verification_status: "verified" | "human_approved" | "inferred" | "unverified" | "rejected";
  content: string;
  content_digest: string;
  estimated_tokens: number;
  priority: number;
  pinned: boolean;
}

export interface ContextSelectionDecision {
  schema_version: 1;
  item_id: string;
  outcome: ContextSelectionOutcome;
  reason: string;
  estimated_tokens: number;
  selected_position: number | null;
}

export interface ContextBuildReport {
  schema_version: 1;
  build_id: string;
  budget_tokens: number;
  candidate_count: number;
  selected_count: number;
  excluded_count: number;
  selected_tokens: number;
  decisions: ContextSelectionDecision[];
}

export interface ContextPack {
  schema_version: 1;
  pack_id: string;
  build_report: ContextBuildReport;
  estimated_tokens: number;
  items: ContextItem[];
}

export interface UsageRecord {
  schema_version: 1;
  availability: "reported" | "unavailable";
  input_tokens: number | null;
  output_tokens: number | null;
  total_tokens: number | null;
  cached_input_tokens: number | null;
  reasoning_output_tokens: number | null;
}

export interface ModelInvocationRecord {
  schema_version: 1;
  invocation_id: string;
  run_id: string;
  thread_id: string;
  turn_number: number;
  sequence: number;
  node_name: string;
  role: "planner" | "explorer" | "candidate_builder" | "evaluator" | "replanner";
  provider: string;
  model: string;
  status: "succeeded" | "failed";
  latency_ms: number;
  usage: UsageRecord;
  plan_step_id: string | null;
  context_pack_id: string;
  input_artifact_id: string;
  output_artifact_id: string | null;
  error_code: string | null;
}

export interface UsageMetricSummary {
  schema_version: 1;
  coverage: "complete" | "partial" | "unavailable";
  value: number | null;
  contributing_invocations: number;
  total_invocations: number;
}

export interface CostSummary {
  schema_version: 1;
  coverage: "complete" | "partial" | "unavailable";
  estimated_cost_usd: string | null;
  estimated_invocations: number;
  unavailable_invocations: number;
  unavailable_reasons: string[];
  disclaimer: "estimate_not_invoice";
}

export interface UsageCostBreakdown {
  schema_version: 1;
  dimension: "attempt" | "role" | "node" | "model";
  key: string;
  invocation_count: number;
  succeeded_count: number;
  failed_count: number;
  latency_ms: number;
  input_tokens: UsageMetricSummary;
  output_tokens: UsageMetricSummary;
  total_tokens: UsageMetricSummary;
  cached_input_tokens: UsageMetricSummary;
  reasoning_output_tokens: UsageMetricSummary;
  cost: CostSummary;
}

export interface InvocationCostEstimate {
  schema_version: 1;
  invocation_id: string;
  pricing_table_id: string;
  provider: string;
  model: string;
  availability: "estimated" | "unavailable";
  input_cost_usd: string | null;
  output_cost_usd: string | null;
  estimated_cost_usd: string | null;
  unavailable_reason: string | null;
  disclaimer: "estimate_not_invoice";
}

export interface AttemptCostBudgetSummary {
  schema_version: 1;
  limit_usd: string;
  status: "not_started" | "within_budget" | "exhausted" | "overshot" | "unavailable";
  reason: string;
  confirmed_cost_usd: string | null;
  remaining_cost_usd: string | null;
  overshoot_usd: string | null;
  next_invocation_allowed: boolean;
}

export interface AttemptUsageCostReport {
  schema_version: 1;
  report_id: string;
  run_id: string;
  thread_id: string;
  turn_number: number;
  pricing_table_id: string;
  pricing_version: string;
  pricing_effective_from: string;
  currency: "USD";
  unit_tokens: 1000000;
  invocation_costs: InvocationCostEstimate[];
  breakdowns: UsageCostBreakdown[];
  budget: AttemptCostBudgetSummary | null;
  disclaimer: "estimate_not_invoice";
}

export interface DashboardRunDetail {
  schema_version: 1;
  attempt: RawRunAttempt;
  trace_availability: Availability;
  trace_events: PublicTraceEvent[];
  plan_timeline: PlanReplayFrame[];
  evaluator_timeline: EvaluatorReplayFrame[];
  context_packs: ContextPack[];
  model_invocations: ModelInvocationRecord[];
  cost_availability: Availability;
  usage_cost_report: AttemptUsageCostReport | null;
}

export interface MemoryEntryMetadata {
  schema_version: 1;
  memory_id: string;
  repository_key: string;
  source_repository_id: string;
  memory_scope: "repository" | "episodic";
  commit_sha: string | null;
  verification_status: "verified" | "human_approved" | "observed";
  write_authority: "validated_evidence" | "human_approval" | "terminal_consolidation";
  created_at_ms: number;
  source_count: number;
  audit_event_count: number;
}

export interface MemoryTombstone {
  schema_version: 1;
  tombstone_id: string;
  memory_id: string;
  repository_key: string;
  memory_scope: "repository" | "episodic";
  deletion_event_id: string;
  reason: "user_request" | "retention_policy";
  deleted_at_ms: number;
}

export interface MemoryAuditEvent {
  schema_version: 1;
  event_id: string;
  memory_id: string;
  repository_key: string;
  action: "created" | "reverified" | "retrieved" | "marked_stale" | "deleted";
  reason: string;
  created_at_ms: number;
  run_id: string | null;
  actor_digest: string | null;
}

export interface DashboardMemoryCatalog {
  schema_version: 1;
  availability: Availability;
  entries: MemoryEntryMetadata[];
  tombstones: MemoryTombstone[];
}

export interface DashboardMemoryAudit {
  schema_version: 1;
  memory_id: string;
  metadata: MemoryEntryMetadata | null;
  tombstone: MemoryTombstone | null;
  events: MemoryAuditEvent[];
}

export interface DashboardMemoryDeleteResult {
  schema_version: 1;
  request_id: string;
  memory_id: string;
  tombstone: MemoryTombstone;
  idempotent_replay: boolean;
}
export interface HumanReviewHistory {
  records: {
    review_number: number;
    request_digest: string;
    candidate_summary: string;
    evaluation_passed: boolean;
    deterministic_issue_count: number;
    allowed_actions: ("approve" | "revise" | "cancel")[];
    action: "approve" | "revise" | "cancel" | null;
    feedback: string | null;
    text_truncated: boolean;
  }[];
  omitted_count: number;
  prior_history_unavailable: boolean;
}

export interface PublicConversationPage {
  schema_version: 1;
  thread_id: string;
  latest_turn_number: number;
  turns: {
    turn_number: number;
    objective: string;
    summary: string | null;
    termination_status: "accepted" | "exhausted" | "cancelled" | null;
    text_truncated: boolean;
    review_history?: HumanReviewHistory | null;
  }[];
  next_before_turn: number | null;
  provenance: "checkpoint_public_turns";
}
