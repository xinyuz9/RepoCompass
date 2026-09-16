export interface EvidenceDriftReport {
  schema_version: 1;
  repository_id: string;
  baseline_commit: string | null;
  current_commit: string | null;
  worktree_clean: boolean;
  snapshot_stable: boolean;
  changes_outside_scope: boolean | null;
  scope_truncated: boolean;
  sources: { path: string; baseline_digests: string[]; current_digest: string | null; status: string; reason: string | null; bytes_read: number }[];
  subjects: { id: string; kind: "claim" | "entity" | "memory"; status: string; paths: string[]; reasons: string[] }[];
  recheck_paths: string[];
  full_recheck_paths: string[];
  bytes_read: number;
  read_budget_used_bytes: number;
  file_reads: number;
  provider_calls: 0;
  disclaimer: "provenance_only_not_semantic_revalidation";
}
