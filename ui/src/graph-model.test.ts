import { describe, expect, it } from "vitest";

import { buildTaskGraph } from "./graph-model";
import type { EvidenceNode, InvestigationStructure } from "./types";

const repositoryNode: EvidenceNode = {
  entity_id: "repo",
  node_id: "repo-node",
  kind: "repository",
  canonical_name: "repository",
  path: null,
  symbol_kind: null,
  is_test: false,
  evidence: [],
};

const functionNode: EvidenceNode = {
  entity_id: "function",
  node_id: "function-node",
  kind: "function",
  canonical_name: "auth.login",
  path: "src/auth.py",
  symbol_kind: "function",
  is_test: false,
  evidence: [
    {
      evidence_id: "evidence",
      kind: "python_symbol",
      location: { path: "src/auth.py", line_start: 10, line_end: 18 },
    },
  ],
};

const structure: InvestigationStructure = {
  artifact_id: "artifact",
  repository_id: "repository-id",
  commit_sha: "commit",
  indexed_paths: ["src/auth.py"],
  reading_path: {
    reading_path_id: "reading-path",
    objective: "Trace login.",
    graph_slice: {
      slice_id: "slice",
      graph_id: "graph",
      seed_entity_ids: ["function"],
      traversal: "both",
      include_inferred: true,
      max_depth: 2,
      max_nodes: 10,
      max_edges: 10,
      truncated: false,
      truncation_reasons: [],
      nodes: [repositoryNode, functionNode],
      edges: [
        {
          edge_id: "edge",
          assertion_id: "assertion",
          relation: "contains",
          source_entity_id: "repo",
          target_entity_id: "function",
          verification_status: "verified",
          evidence: [],
          explanation: null,
        },
      ],
      node_depths: [
        { entity_id: "repo", depth: 1, is_seed: false },
        { entity_id: "function", depth: 0, is_seed: true },
      ],
    },
    steps: [
      {
        reading_step_id: "step",
        ordinal: 1,
        entity_id: "function",
        node_id: "function-node",
        kind: "symbol",
        reason: "seed",
        canonical_name: "auth.login",
        path: "src/auth.py",
        depth: 0,
        path_verification_status: "verified",
        via_assertion_id: null,
        source_locations: [{ path: "src/auth.py", line_start: 10, line_end: 18 }],
      },
    ],
    claim_lineages: [],
  },
};

describe("buildTaskGraph", () => {
  it("renders only the bounded reading-path slice with deterministic depth layout", () => {
    const graph = buildTaskGraph(structure);

    expect(graph.nodes).toHaveLength(2);
    expect(graph.edges).toHaveLength(1);
    expect(graph.readingSteps.map((step) => step.entity_id)).toEqual(["function"]);
    expect(graph.nodes.find((node) => node.id === "function")?.data).toMatchObject({
      depth: 0,
      isSeed: true,
      readingOrdinal: 1,
    });
    expect(graph.nodes.find((node) => node.id === "repo")?.position.x).toBe(320);
    expect(graph.edges[0]).toMatchObject({
      source: "repo",
      target: "function",
      animated: false,
      label: "contains",
    });
  });
});
