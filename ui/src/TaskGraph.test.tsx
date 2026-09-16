import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { TaskGraph } from "./TaskGraph";
import type { InvestigationStructure } from "./types";

const entityId = "d".repeat(64);
const repositoryId = "a".repeat(64);
const commitSha = "b".repeat(40);
const structure = {
  artifact_id: "c".repeat(64),
  repository_id: repositoryId,
  commit_sha: commitSha,
  indexed_paths: ["src/auth.py"],
  reading_path: {
    reading_path_id: "e".repeat(64),
    objective: "Trace login.",
    graph_slice: {
      slice_id: "f".repeat(64),
      graph_id: "1".repeat(64),
      seed_entity_ids: [entityId],
      traversal: "both",
      include_inferred: false,
      max_depth: 2,
      max_nodes: 10,
      max_edges: 10,
      truncated: false,
      truncation_reasons: [],
      nodes: [{
        entity_id: entityId,
        node_id: "2".repeat(64),
        kind: "function",
        canonical_name: "auth.login",
        path: "src/auth.py",
        symbol_kind: "function",
        is_test: false,
        evidence: [{
          evidence_id: "3".repeat(64),
          kind: "python_symbol",
          location: { path: "src/auth.py", line_start: 1, line_end: 2 },
        }],
      }],
      edges: [],
      node_depths: [{ entity_id: entityId, depth: 0, is_seed: true }],
    },
    steps: [{
      reading_step_id: "4".repeat(64),
      ordinal: 1,
      entity_id: entityId,
      node_id: "2".repeat(64),
      kind: "symbol",
      reason: "seed",
      canonical_name: "auth.login",
      path: "src/auth.py",
      depth: 0,
      path_verification_status: "verified",
      via_assertion_id: null,
      source_locations: [{ path: "src/auth.py", line_start: 1, line_end: 2 }],
    }],
    claim_lineages: [],
  },
} satisfies InvestigationStructure;

const source = {
  schema_version: 1,
  thread_id: "thread-1",
  entity_id: entityId,
  repository_id: repositoryId,
  commit_sha: commitSha,
  path: "src/auth.py",
  symbol: "auth.login",
  source_digest: "5".repeat(64),
  evidence_line_start: 1,
  evidence_line_end: 2,
  snippet_line_start: 1,
  snippet_line_end: 2,
  total_lines: 2,
  truncated_before: false,
  truncated_after: false,
  lines: [
    { number: 1, text: "def login():" },
    { number: 2, text: "    return authenticate()" },
  ],
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("TaskGraph source inspector", () => {
  it("never requests a source for an unknown linked node", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const client = new QueryClient();
    render(<QueryClientProvider client={client}><TaskGraph threadId="thread-1" structure={structure} requestedEntityId={"9".repeat(64)} /></QueryClientProvider>);
    expect(screen.getByRole("alert")).toHaveTextContent("not in this saved evidence slice");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("paginates entity tables and reports keyboard selections", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(new Response(JSON.stringify(source)))));
    const many = structuredClone(structure);
    many.reading_path.graph_slice.nodes = Array.from({ length: 41 }, (_, index) => ({ ...structure.reading_path.graph_slice.nodes[0]!, entity_id: index.toString(16).padStart(64, "0"), canonical_name: `node${index}` }));
    const onSelect = vi.fn();
    render(<QueryClientProvider client={new QueryClient()}><TaskGraph threadId="thread-1" structure={many} onEntitySelected={onSelect} /></QueryClientProvider>);
    fireEvent.click(screen.getByRole("button", { name: "Evidence table" }));
    expect(screen.getAllByRole("button", { name: /^Inspect node/ })).toHaveLength(20);
    fireEvent.click(screen.getByRole("button", { name: "Next entities" }));
    fireEvent.click(screen.getByRole("button", { name: "Next entities" }));
    expect(screen.getAllByRole("button", { name: /^Inspect node/ })).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Inspect node40" }));
    expect(onSelect).toHaveBeenCalledWith((40).toString(16).padStart(64, "0"));
  });

  it("isolates source caches by evidence artifact and rejects snapshot mismatches", async () => {
    const fetchMock = vi.fn(() => Promise.resolve(new Response(JSON.stringify(source))));
    vi.stubGlobal("fetch", fetchMock);
    const client = new QueryClient();
    const first = renderGraph(client);
    await screen.findByLabelText("Source preview for src/auth.py");
    first.unmount();
    const changed = { ...structure, artifact_id: "8".repeat(64), commit_sha: "7".repeat(40) };
    render(<QueryClientProvider client={client}><TaskGraph threadId="thread-1" structure={changed} /></QueryClientProvider>);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("source snapshot changed"));
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(screen.queryByLabelText("Source preview for src/auth.py")).not.toBeInTheDocument();
  });

  it("loads one evidence-bound window and reuses it across remounts", async () => {
    const requestedPaths: string[] = [];
    const fetchMock = vi.fn((input: string | URL | Request) => {
      requestedPaths.push(new URL(String(input), "http://dashboard.local").pathname);
      return Promise.resolve(new Response(JSON.stringify(source), {
        status: 200,
        headers: { "content-type": "application/json" },
      }));
    });
    vi.stubGlobal("fetch", fetchMock);
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    const first = renderGraph(client);
    const preview = await screen.findByLabelText("Source preview for src/auth.py");
    expect(preview.querySelector(".source-line.is-evidence code")?.textContent)
      .toBe("def login():");
    expect(preview.querySelectorAll(".source-line")[1]?.textContent)
      .toContain("    return authenticate()");
    expect(requestedPaths).toEqual([
      `/api/v1/threads/thread-1/source/${entityId}`,
    ]);

    first.unmount();
    renderGraph(client);
    await waitFor(() => expect(
      screen.getByLabelText("Source preview for src/auth.py"),
    ).toBeVisible());
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

function renderGraph(client: QueryClient) {
  return render(
    <QueryClientProvider client={client}>
      <TaskGraph threadId="thread-1" structure={structure} />
    </QueryClientProvider>,
  );
}
