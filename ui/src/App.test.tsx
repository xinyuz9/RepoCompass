import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

const system = {
  schema_version: 1,
  product: "RepoCompass",
  product_version: "0.1.0",
  repository_id: "a".repeat(64),
  commit_sha: "b".repeat(40),
  pricing_configured: false,
  lifecycle_mutations_enabled: false,
  memory_mutations_enabled: false,
};

const catalog = {
  schema_version: 1,
  threads: [],
  scanned_attempt_count: 0,
  truncated: false,
};

afterEach(() => vi.restoreAllMocks());

describe("App query shell", () => {
  it("deduplicates cached bootstrap reads and renders the first-run empty state", async () => {
    const fetchMock = vi.fn((input: string | URL | Request) => {
      const url = String(input);
      const body = url.endsWith("/system") ? system : catalog;
      return Promise.resolve(
        new Response(JSON.stringify(body), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );
    });
    vi.stubGlobal("fetch", fetchMock);
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false, staleTime: 60_000 },
      },
    });

    const view = render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    expect(await screen.findByRole("heading", { name: "No investigations recorded" })).toBeVisible();
    expect(screen.getByText(/uv run repocompass/)).toBeVisible();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

    view.rerender(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });
});
