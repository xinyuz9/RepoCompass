import { expect, test } from "@playwright/test";
import axe from "axe-core";

test("reconciles an actual crashed worker with one explicit POST and no model replay", async ({ page, request }, testInfo) => {
  const system = await (await request.get("/api/v1/system")).json();
  const orphan = await (await request.post("/__test__/orphan")).json();
  const posts: string[] = [];
  page.on("request", (entry) => { if (entry.method() === "POST") posts.push(entry.url()); });
  await page.goto("/?compose=1");
  await page.evaluate(({ orphan, system }) => {
    const command = { schema_version: 1, request_id: orphan.request_id, goal: "Explain the flag.", focus_path: ".",
      expected_repository_id: system.repository_id, expected_commit_sha: system.commit_sha,
      human_review_policy: "never", memory_retrieval_enabled: true, max_attempt_cost_usd: null };
    // Use the same public tracker schema as a real request that lost its response.
    window.sessionStorage.setItem("repocompass.pending-lifecycle.v1", JSON.stringify([{ schema_version: 1, recorded_at_ms: Date.now(), kind: "start", thread_id: null, command }]));
  }, { orphan, system });
  await page.reload();
  const reconcile = page.getByRole("button", { name: "Reconcile saved run" });
  await expect(reconcile).toBeVisible();
  expect(posts).toHaveLength(0);
  await expect(page.getByRole("button", { name: "Replay exact request" })).toHaveCount(0);
  await page.reload();
  await expect(reconcile).toBeVisible();
  expect(posts).toHaveLength(0);
  await reconcile.click();
  await expect(page.getByText("Local records reconciled", { exact: true })).toBeVisible();
  expect(posts).toHaveLength(1);
  expect(posts[0]).toContain("/operations/" + orphan.request_id + "/reconcile");
  await page.evaluate(axe.source);
  const violations = await page.evaluate(async () => (await (window as unknown as { axe: typeof import("axe-core") }).axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21aa"] } })).violations.map((v) => v.id));
  expect(violations).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize()!.width);
  await page.screenshot({ path: testInfo.outputPath("reconciled.png"), fullPage: true });
  await page.getByRole("button", { name: "Open reconciled thread" }).click();
  await expect(page.getByRole("button", { name: "Resume", exact: true })).toBeVisible();
  // Resume is a separate user action. This test never invokes it.
  const run = await (await request.get("/api/v1/runs/" + orphan.run_id)).json();
  expect(run.attempt.status).toBe("failed");
  expect(run.model_invocations).toHaveLength(0);
  expect(run.trace_events.at(-1).payload.error_code).toBe("orphan_reconciled");
  expect(posts).toHaveLength(1);
});
