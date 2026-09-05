import { expect, test } from "@playwright/test";
import axe from "axe-core";

test("reads public history from real continued checkpoints without mutations", async ({ page }, testInfo) => {
  const writes: string[] = [];
  page.on("request", (request) => {
    if (!["GET", "HEAD"].includes(request.method())) writes.push(request.url());
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Earlier turns · 1" }).click();
  const turn = page.getByRole("region", { name: "Turn 1", exact: true });
  await expect(turn).toBeVisible();
  await expect(turn.getByText("Trace how proposal verification gates application and identify the regression test.")).toBeVisible();
  await expect(turn.getByText("accepted", { exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize()!.width);
  await page.evaluate(axe.source);
  const scan = await page.evaluate(async () => {
    const browserAxe = (window as unknown as { axe: typeof import("axe-core") }).axe;
    return browserAxe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] } });
  });
  expect(scan.violations.map(({ id }) => id)).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath("conversation-history.png"), fullPage: true });
  await page.reload();
  await page.getByRole("button", { name: "Earlier turns · 1" }).click();
  await expect(turn).toBeVisible();
  expect(writes).toEqual([]);
});

test("keeps persisted recovery inert in the real read-only demo", async ({ page, request }) => {
  const system = await (await request.get("/api/v1/system")).json();
  const tracker = recoveryTracker(system);
  const unsafeRequests: string[] = [];
  page.on("request", (entry) => {
    if (entry.url().includes("/operations/") || entry.method() === "POST") {
      unsafeRequests.push(entry.url());
    }
  });
  await page.goto("/");
  await page.evaluate((value) => {
    sessionStorage.setItem("repocompass.pending-lifecycle.v1", JSON.stringify([value]));
  }, tracker);
  await page.reload();
  await expect(page.getByText("Read-only workspace · recovery paused")).toBeVisible();
  await expect(page.getByRole("button", { name: "Replay exact request" })).toHaveCount(0);
  await page.getByRole("button", { name: "Hide tracker for browser-recovery" }).click();
  await expect(page.getByRole("region", { name: "Lifecycle request recovery" })).toHaveCount(0);
  expect(unsafeRequests).toEqual([]);
});

test("fault-injected 404 recovery stops polling and replays one exact request after refresh", async ({ page, request }, testInfo) => {
  // Browser-contract fault injection only: no live model or mutation-capable backend.
  const system = await (await request.get("/api/v1/system")).json();
  const catalog = await (await request.get("/api/v1/threads")).json();
  const tracker = recoveryTracker(system);
  let statusReads = 0;
  const posts: { body: unknown; key: string | undefined }[] = [];
  await page.route("**/api/v1/system", async (route) => {
    await route.fulfill({ json: { ...system, lifecycle_mutations_enabled: true } });
  });
  await page.route("**/api/v1/operations/browser-recovery", async (route) => {
    statusReads += 1;
    await route.fulfill({ status: 404, contentType: "application/problem+json", body: JSON.stringify({
      schema_version: 1, status: 404, title: "Not found", detail: "No reservation exists.",
      type: "urn:repocompass:problem:operation-not-found", instance: "/api/v1/operations/browser-recovery",
    }) });
  });
  await page.route("**/api/v1/investigations", async (route) => {
    posts.push({ body: route.request().postDataJSON(), key: route.request().headers()["idempotency-key"] });
    await route.fulfill({ json: {
      schema_version: 1, request_id: tracker.command.request_id,
      thread_id: catalog.threads[0].thread_id, execution_state: "completed",
      run_id: catalog.threads[0].latest_attempt.run_id,
      idempotent_replay: true, recovered_from_checkpoint: false,
    } });
  });
  await page.goto("/");
  await page.evaluate((value) => {
    sessionStorage.setItem("repocompass.pending-lifecycle.v1", JSON.stringify([value]));
  }, tracker);
  await page.clock.install();
  await page.reload();
  await expect(page.getByText("Request was not reserved")).toBeVisible();
  await expect(page.getByRole("region", { name: "Investigation result" })).toBeVisible();
  await page.clock.runFor(10_000);
  expect(statusReads).toBe(1);
  expect(posts).toEqual([]);
  await expect(page.getByRole("button", { name: "Retry status" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize()!.width);
  await page.evaluate(axe.source);
  const scan = await page.evaluate(async () => {
    const browserAxe = (window as unknown as { axe: typeof import("axe-core") }).axe;
    return browserAxe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] } });
  });
  expect(scan.violations.map(({ id, nodes }) => ({ id, nodes: nodes.map(({ html, failureSummary }) => ({ html, failureSummary })) }))).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath("recovery.png"), fullPage: true });
  await page.getByRole("button", { name: "Replay exact request" }).click();
  await expect(page.getByRole("region", { name: "Lifecycle request recovery" })).toHaveCount(0);
  expect(posts).toEqual([{ body: tracker.command, key: tracker.command.request_id }]);
  expect(await page.evaluate(() => sessionStorage.getItem("repocompass.pending-lifecycle.v1"))).toBeNull();
});

function recoveryTracker(system: { repository_id: string; commit_sha: string | null }) {
  return {
    schema_version: 1, recorded_at_ms: Date.now(), kind: "start", thread_id: null,
    command: {
      schema_version: 1, request_id: "browser-recovery", goal: "Trace the policy boundary.",
      expected_repository_id: system.repository_id, expected_commit_sha: system.commit_sha,
      focus_path: ".", human_review_policy: "before_finalize",
      memory_retrieval_enabled: true, max_attempt_cost_usd: null,
    },
  };
}

test("replays real demo evidence, navigates inspectors, and keeps a readable workspace", async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  const failedResponses: string[] = [];
  const writes: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("response", (response) => {
    if (response.url().includes("/api/v1/") && response.status() >= 400) failedResponses.push(response.url());
  });
  page.on("request", (request) => {
    if (!["GET", "HEAD"].includes(request.method())) writes.push(request.method() + " " + request.url());
  });
  await page.goto("/");
  const result = page.getByRole("region", { name: "Investigation result" });
  await expect(result.getByText("Accepted", { exact: true })).toBeVisible();
  await expect(result.getByText("pkg/policy.py", { exact: true })).toBeVisible();
  await expect(page.getByText("Explore a completed example")).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Ask a follow-up" })).toHaveCount(0);

  const viewport = page.viewportSize()!;
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(viewport.width);
  if (viewport.width <= 930) {
    await expect(page.getByRole("button", { name: "Show investigations" })).toHaveAttribute("aria-expanded", "false");
    expect((await page.getByRole("main").boundingBox())!.y).toBe(0);
  }
  await page.evaluate(axe.source);
  const scan = await page.evaluate(async () => {
    const browserAxe = (window as unknown as { axe: typeof import("axe-core") }).axe;
    return browserAxe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] } });
  });
  expect(scan.violations.map(({ id, nodes }) => ({ id, targets: nodes.map((node) => node.target) }))).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath("chat.png"), fullPage: true });

  if (await page.getByRole("button", { name: "Show investigations" }).count()) {
    await page.getByRole("button", { name: "Show investigations" }).click();
  }
  await page.getByRole("textbox", { name: "Filter investigations" }).focus();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: "Show investigations" })).toBeFocused();
  await expect(page.getByRole("complementary", { name: "Investigation threads" })).toHaveCount(0);

  await result.getByRole("button", { name: "Explore source evidence" }).click();
  await expect(page.getByText(/snapshot matched/)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Reading path", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Reading route", exact: true })).toHaveAttribute("aria-pressed", "true");
  const focusedNodes = await page.locator(".react-flow__node").count();
  await expect(page.locator(".react-flow__minimap")).toHaveCount(0);
  await page.getByRole("button", { name: /All evidence/ }).click();
  await expect(page.locator(".react-flow__minimap")).toHaveCount(1);
  await expect.poll(() => page.locator(".react-flow__node").count()).toBeGreaterThan(focusedNodes);
  await page.getByRole("button", { name: "Reading route", exact: true }).click();
  await expect(page.locator(".react-flow__node")).toHaveCount(focusedNodes);
  await page.screenshot({ path: testInfo.outputPath("path.png"), fullPage: true });
  const nav = page.getByRole("navigation", { name: "Primary lenses" });
  for (const [label, lens, heading] of [
    ["Plan", "plan", "Plan revisions"],
    ["Run trace", "trace", "Run trace"],
    ["Context", "context", "Context inspector"],
    ["Memory", "memory", "Repository memory"],
    ["Cost", "cost", "Cost analysis"],
  ]) {
    await nav.getByRole("button", { name: label, exact: true }).click();
    await expect(page).toHaveURL(new RegExp("lens=" + lens));
    await expect(nav.getByRole("button", { name: label, exact: true })).toHaveAttribute("aria-current", "page");
    await expect(page.getByRole("heading", { name: heading, exact: true })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(viewport.width);
  }
  await page.reload();
  await expect(page.getByRole("heading", { name: "Cost analysis", exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("cost.png"), fullPage: true });
  expect(pageErrors).toEqual([]);
  expect(failedResponses).toEqual([]);
  expect(writes).toEqual([]);
});
