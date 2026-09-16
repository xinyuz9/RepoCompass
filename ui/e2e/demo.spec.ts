import { expect, test } from "@playwright/test";
import axe from "axe-core";

test("shares selected evidence and keeps chat beside keyboard-readable inspectors", async ({ page }, testInfo) => {
  const writes: string[] = [];
  page.on("request", (request) => { if (!["GET", "HEAD"].includes(request.method())) writes.push(request.url()); });
  await page.goto("/?lens=topology");
  await page.getByRole("button", { name: "Chat + inspector", exact: true }).click();
  await expect(page.getByRole("region", { name: "Conversation beside evidence" })).toBeVisible();
  await page.getByRole("button", { name: "Evidence table", exact: true }).click();
  const inspect = page.getByRole("button", { name: /^Inspect / }).first();
  await inspect.focus();
  await page.keyboard.press("Enter");
  const nodeId = new URL(page.url()).searchParams.get("node");
  expect(nodeId).toMatch(/^[a-f0-9]{64}$/);
  await expect(inspect).toHaveAttribute("aria-pressed", "true");
  expect(new URL(page.url()).searchParams.get("split")).toBe("1");
  await page.reload();
  await expect(page.getByRole("region", { name: "Conversation beside evidence" })).toBeVisible();
  await page.getByRole("button", { name: "Evidence table", exact: true }).click();
  await expect(page.getByRole("button", { name: /^Inspect / }).first()).toHaveAttribute("aria-pressed", "true");
  await page.getByText("中文 / English · terminology", { exact: true }).click();
  await expect(page.getByText("Goal · 当前目标", { exact: true })).toBeVisible();
  await page.evaluate(axe.source);
  const violations = await page.evaluate(async () => (await (window as unknown as { axe: typeof import("axe-core") }).axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21aa"] } })).violations.map((v) => ({ id: v.id, targets: v.nodes.map((node) => node.target) })));
  expect(violations).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize()!.width);
  await page.getByText("中文 / English · terminology", { exact: true }).click();
  const sourcePanel = page.locator(".inspector-section--source");
  const preview = page.locator(".source-preview");
  await expect(preview).toBeVisible();
  const sourceBounds = (await sourcePanel.boundingBox())!;
  const previewBounds = (await preview.boundingBox())!;
  expect(previewBounds.y + previewBounds.height).toBeLessThanOrEqual(sourceBounds.y + sourceBounds.height);
  await page.screenshot({ path: testInfo.outputPath("split-review.png"), fullPage: true });
  const nav = page.getByRole("navigation", { name: "Primary lenses" });
  await nav.getByRole("button", { name: "Cost", exact: true }).click();
  await expect(page.getByRole("region", { name: "Conversation beside evidence" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Cost analysis", exact: true })).toBeVisible();
  expect(new URL(page.url()).searchParams.has("node")).toBe(false);
  expect(writes).toEqual([]);
});

test("compares real saved evidence on demand with accessible drift tables", async ({ page }, testInfo) => {
  const reads: string[] = [], writes: string[] = [];
  page.on("request", (request) => {
    if (request.url().includes("/drift?")) reads.push(request.url());
    if (!["GET", "HEAD"].includes(request.method())) writes.push(request.url());
  });
  await page.goto("/?lens=topology");
  const panel = page.getByRole("region", { name: "Evidence drift", exact: true });
  await expect(panel).toBeVisible();
  expect(reads).toHaveLength(0);
  await panel.getByRole("button", { name: "Compare saved evidence with HEAD" }).click();
  await expect(panel.getByText("Clean worktree")).toBeVisible();
  expect(reads).toHaveLength(1);
  await panel.getByText(/Source fingerprints \(/).click();
  await panel.getByText(/Claims, entities & same-snapshot memory/).click();
  await expect(panel.getByRole("table", { name: "Recorded source files versus current checkout" })).toBeVisible();
  await page.evaluate(axe.source);
  const violations = await page.evaluate(async () => (await (window as unknown as { axe: typeof import("axe-core") }).axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21aa"] } })).violations.map((v) => ({ id: v.id, nodes: v.nodes.map((node) => ({ target: node.target, html: node.html })) })));
  expect(violations).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize()!.width);
  await panel.screenshot({ path: testInfo.outputPath("drift-comparison.png") });
  expect(writes).toHaveLength(0);
});

test("scripted review history is keyboard-readable, bounded and read-only", async ({ page }, testInfo) => {
  // UI contract fixture only; actual SQLite/HTTP lifecycle is covered in Python integration tests.
  const writes: string[] = [];
  page.on("request", (request) => {
    if (!["GET", "HEAD"].includes(request.method())) writes.push(request.url());
  });
  await page.route("**/conversation?*", async (route) => {
    const response = await route.fetch();
    const payload = await response.json();
    const turn = payload.turns.at(-1);
    turn.review_history = {
      omitted_count: 0, prior_history_unavailable: false,
      records: ["revise", "approve"].map((action, index) => ({
        review_number: index + 1, request_digest: "a".repeat(64),
        candidate_summary: index === 0 ? "Verification precedes proposal application." : "Verification and regression evidence now support the result.",
        evaluation_passed: true, deterministic_issue_count: 0,
        allowed_actions: ["approve", "revise", "cancel"], action,
        feedback: index === 0 ? "Please trace the rejected-proposal branch and include its regression test. <script>not executable</script>" : null,
        text_truncated: false,
      })),
    };
    await route.fulfill({ response, json: payload });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Review history", exact: true }).click();
  const reviews = page.getByRole("region", { name: "Current turn review history" });
  await expect(reviews.getByText("Revision requested", { exact: true })).toBeVisible();
  const summary = reviews.locator("summary").first();
  await summary.focus();
  await page.keyboard.press("Enter");
  await expect(reviews.getByText(/Please trace the rejected-proposal branch/)).toBeVisible();
  expect(await reviews.locator("script").count()).toBe(0);
  await reviews.locator("summary").nth(1).click();
  await expect(reviews.getByText("Verification and regression evidence now support the result.")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize()!.width);
  await page.evaluate(axe.source);
  const scan = await page.evaluate(async () => {
    const browserAxe = (window as unknown as { axe: typeof import("axe-core") }).axe;
    return browserAxe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] } });
  });
  expect(scan.violations.map(({ id }) => id)).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath("review-history.png"), fullPage: true, animations: "disabled" });
  await page.reload();
  await page.getByRole("button", { name: "Review history", exact: true }).click();
  await expect(reviews.getByText("Approved", { exact: true })).toBeVisible();
  expect(writes).toEqual([]);
});

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
  await page.getByRole("button", { name: "Review history", exact: true }).click();
  await expect(page.getByText(/Automatic completion is not human approval/).last()).toBeVisible();
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
    if (lens === "trace") {
      const search = page.getByRole("searchbox", { name: "Find node, event kind or #sequence" });
      const rows = page.getByRole("list", { name: "Trace events" }).getByRole("listitem");
      expect(await rows.count()).toBeLessThanOrEqual(50);
      if (viewport.width <= 930) {
        // Chromium animation transforms can add a fractional CSS pixel.
        expect((await page.getByRole("list", { name: "Trace events" }).boundingBox())!.height).toBeLessThanOrEqual(Math.min(viewport.height / 2, 420) + 0.5);
      }
      await search.fill("#0");
      await expect(rows).toHaveCount(1);
      await expect(rows.getByText("#0", { exact: true })).toBeVisible();
      await search.fill("no-matching-node");
      await expect(page.getByRole("heading", { name: "No matching events" })).toBeVisible();
      await search.fill("");
      await page.getByRole("button", { name: "Follow latest" }).focus();
      await page.keyboard.press("Enter");
      await expect(page.getByRole("button", { name: "Follow latest" })).toHaveAttribute("aria-pressed", "true");
      const listBounds = (await page.getByRole("list", { name: "Trace events" }).boundingBox())!;
      const selectedBounds = (await page.getByRole("list", { name: "Trace events" }).locator('button[aria-pressed="true"]').boundingBox())!;
      expect(selectedBounds.y).toBeGreaterThanOrEqual(listBounds.y - 1);
      expect(selectedBounds.y + selectedBounds.height).toBeLessThanOrEqual(listBounds.y + listBounds.height + 1);
      const payload = page.getByRole("region", { name: "Public event payload" });
      await payload.focus();
      await expect(payload).toBeFocused();
      await expect(page.getByRole("link", { name: "Skip to workspace" })).toHaveCSS("opacity", "0");
      if (await payload.evaluate((element) => element.scrollHeight > element.clientHeight)) {
        await payload.press("End");
        await expect.poll(() => payload.evaluate((element) => element.scrollTop)).toBeGreaterThan(0);
        await payload.press("Home");
      }
      await page.evaluate(axe.source);
      const traceScan = await page.evaluate(async () => {
        const browserAxe = (window as unknown as { axe: typeof import("axe-core") }).axe;
        return browserAxe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] } });
      });
      expect(traceScan.violations.map(({ id, nodes }) => ({ id, targets: nodes.map((node) => node.target) }))).toEqual([]);
    }
    if (lens === "trace" || lens === "plan") {
      await page.screenshot({ path: testInfo.outputPath(`${lens}.png`), fullPage: true, animations: "disabled" });
    }
  }
  await page.reload();
  await expect(page.getByRole("heading", { name: "Cost analysis", exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("cost.png"), fullPage: true });
  expect(pageErrors).toEqual([]);
  expect(failedResponses).toEqual([]);
  expect(writes).toEqual([]);
});
