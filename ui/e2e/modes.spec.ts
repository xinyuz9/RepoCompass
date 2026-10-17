import { expect, test } from "@playwright/test";
import axe from "axe-core";

for (const mode of ["quick", "detailed"] as const) {
  test(`real ${mode} mode persists across refresh and follow-up`, async ({ page, request }, testInfo) => {
    const posts: { url: string; body: Record<string, unknown> }[] = [];
    page.on("request", entry => {
      if (entry.method() === "POST") posts.push({ url: entry.url(), body: entry.postDataJSON() });
    });
    await page.goto("/?compose=1");
    const label = mode === "quick" ? "Quick lookup" : "Detailed investigation";
    await page.getByRole("radio", { name: new RegExp(label) }).check();
    await page.getByRole("textbox", { name: "What should RepoCompass investigate?" }).fill("Locate the fixture answer function.");
    await page.getByText("Task settings", { exact: true }).click();
    if (mode === "detailed") await page.getByLabel("Review before finalize", { exact: true }).uncheck();
    else await expect(page.getByLabel(/Review before finalize/)).toBeDisabled();
    await page.getByText("Task settings", { exact: true }).click();
    await page.getByRole("button", { name: "Start investigation", exact: true }).scrollIntoViewIfNeeded();
    await expect(page.getByRole("button", { name: "Start investigation", exact: true })).toBeInViewport({ ratio: 1 });
    await page.evaluate(axe.source);
    const violations = await page.evaluate(async () => {
      const result = await (window as unknown as { axe: typeof import("axe-core") }).axe.run(document,
        { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] } });
      return result.violations.map(item => item.id);
    });
    expect(violations).toEqual([]);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize()!.width);
    await page.screenshot({ path: testInfo.outputPath("mode-picker.png"), fullPage: true, animations: "disabled" });
    await page.getByRole("button", { name: "Start investigation", exact: true }).click();
    await expect(page.getByLabel("Saved investigation mode")).toContainText(label);
    const resultLabel = mode === "quick" ? "Source checks passed" : "Workflow accepted";
    await expect(page.getByText(resultLabel, { exact: true })).toBeVisible();
    expect(posts[0]!.body.investigation_mode).toBe(mode);
    await page.reload();
    await expect(page.getByLabel("Saved investigation mode")).toContainText(label);
    expect(posts.filter(entry => entry.url.endsWith("/investigations"))).toHaveLength(1);
    await page.getByRole("textbox", { name: "Ask a follow-up" }).fill("Explain the same source location.");
    await page.getByRole("button", { name: "Send follow-up" }).click();
    await expect(page.locator(".conversation-goal")).toContainText("turn 2");
    await expect(page.getByText(resultLabel, { exact: true })).toBeVisible();
    const threadId = new URL(page.url()).searchParams.get("thread");
    const detail = await (await request.get(`/api/v1/threads/${threadId}`)).json();
    expect(detail.thread.architecture_preset).toBe(mode === "quick" ? "tool_wrapper" : "evaluator_loop_context_builder");
    expect(detail.thread.final_result.relevant_files).toEqual(["example.py"]);
    const runId = detail.attempts[0].attempt.run_id;
    const run = await (await request.get(`/api/v1/runs/${runId}`)).json();
    const nodes = run.trace_events.map((event: { node: string }) => event.node);
    expect(nodes.includes("evaluate_candidate")).toBe(mode === "detailed");
    expect(nodes.includes("validate_quick_candidate")).toBe(mode === "quick");
    await page.getByText("Inspect this investigation", { exact: true }).click();
    for (const name of ["Memory", "Trace", "Cost"]) {
      await expect(page.getByRole("button", { name, exact: true })).toBeVisible();
    }
    await page.screenshot({ path: testInfo.outputPath("mode-result.png"), fullPage: true, animations: "disabled" });
  });
}
