import { expect, test } from "@playwright/test";
import axe from "axe-core";

test("opens source directly from a result and retains its deep link", async ({ page }, testInfo) => {
  const writes: string[] = [];
  page.on("request", request => { if (request.method() !== "GET") writes.push(request.url()); });
  await page.goto("/");
  const source = page.getByRole("button", { name: "View source: pkg/policy.py", exact: true });
  await expect(source).toBeVisible();
  await source.focus();
  await page.keyboard.press("Enter");
  const url = new URL(page.url());
  expect(url.searchParams.get("node")).toMatch(/^[a-f0-9]{64}$/);
  expect(url.searchParams.get("split")).toBe("1");
  await expect(page.getByRole("region", { name: "Source preview for pkg/policy.py" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Evidence window" })).toBeFocused();
  await expect(page.getByRole("heading", { name: "Evidence window" })).toBeInViewport();
  await page.reload();
  await expect(page.getByRole("region", { name: "Source preview for pkg/policy.py" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Evidence window" })).toBeInViewport();
  await expect(page.locator(".source-preview .source-line").first()).toBeInViewport();
  await page.screenshot({ path: testInfo.outputPath("result-source.png") });
  expect(writes).toEqual([]);
});

test("explains a stopped result without automatic retries or budget changes", async ({ page }, testInfo) => {
  // Presentation fixture only; the demo's actual result and ledger are unchanged.
  const writes: string[] = [];
  page.on("request", request => { if (request.method() !== "GET") writes.push(request.url()); });
  await page.route("**/api/v1/threads/*", async route => {
    const response = await route.fetch();
    const data = await response.json();
    if (data.thread) {
      data.thread.termination_status = "exhausted";
      data.thread.step_stop_reason = "attempt_cost_budget_exhausted";
    }
    await route.fulfill({ response, json: data });
  });
  await page.goto("/");
  const explanation = page.getByRole("complementary", { name: "Why this investigation stopped" });
  await expect(explanation.getByText("Spending limit reached")).toBeVisible();
  await explanation.scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize()!.width);
  await page.evaluate(axe.source);
  const violations = await page.evaluate(async () => (await (window as unknown as { axe: typeof import("axe-core") }).axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21aa"] } })).violations.map(v => v.id));
  expect(violations).toEqual([]);
  const inspectCost = explanation.getByRole("button", { name: "Inspect cost", exact: true });
  await inspectCost.scrollIntoViewIfNeeded();
  await expect(inspectCost).toBeInViewport();
  expect(await inspectCost.evaluate(node => {
    const box = node.getBoundingClientRect();
    return node.contains(document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2));
  })).toBe(true);
  await page.screenshot({ path: testInfo.outputPath("result-stopped.png") });
  await inspectCost.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { name: "Cost analysis", exact: true })).toBeVisible();
  expect(writes).toEqual([]);
});
