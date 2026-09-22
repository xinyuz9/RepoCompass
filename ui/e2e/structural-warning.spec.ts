import { expect, test } from "@playwright/test";
import axe from "axe-core";

test("optional structural warnings remain inspectable without a fatal error", async ({ page }, testInfo) => {
  // Presentation-only fixture; no model call or historical record is changed.
  let injected = 0;
  const writes: string[] = [];
  page.on("request", request => { if (request.method() !== "GET") writes.push(request.url()); });
  await page.route("**/api/v1/threads/*", async route => {
    const response = await route.fetch();
    const data = await response.json();
    for (const frame of data.evaluator_timeline ?? []) {
      frame.report.structural_warnings = [{
        code: "structure_record_limit_exceeded",
        message: "Optional graph enrichment reached the AST record cap.",
        path: "django/db/models/fields/__init__.py",
      }];
      injected += 1;
    }
    await route.fulfill({ response, json: data });
  });
  await page.goto("/");
  await page.getByRole("navigation", { name: "Primary lenses" }).getByRole("button", { name: "Plan", exact: true }).click();
  const summary = page.getByText("Limited structural enrichment (1)").first();
  await expect(summary).toBeVisible();
  expect(injected).toBeGreaterThan(0);
  await expect(summary.locator("..")).not.toHaveAttribute("open");
  await summary.focus();
  await page.keyboard.press("Enter");
  await expect(summary.locator("..").getByText("django/db/models/fields/__init__.py")).toBeVisible();
  await expect(summary.locator("..").getByText(/does not verify missing symbols or relations/)).toBeVisible();
  const warnings = summary.locator("..").locator(".structural-warnings");
  expect(await warnings.evaluate(node => node.scrollWidth <= node.clientWidth)).toBe(true);
  const plan = page.getByRole("region", { name: "Recorded plan steps" });
  if (await plan.evaluate(node => node.scrollHeight > node.clientHeight)) {
    await plan.focus();
    await page.keyboard.press("End");
    await expect.poll(() => plan.evaluate(node => node.scrollTop)).toBeGreaterThan(0);
    await page.keyboard.press("Home");
    await expect.poll(() => plan.evaluate(node => node.scrollTop)).toBe(0);
  }
  await summary.locator("..").scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize()!.width);
  await page.evaluate(axe.source);
  const violations = await page.evaluate(async () => (await (window as unknown as { axe: typeof import("axe-core") }).axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21aa"] } })).violations.map(v => ({ id: v.id, nodes: v.nodes.map(node => ({ target: node.target, html: node.html })) })));
  expect(violations).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath("structural-warning.png"), fullPage: true, animations: "disabled" });
  expect(writes).toEqual([]);
});
