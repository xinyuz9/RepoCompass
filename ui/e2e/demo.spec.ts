import { expect, test } from "@playwright/test";
import axe from "axe-core";

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
