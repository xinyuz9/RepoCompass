import { expect, test, type Locator } from "@playwright/test";
import { writeFile } from "node:fs/promises";
// @ts-expect-error Node-only recording contract is separately unit tested.
import { allowsRecordingRequest, assertRecordingEvidence, DEMO_ORIGIN, RECOVERY_ORIGIN } from "../scripts/recording-contract.mjs";

test("proposal evidence with explicitly separate crash recovery", async ({ page, context, request }, testInfo) => {
  const blocked: string[] = [], posts: string[] = [], errors: string[] = [];
  const shots: { name: string; caption: string; test_elapsed_ms: number }[] = [];
  let reconcileUrl: string | null = null;
  const started = performance.now();
  await context.route("**/*", async route => {
    const req = route.request();
    if (!allowsRecordingRequest(req.url(), req.method(), reconcileUrl)) {
      blocked.push(`${req.method()} ${req.url()}`);
      return route.abort("blockedbyclient");
    }
    if (req.method() === "POST") posts.push(req.url());
    return route.continue();
  });
  page.on("pageerror", error => errors.push(error.message));
  const shot = async (name: string, caption: string, focus?: Locator) => {
    await page.evaluate(text => {
      document.querySelector("[data-recording-annotation]")?.remove();
      if (!document.querySelector("[data-recording-layout]")) {
        const style = document.createElement("style");
        style.dataset.recordingLayout = "true";
        style.textContent = ".app-shell { height: calc(100dvh - 64px); min-height: 0; margin-top: 64px; }";
        document.head.appendChild(style);
      }
      const banner = document.createElement("div");
      banner.dataset.recordingAnnotation = "true";
      banner.textContent = text;
      Object.assign(banner.style, { position: "fixed", top: "0", height: "64px", left: "0", right: "0", zIndex: "9999",
        padding: "14px 22px", background: "#14281f", color: "#e1f5ce", borderBottom: "1px solid #a7d986",
        font: "15px sans-serif", lineHeight: "1.6" });
      document.body.appendChild(banner);
    }, caption);
    if (focus) {
      await focus.scrollIntoViewIfNeeded();
      const bounds = (await focus.boundingBox())!;
      expect(bounds.y).toBeGreaterThanOrEqual(64);
      expect(bounds.y + bounds.height).toBeLessThanOrEqual(1000);
      expect(await focus.evaluate(node => {
        const box = node.getBoundingClientRect();
        return node.contains(document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2));
      })).toBe(true);
    }
    shots.push({ name, caption, test_elapsed_ms: Math.round(performance.now() - started) });
    await page.screenshot({ path: testInfo.outputPath(`${name}.png`) });
    await page.waitForTimeout(8000); // Viewer dwell, not a readiness workaround.
  };
  const demoSystem = await (await request.get(`${DEMO_ORIGIN}/api/v1/system`)).json();
  expect(demoSystem.demo_mode).toBe(true);
  expect(demoSystem.lifecycle_mutations_enabled).toBe(false);
  expect(demoSystem.memory_mutations_enabled).toBe(false);
  const mainUrl = `${DEMO_ORIGIN}/?thread=demo-proposal-gate&lens=chat`;
  await page.goto(mainUrl);
  await expect(page.getByRole("region", { name: "Investigation result", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Hide investigations", exact: true }).click();
  await shot("01-question", "1/8 · 核心问题：proposal 在哪里验证、何时应用？｜确定性模型演示，真实工具／存储；不是实时模型推理");
  await page.getByRole("button", { name: "Explore source evidence", exact: true }).click();
  await page.getByRole("button", { name: "Evidence table", exact: true }).click();
  await page.getByRole("button", { name: "Inspect pkg.api.apply_proposal", exact: true }).click();
  const source = page.getByRole("region", { name: "Source preview for pkg/api.py", exact: true });
  await expect(source).toContainText("verifier.verify");
  await source.scrollIntoViewIfNeeded();
  await shot("02-source", "2/8 · 同一结论回到源码：先 verifier.verify，再写入 workspace；页面展示的是已保存证据",
    source.getByText("workspace[proposal.target] = proposal.replacement", { exact: true }));
  await page.goto(`${DEMO_ORIGIN}/?thread=demo-proposal-gate&lens=plan`);
  const plan = page.getByRole("heading", { name: "Plan revisions", exact: true });
  await plan.scrollIntoViewIfNeeded();
  await shot("03-plan", "3/8 · 查看计划与验收记录；accepted 是流程状态，不代表任意代码语义已被证明");
  await page.goto(`${DEMO_ORIGIN}/?thread=demo-proposal-gate&lens=cost&run=demo-run-001`);
  await page.getByRole("heading", { name: "Cost analysis", exact: true }).scrollIntoViewIfNeeded();
  await shot("04-cost", "4/8 · 费用可追踪：本页 usage／pricing 是确定性演示数据，不是实际付费金额；真实效果见独立 pilot");

  // A distinct fixture, explicitly NOT the proposal task or a real provider failure.
  const system = await (await request.get(`${RECOVERY_ORIGIN}/api/v1/system`)).json();
  const setup = await request.post(`${RECOVERY_ORIGIN}/__test__/orphan`);
  expect(setup.ok()).toBe(true);
  const orphan = await setup.json();
  expect(orphan.request_id).toMatch(/^orphan-request-[a-f0-9]{32}$/);
  await page.goto(`${RECOVERY_ORIGIN}/?compose=1`);
  await page.evaluate(({ orphan, system }) => {
    const command = { schema_version: 1, request_id: orphan.request_id, goal: "Explain the flag.", focus_path: ".",
      expected_repository_id: system.repository_id, expected_commit_sha: system.commit_sha,
      human_review_policy: "never", memory_retrieval_enabled: true, max_attempt_cost_usd: null };
    sessionStorage.setItem("repocompass.pending-lifecycle.v1", JSON.stringify([{ schema_version: 1,
      recorded_at_ms: Date.now(), kind: "start", thread_id: null, command }]));
  }, { orphan, system });
  await page.reload();
  await expect(page.getByRole("button", { name: "Reconcile saved run", exact: true })).toBeVisible();
  expect(posts).toEqual([]);
  await shot("05-crash", "5/8 · 独立故障注入：Explain the flag 子进程真实退出，未调用付费模型；刷新没有自动重发任务");
  reconcileUrl = `${RECOVERY_ORIGIN}/api/v1/operations/${orphan.request_id}/reconcile`;
  await page.getByRole("button", { name: "Reconcile saved run", exact: true }).click();
  await expect(page.getByText("Local records reconciled", { exact: true })).toBeVisible();
  await shot("06-reconcile", "6/8 · 一次显式 Reconcile 只协调本地记录，不把失败改成成功，也没有重跑模型");
  await page.getByRole("button", { name: "Open reconciled thread", exact: true }).click();
  const resume = page.getByRole("button", { name: "Resume", exact: true });
  await expect(resume).toBeVisible();
  await resume.scrollIntoViewIfNeeded();
  await shot("07-resume-boundary", "7/8 · Resume 是另一次用户决定；此录像不点击它，也不声称恢复后答案正确", resume);
  const recovered = await (await request.get(`${RECOVERY_ORIGIN}/api/v1/runs/${orphan.run_id}`)).json();
  await page.goto(mainUrl);
  await page.getByRole("region", { name: "Investigation result", exact: true }).scrollIntoViewIfNeeded();
  await shot("08-limitations", "8/8 · 回到 proposal 示例｜工程可追踪 ≠ 模型效果更好；真实 v11 未证明复杂流程／memory 胜过简单 baseline");
  const report = { schema_version: 1, provenance: "scripted_demo_with_separate_crash_fixture",
    demo_thread_id: "demo-proposal-gate", demo_run_id: "demo-run-001", demo_commit_sha: demoSystem.commit_sha,
    provider_calls: 0, resume_invoked: false, fixture_setup_posts: [`${RECOVERY_ORIGIN}/__test__/orphan`],
    browser_posts: posts, blocked_requests: blocked, browser_errors: errors, reconcile_url: reconcileUrl,
    recovery_model_invocations: recovered.model_invocations.length, recovery_status: recovered.attempt.status,
    recovery_error_code: recovered.trace_events.at(-1)?.payload.error_code, shots,
    annotation: "Recording-only 64px caption strip with reserved space; not product text, model output or real paid usage." };
  await writeFile(testInfo.outputPath("recording.json"), JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  assertRecordingEvidence(report);
});
