import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e", testMatch: "core-case.spec.ts", fullyParallel: false, workers: 1, retries: 0,
  timeout: 180_000, reporter: "list",
  use: { viewport: { width: 1440, height: 1000 }, browserName: "chromium", reducedMotion: "reduce",
    video: { mode: "on", size: { width: 1440, height: 1000 } }, screenshot: "only-on-failure",
    trace: "retain-on-failure", serviceWorkers: "block", acceptDownloads: false },
  webServer: [
    { command: "../.venv/bin/repocompass-dashboard --demo --port 8884 --log-level warning",
      url: "http://127.0.0.1:8884/api/v1/system", reuseExistingServer: false, timeout: 60_000 },
    { command: "../.venv/bin/python e2e/recovery_server.py",
      url: "http://127.0.0.1:8880/api/v1/system", reuseExistingServer: false, timeout: 60_000 },
  ],
});
