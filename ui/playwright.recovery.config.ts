import { defineConfig } from "@playwright/test";
import standard from "./playwright.config";

export default defineConfig({
  ...standard,
  testMatch: "**/recovery.spec.ts", testIgnore: [], outputDir: "test-results-recovery",
  use: { ...standard.use, baseURL: "http://127.0.0.1:8880" },
  webServer: {
    command: "uv run --project .. python e2e/recovery_server.py",
    url: "http://127.0.0.1:8880/api/v1/system", reuseExistingServer: false, timeout: 60_000,
  },
});
