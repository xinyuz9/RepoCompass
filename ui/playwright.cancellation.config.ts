import { defineConfig } from "@playwright/test";
import standard from "./playwright.config";

export default defineConfig({
  ...standard,
  testMatch: "**/cancellation.spec.ts",
  testIgnore: [],
  outputDir: "test-results-cancellation",
  use: { ...standard.use, baseURL: "http://127.0.0.1:8879" },
  webServer: {
    command: "uv run --project .. python e2e/cancellation_server.py",
    url: "http://127.0.0.1:8879/api/v1/system",
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
