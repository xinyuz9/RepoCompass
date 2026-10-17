import { defineConfig } from "@playwright/test";
import standard from "./playwright.config";

export default defineConfig({
  ...standard,
  testMatch: "**/modes.spec.ts",
  testIgnore: [],
  outputDir: "test-results-modes",
  use: { ...standard.use, baseURL: "http://127.0.0.1:8881" },
  webServer: {
    command: "uv run --project .. python e2e/modes_server.py",
    url: "http://127.0.0.1:8881/api/v1/system",
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
