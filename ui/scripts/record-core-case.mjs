// One command, two local fixtures, no provider calls. Prior output is never replaced.
import { execFileSync, spawn } from "node:child_process";
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import process from "node:process";
import { assertRecordingEvidence } from "./recording-contract.mjs";

const { values } = parseArgs({ options: { output: { type: "string" } } });
if (!values.output) throw new Error("Use --output <new directory with an existing parent>");
if (process.platform !== "linux") throw new Error("Run this recording inside Linux / WSL");
execFileSync("ffprobe", ["-version"], { stdio: "ignore" });
const output = resolve(values.output), ui = resolve(dirname(fileURLToPath(import.meta.url)), "..");
// Playwright may clear its output directory; only hand it our newly created child.
await mkdir(output);
const evidence = resolve(output, "evidence");
const child = spawn(process.execPath, [resolve(ui, "node_modules/@playwright/test/cli.js"), "test", "--config", "playwright.portfolio.config.ts", "--output", evidence], { cwd: ui, stdio: "inherit" });
const code = await new Promise((done, reject) => { child.once("error", reject); child.once("exit", done); });
if (code !== 0) throw new Error(`Recording failed (${code}); preserved diagnostics at ${output}`);
const entries = await readdir(evidence, { withFileTypes: true });
const cases = entries.filter(entry => entry.isDirectory());
if (cases.length !== 1) throw new Error("Expected exactly one recorded core case");
const caseDir = resolve(evidence, cases[0].name);
const report = JSON.parse(await readFile(resolve(caseDir, "recording.json"), "utf8"));
assertRecordingEvidence(report);
const video = resolve(caseDir, "video.webm");
const duration = Number(execFileSync("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video], { encoding: "utf8" }).trim());
if (!Number.isFinite(duration) || duration < 64 || duration > 180) throw new Error("Recording duration outside 64–180 seconds");
const metadata = { ...report, duration_seconds: duration, video_sha256: createHash("sha256").update(await readFile(video)).digest("hex"),
  code_commit: execFileSync("git", ["rev-parse", "HEAD"], { cwd: ui, encoding: "utf8" }).trim(),
  worktree_dirty: execFileSync("git", ["status", "--porcelain"], { cwd: ui, encoding: "utf8" }).trim().length > 0,
  video_path: video, evidence_directory: caseDir, recorded_at: new Date().toISOString() };
await writeFile(resolve(output, "recording.json"), JSON.stringify(metadata, null, 2) + "\n", { flag: "wx" });
console.log(JSON.stringify({ video, duration_seconds: duration, provider_calls: 0, output }, null, 2));
