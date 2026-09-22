export const DEMO_ORIGIN = "http://127.0.0.1:8884";
export const RECOVERY_ORIGIN = "http://127.0.0.1:8880";

// Recording is not a capability grant: only public reads and one exact local repair.
export function allowsRecordingRequest(rawUrl, method, reconcileUrl = null) {
  let url;
  try { url = new URL(rawUrl); } catch { return false; }
  if (url.username || url.password || ![DEMO_ORIGIN, RECOVERY_ORIGIN].includes(url.origin)) return false;
  if (["GET", "HEAD"].includes(method)) return true;
  return method === "POST" && rawUrl === reconcileUrl
    && new RegExp(`^${RECOVERY_ORIGIN}/api/v1/operations/orphan-request-[a-f0-9]{32}/reconcile$`).test(rawUrl);
}

export function assertRecordingEvidence(report) {
  if (report.schema_version !== 1 || report.provenance !== "scripted_demo_with_separate_crash_fixture") throw new Error("Recording provenance missing");
  if (report.provider_calls !== 0 || report.resume_invoked !== false) throw new Error("Recording must not replay models or resume work");
  if (report.shots?.length !== 8 || new Set(report.shots.map(shot => shot.name)).size !== 8) throw new Error("Expected eight distinct story frames");
  if (report.blocked_requests?.length !== 0 || report.browser_errors?.length !== 0) throw new Error("Recording encountered blocked traffic or browser errors");
  if (report.browser_posts?.length !== 1 || !allowsRecordingRequest(report.browser_posts[0], "POST", report.reconcile_url)) throw new Error("Expected exactly one explicit repair");
  if (report.recovery_model_invocations !== 0 || report.recovery_status !== "failed" || report.recovery_error_code !== "orphan_reconciled") throw new Error("Recovery evidence does not match the crash fixture");
}
