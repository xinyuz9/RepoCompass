import assert from "node:assert/strict";
import test from "node:test";
import { allowsRecordingRequest, assertRecordingEvidence, DEMO_ORIGIN, RECOVERY_ORIGIN } from "./recording-contract.mjs";

const repair = `${RECOVERY_ORIGIN}/api/v1/operations/orphan-request-${"a".repeat(32)}/reconcile`;
const evidence = () => ({ schema_version: 1, provenance: "scripted_demo_with_separate_crash_fixture", provider_calls: 0,
  resume_invoked: false, shots: Array.from({ length: 8 }, (_, index) => ({ name: String(index) })),
  blocked_requests: [], browser_errors: [], browser_posts: [repair], reconcile_url: repair,
  recovery_model_invocations: 0, recovery_status: "failed", recovery_error_code: "orphan_reconciled" });

test("admits only reads on the two owned loopback origins", () => {
  assert.equal(allowsRecordingRequest(`${DEMO_ORIGIN}/api/v1/system`, "GET"), true);
  assert.equal(allowsRecordingRequest(`${RECOVERY_ORIGIN}/assets/example.css`, "HEAD"), true);
  for (const url of ["https://api.openai.com/v1/models", "http://localhost:8884/", "http://127.0.0.1:8766/", "http://user:pw@127.0.0.1:8884/", "not a URL"]) assert.equal(allowsRecordingRequest(url, "GET"), false);
});
test("repair must match the exact explicitly selected request", () => {
  assert.equal(allowsRecordingRequest(repair, "POST", repair), true);
  for (const url of [repair, `${DEMO_ORIGIN}/api/v1/investigations`, `${RECOVERY_ORIGIN}/__test__/orphan`]) assert.equal(allowsRecordingRequest(url, "POST"), false);
  for (const url of [repair + "?replay=1", repair.replace("a".repeat(32), "b".repeat(32)), `${DEMO_ORIGIN}/api/v1/threads/x/resume`]) assert.equal(allowsRecordingRequest(url, "POST", repair), false);
  assert.equal(allowsRecordingRequest(repair, "DELETE", repair), false);
});
test("accepts an honest complete demo and crash-repair record", () => assert.doesNotThrow(() => assertRecordingEvidence(evidence())));
for (const [field, value] of Object.entries({ provider_calls: 1, resume_invoked: true, provenance: "live_model_success", recovery_status: "completed", recovery_model_invocations: 1, browser_posts: [repair, repair], shots: [], blocked_requests: ["unexpected"], browser_errors: ["error"] })) {
  test(`rejects misleading or incomplete ${field}`, () => assert.throws(() => assertRecordingEvidence({ ...evidence(), [field]: value })));
}
