"""Product modes keep the existing graph and bind its profile to each thread."""

import hashlib
import json
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from repocompass.agent.architecture import profile_for_mode
from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.lifecycle import (
    ContinueRequest,
    HumanReviewPolicy,
    HumanResponseRequest,
    HumanDecision,
    HumanDecisionAction,
)
from repocompass.agent.nodes import finalize_cost_budget_exhausted
from repocompass.agent.runner import AgentRunner, RunRequest, ResumeRequest
from repocompass.dashboard.contracts import DashboardStartCommand, DashboardThreadView
from repocompass.service.contracts import StartInvestigationRequest
from repocompass.service.investigation import InvestigationService
from tests.model_doubles import with_scripted_planner


def candidate(path="example.py", line=1, symbol=None):
    return CandidateResult(
        evidence=[
            EvidenceItem(
                path=path, line_start=line, symbol=symbol, claim="Source location."
            )
        ],
        summary="Located source.",
        unresolved_questions=[],
    )


def models(result=None, explore=None):
    explorer = MagicMock(spec=BaseChatModel)
    explorer.bind_tools.return_value = RunnableLambda(
        explore or (lambda _: AIMessage(content="Done"))
    )
    explorer.with_structured_output.return_value = RunnableLambda(
        lambda _: result or candidate()
    )
    return with_scripted_planner(explorer)


def request(repo, mode="quick", **kwargs):
    return RunRequest(
        repo_path=repo,
        objective="Locate source.",
        investigation_mode=mode,
        max_tool_calls_per_step=2,
        max_explorer_calls_per_step=2,
        **kwargs,
    )


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_mode_survives_sqlite_restart_resume_and_followup(tmp_path, mode):
    (tmp_path / "example.py").write_text("value = 1\n")
    db = tmp_path / "checkpoint.sqlite"

    def fail(_):
        raise TimeoutError("scripted interruption")

    with open_sqlite_checkpointer(db) as saver:
        runner = AgentRunner(models(explore=fail), checkpointer=saver)
        with pytest.raises(TimeoutError):
            runner.run(request(tmp_path, mode, thread_id="saved-mode"))
    # The new operator default is deliberately opposite to the stored mode.
    with open_sqlite_checkpointer(db) as saver:
        runner = AgentRunner(
            models(),
            checkpointer=saver,
            architecture_profile=profile_for_mode(
                "detailed" if mode == "quick" else "quick"
            ),
        )
        resumed = runner.resume(ResumeRequest(thread_id="saved-mode"))
        assert resumed.termination_status == "accepted"
        followup = runner.continue_thread(
            ContinueRequest(thread_id="saved-mode", prompt="Explain the same source.")
        )
        assert followup.termination_status == "accepted"
        view = runner.inspect_thread(ResumeRequest(thread_id="saved-mode"))
        assert view.architecture_preset == profile_for_mode(mode).preset
        assert (
            DashboardThreadView.from_internal(view).architecture_preset
            == view.architecture_preset
        )
        assert view.turn_number == 2
        nodes = {event.node for event in followup.trace.events}
        assert ("evaluate_candidate" in nodes) == (mode == "detailed")
        assert ("validate_quick_candidate" in nodes) == (mode == "quick")


def test_detailed_review_uses_saved_mode_after_restart(tmp_path):
    (tmp_path / "example.py").write_text("value = 1\n")
    db = tmp_path / "checkpoint.sqlite"
    with open_sqlite_checkpointer(db) as saver:
        runner = AgentRunner(models(), checkpointer=saver)
        runner.run(
            request(
                tmp_path,
                "detailed",
                thread_id="review",
                human_review_policy=HumanReviewPolicy.BEFORE_FINALIZE,
            )
        )
    with open_sqlite_checkpointer(db) as saver:
        runner = AgentRunner(
            models(), checkpointer=saver, architecture_profile=profile_for_mode("quick")
        )
        view = runner.inspect_thread(ResumeRequest(thread_id="review"))
        result = runner.respond(
            HumanResponseRequest(
                thread_id="review",
                interrupt_id=view.interrupt_id,
                decision=HumanDecision(
                    action=HumanDecisionAction.APPROVE, feedback=None
                ),
            )
        )
        assert result.termination_status == "accepted"
        assert (
            runner.inspect_thread(ResumeRequest(thread_id="review")).evaluation_report
            is not None
        )


@pytest.mark.parametrize(
    "bad",
    [
        CandidateResult(
            evidence=[], summary="Unsupported answer", unresolved_questions=[]
        ),
        candidate("missing.py"),
        candidate("../outside.py"),
        candidate(line=100),
        candidate(symbol="nonexistent"),
        candidate("link.py"),
    ],
)
def test_quick_rejects_empty_invalid_and_escaping_evidence(tmp_path, bad):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "example.py").write_text("value = 1\n")
    (tmp_path / "outside.py").write_text("private = 1\n")
    (repo / "link.py").symlink_to(tmp_path / "outside.py")
    with open_sqlite_checkpointer(tmp_path / "checkpoint.sqlite") as saver:
        result = AgentRunner(models(bad), checkpointer=saver).run(request(repo))
    assert result.termination_status == "exhausted"
    assert result.step_stop_reason == "invalid_candidate_evidence"
    assert result.final_result.relevant_files == []
    assert result.final_result.evidence == []
    assert result.final_result.structure is None
    assert result.final_result.summary != bad.summary


def test_quick_cost_stop_does_not_publish_unvalidated_candidate(tmp_path):
    state = dict(
        repo_path=tmp_path,
        objective="Locate source",
        architecture_preset="tool_wrapper",
        candidate_result=candidate("../outside.py"),
        cost_budget=None,
        investigation_structure=None,
        step_stop_reason="attempt_cost_reservation_denied",
    )
    result = finalize_cost_budget_exhausted(state)
    assert result["termination_status"] == "exhausted"
    assert result["final_result"].evidence == []
    assert result["final_result"].relevant_files == []


def test_quick_keeps_gateway_and_call_limits(tmp_path):
    (tmp_path / "example.py").write_text("value = 1\n")
    calls = 0

    def explore(_):
        nonlocal calls
        calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                dict(
                    name="read_code",
                    args={
                        "relative_path": "../outside.py",
                        "start_line": 1,
                        "end_line": 1,
                    },
                    id=f"read-{calls}",
                )
            ],
        )

    with open_sqlite_checkpointer(tmp_path / "checkpoint.sqlite") as saver:
        outcome = AgentRunner(models(explore=explore), checkpointer=saver).run(
            request(tmp_path)
        )
    assert calls <= 2
    assert outcome.tool_audit_records
    assert outcome.tool_audit_records[0].decision.reason.value == "path_escape"
    assert all(record.result_digest is None for record in outcome.tool_audit_records)
    assert outcome.step_stop_reason is not None


def test_quick_review_is_rejected_before_running(tmp_path):
    with open_sqlite_checkpointer(tmp_path / "checkpoint.sqlite") as saver:
        runner = AgentRunner(models(), checkpointer=saver)
        with pytest.raises(ValueError, match="review"):
            runner.run(
                request(tmp_path, human_review_policy=HumanReviewPolicy.BEFORE_FINALIZE)
            )
        assert saver.get({"configurable": {"thread_id": "unused"}}) is None


def test_mode_contracts_and_legacy_digest(tmp_path):
    args = dict(
        request_id="mode-request", repo_path=tmp_path, objective="Locate source"
    )
    detailed = StartInvestigationRequest(**args)
    quick = StartInvestigationRequest(**args, investigation_mode="quick")
    legacy = detailed.model_dump(
        mode="json", exclude={"request_id", "investigation_mode"}
    )
    encoded = json.dumps(
        legacy, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    assert (
        InvestigationService.request_digest(detailed)
        == hashlib.sha256(b"repocompass:service-request:v1\0" + encoded).hexdigest()
    )
    assert InvestigationService.request_digest(
        quick
    ) != InvestigationService.request_digest(detailed)
    for invalid in ("auto", "tool_wrapper", None):
        with pytest.raises(ValidationError):
            StartInvestigationRequest(**args, investigation_mode=invalid)
    with pytest.raises(ValidationError, match="review"):
        StartInvestigationRequest(
            **args,
            investigation_mode="quick",
            human_review_policy=HumanReviewPolicy.BEFORE_FINALIZE,
        )
    with pytest.raises(ValidationError, match="review"):
        DashboardStartCommand(
            request_id="quick-review",
            goal="Locate source",
            investigation_mode="quick",
            expected_repository_id="a" * 64,
            expected_commit_sha=None,
        )
