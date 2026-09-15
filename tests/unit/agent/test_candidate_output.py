"""Selected evidence controls the candidate interface, never edge validity."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.candidate_output import EvidenceOnlyCandidateResult
from repocompass.agent.context import (
    ContextSection,
    ContextSourceKind,
    create_context_item,
)
from repocompass.agent.context_builder import ContextBudgetPolicy, build_active_context
from repocompass.agent.context_gathering import (
    gather_candidate_context,
    has_selected_structural_records,
)
from repocompass.agent.nodes import create_build_candidate_node
from repocompass.agent.planning import InvestigationPlan, PlanStep
from repocompass.tools.gateway import ToolExecutionContext
from repocompass.tools.repository_tools import (
    ListSymbolsArguments,
    execute_list_symbols,
)


def _state(path: Path, messages=()):
    return {
        "repo_path": path,
        "objective": "Explain the settings.",
        "messages": list(messages),
        "investigation_plan": InvestigationPlan(
            steps=[
                PlanStep(
                    id="inspect",
                    objective="Read the settings.",
                    expected_evidence=["Source lines."],
                    status="active",
                )
            ]
        ),
        "step_stop_reason": None,
        "context_budget_tokens": 16_384,
        "context_build_count": 0,
    }


def _payload():
    return {
        "evidence": [
            {
                "path": "setup.cfg",
                "line_start": 1,
                "claim": "The settings define metadata.",
            }
        ],
        "summary": "The file contains metadata settings.",
        "unresolved_questions": [],
    }


def _invoke(path, payload, *, name="EvidenceOnlyCandidateResult"):
    raw = AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": payload,
                "id": "candidate",
                "type": "tool_call",
            }
        ],
    )
    model = MagicMock(spec=BaseChatModel)
    model.with_structured_output.return_value = RunnableLambda(
        lambda _: {
            "raw": raw,
            "parsed": None,
            "parsing_error": ValueError("transport"),
        }
    )
    return create_build_candidate_node(model)(_state(path))


def test_evidence_only_schema_does_not_offer_relation_generation():
    schema = EvidenceOnlyCandidateResult.model_json_schema()
    assert set(schema["properties"]) == {"evidence", "summary", "unresolved_questions"}
    assert schema["additionalProperties"] is False
    assert "CandidateRelation" not in schema.get("$defs", {})


def test_evidence_only_json_becomes_checkpoint_domain_result(tmp_path):
    update = _invoke(tmp_path, _payload())
    assert type(update["candidate_result"]) is CandidateResult
    assert update["candidate_result"].relations == []
    assert update["candidate_output_contract"] == "source_evidence_only"


@pytest.mark.parametrize("extra", [[], [{"source": "setup.cfg", "target": "x"}]])
def test_no_structure_rejects_even_an_empty_relations_property(tmp_path, extra):
    payload = _payload() | {"relations": extra}
    with pytest.raises(TypeError, match="did not return CandidateResult"):
        _invoke(tmp_path, payload)


@pytest.mark.parametrize("line", [True, "1", 1.5, 0, -1])
def test_evidence_only_json_does_not_coerce_invalid_line_numbers(tmp_path, line):
    payload = _payload()
    payload["evidence"][0]["line_start"] = line
    with pytest.raises(TypeError, match="did not return CandidateResult"):
        _invoke(tmp_path, payload)


def test_raw_response_cannot_choose_a_more_permissive_schema(tmp_path):
    with pytest.raises(TypeError, match="did not return CandidateResult"):
        _invoke(tmp_path, _payload() | {"relations": []}, name="CandidateResult")


def test_custom_domain_response_cannot_smuggle_relations(tmp_path):
    candidate = CandidateResult.model_validate_json(
        json.dumps(
            _payload()
            | {
                "relations": [
                    {
                        "source": "setup.cfg",
                        "relation": "references",
                        "target": "elsewhere.py",
                        "supporting_evidence_indices": [0],
                        "explanation": "An unsupported configuration relationship.",
                    }
                ],
            }
        )
    )
    model = MagicMock(spec=BaseChatModel)
    model.with_structured_output.return_value = RunnableLambda(lambda _: candidate)
    with pytest.raises(TypeError, match="without selected structural evidence"):
        create_build_candidate_node(model)(_state(tmp_path))


def test_unvalidated_model_instance_is_revalidated(tmp_path):
    candidate = EvidenceOnlyCandidateResult.model_construct(
        evidence=[], summary="", unresolved_questions=[]
    )
    model = MagicMock(spec=BaseChatModel)
    model.with_structured_output.return_value = RunnableLambda(lambda _: candidate)
    with pytest.raises(ValidationError):
        create_build_candidate_node(model)(_state(tmp_path))


@pytest.fixture
def structural_context(tmp_path):
    (tmp_path / "service.py").write_text("class Service: pass\n")
    context = ToolExecutionContext(tmp_path)
    content = execute_list_symbols(
        tmp_path,
        ListSymbolsArguments(relative_path="service.py", max_results=10),
        repository_id=context.repository_id,
        commit_sha=context.commit_sha,
    )
    return gather_candidate_context(
        repo_path=tmp_path,
        objective="Locate Service.",
        active_step=_state(tmp_path)["investigation_plan"].active_step,
        step_stop_reason="none",
        budget_tokens=16_384,
        messages=(
            ToolMessage(content=content, name="list_symbols", tool_call_id="ast"),
        ),
    )


def test_excluded_ast_records_do_not_unlock_proposals(structural_context):
    gathered = structural_context
    assert has_selected_structural_records(gathered.pack)
    # Real ContextBuilder exclusion: objective only, but raw AST history retained.
    candidates = gathered.pack.items
    objective_tokens = sum(
        item.estimated_tokens
        for item in candidates
        if item.pinned or item.section is ContextSection.OBJECTIVE
    )
    pack = build_active_context(
        candidates=candidates,
        policy=ContextBudgetPolicy(budget_tokens=objective_tokens),
    )
    assert not has_selected_structural_records(pack)
    assert gathered.raw_artifacts


@pytest.mark.parametrize(
    "mutation", ["truncated", "call", "location", "tool", "record"]
)
def test_incomplete_or_mismatched_selected_record_does_not_unlock(
    structural_context, mutation
):
    item = next(
        item
        for item in structural_context.pack.items
        if item.source.kind is ContextSourceKind.SOURCE_EXCERPT
    )
    payload = json.loads(item.content)
    if mutation == "call":
        payload["tool_call_id"] = "different-call"
    elif mutation == "location":
        payload["record"]["definition"]["path"] = "different.py"
    elif mutation == "tool":
        payload["tool"] = "read_code"
    elif mutation == "record":
        payload["record"] = {"kind": "pretend-symbol"}
    content = json.dumps(payload) if mutation != "truncated" else item.content[:40]
    changed = create_context_item(
        source=item.source,
        section=item.section,
        content=content,
        verification_status=item.verification_status,
        estimated_tokens=item.estimated_tokens,
        priority=item.priority,
    )
    pack = build_active_context(
        candidates=(changed,), policy=ContextBudgetPolicy(budget_tokens=16_384)
    )
    assert not has_selected_structural_records(pack)
