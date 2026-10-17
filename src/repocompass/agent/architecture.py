"""Content-addressed static feature profiles for Agent architecture ablations."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, model_validator

from repocompass.agent.context_builder import ContextSelectionStrategy
from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.tools.contracts import Sha256Digest

AGENT_ARCHITECTURE_SCHEMA_VERSION = 1

InvestigationMode = Literal["quick", "detailed"]


def profile_for_mode(mode: InvestigationMode) -> AgentArchitectureProfile:
    return create_agent_architecture_profile(
        {
            "quick": AgentArchitecturePreset.TOOL_WRAPPER,
            "detailed": AgentArchitecturePreset.EVALUATOR_LOOP_CONTEXT_BUILDER,
        }[mode]
    )


class AgentArchitecturePreset(StrEnum):
    """Closed graph configurations used by production and M6 ablations."""

    TOOL_WRAPPER = "tool_wrapper"
    EVALUATOR_LOOP = "evaluator_loop"
    EVALUATOR_LOOP_SYMBOL_TOOLS = "evaluator_loop_symbol_tools"
    EVALUATOR_LOOP_CONTEXT_BUILDER = "evaluator_loop_context_builder"


class AgentArchitectureProfile(CheckpointSafeModel):
    """Immutable feature declaration that the graph must actually enforce."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    schema_version: Literal[1] = AGENT_ARCHITECTURE_SCHEMA_VERSION
    profile_id: Sha256Digest
    preset: AgentArchitecturePreset
    planner_enabled: bool
    evaluator_enabled: bool
    structural_tools_enabled: bool
    context_selection_strategy: ContextSelectionStrategy

    @model_validator(mode="after")
    def require_canonical_preset(self) -> "AgentArchitectureProfile":
        expected_features = _features(self.preset)
        observed_features = (
            self.planner_enabled,
            self.evaluator_enabled,
            self.structural_tools_enabled,
            self.context_selection_strategy,
        )
        if observed_features != expected_features:
            raise ValueError("architecture features do not match the declared preset")
        expected_id = _profile_id(self.preset, *expected_features)
        if self.profile_id != expected_id:
            raise ValueError("profile_id does not match architecture contents")
        return self


def create_agent_architecture_profile(
    preset: AgentArchitecturePreset,
) -> AgentArchitectureProfile:
    """Create one canonical architecture feature profile."""

    if not isinstance(preset, AgentArchitecturePreset):
        raise TypeError("preset must be an AgentArchitecturePreset")
    features = _features(preset)
    return AgentArchitectureProfile(
        profile_id=_profile_id(preset, *features),
        preset=preset,
        planner_enabled=features[0],
        evaluator_enabled=features[1],
        structural_tools_enabled=features[2],
        context_selection_strategy=features[3],
    )


def _features(
    preset: AgentArchitecturePreset,
) -> tuple[bool, bool, bool, ContextSelectionStrategy]:
    return {
        AgentArchitecturePreset.TOOL_WRAPPER: (
            False,
            False,
            False,
            ContextSelectionStrategy.FIFO,
        ),
        AgentArchitecturePreset.EVALUATOR_LOOP: (
            True,
            True,
            False,
            ContextSelectionStrategy.FIFO,
        ),
        AgentArchitecturePreset.EVALUATOR_LOOP_SYMBOL_TOOLS: (
            True,
            True,
            True,
            ContextSelectionStrategy.FIFO,
        ),
        AgentArchitecturePreset.EVALUATOR_LOOP_CONTEXT_BUILDER: (
            True,
            True,
            True,
            ContextSelectionStrategy.PRIORITIZED,
        ),
    }[preset]


def _profile_id(
    preset: AgentArchitecturePreset,
    planner_enabled: bool,
    evaluator_enabled: bool,
    structural_tools_enabled: bool,
    context_selection_strategy: ContextSelectionStrategy,
) -> str:
    payload = {
        "schema_version": AGENT_ARCHITECTURE_SCHEMA_VERSION,
        "preset": preset.value,
        "planner_enabled": planner_enabled,
        "evaluator_enabled": evaluator_enabled,
        "structural_tools_enabled": structural_tools_enabled,
        "context_selection_strategy": context_selection_strategy.value,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(b"repocompass:agent-architecture:v1\0" + encoded).hexdigest()


DEFAULT_AGENT_ARCHITECTURE = create_agent_architecture_profile(
    AgentArchitecturePreset.EVALUATOR_LOOP_CONTEXT_BUILDER
)
