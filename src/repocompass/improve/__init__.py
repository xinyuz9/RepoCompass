"""Optional, explicitly scoped code-improvement boundary."""

from repocompass.improve.contracts import (
    FocusedChangeProposal,
    FocusedImproveScope,
    ImproveTargetSelection,
    PythonVerificationProfile,
    ReplacementEdit,
    ScopedImproveTarget,
    VerificationSpec,
    create_change_proposal,
)
from repocompass.improve.proposal import (
    ProposalValidationError,
    ProposalValidationErrorReason,
    ValidatedFocusedChange,
    validate_focused_change_proposal,
)
from repocompass.improve.scope import (
    ImproveScopeError,
    ImproveScopeErrorReason,
    create_focused_improve_scope,
)
from repocompass.improve.sandbox import (
    BubblewrapVerifier,
    ProposalVerificationReport,
    SandboxExecutionError,
    SandboxUnavailableError,
    VerificationDecisionReason,
    VerificationStatus,
)

__all__ = [
    "BubblewrapVerifier",
    "FocusedChangeProposal",
    "FocusedImproveScope",
    "ImproveScopeError",
    "ImproveScopeErrorReason",
    "ImproveTargetSelection",
    "ProposalValidationError",
    "ProposalValidationErrorReason",
    "ProposalVerificationReport",
    "PythonVerificationProfile",
    "ReplacementEdit",
    "ScopedImproveTarget",
    "SandboxExecutionError",
    "SandboxUnavailableError",
    "ValidatedFocusedChange",
    "VerificationSpec",
    "VerificationDecisionReason",
    "VerificationStatus",
    "create_change_proposal",
    "create_focused_improve_scope",
    "validate_focused_change_proposal",
]
