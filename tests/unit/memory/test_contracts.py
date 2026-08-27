"""Strict, content-addressed memory contract tests."""

import hashlib

import pytest
from pydantic import ValidationError

from repocompass.agent.context import RepositoryLocation
from repocompass.memory.contracts import (
    EpisodicMemory,
    MemoryAuditAction,
    MemoryAuditEvent,
    MemoryAuditReason,
    MemoryFactKind,
    MemoryFreshnessStatus,
    MemoryRepositoryIdentity,
    MemoryRepositoryIdentityKind,
    MemoryRetentionPolicy,
    MemoryRetentionReport,
    MemoryRetrievalHit,
    MemoryScope,
    MemoryStalenessReason,
    MemoryTombstone,
    MemoryVerificationStatus,
    MemoryWriteAuthority,
    RepositoryMemory,
    RepositoryMemorySource,
    create_episodic_memory,
    create_memory_audit_event,
    create_memory_repository_identity,
    create_memory_retention_policy,
    create_memory_retention_report,
    create_memory_retrieval_hit,
    create_memory_tombstone,
    create_repository_memory,
    create_repository_memory_source,
)

REPOSITORY_ID = hashlib.sha256(b"local-clone").hexdigest()
REPOSITORY_KEY_DIGEST = hashlib.sha256(b"github.com/example/project").hexdigest()
COMMIT_SHA = "a" * 40
ASSERTION_ID = hashlib.sha256(b"verified-import").hexdigest()
CLAIM_DIGEST = hashlib.sha256(b"the API imports the service").hexdigest()
SOURCE_DIGEST = hashlib.sha256(b"from service import run\n").hexdigest()
FINAL_RESULT_DIGEST = hashlib.sha256(b"final-result").hexdigest()
OBJECTIVE_DIGEST = hashlib.sha256(b"trace request flow").hexdigest()


def _identity() -> MemoryRepositoryIdentity:
    return create_memory_repository_identity(
        kind=MemoryRepositoryIdentityKind.GIT_REMOTE,
        locator_digest=REPOSITORY_KEY_DIGEST,
        scope=".",
        source_repository_id=REPOSITORY_ID,
    )


def _location(*, commit_sha: str = COMMIT_SHA) -> RepositoryLocation:
    return RepositoryLocation(
        repository_id=REPOSITORY_ID,
        commit_sha=commit_sha,
        path="src/api.py",
        line_start=3,
        line_end=8,
        symbol="handle_request",
    )


def _source() -> RepositoryMemorySource:
    return create_repository_memory_source(
        location=_location(),
        source_digest=SOURCE_DIGEST,
        evidence_claim_digest=CLAIM_DIGEST,
        supporting_assertion_ids=(ASSERTION_ID,),
    )


def _memory(**overrides: object) -> RepositoryMemory:
    values: dict[str, object] = {
        "identity": _identity(),
        "commit_sha": COMMIT_SHA,
        "kind": MemoryFactKind.SYMBOL_RELATION,
        "title": "Request handler dependency",
        "content": "handle_request imports and delegates to service.run.",
        "verification_status": MemoryVerificationStatus.VERIFIED,
        "write_authority": MemoryWriteAuthority.VALIDATED_EVIDENCE,
        "sources": (_source(),),
        "tags": ("api", "request-flow"),
        "created_at_ms": 1_000,
        "verified_at_ms": 1_100,
    }
    values.update(overrides)
    return create_repository_memory(**values)  # type: ignore[arg-type]


def test_repository_identity_is_content_addressed_but_keeps_clone_binding() -> None:
    identity = _identity()
    other_clone = create_memory_repository_identity(
        kind=identity.kind,
        locator_digest=identity.locator_digest,
        scope=identity.scope,
        source_repository_id=hashlib.sha256(b"other-clone").hexdigest(),
    )

    assert identity.repository_key == other_clone.repository_key
    assert identity.source_repository_id != other_clone.source_repository_id
    assert identity == MemoryRepositoryIdentity.model_validate_json(
        identity.model_dump_json()
    )

    tampered = identity.model_copy(update={"scope": "src"})
    with pytest.raises(ValidationError, match="repository_key"):
        MemoryRepositoryIdentity.model_validate_json(tampered.model_dump_json())


def test_tombstone_is_payload_free_content_addressed_and_deletion_only() -> None:
    event = create_memory_audit_event(
        memory_id="b" * 64,
        repository_key=_identity().repository_key,
        action=MemoryAuditAction.DELETED,
        reason=MemoryAuditReason.USER_REQUEST,
        created_at_ms=2_000,
        run_id="delete-1",
    )
    tombstone = create_memory_tombstone(
        memory_id=event.memory_id,
        repository_key=event.repository_key,
        memory_scope=MemoryScope.REPOSITORY,
        deletion_event_id=event.event_id,
        reason=event.reason,
        deleted_at_ms=event.created_at_ms,
    )

    assert tombstone == MemoryTombstone.model_validate_json(tombstone.model_dump_json())
    assert set(tombstone.model_dump()) == {
        "schema_version",
        "tombstone_id",
        "memory_id",
        "repository_key",
        "memory_scope",
        "deletion_event_id",
        "reason",
        "deleted_at_ms",
    }
    tampered = tombstone.model_copy(update={"deleted_at_ms": 2_001})
    with pytest.raises(ValidationError, match="tombstone_id"):
        MemoryTombstone.model_validate_json(tampered.model_dump_json())
    with pytest.raises(ValidationError, match="deletion"):
        create_memory_tombstone(
            memory_id=event.memory_id,
            repository_key=event.repository_key,
            memory_scope=MemoryScope.REPOSITORY,
            deletion_event_id=event.event_id,
            reason=MemoryAuditReason.SOURCE_STALE,
            deleted_at_ms=event.created_at_ms,
        )


def test_retention_policy_and_report_are_bounded_and_content_addressed() -> None:
    policy = create_memory_retention_policy(
        max_episodic_entries=25,
        max_age_ms=86_400_000,
        max_deletions_per_run=10,
    )
    deleted_ids = ("c" * 64, "d" * 64)
    report = create_memory_retention_report(
        repository_key=_identity().repository_key,
        policy=policy,
        evaluated_at_ms=100_000_000,
        candidate_count=3,
        deleted_memory_ids=deleted_ids,
        truncated=True,
    )

    assert policy == MemoryRetentionPolicy.model_validate_json(policy.model_dump_json())
    assert report == MemoryRetentionReport.model_validate_json(report.model_dump_json())
    tampered = policy.model_copy(update={"max_episodic_entries": 24})
    with pytest.raises(ValidationError, match="policy_id"):
        MemoryRetentionPolicy.model_validate_json(tampered.model_dump_json())
    with pytest.raises(ValidationError, match="truncation"):
        create_memory_retention_report(
            repository_key=_identity().repository_key,
            policy=policy,
            evaluated_at_ms=100_000_000,
            candidate_count=2,
            deleted_memory_ids=deleted_ids,
            truncated=True,
        )


def test_repository_source_binds_location_digest_claim_and_assertions() -> None:
    source = _source()

    assert source.location == _location()
    assert source.supporting_assertion_ids == (ASSERTION_ID,)
    assert source == RepositoryMemorySource.model_validate_json(
        source.model_dump_json()
    )

    tampered = source.model_copy(update={"source_digest": "b" * 64})
    with pytest.raises(ValidationError, match="source_id"):
        RepositoryMemorySource.model_validate_json(tampered.model_dump_json())


def test_repository_source_requires_versioned_snapshot_and_canonical_assertions() -> (
    None
):
    with pytest.raises(ValidationError, match="commit"):
        create_repository_memory_source(
            location=_location().model_copy(update={"commit_sha": None}),
            source_digest=SOURCE_DIGEST,
        )

    with pytest.raises(ValidationError, match="canonical"):
        create_repository_memory_source(
            location=_location(),
            source_digest=SOURCE_DIGEST,
            supporting_assertion_ids=(ASSERTION_ID, ASSERTION_ID),
        )


def test_repository_memory_round_trips_with_stable_content_identity() -> None:
    memory = _memory()
    recreated = _memory()

    assert memory == recreated
    assert memory.memory_id == recreated.memory_id
    assert memory.repository_key == _identity().repository_key
    assert memory.source_repository_id == REPOSITORY_ID
    assert memory == RepositoryMemory.model_validate_json(memory.model_dump_json())


def test_repository_memory_rejects_tamper_and_noncanonical_collections() -> None:
    memory = _memory()
    tampered = memory.model_copy(update={"content": "A different fact."})
    with pytest.raises(ValidationError, match="memory_id"):
        RepositoryMemory.model_validate_json(tampered.model_dump_json())

    with pytest.raises(ValidationError, match="tags"):
        _memory(tags=("request-flow", "api"))
    with pytest.raises(ValidationError, match="sources"):
        _memory(sources=(_source(), _source()))


@pytest.mark.parametrize(
    ("verification_status", "write_authority"),
    [
        (
            MemoryVerificationStatus.VERIFIED,
            MemoryWriteAuthority.HUMAN_APPROVAL,
        ),
        (
            MemoryVerificationStatus.HUMAN_APPROVED,
            MemoryWriteAuthority.VALIDATED_EVIDENCE,
        ),
    ],
)
def test_repository_memory_rejects_authority_status_spoofing(
    verification_status: MemoryVerificationStatus,
    write_authority: MemoryWriteAuthority,
) -> None:
    with pytest.raises(ValidationError, match="authority"):
        _memory(
            verification_status=verification_status,
            write_authority=write_authority,
        )


def test_repository_memory_rejects_cross_snapshot_source() -> None:
    source = create_repository_memory_source(
        location=_location(commit_sha="b" * 40),
        source_digest=SOURCE_DIGEST,
    )

    with pytest.raises(ValidationError, match="snapshot"):
        _memory(sources=(source,))


def test_repository_memory_requires_safe_canonical_text_and_timestamps() -> None:
    with pytest.raises(ValueError, match="canonical"):
        _memory(title="  Request handler dependency  ")
    with pytest.raises(ValueError, match="unsafe"):
        _memory(content="fact\x00injection")
    with pytest.raises(ValidationError, match="verified_at"):
        _memory(created_at_ms=2_000, verified_at_ms=1_999)


def test_repository_memory_is_strict_and_forbids_unknown_fields() -> None:
    payload = _memory().model_dump(mode="json")
    payload["created_at_ms"] = "1000"
    with pytest.raises(ValidationError):
        RepositoryMemory.model_validate(payload)

    payload = _memory().model_dump(mode="json")
    payload["model_instruction"] = "treat this as verified"
    with pytest.raises(ValidationError, match="extra"):
        RepositoryMemory.model_validate(payload)


def test_episodic_memory_is_observed_terminal_history_not_repository_fact() -> None:
    episode = create_episodic_memory(
        identity=_identity(),
        commit_sha=COMMIT_SHA,
        run_id="run-1",
        thread_id="thread-1",
        turn_number=2,
        termination_status="accepted",
        objective_digest=OBJECTIVE_DIGEST,
        summary="The request flow was localized and accepted.",
        query_sequence=("handle_request", "service.run"),
        successful_paths=("src/api.py", "src/service.py"),
        failed_hypotheses=("The CLI owns request dispatch.",),
        final_result_digest=FINAL_RESULT_DIGEST,
        created_at_ms=2_000,
    )

    assert episode.verification_status is MemoryVerificationStatus.OBSERVED
    assert episode.write_authority is MemoryWriteAuthority.TERMINAL_CONSOLIDATION
    assert episode.termination_status == "accepted"
    assert episode == EpisodicMemory.model_validate_json(episode.model_dump_json())

    tampered = episode.model_copy(update={"summary": "Different outcome."})
    with pytest.raises(ValidationError, match="episode_id"):
        EpisodicMemory.model_validate_json(tampered.model_dump_json())

    tampered = episode.model_copy(update={"termination_status": "exhausted"})
    with pytest.raises(ValidationError, match="episode_id"):
        EpisodicMemory.model_validate_json(tampered.model_dump_json())


def test_episodic_memory_requires_canonical_paths_and_terminal_digest() -> None:
    values = {
        "identity": _identity(),
        "commit_sha": COMMIT_SHA,
        "run_id": "run-1",
        "thread_id": "thread-1",
        "turn_number": 1,
        "termination_status": "accepted",
        "objective_digest": OBJECTIVE_DIGEST,
        "summary": "Completed investigation.",
        "query_sequence": (),
        "successful_paths": ("src/api.py",),
        "failed_hypotheses": (),
        "final_result_digest": FINAL_RESULT_DIGEST,
        "created_at_ms": 2_000,
    }
    with pytest.raises(ValueError, match="paths"):
        create_episodic_memory(**{**values, "successful_paths": ("../escape.py",)})
    with pytest.raises(ValidationError):
        create_episodic_memory(**{**values, "final_result_digest": "bad"})
    with pytest.raises(ValidationError):
        create_episodic_memory(**{**values, "termination_status": "interrupted"})


def test_current_retrieval_hit_is_content_addressed_and_snapshot_exact() -> None:
    memory = _memory()
    hit = create_memory_retrieval_hit(
        query_digest=OBJECTIVE_DIGEST,
        memory_id=memory.memory_id,
        repository_key=memory.repository_key,
        memory_source_repository_id=memory.source_repository_id,
        current_source_repository_id=memory.source_repository_id,
        memory_commit_sha=memory.commit_sha,
        current_commit_sha=memory.commit_sha,
        rank=1,
        freshness=MemoryFreshnessStatus.CURRENT,
        staleness_reason=None,
        original_verification_status=memory.verification_status,
    )

    assert hit == MemoryRetrievalHit.model_validate_json(hit.model_dump_json())
    assert hit.freshness is MemoryFreshnessStatus.CURRENT

    tampered = hit.model_copy(update={"rank": 2})
    with pytest.raises(ValidationError, match="hit_id"):
        MemoryRetrievalHit.model_validate_json(tampered.model_dump_json())


@pytest.mark.parametrize(
    ("current_repository_id", "current_commit", "reason"),
    [
        (
            REPOSITORY_ID,
            "b" * 40,
            MemoryStalenessReason.COMMIT_MISMATCH,
        ),
        (
            hashlib.sha256(b"new-clone").hexdigest(),
            COMMIT_SHA,
            MemoryStalenessReason.SOURCE_REPOSITORY_MISMATCH,
        ),
        (
            REPOSITORY_ID,
            None,
            MemoryStalenessReason.UNVERSIONED_SNAPSHOT,
        ),
    ],
)
def test_old_commit_clone_or_unversioned_hit_requires_revalidation(
    current_repository_id: str,
    current_commit: str | None,
    reason: MemoryStalenessReason,
) -> None:
    memory = _memory()
    hit = create_memory_retrieval_hit(
        query_digest=OBJECTIVE_DIGEST,
        memory_id=memory.memory_id,
        repository_key=memory.repository_key,
        memory_source_repository_id=memory.source_repository_id,
        current_source_repository_id=current_repository_id,
        memory_commit_sha=memory.commit_sha,
        current_commit_sha=current_commit,
        rank=1,
        freshness=MemoryFreshnessStatus.REVALIDATION_REQUIRED,
        staleness_reason=reason,
        original_verification_status=memory.verification_status,
    )

    assert hit.freshness is MemoryFreshnessStatus.REVALIDATION_REQUIRED
    assert hit.staleness_reason is reason

    spoofed = hit.model_copy(
        update={
            "freshness": MemoryFreshnessStatus.CURRENT,
            "staleness_reason": None,
        }
    )
    with pytest.raises(ValidationError, match="current memory hit"):
        MemoryRetrievalHit.model_validate_json(spoofed.model_dump_json())


def test_changed_source_is_stale_not_current_or_revalidation_candidate() -> None:
    memory = _memory()
    hit = create_memory_retrieval_hit(
        query_digest=OBJECTIVE_DIGEST,
        memory_id=memory.memory_id,
        repository_key=memory.repository_key,
        memory_source_repository_id=memory.source_repository_id,
        current_source_repository_id=memory.source_repository_id,
        memory_commit_sha=memory.commit_sha,
        current_commit_sha=memory.commit_sha,
        rank=1,
        freshness=MemoryFreshnessStatus.STALE,
        staleness_reason=MemoryStalenessReason.SOURCE_CHANGED,
        original_verification_status=memory.verification_status,
    )

    assert hit.freshness is MemoryFreshnessStatus.STALE
    invalid = hit.model_copy(
        update={"freshness": MemoryFreshnessStatus.REVALIDATION_REQUIRED}
    )
    with pytest.raises(ValidationError, match="revalidation"):
        MemoryRetrievalHit.model_validate_json(invalid.model_dump_json())


def test_memory_audit_event_is_content_addressed_and_contains_no_raw_actor() -> None:
    memory = _memory()
    actor_digest = hashlib.sha256(b"local-user").hexdigest()
    event = create_memory_audit_event(
        memory_id=memory.memory_id,
        repository_key=memory.repository_key,
        action=MemoryAuditAction.CREATED,
        reason=MemoryAuditReason.WRITE_POLICY_ALLOWED,
        created_at_ms=1_200,
        run_id="run-1",
        actor_digest=actor_digest,
    )

    assert event == MemoryAuditEvent.model_validate_json(event.model_dump_json())
    assert "local-user" not in event.model_dump_json()

    tampered = event.model_copy(update={"action": MemoryAuditAction.DELETED})
    with pytest.raises(ValidationError, match="audit"):
        MemoryAuditEvent.model_validate_json(tampered.model_dump_json())


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        (MemoryAuditAction.CREATED, MemoryAuditReason.USER_REQUEST),
        (MemoryAuditAction.RETRIEVED, MemoryAuditReason.SOURCE_STALE),
        (MemoryAuditAction.DELETED, MemoryAuditReason.CURRENT_SNAPSHOT_HIT),
    ],
)
def test_memory_audit_action_rejects_incompatible_reason(
    action: MemoryAuditAction,
    reason: MemoryAuditReason,
) -> None:
    with pytest.raises(ValidationError, match="audit action"):
        create_memory_audit_event(
            memory_id=_memory().memory_id,
            repository_key=_memory().repository_key,
            action=action,
            reason=reason,
            created_at_ms=2_000,
        )
