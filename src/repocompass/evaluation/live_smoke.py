"""Explicitly enabled, cost-preflighted real-provider smoke harness."""

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from decimal import Decimal, DecimalException, localcontext
from pathlib import Path
from typing import Literal, TextIO

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.cost import (
    COST_DISCLAIMER,
    PRICING_UNIT_TOKENS,
    CostAvailability,
    InvocationCostEstimate,
    PricingTable,
    estimate_invocation_cost,
    validate_cost_budget_usd,
)
from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.agent.usage import (
    PROVIDER_PATTERN,
    ModelInvocationRecord,
    ModelInvocationStatus,
    ModelRole,
    UsageAvailability,
    create_model_invocation_record,
    extract_usage_record,
)
from repocompass.models import create_chat_model
from repocompass.models.config import (
    DeepSeekModelSettings,
    ModelSettings,
    OpenAIModelSettings,
    load_model_settings,
)
from repocompass.pricing import load_pricing_table_file
from repocompass.tools.contracts import (
    IDENTIFIER_PATTERN,
    MAX_SAFE_JSON_INTEGER,
    Sha256Digest,
)

LIVE_SMOKE_SCHEMA_VERSION = 1
LIVE_SMOKE_INPUT_TOKEN_CEILING = 4_096
MAX_LIVE_SMOKE_OUTPUT_TOKENS = 256
LIVE_SMOKE_RESPONSE_MARKER = "REPOCOMPASS_LIVE_OK"
_SYSTEM_PROMPT = "You are a provider connectivity health check. Follow exactly."
_USER_PROMPT = f"Reply with exactly {LIVE_SMOKE_RESPONSE_MARKER} and nothing else."
ModelFactory = Callable[[ModelSettings], BaseChatModel]
Clock = Callable[[], int]


class LiveProviderSmokeError(RuntimeError):
    """A provider call failed without exposing its potentially sensitive body."""


class _LiveSmokeModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class LiveProviderSmokeReport(_LiveSmokeModel):
    """Content-addressed sanitized evidence for one bounded provider call."""

    schema_version: Literal[1] = LIVE_SMOKE_SCHEMA_VERSION
    report_id: Sha256Digest
    provider: str = Field(pattern=PROVIDER_PATTERN)
    model: str = Field(min_length=1, max_length=256)
    pricing_table_id: Sha256Digest
    pricing_version: str = Field(pattern=IDENTIFIER_PATTERN)
    pricing_effective_from: str
    max_input_tokens_assumed: Literal[4096] = LIVE_SMOKE_INPUT_TOKEN_CEILING
    max_output_tokens: int = Field(ge=1, le=MAX_LIVE_SMOKE_OUTPUT_TOKENS)
    max_cost_usd: str
    preflight_max_cost_usd: str
    invocation: ModelInvocationRecord
    cost: InvocationCostEstimate
    response_marker_matched: Literal[True] = True
    disclaimer: Literal["estimate_not_invoice"] = COST_DISCLAIMER

    @field_validator("max_cost_usd", mode="before")
    @classmethod
    def require_positive_cost_limit(cls, value: object) -> object:
        validate_cost_budget_usd(value)
        return value

    @field_validator("preflight_max_cost_usd", mode="before")
    @classmethod
    def require_canonical_preflight_cost(cls, value: object) -> object:
        if not _is_canonical_non_negative_decimal(value):
            raise ValueError("preflight cost must be a canonical decimal")
        return value

    @model_validator(mode="after")
    def reconcile_sanitized_live_evidence(self) -> "LiveProviderSmokeReport":
        invocation = self.invocation
        cost = self.cost
        if invocation.provider != self.provider or invocation.model != self.model:
            raise ValueError("invocation provider/model does not match report")
        if invocation.status is not ModelInvocationStatus.SUCCEEDED:
            raise ValueError("live smoke report requires a successful invocation")
        if cost.invocation_id != invocation.invocation_id:
            raise ValueError("cost estimate does not match invocation")
        if cost.provider != self.provider or cost.model != self.model:
            raise ValueError("cost provider/model does not match report")
        if cost.pricing_table_id != self.pricing_table_id:
            raise ValueError("cost pricing table does not match report")
        usage = invocation.usage
        if (
            usage.availability is not UsageAvailability.REPORTED
            or usage.input_tokens is None
            or usage.output_tokens is None
        ):
            raise ValueError("live smoke requires reported input and output usage")
        if usage.input_tokens > self.max_input_tokens_assumed:
            raise ValueError("reported input usage exceeds the preflight assumption")
        if usage.output_tokens > self.max_output_tokens:
            raise ValueError("reported output usage exceeds the provider call limit")
        if cost.availability is not CostAvailability.ESTIMATED:
            raise ValueError("live smoke requires a complete cost estimate")
        actual_cost = Decimal(cost.estimated_cost_usd or "0")
        limit = Decimal(self.max_cost_usd)
        preflight = Decimal(self.preflight_max_cost_usd)
        if preflight > limit:
            raise ValueError("preflight cost exceeds configured cost limit")
        if actual_cost > limit:
            raise ValueError("observed provider cost exceeds configured cost limit")
        expected_id = _report_id(self.model_dump(mode="json", exclude={"report_id"}))
        if self.report_id != expected_id:
            raise ValueError("report_id does not match live smoke contents")
        return self


class _LiveProviderSmokeReservation:
    def __init__(self, path: Path, stream: TextIO) -> None:
        self.path = path
        self._stream = stream
        self.committed = False

    def write(self, report: LiveProviderSmokeReport) -> Path:
        if not isinstance(report, LiveProviderSmokeReport):
            raise TypeError("report must be a LiveProviderSmokeReport")
        if self.committed:
            raise RuntimeError("live smoke reservation already contains a report")
        self._stream.write(report.model_dump_json(indent=2))
        self._stream.write("\n")
        self._stream.flush()
        self.committed = True
        return self.path

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()


class LiveProviderSmokeReportSink:
    """Persist sanitized live evidence without replacing an earlier report."""

    def write(self, report: LiveProviderSmokeReport, destination: Path) -> Path:
        with self.reserve(destination) as reservation:
            return reservation.write(report)

    @contextmanager
    def reserve(self, destination: Path) -> Iterator[_LiveProviderSmokeReservation]:
        if not isinstance(destination, Path):
            raise TypeError("destination must be a Path")
        resolved = destination.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        stream = resolved.open("x", encoding="utf-8", newline="\n")
        reservation = _LiveProviderSmokeReservation(resolved, stream)
        try:
            yield reservation
        except BaseException:
            reservation.close()
            if not reservation.committed:
                try:
                    resolved.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        else:
            reservation.close()
            if not reservation.committed:
                resolved.unlink(missing_ok=True)
                raise RuntimeError("live smoke reservation completed without a report")


def run_live_provider_smoke(
    *,
    settings: ModelSettings,
    pricing_table: PricingTable,
    max_cost_usd: str,
    max_output_tokens: int = 32,
    model_factory: ModelFactory | None = None,
    monotonic_ns: Clock | None = None,
) -> LiveProviderSmokeReport:
    """Make one fixed, output-bounded call after exact-price cost preflight."""

    if not isinstance(settings, (DeepSeekModelSettings, OpenAIModelSettings)):
        raise TypeError("settings must be a supported model settings value")
    if not isinstance(pricing_table, PricingTable):
        raise TypeError("pricing_table must be a PricingTable")
    if (
        not isinstance(max_output_tokens, int)
        or isinstance(max_output_tokens, bool)
        or not 1 <= max_output_tokens <= MAX_LIVE_SMOKE_OUTPUT_TOKENS
    ):
        raise ValueError(
            f"max_output_tokens must be between 1 and {MAX_LIVE_SMOKE_OUTPUT_TOKENS}"
        )
    limit_text = validate_cost_budget_usd(max_cost_usd)
    provider = settings.provider.value
    price = pricing_table.lookup(provider, settings.model)
    if price is None:
        raise ValueError(
            "Pricing snapshot has no exact entry for "
            f"{provider}:{settings.model}; refusing live API call"
        )
    preflight = _preflight_max_cost(
        input_tokens=LIVE_SMOKE_INPUT_TOKEN_CEILING,
        output_tokens=max_output_tokens,
        input_rate=price.input_usd_per_million_tokens,
        cached_input_rate=price.cached_input_usd_per_million_tokens,
        output_rate=price.output_usd_per_million_tokens,
        reasoning_output_rate=price.reasoning_output_usd_per_million_tokens,
    )
    if Decimal(preflight) > Decimal(limit_text):
        raise ValueError(
            f"Live smoke preflight maximum cost USD {preflight} exceeds "
            f"configured limit USD {limit_text}; refusing API call"
        )

    resolved_factory = model_factory or create_chat_model
    clock = monotonic_ns or time.perf_counter_ns
    model = resolved_factory(settings)
    bounded_model = model.bind(max_tokens=max_output_tokens)
    messages = (
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_USER_PROMPT),
    )
    started_ns = clock()
    try:
        response = bounded_model.invoke(messages)
    except Exception as error:
        raise LiveProviderSmokeError(
            f"provider call failed with {type(error).__name__}"
        ) from error
    elapsed_ns = max(0, clock() - started_ns)
    if not isinstance(response, AIMessage) or not isinstance(response.content, str):
        raise ValueError("provider smoke response must be one text AIMessage")
    if response.content.strip() != LIVE_SMOKE_RESPONSE_MARKER:
        raise ValueError("provider smoke response marker did not match")
    usage = extract_usage_record(response)
    if (
        usage.availability is not UsageAvailability.REPORTED
        or usage.input_tokens is None
        or usage.output_tokens is None
    ):
        raise ValueError("provider did not report complete input/output token usage")
    if usage.input_tokens > LIVE_SMOKE_INPUT_TOKEN_CEILING:
        raise ValueError("provider input usage exceeded the preflight assumption")
    if usage.output_tokens > max_output_tokens:
        raise ValueError("provider output usage exceeded max_output_tokens")

    input_payload = json.dumps(
        [
            {"content": _SYSTEM_PROMPT, "role": "system"},
            {"content": _USER_PROMPT, "role": "user"},
        ],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    input_artifact_id = hashlib.sha256(input_payload).hexdigest()
    output_artifact_id = hashlib.sha256(response.content.encode("utf-8")).hexdigest()
    invocation = create_model_invocation_record(
        run_id="live-smoke",
        thread_id="live-smoke",
        turn_number=1,
        sequence=0,
        node_name="live_smoke",
        role=ModelRole.EVALUATOR,
        provider=provider,
        model=settings.model,
        status=ModelInvocationStatus.SUCCEEDED,
        latency_ms=min(elapsed_ns // 1_000_000, MAX_SAFE_JSON_INTEGER),
        usage=usage,
        context_pack_id=hashlib.sha256(
            b"repocompass:live-smoke-context:v1\0" + input_payload
        ).hexdigest(),
        input_artifact_id=input_artifact_id,
        output_artifact_id=output_artifact_id,
    )
    cost = estimate_invocation_cost(invocation, pricing_table)
    if cost.availability is not CostAvailability.ESTIMATED:
        reason = (
            "unknown" if cost.unavailable_reason is None else cost.unavailable_reason
        )
        raise ValueError(f"live smoke cost estimate is unavailable: {reason}")
    if Decimal(cost.estimated_cost_usd or "0") > Decimal(limit_text):
        raise ValueError("observed provider cost exceeded configured live smoke limit")
    payload = {
        "schema_version": LIVE_SMOKE_SCHEMA_VERSION,
        "provider": provider,
        "model": settings.model,
        "pricing_table_id": pricing_table.table_id,
        "pricing_version": pricing_table.version,
        "pricing_effective_from": pricing_table.effective_from,
        "max_input_tokens_assumed": LIVE_SMOKE_INPUT_TOKEN_CEILING,
        "max_output_tokens": max_output_tokens,
        "max_cost_usd": limit_text,
        "preflight_max_cost_usd": preflight,
        "invocation": invocation.model_dump(mode="json"),
        "cost": cost.model_dump(mode="json"),
        "response_marker_matched": True,
        "disclaimer": COST_DISCLAIMER,
    }
    return LiveProviderSmokeReport(
        report_id=_report_id(payload),
        provider=provider,
        model=settings.model,
        pricing_table_id=pricing_table.table_id,
        pricing_version=pricing_table.version,
        pricing_effective_from=pricing_table.effective_from,
        max_input_tokens_assumed=LIVE_SMOKE_INPUT_TOKEN_CEILING,
        max_output_tokens=max_output_tokens,
        max_cost_usd=limit_text,
        preflight_max_cost_usd=preflight,
        invocation=invocation,
        cost=cost,
        response_marker_matched=True,
        disclaimer=COST_DISCLAIMER,
    )


def _preflight_max_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    input_rate: str,
    cached_input_rate: str,
    output_rate: str,
    reasoning_output_rate: str,
) -> str:
    cached_rate = (
        input_rate if cached_input_rate == "same_as_input" else cached_input_rate
    )
    reasoning_rate = (
        output_rate
        if reasoning_output_rate == "same_as_output"
        else reasoning_output_rate
    )
    with localcontext() as context:
        context.prec = 128
        amount = (
            Decimal(input_tokens) * max(Decimal(input_rate), Decimal(cached_rate))
            + Decimal(output_tokens)
            * max(Decimal(output_rate), Decimal(reasoning_rate))
        ) / Decimal(PRICING_UNIT_TOKENS)
    return _decimal_string(amount)


def _is_canonical_non_negative_decimal(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return _decimal_string(Decimal(value)) == value
    except (DecimalException, ValueError):
        return False


def _decimal_string(value: Decimal) -> str:
    if not value.is_finite() or value < 0:
        raise ValueError("decimal amount must be finite and non-negative")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _report_id(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(b"repocompass:live-provider-smoke:v1\0" + encoded).hexdigest()


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repocompass-live-smoke",
        description="Make one sanitized, output-bounded real-provider API call.",
    )
    parser.add_argument("--acknowledge-live-api-costs", action="store_true")
    parser.add_argument("--pricing-file", type=Path, required=True)
    parser.add_argument("--max-cost-usd", required=True)
    parser.add_argument("--max-output-tokens", type=_positive_int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    arguments = parser.parse_args(argv)
    if not arguments.acknowledge_live_api_costs:
        parser.error("--acknowledge-live-api-costs is required")
    if os.getenv("REPOCOMPASS_RUN_LIVE_TESTS", "").strip() != "1":
        print(
            "repocompass-live-smoke: error: set REPOCOMPASS_RUN_LIVE_TESTS=1 "
            "to explicitly enable a paid provider call",
            file=sys.stderr,
        )
        return 1
    try:
        sink = LiveProviderSmokeReportSink()
        with sink.reserve(arguments.output) as reservation:
            report = run_live_provider_smoke(
                settings=load_model_settings(),
                pricing_table=load_pricing_table_file(arguments.pricing_file),
                max_cost_usd=arguments.max_cost_usd,
                max_output_tokens=arguments.max_output_tokens,
            )
            report_path = reservation.write(report)
    except FileExistsError:
        print(
            f"repocompass-live-smoke: error: report already exists: "
            f"{arguments.output.resolve()}",
            file=sys.stderr,
        )
        return 1
    except (LiveProviderSmokeError, OSError, TypeError, ValueError) as error:
        print(f"repocompass-live-smoke: error: {error}", file=sys.stderr)
        return 1
    print(report.model_dump_json(indent=2))
    print(
        f"Live provider smoke passed: {report.provider}:{report.model}; "
        f"latency {report.invocation.latency_ms} ms; estimated cost USD "
        f"{report.cost.estimated_cost_usd}; cap USD {report.max_cost_usd}; "
        "estimate, not provider invoice",
        file=sys.stderr,
    )
    print(f"Report: {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
