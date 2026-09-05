"""Durable, conservative pre-call cost admission for live benchmark batches.

Reservations survive crashes. Unknown usage is never refunded. Costs use a
reviewed peak-price ceiling, not a claim about the provider invoice.
"""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from threading import RLock

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from repocompass.agent.cost import ModelPrice
from repocompass.agent.admission import PrecallBudgetDenied

NANODOLLARS = 1_000_000_000
MAX_INPUT_BYTES = 262_144


class BudgetAdmissionError(RuntimeError):
    """No provider call may proceed under the current batch budget."""


class BudgetReservationDenied(PrecallBudgetDenied, BudgetAdmissionError):
    """Expected, non-billable stop on insufficient batch or attempt capacity."""


def _units(amount: Decimal) -> int:
    if not isinstance(amount, Decimal) or not amount.is_finite() or amount < 0:
        raise ValueError("cost must be a finite non-negative Decimal")
    return int((amount * NANODOLLARS).to_integral_value(rounding=ROUND_CEILING))


def _dollars(value: int) -> str:
    return format(Decimal(value) / NANODOLLARS, "f")


class SQLiteSpendLedger:
    """Atomic reservations shared by concurrent processes and restarts."""

    def __init__(
        self,
        path: Path,
        *,
        limit_usd: str,
        pricing_id: str,
        allow_pricing_extension: bool = False,
    ) -> None:
        limit = _units(Decimal(limit_usd))
        if limit <= 0 or limit > 1000 * NANODOLLARS:
            raise ValueError("batch limit must be positive and at most 1000 USD")
        if len(pricing_id) != 64 or any(
            c not in "0123456789abcdef" for c in pricing_id
        ):
            raise ValueError("pricing identity must be a SHA256")
        if path.is_symlink():
            raise ValueError("budget path must not be a symlink")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.pricing_id = pricing_id
        self._db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        try:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS budget "
                "(id INTEGER PRIMARY KEY CHECK(id=1), ceiling INTEGER, pricing TEXT, blocked INTEGER)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS calls "
                "(id TEXT PRIMARY KEY, reserved INTEGER, charged INTEGER, status TEXT)"
            )
            self._db.execute("BEGIN IMMEDIATE")
            self._db.execute(
                "INSERT OR IGNORE INTO budget VALUES (1,?,?,0)", (limit, pricing_id)
            )
            row = self._db.execute(
                "SELECT ceiling,pricing FROM budget WHERE id=1"
            ).fetchone()
            if row[0] != limit:
                raise ValueError("existing ledger budget/pricing identity differs")
            # Preserve the original table identity and every historical charge.
            # Separate tables keep the original calls schema readable by old tools.
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS authorized_pricing (id TEXT PRIMARY KEY)"
            )
            self._db.execute(
                "INSERT OR IGNORE INTO authorized_pricing VALUES (?)", (row[1],)
            )
            known = self._db.execute(
                "SELECT 1 FROM authorized_pricing WHERE id=?", (pricing_id,)
            ).fetchone()
            if known is None and not allow_pricing_extension:
                raise ValueError("existing ledger budget/pricing identity differs")
            self._db.execute(
                "INSERT OR IGNORE INTO authorized_pricing VALUES (?)", (pricing_id,)
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS call_pricing "
                "(call_id TEXT PRIMARY KEY, pricing TEXT NOT NULL)"
            )
            self._db.execute(
                "INSERT OR IGNORE INTO call_pricing SELECT id,? FROM calls", (row[1],)
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS attempt_budgets "
                "(id TEXT PRIMARY KEY, ceiling INTEGER NOT NULL, pricing TEXT NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS call_attempts "
                "(call_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL)"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS call_attempts_by_attempt ON call_attempts(attempt_id)"
            )
            self._db.commit()
        except Exception:
            self._db.rollback()
            self._db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._db.close()

    def begin_attempt(self, attempt_id: str, *, limit_usd: str) -> None:
        """Bind an immutable attempt ceiling; reopening never resets its spending."""
        ceiling = _units(Decimal(limit_usd))
        if not attempt_id or len(attempt_id) > 128 or ceiling <= 0:
            raise ValueError("invalid attempt budget identity or ceiling")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                batch_limit = self._db.execute(
                    "SELECT ceiling FROM budget WHERE id=1"
                ).fetchone()[0]
                if ceiling > batch_limit:
                    raise ValueError("attempt limit must not exceed batch limit")
                self._db.execute(
                    "INSERT OR IGNORE INTO attempt_budgets VALUES (?,?,?)",
                    (attempt_id, ceiling, self.pricing_id),
                )
                row = self._db.execute(
                    "SELECT ceiling,pricing FROM attempt_budgets WHERE id=?",
                    (attempt_id,),
                ).fetchone()
                if row != (ceiling, self.pricing_id):
                    raise ValueError("existing attempt budget/pricing identity differs")
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def attempt_summary(self, attempt_id: str) -> dict:
        with self._lock:
            row = self._db.execute(
                "SELECT ceiling,pricing FROM attempt_budgets WHERE id=?", (attempt_id,)
            ).fetchone()
            if row is None or row[1] != self.pricing_id:
                raise ValueError("unknown attempt budget/pricing identity")
            count, spent = self._db.execute(
                "SELECT COUNT(*),COALESCE(SUM(charged),0) FROM calls "
                "JOIN call_attempts ON calls.id=call_attempts.call_id WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
        return {
            "attempt_id": attempt_id,
            "limit_usd": _dollars(row[0]),
            "charged_or_reserved_usd": _dollars(spent),
            "remaining_usd": _dollars(max(0, row[0] - spent)),
            "invocations": count,
        }

    def reserve(
        self, call_id: str, maximum_cost: Decimal, *, attempt_id: str | None = None
    ) -> None:
        amount = _units(maximum_cost)
        if not call_id or len(call_id) > 128 or amount <= 0:
            raise ValueError("invalid provider reservation")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                limit, blocked = self._db.execute(
                    "SELECT ceiling,blocked FROM budget"
                ).fetchone()
                spent = self._db.execute(
                    "SELECT COALESCE(SUM(charged),0) FROM calls"
                ).fetchone()[0]
                if blocked or spent + amount > limit:
                    raise BudgetReservationDenied("batch_cost_reservation_denied")
                if attempt_id is not None:
                    attempt = self._db.execute(
                        "SELECT ceiling,pricing FROM attempt_budgets WHERE id=?",
                        (attempt_id,),
                    ).fetchone()
                    if attempt is None or attempt[1] != self.pricing_id:
                        raise ValueError("unknown attempt budget/pricing identity")
                    attempt_spent = self._db.execute(
                        "SELECT COALESCE(SUM(charged),0) FROM calls "
                        "JOIN call_attempts ON calls.id=call_attempts.call_id WHERE attempt_id=?",
                        (attempt_id,),
                    ).fetchone()[0]
                    if attempt_spent + amount > attempt[0]:
                        raise BudgetReservationDenied("attempt_cost_reservation_denied")
                if self._db.execute(
                    "SELECT 1 FROM calls WHERE id=?", (call_id,)
                ).fetchone():
                    raise BudgetAdmissionError(
                        "provider invocation identity is already reserved"
                    )
                self._db.execute(
                    "INSERT INTO calls VALUES (?,?,?,'reserved')",
                    (call_id, amount, amount),
                )
                self._db.execute(
                    "INSERT INTO call_pricing VALUES (?,?)", (call_id, self.pricing_id)
                )
                if attempt_id is not None:
                    self._db.execute(
                        "INSERT INTO call_attempts VALUES (?,?)", (call_id, attempt_id)
                    )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def settle(
        self, call_id: str, cost: Decimal, *, boundary_violated: bool = False
    ) -> None:
        amount = _units(cost)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT reserved,status FROM calls JOIN call_pricing "
                    "ON calls.id=call_pricing.call_id WHERE calls.id=? AND pricing=?",
                    (call_id, self.pricing_id),
                ).fetchone()
                if row is None or row[1] != "reserved":
                    raise BudgetAdmissionError("unknown or already settled invocation")
                overshot = amount > row[0] or boundary_violated
                self._db.execute(
                    "UPDATE calls SET charged=?,status=? WHERE id=?",
                    (
                        max(row[0], amount) if overshot else amount,
                        "boundary_violation" if overshot else "settled",
                        call_id,
                    ),
                )
                if overshot:
                    self._db.execute("UPDATE budget SET blocked=1")
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        if overshot:
            raise BudgetAdmissionError(
                "provider exceeded preflight assumptions; batch blocked"
            )

    def uncertain(self, call_id: str, status: str) -> None:
        if status not in ("failed_unknown_usage", "missing_usage"):
            raise ValueError("invalid uncertainty status")
        with self._lock, self._db:
            self._db.execute(
                "UPDATE calls SET status=? WHERE id=? AND status='reserved' "
                "AND id IN (SELECT call_id FROM call_pricing WHERE pricing=?)",
                (status, call_id, self.pricing_id),
            )

    def summary(self) -> dict:
        with self._lock:
            limit, pricing, blocked = self._db.execute(
                "SELECT ceiling,pricing,blocked FROM budget"
            ).fetchone()
            count, spent, settled = self._db.execute(
                "SELECT COUNT(*),COALESCE(SUM(charged),0),"
                "COALESCE(SUM(CASE WHEN status='settled' THEN 1 ELSE 0 END),0) FROM calls"
            ).fetchone()
            pricing_ids = [
                row[0]
                for row in self._db.execute(
                    "SELECT id FROM authorized_pricing ORDER BY id"
                )
            ]
        return {
            "limit_usd": _dollars(limit),
            "pricing_table_id": pricing,
            "active_pricing_table_id": self.pricing_id,
            "authorized_pricing_table_ids": pricing_ids,
            "charged_or_reserved_usd": _dollars(spent),
            "remaining_usd": _dollars(max(0, limit - spent)),
            "invocations": count,
            "settled_invocations": settled,
            "unsettled_invocations": count - settled,
            "blocked": bool(blocked),
            "basis": "conservative_peak_estimate_not_invoice",
        }


class BudgetCallback(BaseCallbackHandler):
    """Reserve before LangChain calls the provider; never automatically retry."""

    raise_error = True
    run_inline = True

    def __init__(
        self,
        ledger: SQLiteSpendLedger,
        price: ModelPrice,
        *,
        max_output_tokens: int,
        attempt_id: str | None = None,
    ):
        if not 64 <= max_output_tokens <= 8192:
            raise ValueError("output cap must be between 64 and 8192")
        self.ledger = ledger
        self.price = price
        self.max_output_tokens = max_output_tokens
        self.attempt_id = attempt_id
        self._ceilings: dict[str, int] = {}

    def _cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        return (
            Decimal(self.price.input_usd_per_million_tokens) * input_tokens
            + Decimal(self.price.output_usd_per_million_tokens) * output_tokens
        ) / 1_000_000

    def on_chat_model_start(
        self, serialized, messages: list[list[BaseMessage]], *, run_id, **kwargs
    ) -> None:
        if len(messages) != 1:
            raise BudgetAdmissionError("batched provider requests are not admitted")
        params = kwargs.get("invocation_params", {})
        # Account for text, message framing, tool schemas and provider formatting.
        # UTF-8 bytes plus 4096 framing tokens is deliberately conservative for
        # this text-only BPE provider; token-ceiling violations block the batch.
        payload = {
            "messages": [message.model_dump(mode="json") for message in messages[0]],
            "parameters": params,
        }
        wire_bytes = len(
            json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        )
        if wire_bytes > MAX_INPUT_BYTES:
            raise BudgetAdmissionError(
                "text request exceeds the live batch input boundary"
            )
        actual_cap = params.get("max_tokens", params.get("max_completion_tokens"))
        if actual_cap is not None and actual_cap != self.max_output_tokens:
            raise BudgetAdmissionError("provider output cap differs from reserved cap")
        if params.get("n", 1) != 1:
            raise BudgetAdmissionError("multiple completions are not admitted")
        ceiling = wire_bytes + 4096
        self.ledger.reserve(
            str(run_id),
            self._cost(ceiling, self.max_output_tokens),
            attempt_id=self.attempt_id,
        )
        self._ceilings[str(run_id)] = ceiling

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs) -> None:
        identity = str(run_id)
        ceiling = self._ceilings.pop(identity, None)
        if ceiling is None:
            raise BudgetAdmissionError("response has no matching pre-call reservation")
        try:
            if len(response.generations) != 1 or len(response.generations[0]) != 1:
                raise ValueError("unexpected response batch")
            usage = response.generations[0][0].message.usage_metadata
            inputs, outputs = usage["input_tokens"], usage["output_tokens"]
            if any(type(value) is not int or value < 0 for value in (inputs, outputs)):
                raise ValueError("invalid token usage")
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            self.ledger.uncertain(identity, "missing_usage")
            return
        self.ledger.settle(
            identity,
            self._cost(inputs, outputs),
            boundary_violated=inputs > ceiling or outputs > self.max_output_tokens,
        )

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._ceilings.pop(str(run_id), None)
        self.ledger.uncertain(str(run_id), "failed_unknown_usage")
