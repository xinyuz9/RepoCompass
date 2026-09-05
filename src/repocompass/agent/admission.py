"""Provider-independent, pre-request spending denial contract."""

PRECALL_BUDGET_STOP_REASONS = frozenset(
    {"attempt_cost_reservation_denied", "batch_cost_reservation_denied"}
)


class PrecallBudgetDenied(RuntimeError):
    """A spending reservation failed before a provider request was sent."""

    def __init__(self, stop_reason: str) -> None:
        if stop_reason not in PRECALL_BUDGET_STOP_REASONS:
            raise ValueError("unknown pre-request budget stop reason")
        self.stop_reason = stop_reason
        super().__init__(stop_reason)
