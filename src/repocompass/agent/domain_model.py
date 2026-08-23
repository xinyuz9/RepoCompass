"""Shared Pydantic behavior for checkpointed RepoCompass domain aggregates."""

from typing import Any

from pydantic import BaseModel


class CheckpointSafeModel(BaseModel):
    """Preserve nested models for LangGraph's no-argument msgpack dump path.

    ``JsonPlusSerializer`` calls ``model_dump()`` without arguments. Standard
    recursive dumping turns strict nested models into dictionaries, so they
    cannot be reconstructed by a strict aggregate on the next checkpoint read.
    Explicit public dumps such as ``model_dump(mode="json")`` retain normal
    Pydantic behavior.
    """

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Return checkpoint-safe Python fields only for an implicit dump."""

        if not kwargs:
            return dict(self.__dict__)

        return super().model_dump(**kwargs)
