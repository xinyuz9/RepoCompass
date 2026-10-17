"""Bound non-streaming OpenAI-compatible calls without replacing LangChain."""

import asyncio

import httpx
from langchain_openai import ChatOpenAI
from openai import APITimeoutError
from pydantic import Field

from repocompass.models.control import ModelCallTimeout, await_model_request


class ControlledChatOpenAI(ChatOpenAI):
    model_call_timeout: float = Field(default=60.0, gt=0, allow_inf_nan=False)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Graph nodes are synchronous; the request itself remains cancellable.
        return asyncio.run(self._agenerate(messages, stop=stop, **kwargs))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        # A per-call client is closed on success, timeout and cancellation. It also
        # avoids sharing a connection pool across the sync bridge's event loops.
        async with self.root_async_client.copy(
            http_client=httpx.AsyncClient(timeout=self.request_timeout),
            max_retries=0,
        ) as client:
            model = self.model_copy(
                update={
                    "root_async_client": client,
                    "async_client": client.chat.completions,
                }
            )
            try:
                return await await_model_request(
                    lambda: ChatOpenAI._agenerate(
                        model, messages, stop=stop, run_manager=run_manager, **kwargs
                    ),
                    timeout_seconds=self.model_call_timeout,
                )
            except APITimeoutError:
                raise ModelCallTimeout("model transport timed out") from None
