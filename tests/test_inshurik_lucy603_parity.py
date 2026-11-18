#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fork behaviour the inshurik voice app relies on (LUCY-603).

* The async-tool guidance joins the system instruction only with
  ``enable_async_tool_cancellation``: a tool registered with
  ``cancel_on_interruption=False`` just to survive barge-in leaves the prompt as given.
* The xAI receive loop hands the errors it reads past to an overridable hook.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.async_tool_messages import ASYNC_TOOL_INSTRUCTIONS
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import LLMSettings
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService

PROMPT = "You are a phone agent. Ask for the caller's first name."


async def _composed(**kwargs) -> str:
    with patch.object(OpenAILLMService, "create_client"):
        llm = OpenAILLMService(api_key="test-key", **kwargs)
    llm.register_function("transfer", AsyncMock(), cancel_on_interruption=False)
    await llm._update_settings(LLMSettings(system_instruction=PROMPT))
    return llm._settings.system_instruction


@pytest.mark.asyncio
async def test_an_async_tool_leaves_the_system_instruction_as_given():
    assert await _composed() == PROMPT


@pytest.mark.asyncio
async def test_async_tool_cancellation_still_adds_the_guidance():
    with pytest.warns(DeprecationWarning):
        composed = await _composed(enable_async_tool_cancellation=True)
    assert composed.startswith(PROMPT)
    assert ASYNC_TOOL_INSTRUCTIONS in composed


class _Socket:
    def __init__(self, *events):
        self._msgs = [json.dumps(e) for e in events]
        self.read = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.read >= len(self._msgs):
            raise StopAsyncIteration
        self.read += 1
        return self._msgs[self.read - 1]


@pytest.mark.asyncio
async def test_a_tolerated_xai_error_reaches_the_hook_and_the_loop_reads_on():
    llm = GrokRealtimeLLMService(api_key="test-key")
    llm._handle_evt_tolerated_error = AsyncMock()
    llm._handle_evt_error = AsyncMock()
    refusal = {
        "type": "error",
        "event_id": "e1",
        "error": {
            "type": "invalid_request_error",
            "code": None,
            "message": "Cancellation failed: no active response found",
        },
    }
    deleted = {"type": "conversation.item.deleted", "event_id": "e2", "item_id": "i1"}
    llm._websocket = _Socket(refusal, deleted)
    await llm._receive_task_handler()
    assert llm._websocket.read == 2
    llm._handle_evt_tolerated_error.assert_awaited_once()
    llm._handle_evt_error.assert_not_awaited()
