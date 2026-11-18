#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inshurik patches on BaseOpenAILLMService (patch 7, LUCY-603 log privacy)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from loguru import logger

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService

# Stand-ins for a caller's details: a date of birth and account digits.
_SECRET_ARGS = '{"dob": "04/17/1951", "acct": "6623"'  # truncated JSON: unparsable


class _Stream:
    def __init__(self, chunks, stall_first: float | None = None):
        self._chunks = list(chunks)
        self._stall_first = stall_first
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        if self._stall_first:
            await asyncio.sleep(self._stall_first)
        for chunk in self._chunks:
            yield chunk

    async def close(self):
        self.closed = True


def _content(text):
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(tool_calls=None, content=text))],
    )


def _tool_call(name, arguments):
    call = SimpleNamespace(
        index=0, id="call-1", function=SimpleNamespace(name=name, arguments=arguments)
    )
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(tool_calls=[call], content=None))],
    )


def _service(**kwargs) -> OpenAILLMService:
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(api_key="test-key", **kwargs)
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.stop_ttfat_metrics = AsyncMock()
    service._push_llm_text = AsyncMock()
    service.push_frame = AsyncMock()
    service.run_function_calls = AsyncMock()
    return service


@pytest.fixture
def messages():
    records: list[str] = []
    sink = logger.add(lambda m: records.append(m.record["message"]), level=0)
    yield records
    logger.remove(sink)


# ---------------------------------------------------------------------------
# Patch 7: first_token_timeout_secs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stream_stalled_before_its_first_token_is_retried_once():
    service = _service(first_token_timeout_secs=0.05)
    stalled = _Stream([_content("late")], stall_first=5)
    healthy = _Stream([_content("hello")])
    service.get_chat_completions = AsyncMock(side_effect=[stalled, healthy])

    await asyncio.wait_for(service._process_context(LLMContext()), 2.0)

    assert service.get_chat_completions.await_count == 2
    assert stalled.closed
    service._push_llm_text.assert_awaited_once_with("hello")


@pytest.mark.asyncio
async def test_without_the_bound_a_slow_first_token_is_waited_for():
    service = _service()
    slow = _Stream([_content("hello")], stall_first=0.1)
    service.get_chat_completions = AsyncMock(return_value=slow)

    await asyncio.wait_for(service._process_context(LLMContext()), 2.0)

    assert service.get_chat_completions.await_count == 1
    service._push_llm_text.assert_awaited_once_with("hello")


@pytest.mark.asyncio
async def test_the_retry_itself_is_not_bounded():
    service = _service(first_token_timeout_secs=0.05)
    stalled = _Stream([_content("late")], stall_first=5)
    slow = _Stream([_content("hello")], stall_first=0.15)
    service.get_chat_completions = AsyncMock(side_effect=[stalled, slow])

    await asyncio.wait_for(service._process_context(LLMContext()), 2.0)

    service._push_llm_text.assert_awaited_once_with("hello")


# ---------------------------------------------------------------------------
# LUCY-603: no caller or LLM text in logs, lengths only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unparsable_tool_arguments_are_not_logged(messages):
    service = _service()
    service.get_chat_completions = AsyncMock(
        return_value=_Stream([_tool_call("look_up", _SECRET_ARGS)])
    )

    await service._process_context(LLMContext())

    assert any("Failed to parse function call arguments" in m for m in messages)
    assert any(f"({len(_SECRET_ARGS)} chars)" in m for m in messages)
    for fragment in ("04/17/1951", "6623", "dob"):
        assert not any(fragment in m for m in messages), fragment


@pytest.mark.asyncio
async def test_punctuation_only_content_logs_its_length_not_its_text(messages):
    service = _service()
    service.get_chat_completions = AsyncMock(
        return_value=_Stream([_content("?!..."), _tool_call("transition", "{}")])
    )

    await service._process_context(LLMContext())

    lines = [m for m in messages if "no speakable content" in m]
    assert lines, "the unspeakable-content line is still logged"
    assert all("?!..." not in m for m in lines)
    assert any("(5 chars)" in m for m in lines)
