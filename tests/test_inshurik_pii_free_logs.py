#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Warnings that name what the bot was about to say, or a tool call's arguments, log lengths only (inshurik).

A reply can carry a caller's name, date of birth or phone number read back to them, and tool
arguments carry the same fields. Such warnings say how much went unspoken or unparsed.
"""

import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from loguru import logger

from pipecat.frames.frames import Frame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import run_test

_PII = "Sarah Johnson, born March 3 1951, at 786 555 0100"


def _capture():
    lines: list[str] = []
    sink_id = logger.add(lambda m: lines.append(str(m)), level="TRACE")
    return lines, lambda: logger.remove(sink_id)


class _SilentTTSService(TTSService):
    def __init__(self, **kwargs):
        super().__init__(push_start_frame=True, push_stop_frames=True, sample_rate=16000, **kwargs)

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if False:
            yield


@pytest.mark.asyncio
async def test_an_unusable_tts_names_the_length_of_the_line_not_the_line():
    tts = _SilentTTSService()
    tts._is_usable = False
    lines, stop = _capture()
    try:
        await asyncio.wait_for(run_test(tts, frames_to_send=[TTSSpeakFrame(_PII)]), timeout=5.0)
    finally:
        stop()
    warnings = [line for line in lines if "no longer usable" in line]
    assert warnings, "the unspoken line is no longer reported at all"
    assert all("Sarah" not in w and "1951" not in w and "0100" not in w for w in warnings)
    assert all("chars)" in w for w in warnings)


def _tool_call_chunk(name: str, arguments: str):
    tool_call = SimpleNamespace(
        index=0,
        id="call_1",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    delta = SimpleNamespace(tool_calls=[tool_call], content=None)
    return SimpleNamespace(usage=None, model=None, choices=[SimpleNamespace(delta=delta)])


class _Stream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


@pytest.mark.asyncio
async def test_unparseable_tool_arguments_are_logged_by_length():
    with patch.object(OpenAILLMService, "create_client"):
        llm = OpenAILLMService(api_key="test-key")
    broken = '{"name": "' + _PII  # cut off mid-stream: not valid JSON
    llm.get_chat_completions = AsyncMock(
        return_value=_Stream([_tool_call_chunk("save_applicant", broken)])
    )
    llm.push_frame = AsyncMock()
    llm._run_or_defer_function_calls = AsyncMock()
    lines, stop = _capture()
    try:
        await llm._process_context(LLMContext())
    finally:
        stop()
    warnings = [line for line in lines if "Failed to parse function call arguments" in line]
    assert len(warnings) == 1
    assert "Sarah" not in warnings[0] and "1951" not in warnings[0]
    assert f"({len(broken)} chars)" in warnings[0]


def _only_parse_warning(lines: list[str], broken: str) -> str:
    warnings = [line for line in lines if "Failed to parse function call arguments" in line]
    assert len(warnings) == 1
    assert "Sarah" not in warnings[0] and "1951" not in warnings[0]
    assert f"({len(broken)} chars)" in warnings[0]
    return warnings[0]


@pytest.mark.asyncio
async def test_sambanova_logs_unparseable_tool_arguments_by_length():
    from pipecat.services.sambanova.llm import SambaNovaLLMService

    with patch.object(SambaNovaLLMService, "create_client"):
        llm = SambaNovaLLMService(api_key="test-key")
    broken = '{"name": "' + _PII
    llm.get_chat_completions = AsyncMock(
        return_value=_Stream([_tool_call_chunk("save_applicant", broken)])
    )
    llm.push_frame = AsyncMock()
    llm.run_function_calls = AsyncMock()
    lines, stop = _capture()
    try:
        await llm._process_context(LLMContext())
    finally:
        stop()
    _only_parse_warning(lines, broken)


def test_openai_responses_logs_unparseable_tool_arguments_by_length():
    from pipecat.services.openai.responses.llm import OpenAIResponsesHttpLLMService

    llm = OpenAIResponsesHttpLLMService(api_key="test-key")
    broken = '{"name": "' + _PII
    lines, stop = _capture()
    try:
        calls = llm._process_function_calls(
            LLMContext(),
            {"item_1": {"name": "save_applicant", "call_id": "c1", "arguments": broken}},
        )
    finally:
        stop()
    _only_parse_warning(lines, broken)
    assert calls[0].arguments == {}


def test_google_interactions_logs_unparseable_tool_arguments_by_length():
    from pipecat.services.google.interactions.llm import GoogleInteractionsLLMService

    llm = GoogleInteractionsLLMService(api_key="test-key")
    broken = '{"name": "' + _PII
    lines, stop = _capture()
    try:
        llm._function_call(
            LLMContext(),
            {"name": "save_applicant", "id": "c1", "initial": {}, "arguments": broken},
        )
    finally:
        stop()
    _only_parse_warning(lines, broken)
