#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Grok's provisional ``...transcription.completed`` snapshots are interim.

Ported from dograh-hq/pipecat cb766496 (28ce4694 + 385b53e1). Grok sends
cumulative provisional readings of one user item under the *completed* event
name with ``status="in_progress"``; only the final one (``status="completed"``,
or no status on older servers) may become a ``TranscriptionFrame``.
"""

import json
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.xai.realtime import events
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService


def _event(**fields):
    return events.parse_server_event(
        json.dumps(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "event_id": "event-1",
                "item_id": "user-item-1",
                "content_index": 0,
                "transcript": "Can you hear me?",
                **fields,
            }
        )
    )


def _service():
    service = GrokRealtimeLLMService(api_key="test-key")
    service.push_frame = AsyncMock()
    service._call_event_handler = AsyncMock()
    return service


def test_completed_event_keeps_status_field():
    assert _event(status="in_progress").status == "in_progress"
    assert _event(status="completed").status == "completed"
    assert _event().status is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_fields, frame_type",
    [
        ({"status": "in_progress"}, InterimTranscriptionFrame),
        ({"status": "completed"}, TranscriptionFrame),
        ({}, TranscriptionFrame),
        ({"status": None}, TranscriptionFrame),
    ],
    ids=["provisional", "final", "legacy-omitted", "legacy-null"],
)
async def test_completed_transcription_uses_status_to_select_frame(status_fields, frame_type):
    service = _service()

    await service._handle_evt_input_audio_transcription_completed(_event(**status_fields))

    service.push_frame.assert_awaited_once()
    frame, direction = service.push_frame.await_args.args
    assert type(frame) is frame_type
    assert frame.text == "Can you hear me?"
    assert frame.result.status == status_fields.get("status")
    assert direction == FrameDirection.UPSTREAM
    if frame_type is InterimTranscriptionFrame:
        service._call_event_handler.assert_not_awaited()
    else:
        service._call_event_handler.assert_awaited_once_with(
            "on_conversation_item_updated", "user-item-1", None
        )


@pytest.mark.asyncio
async def test_one_spoken_phrase_yields_one_final_transcript():
    # The sequence dograh captured from one spoken "Can you hear me?" on Grok.
    service = _service()
    for fields in (
        {"status": "in_progress", "transcript": "Can you hear"},
        {"status": "in_progress", "transcript": "Can you hear me?"},
        {"status": "completed", "transcript": "Can you hear me?"},
    ):
        await service._handle_evt_input_audio_transcription_completed(_event(**fields))

    pushed = [call.args[0] for call in service.push_frame.await_args_list]
    assert [type(f) for f in pushed] == [
        InterimTranscriptionFrame,
        InterimTranscriptionFrame,
        TranscriptionFrame,
    ]
    assert [f.text for f in pushed if isinstance(f, TranscriptionFrame)] == ["Can you hear me?"]
