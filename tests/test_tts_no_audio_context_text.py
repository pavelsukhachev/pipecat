#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A TTS audio context that produces no audio adds none of its text to the context (inshurik).

inshurik LUCY-599, ported from upstream pipecat-ai/pipecat#6234 (dograh-hq/pipecat
9ac08e90). Before the fix, a context that ended without audio (a provider error, a
provider that returned nothing, or first audio arriving after stop_frame_timeout_s)
still emitted the reply's text at force-complete, so the LLM context recorded a reply
the caller never heard.
"""

import asyncio
from collections.abc import AsyncGenerator, Sequence

import pytest
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

_FAKE_AUDIO = b"\x00\x01" * 320
_SAMPLE_RATE = 16000
_STOP_FRAME_TIMEOUT_S = 0.2


def _audio(context_id: str) -> TTSAudioRawFrame:
    return TTSAudioRawFrame(
        audio=_FAKE_AUDIO, sample_rate=_SAMPLE_RATE, num_channels=1, context_id=context_id
    )


class MockTTSService(TTSService):
    """HTTP-style service: audio, if any, is yielded from run_tts.

    Args:
        speaks: Whether the service yields any audio.
        push_text_frames: False models a word-timestamp service that never sends
            timestamps, so its text comes out at force-complete.
    """

    def __init__(self, speaks: bool = True, push_text_frames: bool = True, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=push_text_frames,
            sample_rate=_SAMPLE_RATE,
            stop_frame_timeout_s=_STOP_FRAME_TIMEOUT_S,
            **kwargs,
        )
        self._speaks = speaks

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if self._speaks:
            yield _audio(context_id)


class MockLateAudioTTSService(TTSService):
    """WebSocket-style service whose audio arrives after its text is queued.

    Args:
        audio_delay_s: When the audio arrives. Past the stop-frame timeout, the
            context has already ended and the audio comes too late to be played.
    """

    def __init__(self, audio_delay_s: float = _STOP_FRAME_TIMEOUT_S / 2, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=True,
            sample_rate=_SAMPLE_RATE,
            stop_frame_timeout_s=_STOP_FRAME_TIMEOUT_S,
            **kwargs,
        )
        self._audio_delay_s = audio_delay_s

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        async def _deliver_audio():
            await asyncio.sleep(self._audio_delay_s)
            if self.audio_context_available(context_id):
                await self.append_to_audio_context(context_id, _audio(context_id))

        self.create_task(_deliver_audio(), name=f"late_audio_{context_id}")
        if False:
            yield


class MockStoppedWithoutAudioTTSService(TTSService):
    """Word-timestamp service whose provider ends the context with no audio at all.

    The provider's "done" arrives as a TTSStoppedFrame in the audio context, so the
    context is closed by the TTSStoppedFrame branch of the playback loop rather than
    by the end-of-context path.
    """

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            stop_frame_timeout_s=_STOP_FRAME_TIMEOUT_S,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        yield TTSStoppedFrame(context_id=context_id)


def _speak(*texts: str) -> list[Frame]:
    frames: list[Frame] = []
    for text in texts:
        frames += [
            LLMFullResponseStartFrame(),
            TTSSpeakFrame(text=text),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=_STOP_FRAME_TIMEOUT_S * 4),
        ]
    return frames


def _context_text(down: Sequence[Frame]) -> list[str]:
    return [f.text for f in down if isinstance(f, TTSTextFrame) and f.append_to_context]


@pytest.mark.asyncio
@pytest.mark.parametrize("push_text_frames", [True, False])
async def test_a_silent_context_adds_no_text_to_the_context(push_text_frames):
    tts = MockTTSService(speaks=False, push_text_frames=push_text_frames)

    down, _ = await asyncio.wait_for(run_test(tts, frames_to_send=_speak("Hello there.")), 5.0)

    assert not any(isinstance(f, TTSTextFrame) for f in down)
    # The context still ends cleanly.
    assert any(isinstance(f, TTSStoppedFrame) for f in down)


@pytest.mark.asyncio
@pytest.mark.parametrize("push_text_frames", [True, False])
async def test_a_spoken_context_adds_its_text_to_the_context(push_text_frames):
    tts = MockTTSService(speaks=True, push_text_frames=push_text_frames)

    down, _ = await asyncio.wait_for(run_test(tts, frames_to_send=_speak("Hello there.")), 5.0)

    assert "".join(_context_text(down)).strip() == "Hello there."


@pytest.mark.asyncio
async def test_text_queued_before_its_audio_is_pushed_ahead_of_it():
    tts = MockLateAudioTTSService()

    down, _ = await asyncio.wait_for(run_test(tts, frames_to_send=_speak("Hello there.")), 5.0)

    assert _context_text(down) == ["Hello there."]
    kinds = [type(f) for f in down if isinstance(f, (TTSTextFrame, TTSAudioRawFrame))]
    assert kinds.index(TTSTextFrame) < kinds.index(TTSAudioRawFrame)


@pytest.mark.asyncio
async def test_audio_arriving_after_the_timeout_adds_no_text():
    # The MiniMax case upstream: first audio after stop_frame_timeout_s. The context
    # ended silent, so the reply the caller never heard stays out of the context.
    tts = MockLateAudioTTSService(audio_delay_s=_STOP_FRAME_TIMEOUT_S * 2)

    down, _ = await asyncio.wait_for(run_test(tts, frames_to_send=_speak("Hello there.")), 5.0)

    assert _context_text(down) == []


@pytest.mark.asyncio
async def test_a_stopped_frame_without_audio_keeps_the_text_out_of_the_context():
    # Word-timestamp service, provider ends the context with TTSStoppedFrame and no audio:
    # the force-complete in the TTSStoppedFrame branch must drop the unheard text, not
    # emit it (second review of LUCY-599: that call was otherwise unprotected).
    tts = MockStoppedWithoutAudioTTSService()

    down, _ = await asyncio.wait_for(run_test(tts, frames_to_send=_speak("Hello there.")), 5.0)

    assert any(isinstance(f, TTSStoppedFrame) for f in down)
    assert not any(isinstance(f, TTSTextFrame) for f in down)
    assert _context_text(down) == []


# A reply can carry a customer's personal details read back (a name, a date of birth,
# bank digits). The no-audio warnings must say how much was dropped, never what.
_SECRET = "Your date of birth is 04/17/1951 and the account ends 6623."


@pytest.fixture
def log_records():
    records = []
    sink_id = logger.add(lambda m: records.append(m.record), level=0, format="{message}")
    yield records
    logger.remove(sink_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("push_text_frames", [True, False])
async def test_the_dropped_text_is_never_logged(push_text_frames, log_records):
    tts = MockTTSService(speaks=False, push_text_frames=push_text_frames)

    down, _ = await asyncio.wait_for(run_test(tts, frames_to_send=_speak(_SECRET)), 5.0)

    assert not any(isinstance(f, TTSTextFrame) for f in down)
    # DEBUG and up: pipecat's generic TRACE frame dump (off in production) prints every
    # frame's text and is out of scope here. The "Generating TTS" line every synthesis
    # logs is in scope: on 1.16 it logs the length only (inshurik LUCY-603).
    messages = [r["message"] for r in log_records if r["level"].no >= 10]
    for fragment in ("04/17/1951", "6623", "date of birth"):
        assert not any(fragment in m for m in messages), fragment
    warnings = [
        r["message"]
        for r in log_records
        if r["level"].name == "WARNING" and "produced no audio" in r["message"]
    ]
    assert warnings, "the drop is still reported"
    assert any(f"({len(_SECRET)} chars)" in w for w in warnings)
