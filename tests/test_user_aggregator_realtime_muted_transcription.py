#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Realtime transcripts that arrive while the user is muted reach the context.

Ported from dograh-hq/pipecat d03e2292 (inshurik LUCY-598). A speech-to-speech
service transcribes audio it has already heard and often delivers the
transcript after its reply has started, when a mute strategy (bot speaking,
a function call in flight) has muted the user. The transcript must still be
recorded in the context, while turn-taking stays muted: it may not start or
stop a user turn.

The turn strategies below are the ones the Inshurik Grok pipeline uses
(``ExternalUserTurnStartStrategy`` + ``ExternalUserTurnStopStrategy``).
"""

import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_mute import AlwaysUserMuteStrategy, BaseUserMuteStrategy
from pipecat.turns.user_start import ExternalUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies


class _MuteWhileBotSpeaks(BaseUserMuteStrategy):
    """Muted from BotStartedSpeaking to BotStoppedSpeaking (a stand-in for the
    function-call / first-bot-complete strategies the pipeline really uses)."""

    def __init__(self):
        super().__init__()
        self._muted = False

    async def process_frame(self, frame) -> bool:
        await super().process_frame(frame)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._muted = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._muted = False
        return self._muted


def _pair(*, realtime_service_mode, mute_strategy):
    context = LLMContext()
    pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[ExternalUserTurnStartStrategy()],
                stop=[ExternalUserTurnStopStrategy(timeout=0.05)],
            ),
            user_mute_strategies=[mute_strategy],
        ),
        realtime_service_mode=realtime_service_mode,
    )
    return context, pair


def _transcript(text):
    return TranscriptionFrame(text=text, user_id="", timestamp="now")


def _user_texts(context):
    return [m["content"] for m in context.get_messages() if m["role"] == "user"]


class TestRealtimeMutedTranscription(unittest.IsolatedAsyncioTestCase):
    async def test_muted_transcription_is_dropped_without_realtime_mode(self):
        # The pre-port behavior, still the default.
        context, pair = _pair(realtime_service_mode=None, mute_strategy=AlwaysUserMuteStrategy())
        user, assistant = pair

        await run_test(
            Pipeline([user, assistant]),
            frames_to_send=[
                BotStartedSpeakingFrame(),
                SleepFrame(),
                _transcript("building number 35"),
                SleepFrame(),
                BotStoppedSpeakingFrame(),
                SleepFrame(),
            ],
            expected_down_frames=None,
        )

        self.assertEqual(_user_texts(context), [])

    async def test_muted_transcription_reaches_the_context_in_realtime_mode(self):
        context, pair = _pair(realtime_service_mode=True, mute_strategy=AlwaysUserMuteStrategy())
        user, assistant = pair
        turns_started = []
        turns_stopped = []

        @user.event_handler("on_user_turn_started")
        async def _on_start(_aggregator, _strategy):
            turns_started.append(True)

        @user.event_handler("on_user_turn_stopped")
        async def _on_stop(_aggregator, _strategy, message):
            turns_stopped.append(message.content)

        await run_test(
            Pipeline([user, assistant]),
            frames_to_send=[
                BotStartedSpeakingFrame(),
                SleepFrame(),
                _transcript("building number 35"),
                SleepFrame(sleep=0.2),
                BotStoppedSpeakingFrame(),
                SleepFrame(),
            ],
            expected_down_frames=None,
        )

        # Kept for the context ...
        self.assertEqual(_user_texts(context), ["building number 35"])
        # ... but turn-taking stayed muted: the transcript started no turn,
        # and only the session end flushed it.
        self.assertEqual(turns_started, [])
        self.assertEqual(turns_stopped, ["building number 35"])

    async def test_muted_transcription_is_committed_ahead_of_the_next_turn(self):
        # Lucy's shape: the caller's answer lands while she is muted, then the
        # caller speaks again once she is unmuted. Nothing is lost; the held
        # words close with the next turn, in the order they were said.
        context, pair = _pair(realtime_service_mode=True, mute_strategy=_MuteWhileBotSpeaks())
        user, assistant = pair

        await run_test(
            Pipeline([user, assistant]),
            frames_to_send=[
                BotStartedSpeakingFrame(),
                SleepFrame(),
                _transcript("Yes."),
                SleepFrame(sleep=0.2),
                BotStoppedSpeakingFrame(),
                SleepFrame(),
                UserStartedSpeakingFrame(),
                SleepFrame(),
                UserStoppedSpeakingFrame(),
                _transcript("It is 35 Main Street."),
                SleepFrame(sleep=0.2),
            ],
            expected_down_frames=None,
        )

        self.assertEqual(_user_texts(context), ["Yes. It is 35 Main Street."])


if __name__ == "__main__":
    unittest.main()
