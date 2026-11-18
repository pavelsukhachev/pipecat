#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inshurik fork patches 1 and 4 on LLMAssistantAggregator.

Patch 1: a final function-call result whose entry another task already removed from
``_function_calls_in_progress`` is absorbed, not raised as ``KeyError`` (prod ErrorFrame
"Error processing frame: 'call_<id>'", ~5/day before the fix).

Patch 4: tokens from an LLM response that started before an interruption are dropped
once ``reset()`` has bumped the aggregation generation, so they never interleave into
the next reply (inshurik run 1274).
"""

import unittest

from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.tests.utils import SleepFrame, run_test


class TestFinishedResultRace(unittest.IsolatedAsyncioTestCase):
    async def test_result_for_a_call_already_removed_is_absorbed(self):
        aggregator = LLMAssistantAggregator(LLMContext())
        in_progress = FunctionCallInProgressFrame(
            function_name="f", tool_call_id="call_1", arguments={}
        )
        result = FunctionCallResultFrame(
            function_name="f", tool_call_id="call_1", arguments={}, result={"ok": True}
        )
        # A concurrent handler removed the entry between the caller's membership check
        # and the removal inside _handle_function_call_finished.
        self.assertNotIn("call_1", aggregator._function_calls_in_progress)
        await aggregator._handle_function_call_finished(result, in_progress)
        self.assertNotIn("call_1", aggregator._function_calls_in_progress)


class TestStaleTokensAfterInterruption(unittest.IsolatedAsyncioTestCase):
    async def test_tokens_from_the_interrupted_response_never_reach_the_next_reply(self):
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)
        await run_test(
            aggregator,
            frames_to_send=[
                LLMFullResponseStartFrame(),
                LLMTextFrame("Hello "),
                SleepFrame(),
                InterruptionFrame(),
                # Still draining from the interrupted response: no new start frame yet.
                LLMTextFrame("stale "),
                LLMTextFrame("words "),
                SleepFrame(),
                LLMFullResponseStartFrame(),
                LLMTextFrame("Fresh "),
                LLMTextFrame("reply."),
                LLMFullResponseEndFrame(),
            ],
        )
        contents = [m["content"] for m in context.messages]
        self.assertEqual(contents[-1], "Fresh reply.")
        self.assertFalse(any("stale" in c for c in contents), contents)

    async def test_reset_moves_the_generation_on(self):
        aggregator = LLMAssistantAggregator(LLMContext())
        before = aggregator._aggregation_generation
        await aggregator.reset()
        self.assertEqual(aggregator._aggregation_generation, before + 1)


if __name__ == "__main__":
    unittest.main()
