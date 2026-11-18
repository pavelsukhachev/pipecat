#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A graceful close that cannot finish escalates to a hard cancel (inshurik patch 11a).

On 2026-08-26 a processor blocking the EndFrame (a dead telephony websocket, a wedged
LLM service) parked PipelineWorker.run() forever, and 21 zombie workers accumulated in
production (inshurik voice#200). The wait for the EndFrame is bounded by
``end_timeout_secs``; on timeout the worker cancels instead.
"""

import asyncio
import unittest

from pipecat.frames.frames import EndFrame, Frame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, WorkerParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.asyncio.task_manager import TaskManager


class EndBlocker(FrameProcessor):
    """Swallows the EndFrame forever, like a processor wedged on a dead socket."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._block = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, EndFrame):
            await self._block.wait()
        await self.push_frame(frame, direction)


class TestPipelineWorkerEndTimeout(unittest.IsolatedAsyncioTestCase):
    async def _run(self, **worker_kwargs):
        worker = PipelineWorker(Pipeline([EndBlocker()]), cancel_timeout_secs=0.2, **worker_kwargs)
        finished: list[Frame] = []
        self.timeouts: list[Frame] = []

        @worker.event_handler("on_pipeline_started")
        async def on_pipeline_started(worker: PipelineWorker, frame: StartFrame):
            await worker.queue_frame(EndFrame())

        @worker.event_handler("on_pipeline_finished")
        async def on_pipeline_finished(worker: PipelineWorker, frame: Frame):
            finished.append(frame)

        @worker.event_handler("on_pipeline_timeout")
        async def on_pipeline_timeout(worker: PipelineWorker, frame: Frame):
            self.timeouts.append(frame)

        run_task = asyncio.create_task(worker.run(WorkerParams(task_manager=TaskManager())))
        return worker, run_task, finished

    async def test_blocked_end_escalates_to_cancel(self):
        worker, run_task, finished = await self._run(end_timeout_secs=0.2)

        await asyncio.wait_for(run_task, timeout=3.0)

        self.assertTrue(worker.has_finished())
        # on_pipeline_finished still fires exactly once, from the cancel path.
        self.assertEqual(len(finished), 1)
        # The escalation queued a CancelFrame, which reached the end of the pipeline
        # where the EndFrame could not: the cancel wait itself did not time out.
        self.assertEqual(self.timeouts, [])

    def test_graceful_close_is_bounded_by_default(self):
        worker = PipelineWorker(Pipeline([EndBlocker()]))
        self.assertEqual(worker._end_timeout_secs, 30.0)


if __name__ == "__main__":
    unittest.main()
