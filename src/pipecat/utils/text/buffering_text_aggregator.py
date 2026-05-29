#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Buffering sentence aggregator (Inshurik) — merges short sentences before TTS.

Wraps :class:`SimpleTextAggregator`. Completed sentences are held and
concatenated until the pending text reaches ``min_chars`` characters, then
emitted as a single :class:`Aggregation`. This reduces the number of tiny,
independent TTS generations (e.g. a lone ``"yes"`` or ``"Then hit NEXT."`` in a
form-walk) that ElevenLabs renders with inconsistent prosody on phone calls —
each fragment, sent as its own context, has no surrounding sentence context.

Set ``min_chars=0`` to disable (behaves exactly like ``SimpleTextAggregator``).
In ``TOKEN`` aggregation mode buffering is bypassed entirely.

Trade-off: holding completed sentences adds a small amount of time-to-first-audio
(the bot waits to accumulate ``min_chars`` before speaking). Tunable per stack via
the ``TTS_MIN_AGGREGATION_CHARS`` env var (read in ``service_factory``); default
off so production behaviour is unchanged until explicitly enabled.
"""

from collections.abc import AsyncIterator

from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


class BufferingSentenceAggregator(SimpleTextAggregator):
    """Sentence aggregator that merges short sentences up to a minimum length."""

    def __init__(self, *, min_chars: int = 0, **kwargs):
        """Initialize the buffering aggregator.

        Args:
            min_chars: Minimum character length to accumulate before emitting a
                merged aggregation. ``0`` disables buffering.
            **kwargs: Forwarded to :class:`SimpleTextAggregator` /
                :class:`BaseTextAggregator` (e.g. ``aggregation_type``).
        """
        super().__init__(**kwargs)
        self._min_chars = max(0, int(min_chars))
        self._pending = ""

    @property
    def text(self) -> Aggregation:
        """Return the held pending text plus any in-progress sentence."""
        base = super().text.text
        combined = " ".join(p for p in (self._pending, base) if p).strip(" ")
        return Aggregation(text=combined, type=AggregationType.SENTENCE)

    def _merge(self, addition: str) -> None:
        addition = addition.strip(" ")
        if not addition:
            return
        self._pending = f"{self._pending} {addition}".strip(" ") if self._pending else addition

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        """Aggregate text, holding completed sentences until ``min_chars``.

        When buffering is disabled (``min_chars <= 0``) or in ``TOKEN`` mode,
        this defers entirely to :class:`SimpleTextAggregator`.
        """
        if self._min_chars <= 0 or self._aggregation_type == AggregationType.TOKEN:
            async for agg in super().aggregate(text):
                yield agg
            return

        async for agg in super().aggregate(text):
            self._merge(agg.text)
            if len(self._pending) >= self._min_chars:
                yield Aggregation(text=self._pending, type=AggregationType.SENTENCE)
                self._pending = ""

    async def flush(self) -> Aggregation | None:
        """Emit any held pending text plus the in-progress remainder."""
        base = await super().flush()
        if base and base.text:
            self._merge(base.text)
        if self._pending:
            result = self._pending
            self._pending = ""
            return Aggregation(text=result, type=AggregationType.SENTENCE)
        return None

    async def handle_interruption(self):
        """Clear pending text on interruption."""
        await super().handle_interruption()
        self._pending = ""

    async def reset(self):
        """Clear pending text."""
        await super().reset()
        self._pending = ""
