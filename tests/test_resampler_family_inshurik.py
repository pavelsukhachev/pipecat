#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inshurik fork: the resampler factories build the family the application chose.

pipecat 1.16 moved the factories from SoX to sinc. The fork keeps that default
but lets an application keep SoX, process-wide or per task, so a call's audio
path changes only when someone decides it should (LUCY-603 hop 2).
"""

import asyncio
import warnings

import numpy as np
import pytest

from pipecat.audio import utils as audio_utils
from pipecat.audio.resamplers.sinc_resampler import SincAudioResampler
from pipecat.audio.resamplers.sinc_stream_resampler import SincStreamAudioResampler
from pipecat.audio.resamplers.soxr_resampler import SOXRAudioResampler
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler


@pytest.fixture(autouse=True)
def _restore_default():
    before = audio_utils._default_resampler_family
    yield
    audio_utils._default_resampler_family = before


def test_upstream_default_is_sinc():
    assert audio_utils.resampler_family() == "sinc"
    assert isinstance(audio_utils.create_stream_resampler(), SincStreamAudioResampler)
    assert isinstance(audio_utils.create_file_resampler(), SincAudioResampler)


def test_process_default_soxr_builds_soxr_without_a_deprecation_warning():
    audio_utils.set_default_resampler_family("soxr")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        stream = audio_utils.create_stream_resampler()
        file = audio_utils.create_file_resampler()
    assert isinstance(stream, SOXRStreamAudioResampler)
    assert isinstance(file, SOXRAudioResampler)


def test_soxr_stream_keeps_the_pre_116_defaults():
    # 1.12's create_stream_resampler() built SOXRStreamAudioResampler() as is:
    # VHQ, and the history cleared after 0.2 s without audio.
    audio_utils.set_default_resampler_family("soxr")
    stream = audio_utils.create_stream_resampler()
    assert stream._quality == "VHQ"
    assert stream._clear_after_secs == 0.2


def test_kwargs_reach_the_soxr_resampler():
    audio_utils.set_default_resampler_family("soxr")
    stream = audio_utils.create_stream_resampler(quality="QQ", clear_after_secs=None)
    assert stream._quality == "QQ"
    assert stream._clear_after_secs is None


def test_task_choice_overrides_the_process_default_and_resets():
    audio_utils.set_default_resampler_family("soxr")
    token = audio_utils.use_resampler_family("sinc")
    try:
        assert isinstance(audio_utils.create_stream_resampler(), SincStreamAudioResampler)
    finally:
        audio_utils.reset_resampler_family(token)
    assert isinstance(audio_utils.create_stream_resampler(), SOXRStreamAudioResampler)


def test_none_falls_back_to_the_process_default():
    audio_utils.set_default_resampler_family("soxr")
    token = audio_utils.use_resampler_family(None)
    try:
        assert audio_utils.resampler_family() == "soxr"
    finally:
        audio_utils.reset_resampler_family(token)


@pytest.mark.asyncio
async def test_task_choice_reaches_tasks_started_after_it_only():
    audio_utils.set_default_resampler_family("soxr")

    async def family_in_child():
        return audio_utils.resampler_family()

    token = audio_utils.use_resampler_family("sinc")
    try:
        assert await asyncio.create_task(family_in_child()) == "sinc"
    finally:
        audio_utils.reset_resampler_family(token)
    assert await asyncio.create_task(family_in_child()) == "soxr"


@pytest.mark.parametrize("family", ["SOXR", "speex", ""])
def test_unknown_family_is_refused(family):
    with pytest.raises(ValueError):
        audio_utils.set_default_resampler_family(family)
    with pytest.raises(ValueError):
        audio_utils.use_resampler_family(family)


@pytest.mark.asyncio
async def test_soxr_stream_resamples_8k_to_16k():
    audio_utils.set_default_resampler_family("soxr")
    stream = audio_utils.create_stream_resampler()
    pcm = (np.sin(np.arange(1600) * 2 * np.pi * 440 / 8000) * 8000).astype(np.int16).tobytes()
    out = await stream.resample(pcm, 8000, 16000)
    assert 0 < len(out) <= 2 * len(pcm)
