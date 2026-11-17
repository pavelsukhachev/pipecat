#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for WebsocketService reconnection and lifecycle behavior."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from websockets.asyncio.client import connect as websocket_connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame
from pipecat.services.websocket_service import WebsocketService

# `_no_sleep` patches asyncio.sleep for the whole process; the peer tests need a real one.
_real_sleep = asyncio.sleep


class ConcreteWebsocketService(WebsocketService):
    """Minimal concrete implementation for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._receive_messages_impl: AsyncMock | None = None

    async def _connect_websocket(self):
        pass

    async def _disconnect_websocket(self):
        pass

    async def _receive_messages(self):
        if self._receive_messages_impl:
            await self._receive_messages_impl()


@pytest.fixture
def service():
    return ConcreteWebsocketService()


@pytest.fixture
def report_error():
    return AsyncMock()


@pytest.fixture(autouse=True)
def _no_sleep():
    """Patch asyncio.sleep globally to avoid real backoff waits."""
    with patch("pipecat.services.websocket_service.asyncio.sleep", new_callable=AsyncMock):
        yield


# ---------------------------------------------------------------------------
# Receive loop — how each exception type is handled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connection_closed_ok_exits_cleanly(service, report_error):
    """ConnectionClosedOK exits the loop with no error and no reconnection."""
    service._receive_messages_impl = AsyncMock(
        side_effect=ConnectionClosedOK(Close(1000, "Normal closure"), None)
    )
    service._try_reconnect = AsyncMock()

    await service._receive_task_handler(report_error)

    report_error.assert_not_called()
    service._try_reconnect.assert_not_called()


@pytest.mark.asyncio
async def test_connection_closed_error_triggers_reconnect(service, report_error):
    """ConnectionClosedError triggers reconnection; loop continues after success."""
    call_count = 0

    async def fail_then_exit():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionClosedError(Close(1006, "Abnormal closure"), None)
        service._disconnecting = True

    service._receive_messages_impl = AsyncMock(side_effect=fail_then_exit)
    service._try_reconnect = AsyncMock(return_value=True)

    await service._receive_task_handler(report_error)

    assert call_count == 2
    service._try_reconnect.assert_called_once()


@pytest.mark.asyncio
async def test_graceful_server_close_triggers_reconnect(service, report_error):
    """Normal return from _receive_messages (server close frame) triggers reconnection."""
    call_count = 0

    async def return_then_exit():
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            service._disconnecting = True

    service._receive_messages_impl = AsyncMock(side_effect=return_then_exit)
    service._try_reconnect = AsyncMock(return_value=True)

    await service._receive_task_handler(report_error)

    assert call_count == 2
    service._try_reconnect.assert_called_once()


@pytest.mark.asyncio
async def test_general_exception_triggers_reconnect(service, report_error):
    """A general exception in _receive_messages triggers reconnection."""
    call_count = 0

    async def fail_then_exit():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("something broke")
        service._disconnecting = True

    service._receive_messages_impl = AsyncMock(side_effect=fail_then_exit)
    service._try_reconnect = AsyncMock(return_value=True)

    await service._receive_task_handler(report_error)

    assert call_count == 2
    service._try_reconnect.assert_called_once()


# ---------------------------------------------------------------------------
# Exponential backoff — server unreachable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_succeeds_on_later_attempt(service, report_error):
    """_try_reconnect retries and succeeds on a later attempt."""
    service._reconnect_websocket = AsyncMock(
        side_effect=[ConnectionError("fail"), ConnectionError("fail"), True]
    )

    result = await service._try_reconnect(report_error=report_error)

    assert result is True
    assert service._reconnect_websocket.call_count == 3


@pytest.mark.asyncio
async def test_reconnect_exhausted_emits_non_fatal_error(service, report_error):
    """Exhausting all retries returns False and emits a non-fatal ErrorFrame."""
    service._reconnect_websocket = AsyncMock(side_effect=ConnectionError("Connection refused"))

    result = await service._try_reconnect(report_error=report_error)

    assert result is False
    assert service._reconnect_websocket.call_count == 3
    final_error = report_error.call_args_list[-1][0][0]
    assert isinstance(final_error, ErrorFrame)
    assert final_error.fatal is False
    assert "Connection refused" in final_error.error


@pytest.mark.asyncio
async def test_reconnect_exhausted_when_connect_does_not_raise(service, report_error):
    """A non-raising failed connect is treated as a failed reconnect attempt."""
    result = await service._try_reconnect(report_error=report_error)

    assert result is False
    assert report_error.call_count == 4
    final_error = report_error.call_args_list[-1][0][0]
    assert isinstance(final_error, ErrorFrame)
    assert final_error.fatal is False
    assert "websocket reconnection failed verification" in final_error.error


# ---------------------------------------------------------------------------
# Quick failure detection — accept then immediately close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quick_failures_emit_error(service, report_error):
    """Connections failing immediately after establishment emit error after 3 cycles."""
    call_count = 0

    async def fail_immediately():
        nonlocal call_count
        call_count += 1
        raise ConnectionClosedError(Close(1008, "Invalid API key"), None)

    service._receive_messages_impl = AsyncMock(side_effect=fail_immediately)
    service._try_reconnect = AsyncMock(return_value=True)

    await service._receive_task_handler(report_error)

    assert call_count == service._MAX_CONSECUTIVE_QUICK_FAILURES
    report_error.assert_called_once()
    error_frame = report_error.call_args[0][0]
    assert isinstance(error_frame, ErrorFrame)
    assert error_frame.fatal is False
    assert "failed 3 times immediately after connecting" in error_frame.error


@pytest.mark.asyncio
async def test_stable_connection_resets_quick_failure_counter(service, report_error):
    """A stable connection resets the quick failure counter; needs 3 new failures to trigger."""
    call_count = 0

    async def always_fail():
        nonlocal call_count
        call_count += 1
        raise ConnectionClosedError(Close(1006, "Abnormal closure"), None)

    service._receive_messages_impl = AsyncMock(side_effect=always_fail)
    service._try_reconnect = AsyncMock(return_value=True)

    base_time = 1000.0
    time_values = iter(
        [
            # Call 1: set _last_connect_time, check in _maybe_try_reconnect (quick) -> count=1
            base_time,
            base_time,
            # Call 2: quick -> count=2
            base_time + 1.0,
            base_time + 1.0,
            # Call 3: stable (10s elapsed) -> count=0
            base_time + 2.0,
            base_time + 12.0,
            # Call 4: quick -> count=1
            base_time + 13.0,
            base_time + 13.0,
            # Call 5: quick -> count=2
            base_time + 14.0,
            base_time + 14.0,
            # Call 6: quick -> count=3 -> error emitted, loop stops
            base_time + 15.0,
            base_time + 15.0,
        ]
    )

    with patch("pipecat.services.websocket_service.time") as mock_time:
        mock_time.monotonic = lambda: next(time_values)
        await service._receive_task_handler(report_error)

    assert call_count == 6
    report_error.assert_called_once()
    error_frame = report_error.call_args[0][0]
    assert error_frame.fatal is False


# ---------------------------------------------------------------------------
# Lifecycle and guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_prevents_reconnection(service, report_error):
    """After _disconnect(), errors exit the loop without reconnecting or emitting errors."""
    await service._disconnect()

    service._receive_messages_impl = AsyncMock(
        side_effect=ConnectionClosedError(Close(1006, "Abnormal closure"), None)
    )
    service._try_reconnect = AsyncMock()

    await service._receive_task_handler(report_error)

    report_error.assert_not_called()
    service._try_reconnect.assert_not_called()


@pytest.mark.asyncio
async def test_connect_resets_state(service):
    """_connect() resets _disconnecting and _quick_failure_count."""
    service._disconnecting = True
    service._quick_failure_count = 5

    await service._connect()

    assert service._disconnecting is False
    assert service._quick_failure_count == 0


# ---------------------------------------------------------------------------
# Dropped connections — replacing or clearing an open connection aborts it
# (inshurik LUCY-599; upstream pipecat-ai/pipecat#6490 via dograh-hq/pipecat 77e137d0)
# ---------------------------------------------------------------------------


class _Peer:
    """A local WebSocket server that records which connections are still open."""

    def __init__(self):
        self.open = set()
        self.close_codes: list[int | None] = []
        self.url = ""

    async def handle(self, websocket):
        self.open.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.open.discard(websocket)
            self.close_codes.append(websocket.close_code)

    async def wait_until_all_closed(self, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        while self.open and time.monotonic() < deadline:
            await _real_sleep(0.01)
        return not self.open


@pytest_asyncio.fixture
async def peer():
    peer = _Peer()
    async with serve(peer.handle, "127.0.0.1", 0) as server:
        peer.url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        yield peer


class _CancellableDisconnectService(ConcreteWebsocketService):
    """Disconnects the way most services do, with an await before the close."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.disconnecting = asyncio.Event()

    async def _disconnect_websocket(self):
        try:
            if self._websocket:
                # Stands in for a final message to the provider.
                self.disconnecting.set()
                await asyncio.Event().wait()
                await self._websocket.close()
        finally:
            self._websocket = None


@pytest.mark.asyncio
async def test_clearing_an_open_connection_aborts_it(service, peer):
    """The service never leaves an open connection behind when it clears the reference."""
    service._websocket = await websocket_connect(peer.url)
    websocket = service._websocket

    service._websocket = None

    assert await peer.wait_until_all_closed()
    assert websocket.state is State.CLOSED


@pytest.mark.asyncio
async def test_clearing_a_closed_connection_leaves_its_close_alone(service, peer):
    """A connection closed before it is cleared keeps its clean close."""
    service._websocket = await websocket_connect(peer.url)
    websocket = service._websocket
    await websocket.close()

    with patch.object(websocket.transport, "abort") as abort:
        service._websocket = None

    abort.assert_not_called()
    assert await peer.wait_until_all_closed()
    assert peer.close_codes == [1000]


@pytest.mark.asyncio
async def test_replacing_a_connection_aborts_only_the_old_one(service, peer):
    """The replacement stays open while the connection it replaces is aborted."""
    old = await websocket_connect(peer.url)
    new = await websocket_connect(peer.url)
    service._websocket = old

    service._websocket = new

    await _real_sleep(0.05)
    assert old.state is State.CLOSED
    assert new.state is State.OPEN
    assert len(peer.open) == 1
    await new.close()


@pytest.mark.asyncio
async def test_a_cancelled_disconnect_still_ends_the_connection(peer):
    """A disconnect cancelled before its close ends the connection as it clears it."""
    service = _CancellableDisconnectService()
    service._websocket = await websocket_connect(peer.url)
    disconnect = asyncio.create_task(service._disconnect_websocket())
    await service.disconnecting.wait()

    disconnect.cancel()
    with pytest.raises(asyncio.CancelledError):
        await disconnect

    assert service._websocket is None
    assert await peer.wait_until_all_closed()


@pytest.mark.asyncio
async def test_clearing_something_other_than_a_connection_does_nothing(service):
    """Only real connections are aborted, so test doubles pass through untouched."""
    fake = MagicMock()
    service._websocket = fake

    service._websocket = None

    fake.transport.abort.assert_not_called()


def test_the_connection_reads_as_none_before_init():
    """Subclasses that initialize their other bases first can read it early."""
    service = ConcreteWebsocketService.__new__(ConcreteWebsocketService)

    assert service._websocket is None
