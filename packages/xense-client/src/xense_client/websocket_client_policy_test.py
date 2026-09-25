"""Bounded requests must close the socket instead of reusing late replies."""

from unittest.mock import Mock

import numpy as np
import pytest

from xense_client import msgpack_numpy
from xense_client.websocket_client_policy import WebsocketClientPolicy


def make_client(monkeypatch, timeout):
    connection = Mock()
    monkeypatch.setattr(WebsocketClientPolicy, "_wait_for_server", lambda _: (connection, {}))
    client = WebsocketClientPolicy(request_timeout_s=timeout)
    return client, connection


@pytest.mark.parametrize("timeout", [None, 0.2])
def test_recv_timeout_and_action_protocol(monkeypatch, timeout):
    client, connection = make_client(monkeypatch, timeout)
    connection.recv.return_value = msgpack_numpy.packb({"actions": np.zeros((50, 58), dtype=np.float32)})
    result = client.infer({"state": np.zeros(58)})
    connection.recv.assert_called_once_with(timeout=timeout)
    assert result["actions"].shape == (50, 58)
    sent = msgpack_numpy.unpackb(connection.send.call_args.args[0])
    assert "__rtc_kwargs__" not in sent
    client.disconnect()
    connection.close.assert_called_once()


def test_inference_timeout_closes_connection(monkeypatch):
    client, connection = make_client(monkeypatch, 0.1)
    connection.recv.side_effect = TimeoutError("expired")
    with pytest.raises(TimeoutError):
        client.infer({"state": np.zeros(58)})
    connection.close.assert_called_once()


def test_metadata_timeout_closes_connection(monkeypatch):
    connection = Mock(recv=Mock(side_effect=TimeoutError("metadata")))
    monkeypatch.setattr("websockets.sync.client.connect", lambda *_, **__: connection)
    with pytest.raises(TimeoutError):
        WebsocketClientPolicy(request_timeout_s=0.1)
    connection.close.assert_called_once()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="request_timeout_s"):
        WebsocketClientPolicy(request_timeout_s=timeout)
