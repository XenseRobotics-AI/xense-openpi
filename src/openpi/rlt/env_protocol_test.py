import socket
import threading

import numpy as np
import pytest
import websockets.sync.client
from xense_client import msgpack_numpy

from openpi.rlt import env_protocol


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _robot(port, hello, replies, received):
    packer = msgpack_numpy.Packer()
    with websockets.sync.client.connect(f"ws://127.0.0.1:{port}", max_size=None) as conn:
        conn.send(packer.pack(hello))
        for reply in replies:
            received.append(msgpack_numpy.unpackb(conn.recv()))
            if received[-1].get("op") == "status":
                received.append(msgpack_numpy.unpackb(conn.recv()))
            conn.send(packer.pack(reply))


def test_request_reply_and_error_handling():
    port = _free_port()
    env = env_protocol.RemoteEnv("127.0.0.1", port, state_dim=20, action_dim=20)
    received = []
    hello = {"protocol": env_protocol.PROTOCOL, "state_dim": 20, "action_dim": 20}
    replies = [{"obs": {"state": np.arange(20.0)}, "recording": False}, {"error": "estop"}]
    robot = threading.Thread(target=_robot, args=(port, hello, replies, received))
    robot.start()

    reply = env.request({"op": "reset"})
    np.testing.assert_array_equal(reply["obs"]["state"], np.arange(20.0))
    env.status("hello operator")
    with pytest.raises(env_protocol.EnvConnectionLostError, match="estop"):
        env.request({"op": "chunk", "actions": np.zeros((2, 20))})
    robot.join(timeout=5)
    assert [m["op"] for m in received] == ["reset", "status", "chunk"]
    env.close()


def test_mismatched_robot_is_refused():
    port = _free_port()
    env = env_protocol.RemoteEnv("127.0.0.1", port, state_dim=20, action_dim=20)
    packer = msgpack_numpy.Packer()
    with websockets.sync.client.connect(f"ws://127.0.0.1:{port}") as conn:
        conn.send(packer.pack({"protocol": env_protocol.PROTOCOL, "state_dim": 14, "action_dim": 20}))
        assert "expected" in msgpack_numpy.unpackb(conn.recv())["error"]
    assert env._connections.empty()
    env.close()
