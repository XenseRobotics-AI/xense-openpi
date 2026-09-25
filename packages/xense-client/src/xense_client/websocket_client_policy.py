import math
import time
from typing import override

import websockets.sync.client

from xense_client import base_policy as _base_policy
from xense_client import msgpack_numpy
from xense_client.logger import get_logger

logger = get_logger("WebsocketClientPolicy")


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
        request_timeout_s: float | None = None,
    ) -> None:
        if request_timeout_s is not None and (not math.isfinite(request_timeout_s) or request_timeout_s <= 0):
            raise ValueError("request_timeout_s must be finite and positive")
        self._request_timeout_s = request_timeout_s
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict]:
        logger.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                try:
                    metadata = msgpack_numpy.unpackb(conn.recv(timeout=self._request_timeout_s))
                except BaseException:
                    conn.close()
                    raise
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: dict, **kwargs) -> dict:
        # Pack kwargs into obs if present
        if kwargs:
            obs["__rtc_kwargs__"] = kwargs

        data = self._packer.pack(obs)
        self._ws.send(data)
        try:
            response = self._ws.recv(timeout=self._request_timeout_s)
        except TimeoutError:
            # A late reply must never become the next observation's action.
            self._ws.close()
            raise
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass

    def disconnect(self) -> None:
        """Close the transport."""
        self._ws.close()
