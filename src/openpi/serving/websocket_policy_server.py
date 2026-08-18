import asyncio
import http
import logging
import time
import traceback

import websockets.asyncio.server as _server
import websockets.frames
from xense_client import base_policy as _base_policy
from xense_client import msgpack_numpy

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                message = await websocket.recv()
                logger.info(f"Received message of size {len(message)} bytes from {websocket.remote_address}")
                obs = msgpack_numpy.unpackb(message)
                logger.info(f"Unpacked observation keys: {list(obs.keys())}")

                # Extract RTC kwargs if present
                rtc_kwargs = {}
                if isinstance(obs, dict) and "__rtc_kwargs__" in obs:
                    rtc_kwargs = obs.pop("__rtc_kwargs__")

                start = time.perf_counter()
                action = self._policy.infer(obs, **rtc_kwargs)
                infer_time = time.perf_counter() - start
                logger.info(f"Inference took {infer_time * 1000:.2f} ms")

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                error_msg = traceback.format_exc()
                logger.error(f"Error processing request from {websocket.remote_address}: {error_msg}")
                await websocket.send(error_msg)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
