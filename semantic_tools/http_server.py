"""Small localhost-only JSON service transport for heavy perception models."""

from __future__ import annotations

import ipaddress
import json
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol


MAX_REQUEST_BYTES = 32 * 1024 * 1024


class JsonService(Protocol):
    route: str

    def health(self) -> dict[str, Any]: ...

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]: ...


def serve(service: JsonService, *, host: str, port: int) -> None:
    """Serve one typed endpoint plus ``/health`` without framework dependencies."""

    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.strip().lower() == "localhost"
    if not loopback:
        raise ValueError("semantic model services may bind only to a loopback address")

    class Handler(BaseHTTPRequestHandler):
        server_version = "UniVTACSemanticTool/1"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send(HTTPStatus.NOT_FOUND, {"success": False, "error": "not_found"})
                return
            self._send(HTTPStatus.OK, service.health())

        def do_POST(self) -> None:  # noqa: N802
            if self.path != service.route:
                self._send(HTTPStatus.NOT_FOUND, {"success": False, "error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length < 2 or length > MAX_REQUEST_BYTES:
                self._send(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"success": False, "error": "invalid_request_size"},
                )
                return
            try:
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                response = service.handle(payload)
                if not isinstance(response, dict):
                    raise TypeError("service returned a non-object")
            except (json.JSONDecodeError, ValueError) as exc:
                self._send(
                    HTTPStatus.BAD_REQUEST,
                    {"success": False, "error": "invalid_request", "message": str(exc)},
                )
                return
            except Exception as exc:  # noqa: BLE001 - model boundary.
                print(
                    f"{type(service).__name__} request failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc(file=sys.stderr)
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "success": False,
                        "error": "service_failure",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                return
            self._send(HTTPStatus.OK, response)

        def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, int(port)), Handler)
    server.daemon_threads = True
    server.serve_forever()
