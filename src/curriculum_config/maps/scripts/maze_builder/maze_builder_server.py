"""Local web server for the SwarmEcho maze builder.

Run from the repository root:

    uv run python src/curriculum_config/maps/scripts/maze_builder/maze_builder_server.py

Then open http://127.0.0.1:8765/ in a browser.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from maze_builder_core import create_map_and_level, crop_canvas_payload, normalize_machine_maze

_SCRIPT_DIR = Path(__file__).resolve().parent
_INDEX = _SCRIPT_DIR / "index.html"


class MazeBuilderHandler(BaseHTTPRequestHandler):
    server_version = "SwarmEchoMazeBuilder/1.0"

    def log_message(self, fmt: str, *args):  # noqa: D401 - stdlib override
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _send_json(self, status: HTTPStatus, payload: dict):
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):  # noqa: N802 - stdlib API
        path = urlparse(self.path).path
        if path not in {"/", "/index.html"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = _INDEX.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802 - stdlib API
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/export-machine":
                machine = crop_canvas_payload(payload)
                self._send_json(HTTPStatus.OK, {"ok": True, "machine_maze": machine})
                return
            if path == "/api/normalize-machine":
                machine = normalize_machine_maze(payload)
                self._send_json(HTTPStatus.OK, {"ok": True, "machine_maze": machine})
                return
            if path == "/api/create":
                result = create_map_and_level(payload, overwrite=bool(payload.get("overwrite", False)))
                self._send_json(HTTPStatus.OK, {"ok": True, **result})
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:  # return user-facing GUI error
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SwarmEcho maze builder web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MazeBuilderHandler)
    print(f"SwarmEcho maze builder running at http://{args.host}:{args.port}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping maze builder.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
