"""Local web server for the SwarmEcho maze builder.

Run from the repository root:

    uv run swarmecho-maze-builder

Then open http://127.0.0.1:8765/ in a browser.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from swarmecho.curriculum_config.maps.scripts.maze_builder.maze_builder_core import (
    create_map_and_level,
    crop_canvas_payload,
    normalize_machine_maze,
)
from swarmecho.curriculum_config.maps.scripts.maze_builder.building_builder_core import (
    add_outer_walls,
    add_roof,
    add_layer,
    available_maps,
    delete_layer,
    document_yaml,
    load_document,
    new_document,
    expand_document,
    save_document,
    validate_document,
)
from swarmecho.core.config import MAP_DIR
from swarmecho.curriculum_config.maps.scripts.maze_builder.maze_builder_core import validate_map_name

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
        if path == "/api/buildings":
            self._send_json(HTTPStatus.OK, {"ok": True, "maps": list(available_maps())})
            return
        if path == "/api/buildings/new":
            self._send_json(HTTPStatus.OK, {"ok": True, "document": new_document()})
            return
        if path.startswith("/api/buildings/"):
            try:
                name = validate_map_name(path.removeprefix("/api/buildings/"))
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "document": load_document(MAP_DIR / f"{name}.yaml")},
                )
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return
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
            if path == "/api/buildings/new":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "document": new_document(
                            payload.get("cols", 6),
                            payload.get("rows", 4),
                            payload.get("layers", 1),
                        ),
                    },
                )
                return
            if path == "/api/buildings/validate":
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "report": validate_document(payload)},
                )
                return
            if path == "/api/buildings/outer-walls":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "document": add_outer_walls(payload, payload.get("selected_layer", 0)),
                    },
                )
                return
            if path == "/api/buildings/add-layer":
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "document": add_layer(payload)},
                )
                return
            if path == "/api/buildings/delete-layer":
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "document": delete_layer(payload, payload.get("selected_layer", 0))},
                )
                return
            if path == "/api/buildings/expand":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "document": expand_document(
                            payload, payload.get("direction"), payload.get("delta", 1)
                        ),
                    },
                )
                return
            if path == "/api/buildings/code":
                self._send_json(
                    HTTPStatus.OK, {"ok": True, "code": document_yaml(payload)}
                )
                return
            if path == "/api/buildings/roof":
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "document": add_roof(payload)},
                )
                return
            if path == "/api/buildings/save":
                name = validate_map_name(payload.get("name"))
                target = MAP_DIR / f"{name}.yaml"
                if target.exists() and not payload.get("overwrite", False):
                    raise FileExistsError(f"Map already exists: {target}")
                saved = save_document(payload, target)
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "path": str(saved),
                        "report": validate_document(payload),
                    },
                )
                return
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
