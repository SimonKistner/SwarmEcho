"""Independent incremental browser dashboard for adaptive-spawn journals."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse
import webbrowser


def discover_runs(root: Path):
    candidates = [root / "metadata.json", root / "adaptive_spawn" / "metadata.json"]
    candidates.extend(root.glob("*/adaptive_spawn/metadata.json"))
    candidates.extend(root.glob("*/*/adaptive_spawn/metadata.json"))
    candidates.extend(root.glob("adaptive_spawn_maps/*/adaptive_spawn/metadata.json"))
    candidates.extend(root.glob("*/adaptive_spawn_maps/*/adaptive_spawn/metadata.json"))
    candidates.extend(root.glob("*/*/adaptive_spawn_maps/*/adaptive_spawn/metadata.json"))
    runs = {}
    for path in candidates:
        if path.is_file():
            try:
                directory = path.parent.resolve()
                key = directory.relative_to(root).as_posix()
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if metadata.get("format") == "swarmecho-adaptive-spawn/v1":
                    runs[key] = (directory, metadata)
            except (OSError, ValueError, KeyError):
                continue
    return runs


def read_increment(directory: Path, offset: int):
    """Only complete newline-terminated records are visible, including after a crash."""
    events = []
    skipped = 0
    path = directory / "history.jsonl"
    if not path.is_file():
        return {"events": [], "offset": 0, "reset": offset != 0, "skipped": 0}
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        reset = offset > size
        if reset:
            offset = 0
        stream.seek(offset)
        limit = offset + 4 * 1024 * 1024
        while stream.tell() < limit:
            line = stream.readline(1024 * 1024)
            if not line or not line.endswith(b"\n"):
                break
            offset = stream.tell()
            try:
                event = json.loads(line)
                if (event.get("kind") not in ("session", "update")
                        or type(event.get("steps")) is not int
                        or type(event.get("update")) is not int):
                    raise ValueError("Unknown journal event")
                if event["kind"] == "update":
                    columns = [event.get(key) for key in ("probabilities", "success_rates", "window_counts")]
                    if (any(not isinstance(column, list) or not column for column in columns)
                            or len({len(column) for column in columns}) != 1
                            or any(type(value) not in (int, float) or not math.isfinite(value)
                                   for column in columns for value in column)
                            or any(not 0. <= value <= 1. for column in columns[:2] for value in column)):
                        raise ValueError("Invalid history values")
                events.append(event)
            except (ValueError, UnicodeDecodeError, AttributeError):
                skipped += 1
    return {"events": events, "offset": offset, "reset": reset, "skipped": skipped}


def handler_for(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, payload, content_type="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            request = urlparse(self.path)
            try:
                if request.path == "/":
                    page = Path(__file__).with_name("adaptive_spawn_dashboard.html").read_bytes()
                    self.respond(page, "text/html; charset=utf-8")
                    return
                runs = discover_runs(root)
                if request.path == "/api/runs":
                    self.respond([{"id": key, "name": metadata["run_name"]}
                                  for key, (_, metadata) in sorted(runs.items())])
                    return
                query = parse_qs(request.query)
                key = query.get("run", [""])[0]
                if key not in runs:
                    self.send_error(404, "Run not found; refresh the run list")
                    return
                directory, metadata = runs[key]
                if request.path == "/api/history":
                    offset = int(query.get("offset", ["0"])[0])
                    if offset < 0:
                        raise ValueError("Negative journal offset")
                    self.respond({"metadata": metadata, **read_increment(directory, offset)})
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (OSError, ValueError, KeyError) as exc:
                self.send_error(400, str(exc))
    return Handler


def main():
    options = {"root": "outputs", "port": "8767", "open": "true"}
    for argument in sys.argv[1:]:
        name, separator, value = argument.partition("=")
        if not separator or name not in options:
            raise SystemExit("Usage: swarmecho-spawn-dashboard root=outputs port=8767 open=false")
        options[name] = value
    if options["open"].lower() not in {"true", "false"}:
        raise SystemExit("open must be true or false")
    port = int(options["port"])
    if not 1 <= port <= 65535:
        raise SystemExit("port must be between 1 and 65535")
    root = Path(options["root"]).resolve()
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(root))
    address = f"http://127.0.0.1:{port}"
    print(f"Adaptive spawn dashboard: {address} (root: {root})", flush=True)
    if options["open"].lower() == "true":
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
