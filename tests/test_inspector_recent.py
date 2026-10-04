import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from swarmecho.visualize import inspector


def test_recent_listing_returns_three_runs_then_full_inventory(tmp_path, monkeypatch):
    replays = []
    for index in range(4):
        path = tmp_path / f"run_{index}" / "artifacts" / "train" / "replays" / "eval.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
        os.utime(path, ns=((index + 1) * 1_000_000_000,) * 2)
        replays.append(path.resolve())
    roadmap = tmp_path / "testresults" / "route.roadmap.json"
    roadmap.parent.mkdir()
    roadmap.write_text("{}", encoding="utf-8")

    def discover_replays(root, *, single_run=False):
        if single_run:
            return [path for path in replays if path.parents[3] == root.resolve()]
        return list(reversed(replays))

    monkeypatch.setattr(inspector, "discover_replays", discover_replays)
    monkeypatch.setattr(inspector, "discover_heatmaps", lambda *_: [])
    monkeypatch.setattr(inspector, "discover_roadmap_tests", lambda *_: [roadmap.resolve()])
    monkeypatch.setattr(inspector, "replay_label", lambda path: f"{path.parents[3].name}_[1M]_[Replay]")
    monkeypatch.setattr(inspector, "roadmap_label", lambda _: "route_[Roadmap]")

    server = ThreadingHTTPServer(("127.0.0.1", 0), inspector.make_handler([], root=tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}/api/replays"
        with urllib.request.urlopen(base + "?recent=3") as response:
            recent = json.load(response)
            assert response.headers["X-Inspector-Has-More"] == "true"
        assert len({item["run_id"] for item in recent}) == 3
        assert all(item["kind"] == "replay" for item in recent)
        assert recent[0]["selected"] is True

        with urllib.request.urlopen(base) as response:
            complete = json.load(response)
            assert response.headers["X-Inspector-Has-More"] == "false"
        assert len(complete) == 5
        assert {item["kind"] for item in complete} == {"replay", "roadmap"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
