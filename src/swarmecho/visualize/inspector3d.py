"""Standalone browser inspector for completed 3D replay artifacts."""

from __future__ import annotations

import json
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from swarmecho.visualize.replay3d import load_replay


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>SwarmEcho 3D Inspector</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>
:root{color-scheme:dark;--bg:#090e18;--panel:#111a2a;--line:#26344d;--cyan:#22d3ee;--text:#e5eefc;--muted:#91a4c3}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px Inter,system-ui,sans-serif;overflow:hidden}
header{height:58px;display:flex;align-items:center;gap:18px;padding:0 20px;background:#0d1523;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0;letter-spacing:.04em}.tag{color:var(--cyan);font-weight:700}.meta{color:var(--muted)}
#layout{display:grid;grid-template-columns:1fr 310px;height:calc(100vh - 58px)}#scene{min-width:0}.panel{padding:18px;background:var(--panel);border-left:1px solid var(--line);overflow:auto}
.card{padding:14px;margin-bottom:12px;background:#0c1422;border:1px solid var(--line);border-radius:10px}.label{font-size:11px;text-transform:uppercase;color:var(--muted);letter-spacing:.1em;margin-bottom:8px}
.value{font-size:24px;font-weight:750}.row{display:flex;justify-content:space-between;gap:12px;margin:7px 0;color:var(--muted)}.row b{color:var(--text)}
button{border:1px solid #325173;background:#15253a;color:var(--text);padding:8px 12px;border-radius:7px;cursor:pointer}button:hover{border-color:var(--cyan)}
input[type=range]{width:100%;accent-color:var(--cyan)}select{width:100%;background:#15253a;color:var(--text);border:1px solid #325173;padding:8px;border-radius:7px}
.controls{display:flex;gap:8px;margin-bottom:10px}.legend span{display:block;margin:7px 0}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
</style></head><body>
<header><h1><span class="tag">SwarmEcho</span> 3D Inspector</h1><select id="replaySelect" style="width:min(520px,40vw)"></select><span id="replayName" class="meta"></span><span class="meta">Drag to orbit · wheel to zoom · right-drag to pan</span></header>
<div id="layout"><div id="scene"></div><aside class="panel">
<div class="card"><div class="label">Playback</div><div class="controls"><button id="play">▶ Play</button><button id="step">Step</button></div><input id="timeline" type="range" min="0" value="0"><div class="row"><span>Frame</span><b id="frame">0</b></div><div class="row"><span>Time</span><b id="time">0.0 s</b></div><select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option selected value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></div>
<div class="card"><div class="label">Episode</div><div id="status" class="value">Exploring</div><div class="row"><span>Reward</span><b id="reward">—</b></div><div class="row"><span>Coverage</span><b id="coverage">0%</b></div><div class="row"><span>Known agents</span><b id="known">0</b></div><div class="row"><span>Base connected</span><b id="baseConn">0</b></div></div>
<div class="card"><div class="label">Layers</div><label><input id="showCoverage" type="checkbox" checked> Coverage voxels</label><br><label><input id="showLinks" type="checkbox" checked> Communication links</label><br><label><input id="showShell" type="checkbox" checked> Transparent shell</label></div>
<div class="card legend"><div class="label">Legend</div><span><i class="dot" style="background:#22d3ee"></i>Drone</span><span><i class="dot" style="background:#60a5fa"></i>Base</span><span><i class="dot" style="background:#fb7185"></i>Target</span><span><i class="dot" style="background:#a3e635"></i>Target-knowing drone</span></div>
</aside></div><script>
let D,frame=0,playing=false,timer=null;const $=id=>document.getElementById(id);
function loadReplay(id){fetch('/api/replay?id='+encodeURIComponent(id)).then(r=>{if(!r.ok)throw Error('Replay failed to load');return r.json()}).then(d=>{D=d;frame=0;$('timeline').max=d.manifest.frames-1;$('replayName').textContent=d.manifest.map_name+' · '+d.manifest.frames+' frames';draw(0)})}
fetch('/api/replays').then(r=>r.json()).then(items=>{let s=$('replaySelect');s.innerHTML=items.map(x=>'<option value="'+x.id+'"'+(x.selected?' selected':'')+'>'+x.label+'</option>').join('');if(items.length)loadReplay(s.value);else $('replayName').textContent='No completed replays found';s.onchange=()=>loadReplay(s.value)});
function lineTrace(points,color,width=3){return {type:'scatter3d',mode:'lines',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),line:{color,width},hoverinfo:'skip'}}
function boxTrace(s){let [x,y,z]=s,p=[[0,0,0],[x,0,0],[x,y,0],[0,y,0],[0,0,0],[0,0,z],[x,0,z],[x,y,z],[0,y,z],[0,0,z],[null,null,null],[x,0,0],[x,0,z],[null,null,null],[x,y,0],[x,y,z],[null,null,null],[0,y,0],[0,y,z]];return lineTrace(p,'rgba(120,155,205,.48)',2)}
function draw(f){frame=+f;let p=D.position[f],active=D.active[f],known=D.target_known[f],tr=[];if($('showShell').checked)tr.push(boxTrace(D.manifest.world_size_m||[20,20,20]));
let colors=p.map((_,i)=>known[i]?'#a3e635':'#22d3ee');tr.push({type:'scatter3d',mode:'markers+text',x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),text:p.map((_,i)=>'D'+(i+1)),textposition:'top center',marker:{size:7,color:colors,line:{color:'#e5eefc',width:1}},hovertemplate:'%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>'});
let b=D.base_position[f],t=D.target_position[f];tr.push({type:'scatter3d',mode:'markers+text',x:[b[0],t[0]],y:[b[1],t[1]],z:[b[2],t[2]],text:['BASE','TARGET'],textposition:'top center',marker:{size:[9,9],color:['#60a5fa','#fb7185'],symbol:['diamond','circle']}});
if($('showLinks').checked){let r=D.manifest.comm_radius_m||5;for(let i=0;i<p.length;i++)for(let j=i+1;j<p.length;j++)if(active[i]&&active[j]&&Math.hypot(...p[i].map((v,k)=>v-p[j][k]))<=r)tr.push(lineTrace([p[i],p[j]],'rgba(34,211,238,.35)',3))}
if($('showCoverage').checked){let c=D.coverage[f],cs=D.manifest.cell_size_m||5,x=[],y=[],z=[];for(let i=0;i<c.length;i++)for(let j=0;j<c[i].length;j++)for(let k=0;k<c[i][j].length;k++)if(c[i][j][k]){x.push((i+.5)*cs);y.push((j+.5)*cs);z.push((k+.5)*cs)}tr.push({type:'scatter3d',mode:'markers',x,y,z,marker:{size:5,color:'rgba(56,189,248,.18)',symbol:'square'},hoverinfo:'skip'})}
Plotly.react('scene',tr,{margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'#090e18',scene:{bgcolor:'#090e18',aspectmode:'data',xaxis:{title:'X',gridcolor:'#22304a'},yaxis:{title:'Y',gridcolor:'#22304a'},zaxis:{title:'Z',gridcolor:'#22304a'},camera:{projection:{type:'perspective'}}},showlegend:false},{responsive:true,displaylogo:false});
$('timeline').value=f;$('frame').textContent=f+'/'+(D.manifest.frames-1);$('time').textContent=(f*D.manifest.dt).toFixed(1)+' s';let cov=D.coverage[f].flat(2).filter(Boolean).length,total=D.coverage[f].flat(2).length;$('coverage').textContent=(100*cov/total).toFixed(1)+'%';$('known').textContent=known.filter(Boolean).length;$('baseConn').textContent=D.connected_to_base[f].filter(Boolean).length;$('status').textContent=D.success[f]?'SUCCESS':D.fully_connected[f]?'CHAIN HELD':known.some(Boolean)?'TARGET KNOWN':'EXPLORING';$('status').style.color=D.success[f]?'#a3e635':'#e5eefc';$('reward').textContent=D.reward_terms?D.reward_terms[f].flat().reduce((a,b)=>a+b,0).toFixed(2):'—'}
$('timeline').oninput=e=>draw(e.target.value);['showCoverage','showLinks','showShell'].forEach(id=>$(id).onchange=()=>draw(frame));$('step').onclick=()=>draw(Math.min(frame+1,D.manifest.frames-1));$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'❚❚ Pause':'▶ Play';if(playing)tick();else clearTimeout(timer)};function tick(){if(!playing)return;draw(frame>=D.manifest.frames-1?0:frame+1);timer=setTimeout(tick,1000*D.manifest.dt/+$('speed').value)}
</script></body></html>"""


def replay_payload(manifest_path: str | Path) -> dict:
    manifest, arrays = load_replay(manifest_path)
    return {"manifest": manifest, **{name: value.tolist() for name, value in arrays.items()}}


def inspector_html() -> str:
    """Embed the installed Plotly runtime so inspection also works offline."""
    from plotly.offline import get_plotlyjs

    external = '<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>'
    return HTML.replace(external, f"<script>{get_plotlyjs()}</script>")


def discover_replays(root: str | Path = "outputs") -> list[Path]:
    """Find completed replay manifests, newest first, below an output root."""
    root = Path(root)
    manifests: list[Path] = []
    if not root.exists():
        return manifests
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("format") == "swarmecho-replay/v1" and (path.parent / data.get("data_file", "")).exists():
            manifests.append(path.resolve())
    return sorted(manifests, key=lambda path: path.stat().st_mtime, reverse=True)


def make_handler(manifests: list[Path], initial: Path | None = None):
    resolved = [path.resolve() for path in manifests]
    initial = initial.resolve() if initial is not None else None
    encoded_html = inspector_html().encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/replays":
                items = [
                    {
                        "id": str(index),
                        "label": str(path),
                        "selected": path == initial or (initial is None and index == 0),
                    }
                    for index, path in enumerate(resolved)
                ]
                body, content_type = json.dumps(items).encode(), "application/json"
            elif parsed.path == "/api/replay":
                query = urllib.parse.parse_qs(parsed.query)
                try:
                    path = resolved[int(query.get("id", ["0"])[0])]
                    body = json.dumps(replay_payload(path)).encode()
                except (IndexError, ValueError):
                    self.send_error(404, "Replay not found")
                    return
                content_type = "application/json"
            elif parsed.path in {"/", "/index.html"}:
                body, content_type = encoded_html, "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            return

    return Handler


def main() -> None:
    manifest: Path | None = None
    root = Path("outputs")
    host = "127.0.0.1"
    port = 8765
    open_browser = True
    for argument in sys.argv[1:]:
        if "=" not in argument:
            raise ValueError("Use replay=<path>, root=<path>, host=<host>, port=<port>, or open=false.")
        key, value = argument.split("=", 1)
        if key in {"replay", "manifest"}:
            manifest = Path(value)
        elif key == "root":
            root = Path(value)
        elif key == "host":
            host = value
        elif key == "port":
            port = int(value)
        elif key == "open":
            open_browser = value.lower() not in {"false", "0", "no"}
        else:
            raise ValueError(f"Unknown inspector option {key!r}.")
    manifests = discover_replays(root)
    if manifest is not None:
        manifest = manifest.resolve()
        if manifest not in manifests:
            manifests.insert(0, manifest)
    if not manifests:
        raise FileNotFoundError(f"No completed 3D replays found below {root}.")
    server = ThreadingHTTPServer(
        (host, port), make_handler(manifests, manifest)
    )
    url = f"http://{host}:{port}"
    print(f"SwarmEcho 3D Inspector: {url}")
    print("Press Ctrl+C to stop. Training is not coupled to this process.")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
