"""Standalone browser inspector for completed 3D replay artifacts."""

from __future__ import annotations

import json
import re
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from swarmecho.training.artifacts import load_eval_info_csv
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
.loading{display:inline-flex;align-items:center;gap:7px;color:var(--muted);white-space:nowrap}.loading.hidden{display:none}.spinner{width:13px;height:13px;border:2px solid #34506f;border-top-color:var(--cyan);border-radius:50%;animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}
input[type=range]{width:100%;accent-color:var(--cyan)}select,input[type=text]{width:100%;background:#15253a;color:var(--text);border:1px solid #325173;padding:8px;border-radius:7px}.hidden{display:none!important}
.controls{display:flex;gap:8px;margin-bottom:10px}.layer-control{display:flex;justify-content:space-between;gap:8px;align-items:center}.opacity-value{color:var(--muted);font-size:12px}.opacity-control{display:block;margin:4px 0 8px}.legend span{display:block;margin:7px 0}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
.known-list{display:grid;gap:6px}.known-item{width:100%;font:inherit;text-align:left;color:#667893;background:#0c1422;border:1px solid var(--line);border-radius:7px;padding:7px 10px;pointer-events:none}.known-item.known{color:var(--text);border-color:#a3e635;background:rgba(163,230,53,.14);box-shadow:0 0 0 1px rgba(163,230,53,.25)}
</style></head><body>
<header><h1><span class="tag">SwarmEcho</span> 3D Inspector</h1><select id="replaySelect" style="width:min(520px,40vw)"></select><button id="loadReplay" type="button" title="Load selected replay">Load</button><button id="refreshReplays" type="button" title="Refresh available replays">Refresh</button><span id="loading" class="loading hidden" role="status" aria-live="polite"><span class="spinner"></span><span id="loadingText">Loading...</span></span><span id="replayDiscovery" class="meta"></span><span id="replayName" class="meta"></span><span class="meta">Drag to orbit · wheel to zoom · right-drag to pan</span></header>
<div id="layout"><div id="scene"></div><aside class="panel">
<div class="card"><div class="label">Playback</div><div class="controls"><button id="play">▶ Play</button><button id="step">Step</button></div><input id="timeline" type="range" min="0" value="0"><div class="row"><span>Frame</span><b id="frame">0</b></div><div class="row"><span>Time</span><b id="time">0.0 s</b></div><select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option selected value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></div>
<div class="card"><div class="label">Episode</div><div id="status" class="value">Exploring</div><div class="row"><span>Reward</span><b id="reward">—</b></div><div class="row"><span>Coverage</span><b id="coverage">0%</b></div><div class="row"><span>Known agents</span><b id="known">0</b></div><div class="row"><span>Base connected</span><b id="baseConn">0</b></div><div class="label" style="margin-top:16px">Target known</div><div id="knownList" class="known-list"></div></div>
<div class="card"><div class="label">Layers</div><label class="layer-control"><span><input id="showCoverage" type="checkbox"> Coverage voxels</span><span id="coverageOpacityValue" class="opacity-value">2%</span></label><input id="coverageOpacity" class="opacity-control" type="range" min="0" max="1" step="0.001" value="0.02" aria-label="Coverage voxel opacity"><label class="layer-control"><span><input id="showVisualRange" type="checkbox" checked> Visual range</span><span id="visualOpacityValue" class="opacity-value">4.5%</span></label><input id="visualOpacity" class="opacity-control" type="range" min="0" max="1" step="0.001" value="0.045" aria-label="Visual range opacity"><label class="layer-control"><span><input id="showCommRange" type="checkbox" checked> Communication range</span><span id="commOpacityValue" class="opacity-value">2.5%</span></label><input id="commOpacity" class="opacity-control" type="range" min="0" max="1" step="0.001" value="0.025" aria-label="Communication range opacity"><label><input id="showLinks" type="checkbox" checked> Communication links</label><br><label><input id="showShell" type="checkbox" checked> Transparent shell</label></div>
<div id="heatmapControls" class="card hidden"><div class="label">Heatmap rendering</div><label><input id="heatmapSpheres" type="checkbox" checked> Render spheres</label><label class="layer-control" style="margin-top:12px"><span>Sphere opacity</span><span id="heatmapOpacityValue" class="opacity-value">55%</span></label><input id="heatmapOpacity" class="opacity-control" type="range" min="0" max="1" step="0.01" value="0.55"><label class="layer-control"><span>Sphere radius</span><span id="heatmapRadiusValue" class="opacity-value">0.35 m</span></label><input id="heatmapRadius" class="opacity-control" type="range" min="0.05" max="2" step="0.05" value="0.35"><div class="label" style="margin-top:16px">Selected target</div><div id="selectedPoint" class="meta">Click a point to create a replay command.</div><input id="heatmapCommand" type="text" readonly style="margin-top:8px" value=""><button id="copyHeatmapCommand" type="button" style="margin-top:8px">Copy command</button></div>
<div id="replayLegend" class="card legend"><div class="label">Legend</div><span><i class="dot" style="background:#22d3ee"></i>Drone</span><span><i class="dot" style="background:#60a5fa"></i>Base</span><span><i class="dot" style="background:#fb7185"></i>Target</span><span><i class="dot" style="background:#fb7185"></i>Target-known drone / base</span><span><i class="dot" style="background:#a3e635"></i>Known area highlight</span></div>
<div id="heatmapLegend" class="card legend hidden"><div class="label">Evaluation result</div><span><i class="dot" style="background:#22c55e"></i>Chain success</span><span><i class="dot" style="background:#3b82f6"></i>Found and delivered</span><span><i class="dot" style="background:#fbbf24"></i>Visually found</span><span><i class="dot" style="background:#ef4444"></i>Not found</span></div>
</aside></div><script>
let D,H,mode='replay',frame=0,playing=false,timer=null,camera=null;const $=id=>document.getElementById(id);const DEFAULT_CAMERA={projection:{type:'perspective'}};
function setLoading(active,message){$('loading').classList.toggle('hidden',!active);if(active)$('loadingText').textContent=message}
function setMode(nextMode){mode=nextMode;playing=false;clearTimeout(timer);$('play').textContent='▶ Play';let replayCards=[$('timeline').closest('.card'),$('status').closest('.card'),$('showCoverage').closest('.card')];replayCards.forEach(card=>card.classList.toggle('hidden',mode==='heatmap'));$('replayLegend').classList.toggle('hidden',mode==='heatmap');$('heatmapControls').classList.toggle('hidden',mode!=='heatmap');$('heatmapLegend').classList.toggle('hidden',mode!=='heatmap')}
function loadReplay(id){setLoading(true,'Loading artifact...');return fetch('/api/replay?id='+encodeURIComponent(id)).then(r=>{if(!r.ok)throw Error('Artifact failed to load');return r.json()}).then(d=>{camera=null;if(d.kind==='heatmap'){H=d;D=null;setMode('heatmap');$('replayName').textContent=(d.manifest.map_name||'3D evaluation')+' · '+d.positions.length+' targets';return drawHeatmap()}D=d;H=null;setMode('replay');frame=0;$('timeline').max=d.manifest.frames-1;$('replayName').textContent=d.manifest.map_name+' · '+d.manifest.frames+' frames';renderKnownList(d.manifest.agents||d.position[0].length);return draw(0)}).finally(()=>setLoading(false))}
function renderReplayList(items,preferredLabel){let s=$('replaySelect');s.replaceChildren();items.forEach(x=>{let option=document.createElement('option');option.value=x.id;option.textContent=x.label;s.appendChild(option)});let choice=items.find(x=>x.label===preferredLabel)||items.find(x=>x.selected)||items[0];if(choice)s.value=choice.id;return Promise.resolve()}
function refreshReplays(){let s=$('replaySelect'),preferredLabel=s.options[s.selectedIndex]?.textContent,b=$('refreshReplays');b.disabled=true;setLoading(true,'Finding artifacts...');$('replayDiscovery').textContent='Finding artifacts...';fetch('/api/replays').then(r=>{if(!r.ok)throw Error('Artifact list failed to load');return r.json()}).then(items=>renderReplayList(items,preferredLabel).then(()=>{$('replayDiscovery').textContent=items.length+' artifact'+(items.length===1?'':'s')+' found'})).catch(()=>{$('replayDiscovery').textContent='Unable to find artifacts'}).finally(()=>{b.disabled=false;setLoading(false)})}
function loadSelectedReplay(){let id=$('replaySelect').value;if(id)loadReplay(id)}
$('replaySelect').onchange=loadSelectedReplay;$('loadReplay').onclick=loadSelectedReplay;$('refreshReplays').onclick=refreshReplays;refreshReplays();
function lineTrace(points,color,width=3){return {type:'scatter3d',mode:'lines',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),line:{color,width},hoverinfo:'skip'}}
function boxTrace(s){let [x,y,z]=s,p=[[0,0,0],[x,0,0],[x,y,0],[0,y,0],[0,0,0],[0,0,z],[x,0,z],[x,y,z],[0,y,z],[0,0,z],[null,null,null],[x,0,0],[x,0,z],[null,null,null],[x,y,0],[x,y,z],[null,null,null],[0,y,0],[0,y,z]];return lineTrace(p,'rgba(120,155,205,.48)',2)}
const CUBE_FACES=[[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]];
function coverageTrace(coverage,cellSize,opacity){let x=[],y=[],z=[],i=[],j=[],k=[];for(let a=0;a<coverage.length;a++)for(let b=0;b<coverage[a].length;b++)for(let c=0;c<coverage[a][b].length;c++)if(coverage[a][b][c]){let o=x.length,px=a*cellSize,py=b*cellSize,pz=c*cellSize;[[px,py,pz],[px+cellSize,py,pz],[px+cellSize,py+cellSize,pz],[px,py+cellSize,pz],[px,py,pz+cellSize],[px+cellSize,py,pz+cellSize],[px+cellSize,py+cellSize,pz+cellSize],[px,py+cellSize,pz+cellSize]].forEach(v=>{x.push(v[0]);y.push(v[1]);z.push(v[2])});CUBE_FACES.forEach(q=>{i.push(o+q[0]);j.push(o+q[1]);k.push(o+q[2])})}return x.length?{type:'mesh3d',x,y,z,i,j,k,color:'#38bdf8',opacity,flatshading:true,hoverinfo:'skip',lighting:{ambient:.8,diffuse:.25,specular:0}}:null}
function sphereTrace(centres,radius,color,opacity){let x=[],y=[],z=[],i=[],j=[],k=[],lon=12,lat=8;centres.forEach(centre=>{let o=x.length;for(let row=0;row<=lat;row++)for(let col=0;col<=lon;col++){let theta=Math.PI*row/lat,phi=2*Math.PI*col/lon;x.push(centre[0]+radius*Math.sin(theta)*Math.cos(phi));y.push(centre[1]+radius*Math.sin(theta)*Math.sin(phi));z.push(centre[2]+radius*Math.cos(theta))}for(let row=0;row<lat;row++)for(let col=0;col<lon;col++){let a=o+row*(lon+1)+col,b=a+lon+1;i.push(a,b,a+1);j.push(b,b+1,b);k.push(a+1,a+1,b+1)}});return {type:'mesh3d',x,y,z,i,j,k,color,opacity,flatshading:true,hoverinfo:'skip',lighting:{ambient:1,diffuse:0,specular:0}}}
const HEATMAP_COLORS={chain_success:'#22c55e',found_and_delivered:'#3b82f6',visually_found:'#fbbf24',not_found:'#ef4444'};
function showHeatmapCommand(point,stage){let checkpoint=H.manifest.checkpoint||'<checkpoint-path>';let coords=point.map(value=>Number(value).toFixed(6)).join(',');let command='uv run swarmecho-evaluate-3d checkpoint='+checkpoint+' mode=selective_manual_pick target_position='+coords;$('selectedPoint').textContent=stage+' · ('+coords+')';$('heatmapCommand').value=command}
function drawHeatmap(){let traces=[],groups={chain_success:[],found_and_delivered:[],visually_found:[],not_found:[]},radius=+$('heatmapRadius').value,opacity=+$('heatmapOpacity').value;H.positions.forEach((point,index)=>{let stage=H.stages[index];(groups[stage]||groups.not_found).push({point,stage})});if(H.manifest.world_size_m)traces.push(boxTrace(H.manifest.world_size_m));Object.entries(groups).forEach(([stage,entries])=>{if(!entries.length)return;let points=entries.map(entry=>entry.point),color=HEATMAP_COLORS[stage];if($('heatmapSpheres').checked)traces.push(sphereTrace(points,radius,color,opacity));traces.push({type:'scatter3d',mode:'markers',x:points.map(point=>point[0]),y:points.map(point=>point[1]),z:points.map(point=>point[2]),customdata:entries.map(entry=>[entry.point,entry.stage]),marker:{size:$('heatmapSpheres').checked?13:8,color,opacity:$('heatmapSpheres').checked?0.035:opacity,line:{color:'#e5eefc',width:$('heatmapSpheres').checked?0:1}},hovertemplate:stage+'<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra>Click to create replay command</extra>'})});let renderPromise=Plotly.react('scene',traces,{margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'#090e18',uirevision:'heatmap-camera',scene:{bgcolor:'#090e18',uirevision:'heatmap-camera',aspectmode:'data',xaxis:{title:'X',gridcolor:'#22304a'},yaxis:{title:'Y',gridcolor:'#22304a'},zaxis:{title:'Z',gridcolor:'#22304a'},camera:camera||DEFAULT_CAMERA},showlegend:false},{responsive:true,displaylogo:false}).then(captureCamera);return renderPromise}
function captureCamera(){let scene=$('scene');if(!scene.on)return;if(!scene.__cameraListener){scene.on('plotly_relayout',event=>{if(event['scene.camera'])camera=event['scene.camera']});scene.__cameraListener=true}if(!scene.__heatmapClickListener){scene.on('plotly_click',event=>{if(mode!=='heatmap')return;let point=event.points.find(item=>item.customdata);if(point)showHeatmapCommand(point.customdata[0],point.customdata[1])});scene.__heatmapClickListener=true}}
function distance3(a,b){return Math.hypot(...a.map((v,k)=>v-b[k]))}
function baseKnowsTarget(f,base){if(D.base_target_known)return Boolean(D.base_target_known[f]);let radius=D.manifest.comm_radius_base_m||6;for(let q=0;q<=f;q++)for(let i=0;i<D.position[q].length;i++)if(D.active[q][i]&&D.target_known[q][i]&&distance3(D.position[q][i],base)<=radius)return true;return false}
function renderKnownList(agentCount){let list=$('knownList');list.replaceChildren();['BASE',...Array.from({length:agentCount},(_,i)=>'D'+(i+1))].forEach(name=>{let item=document.createElement('button');item.type='button';item.className='known-item';item.textContent=name;list.appendChild(item)})}
function updateKnownList(agentKnown,baseKnown){let statuses=[baseKnown,...agentKnown];$('knownList').querySelectorAll('.known-item').forEach((item,i)=>item.classList.toggle('known',Boolean(statuses[i])))}
function draw(f){frame=+f;let p=D.position[f],active=D.active[f],known=D.target_known[f],tr=[],agents=p.filter((_,i)=>active[i]),b=D.base_position[f],t=D.target_position[f],visualRadius=D.manifest.visual_radius_m||4,commRadius=D.manifest.comm_radius_m||5,baseCommRadius=D.manifest.comm_radius_base_m||6,baseKnown=baseKnowsTarget(f,b);if($('showShell').checked)tr.push(boxTrace(D.manifest.world_size_m||[20,20,20]));if($('showVisualRange').checked&&agents.length)tr.push(sphereTrace(agents,visualRadius,'#fbbf24',+$('visualOpacity').value));if($('showCommRange').checked&&agents.length)tr.push(sphereTrace(agents,commRadius,'#a78bfa',+$('commOpacity').value));if($('showCommRange').checked)tr.push(sphereTrace([b],baseCommRadius,'#a78bfa',+$('commOpacity').value));updateKnownList(known,baseKnown);
let colors=p.map((_,i)=>known[i]?'#fb7185':'#22d3ee');tr.push({type:'scatter3d',mode:'markers+text',x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),text:p.map((_,i)=>'D'+(i+1)),textposition:'top center',marker:{size:7,color:colors,line:{color:'#e5eefc',width:1}},hovertemplate:'%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>'});
tr.push({type:'scatter3d',mode:'markers+text',x:[b[0],t[0]],y:[b[1],t[1]],z:[b[2],t[2]],text:['BASE','TARGET'],textposition:'top center',marker:{size:[9,9],color:[baseKnown?'#fb7185':'#60a5fa','#fb7185'],symbol:['diamond','circle']}});
if($('showLinks').checked){for(let i=0;i<p.length;i++)for(let j=i+1;j<p.length;j++)if(active[i]&&active[j]&&distance3(p[i],p[j])<=commRadius)tr.push(lineTrace([p[i],p[j]],'rgba(34,211,238,.35)',3));for(let i=0;i<p.length;i++)if(active[i]&&distance3(p[i],b)<=baseCommRadius)tr.push(lineTrace([b,p[i]],'rgba(167,139,250,.55)',3));for(let i=0;i<p.length;i++){let seesTarget=D.directly_sees_target?D.directly_sees_target[f][i]:distance3(p[i],t)<=visualRadius;if(active[i]&&seesTarget)tr.push(lineTrace([t,p[i]],'rgba(251,113,133,.65)',3))}}
if($('showCoverage').checked){let coverage=coverageTrace(D.coverage[f],D.manifest.coverage_voxel_size_m||D.manifest.cell_size_m||5,+$('coverageOpacity').value);if(coverage)tr.push(coverage)}
const renderPromise=Plotly.react('scene',tr,{margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'#090e18',uirevision:'replay-camera',scene:{bgcolor:'#090e18',uirevision:'replay-camera',aspectmode:'data',xaxis:{title:'X',gridcolor:'#22304a'},yaxis:{title:'Y',gridcolor:'#22304a'},zaxis:{title:'Z',gridcolor:'#22304a'},camera:camera||DEFAULT_CAMERA},showlegend:false},{responsive:true,displaylogo:false}).then(captureCamera);
$('timeline').value=f;$('frame').textContent=f+'/'+(D.manifest.frames-1);$('time').textContent=(f*D.manifest.dt).toFixed(1)+' s';let cov=D.coverage[f].flat(2).filter(Boolean).length,total=D.coverage[f].flat(2).length;$('coverage').textContent=(100*cov/total).toFixed(1)+'%';$('known').textContent=known.filter(Boolean).length;$('baseConn').textContent=D.connected_to_base[f].filter(Boolean).length;$('status').textContent=D.success[f]?'SUCCESS':D.fully_connected[f]?'CHAIN HELD':known.some(Boolean)?'TARGET KNOWN':'EXPLORING';$('status').style.color=D.success[f]?'#a3e635':'#e5eefc';$('reward').textContent=D.reward_terms?D.reward_terms[f].flat().reduce((a,b)=>a+b,0).toFixed(2):'—';return renderPromise}
function updateOpacityValue(input,output){$(output).textContent=(Math.round(1000*+$(input).value)/10)+'%'}
function updateRadiusValue(){$('heatmapRadiusValue').textContent=(+$('heatmapRadius').value).toFixed(2)+' m'}
$('timeline').oninput=e=>draw(e.target.value);['showCoverage','showVisualRange','showCommRange','showLinks','showShell'].forEach(id=>$(id).onchange=()=>draw(frame));[['coverageOpacity','coverageOpacityValue'],['visualOpacity','visualOpacityValue'],['commOpacity','commOpacityValue']].forEach(([input,output])=>{updateOpacityValue(input,output);$(input).oninput=()=>{updateOpacityValue(input,output);draw(frame)}});updateOpacityValue('heatmapOpacity','heatmapOpacityValue');updateRadiusValue();['heatmapSpheres','heatmapOpacity','heatmapRadius'].forEach(id=>$(id).oninput=()=>{updateOpacityValue('heatmapOpacity','heatmapOpacityValue');updateRadiusValue();if(mode==='heatmap')drawHeatmap()});$('copyHeatmapCommand').onclick=()=>{let command=$('heatmapCommand').value;if(command)navigator.clipboard.writeText(command).then(()=>{$('copyHeatmapCommand').textContent='Copied';setTimeout(()=>{$('copyHeatmapCommand').textContent='Copy command'},1200)})};$('step').onclick=()=>draw(Math.min(frame+1,D.manifest.frames-1));$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'❚❚ Pause':'▶ Play';if(playing)tick();else clearTimeout(timer)};function tick(){if(!playing)return;draw(frame>=D.manifest.frames-1?0:frame+1);timer=setTimeout(tick,1000*D.manifest.dt/+$('speed').value)}
</script></body></html>"""


def replay_payload(manifest_path: str | Path) -> dict:
    manifest, arrays = load_replay(manifest_path)
    return {
        "kind": "replay",
        "manifest": manifest,
        **{name: value.tolist() for name, value in arrays.items()},
    }


def heatmap_payload(info_path: str | Path) -> dict:
    """Load a 3D evaluation CSV and its optional inspector metadata sidecar."""
    path = Path(info_path)
    records = load_eval_info_csv(path)
    if records["positions"].shape[-1] != 3:
        raise ValueError(f"3D heatmaps require x,y,z coordinates: {path}")
    manifest = {
        "format": "swarmecho-3d-eval-heatmap/v1",
        "data_file": path.name,
        "map_name": path.stem,
    }
    sidecar = path.with_suffix(".heatmap.json")
    if sidecar.exists():
        try:
            manifest.update(json.loads(sidecar.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "kind": "heatmap",
        "manifest": manifest,
        "positions": records["positions"].tolist(),
        "stages": records["stages"].tolist(),
        "distance_to_base": records["distance_to_base"].tolist(),
    }


def inspector_html() -> str:
    """Embed the installed Plotly runtime so inspection also works offline."""
    from plotly.offline import get_plotlyjs

    external = '<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>'
    return HTML.replace(external, f"<script>{get_plotlyjs()}</script>")


def discover_replays(root: str | Path = "outputs") -> list[Path]:
    """Find completed replay manifests in the known replay folders.

    Replay artifacts use a fixed layout, so scanning only those folders avoids
    walking checkpoints, W&B data, logs, and unrelated JSON files.
    """
    root = Path(root)
    manifests: list[Path] = []
    if not root.exists():
        return manifests
    replay_dirs: set[Path] = set()
    for pattern in (
        "*/replays",
        "*/artifacts/train/replays",
        "*/artifacts/eval/replays",
        "*/artifacts/eval/*/replays",
    ):
        replay_dirs.update(path for path in root.glob(pattern) if path.is_dir())
    for replay_dir in replay_dirs:
        for path in replay_dir.glob("*.json"):
            data_path: Path | None = None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data_file = data.get("data_file") if isinstance(data, dict) else None
                if isinstance(data_file, str) and data_file:
                    data_path = path.parent / data_file
            except (OSError, json.JSONDecodeError):
                continue
            if (
                not isinstance(data, dict)
                or data.get("format") != "swarmecho-replay/v1"
                or data_path is None
                or not data_path.exists()
            ):
                continue
            manifests.append(path.resolve())
    return sorted(set(manifests), key=lambda path: path.stat().st_mtime, reverse=True)


def discover_heatmaps(root: str | Path = "outputs") -> list[Path]:
    """Find complete 3D evaluation CSVs in the standard artifact data folders."""
    root = Path(root)
    if not root.exists():
        return []
    data_dirs: set[Path] = set()
    for pattern in (
        "*/artifacts/train/data",
        "*/artifacts/eval/*/data",
    ):
        data_dirs.update(path for path in root.glob(pattern) if path.is_dir())
    heatmaps: list[Path] = []
    for data_dir in data_dirs:
        for path in data_dir.glob("eval_info_*.csv"):
            try:
                if load_eval_info_csv(path)["positions"].shape[-1] == 3:
                    heatmaps.append(path.resolve())
            except (OSError, ValueError):
                continue
    return sorted(set(heatmaps), key=lambda path: path.stat().st_mtime, reverse=True)


_REPLAY_STEPS = re.compile(r"(?:^|_)s0*(\d+)([Mk])(?:$|[._])", re.IGNORECASE)


def _replay_steps_label(manifest_path: Path) -> str | None:
    """Return the compact, unpadded step suffix for a replay."""
    match = _REPLAY_STEPS.search(manifest_path.stem)
    if match:
        return f"{int(match.group(1))}{match.group(2)}"
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    steps = metadata.get("environment_steps")
    if steps is None:
        return None
    steps = int(steps)
    if steps >= 1_000_000:
        return f"{round(steps / 1_000_000):g}M"
    if steps >= 1_000:
        return f"{round(steps / 1_000):g}k"
    return str(steps)


def replay_label(manifest_path: str | Path) -> str:
    """Return the run name and compact training steps for the replay picker."""
    path = Path(manifest_path)
    parts = path.parts
    try:
        run_name = parts[parts.index("artifacts") - 1]
    except ValueError:
        run_name = path.parent.parent.name if path.parent.name == "replays" else path.stem
    steps = _replay_steps_label(path)
    target_replay = _is_target_replay(path)
    suffix = " TARGET-REPLAY" if target_replay else ""
    return f"{run_name}-[{steps}]{suffix}" if steps else f"{run_name}{suffix}"


def _is_target_replay(manifest_path: str | Path) -> bool:
    """Identify both legacy and coordinate-qualified manual target replays."""
    path = Path(manifest_path)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("target_selection") in {"TARGET", "TARGET-REPLAY"}:
            return True
    except (OSError, json.JSONDecodeError):
        pass
    return bool(re.search(r"(?:^|_)TARGET(?:-REPLAY)?(?:_|$)", path.stem))


def heatmap_label(info_path: str | Path) -> str:
    """Return the matching compact label for a 3D evaluation heatmap CSV."""
    path = Path(info_path)
    parts = path.parts
    try:
        run_name = parts[parts.index("artifacts") - 1]
    except ValueError:
        run_name = path.stem
    steps = _replay_steps_label(path)
    return f"{run_name}-[{steps}] heatmap" if steps else f"{run_name} heatmap"


def make_handler(
    manifests: list[Path], initial: Path | None = None, root: str | Path | None = None
):
    resolved = [path.resolve() for path in manifests]
    initial = initial.resolve() if initial is not None else None
    replay_root = Path(root).resolve() if root is not None else None
    encoded_html = inspector_html().encode()
    manifests_lock = threading.Lock()
    cached_sources: list[tuple[str, Path]] = [("replay", path) for path in resolved]

    def refresh_sources() -> list[tuple[str, Path]]:
        nonlocal cached_sources
        replays = discover_replays(replay_root) if replay_root is not None else list(resolved)
        heatmaps = discover_heatmaps(replay_root) if replay_root is not None else []
        sources = [("replay", path) for path in replays] + [
            ("heatmap", path) for path in heatmaps
        ]
        if initial is not None and initial not in replays:
            sources.append(("replay", initial))

        # Replays and heatmaps share one chronological ordering. The run name
        # and artifact kind must never determine where an item appears.
        current = sorted(
            sources,
            key=lambda source: source[1].stat().st_mtime_ns,
            reverse=True,
        )
        with manifests_lock:
            cached_sources = current
            return list(cached_sources)

    def available_sources() -> list[tuple[str, Path]]:
        with manifests_lock:
            return list(cached_sources)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/replays":
                current = refresh_sources()
                items = [
                    {
                        "id": str(index),
                        "kind": kind,
                        "label": (
                            f"[Replay] {replay_label(path)}"
                            if kind == "replay"
                            else f"[Heatmap] {heatmap_label(path)}"
                        ),
                        "selected": kind == "replay" and (
                            path == initial or (initial is None and index == 0)
                        ),
                    }
                    for index, (kind, path) in enumerate(current)
                ]
                body, content_type = json.dumps(items).encode(), "application/json"
            elif parsed.path == "/api/replay":
                query = urllib.parse.parse_qs(parsed.query)
                try:
                    kind, path = available_sources()[int(query.get("id", ["0"])[0])]
                    payload = replay_payload(path) if kind == "replay" else heatmap_payload(path)
                    body = json.dumps(payload).encode()
                except (IndexError, ValueError, OSError):
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


def default_replay_root() -> Path:
    """Locate this checkout's outputs directory when launched from any cwd."""
    project_outputs = Path(__file__).resolve().parents[3] / "outputs"
    return project_outputs if project_outputs.exists() else Path.cwd() / "outputs"


def main() -> None:
    manifest: Path | None = None
    root = default_replay_root()
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
    manifests: list[Path] = []
    if manifest is not None:
        manifest = manifest.resolve()
    server = ThreadingHTTPServer(
        (host, port), make_handler(manifests, manifest, root=root)
    )
    url = f"http://{host}:{port}"
    print(f"SwarmEcho 3D Inspector: {url}")
    print(f"Replay root: {root.resolve()}")
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
