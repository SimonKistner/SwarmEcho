"""Standalone browser inspector for completed replay artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import yaml
from swarmecho.core.compatibility import canonical_marker, resolve_map_file

from swarmecho.training.artifacts import load_eval_info_csv, parse_checkpoint_update
from swarmecho.visualize.replay import load_replay


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>SwarmEcho Inspector</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>
:root{color-scheme:dark;--bg:#090e18;--panel:#111a2a;--line:#26344d;--cyan:#22d3ee;--text:#e5eefc;--muted:#91a4c3;--heat-chain:#4ade80;--heat-delivered:#60a5fa;--heat-visual:#fcd34d;--heat-missed:#fb7185}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px Inter,system-ui,sans-serif;overflow:hidden}
header{height:116px;display:grid;grid-template-columns:230px minmax(0,1fr) 310px;grid-template-rows:40px 37px 38px;align-items:center;background:#0d1523;border-bottom:1px solid var(--line)}
header.has-map-navigation{height:154px;grid-template-rows:40px 37px 38px 38px}header.has-map-navigation+#layout{height:calc(100vh - 154px)}
h1{font-size:18px;margin:0;letter-spacing:.04em}.tag{color:var(--cyan);font-weight:700}.meta{color:var(--muted)}
#layout{display:grid;grid-template-columns:230px minmax(0,1fr) 310px;height:calc(100vh - 116px)}#scene{min-width:0}.panel{padding:18px;background:var(--panel);border-left:1px solid var(--line);overflow:auto}
.card{padding:14px;margin-bottom:12px;background:#0c1422;border:1px solid var(--line);border-radius:10px}.label{font-size:11px;text-transform:uppercase;color:var(--muted);letter-spacing:.1em;margin-bottom:8px}
.value{font-size:24px;font-weight:750}.row{display:flex;justify-content:space-between;gap:12px;margin:7px 0;color:var(--muted)}.row b{color:var(--text)}
button{border:1px solid #325173;background:#15253a;color:var(--text);padding:8px 12px;border-radius:7px;cursor:pointer}button:hover{border-color:var(--cyan)}
.loading{display:inline-flex;align-items:center;gap:7px;color:var(--muted);white-space:nowrap}.loading.hidden{display:none}.spinner{width:13px;height:13px;border:2px solid #34506f;border-top-color:var(--cyan);border-radius:50%;animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}
input[type=range]{width:100%;accent-color:var(--cyan)}select,input[type=text]{width:100%;background:#15253a;color:var(--text);border:1px solid #325173;padding:8px;border-radius:7px}.hidden{display:none!important}
.controls{display:flex;gap:8px;margin-bottom:10px}.layer-control{display:flex;justify-content:space-between;gap:8px;align-items:center}.opacity-value{color:var(--muted);font-size:12px}.opacity-control{display:block;margin:4px 0 8px}.legend span{display:block;margin:7px 0}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
.agreement-buttons{display:flex;flex-wrap:wrap;gap:6px}.agreement-buttons button{padding:6px 9px;font-size:12px}.agreement-buttons button.selected{border-color:var(--cyan);background:#16445a;color:#fff}.stat-row{display:grid;grid-template-columns:1fr auto;align-items:baseline;gap:10px;margin:7px 0}.stat-row>span{color:var(--muted)}.stat-row b{display:grid;grid-template-columns:4ch auto;gap:5px;color:var(--text);min-width:90px;text-align:right;white-space:nowrap}.stat-row b span:first-child{text-align:right}.stat-row b span:last-child{color:var(--muted);font-weight:400}
.heatmap-divider{border:0;border-top:1px solid var(--line);margin:16px 0}.heatmap-category{display:flex;align-items:center;gap:7px;margin:8px 0}.heatmap-category input,.heatmap-category .dot{margin:0}
.known-list{display:grid;gap:6px}.known-item{width:100%;font:inherit;text-align:left;color:#667893;background:#0c1422;border:1px solid var(--line);border-radius:7px;padding:7px 10px;pointer-events:none}.known-item.known{color:var(--text);border-color:#a3e635;background:rgba(163,230,53,.14);box-shadow:0 0 0 1px rgba(163,230,53,.25)}
#artifactPicker{position:relative;min-width:180px;width:min(440px,32vw)}#artifactPicker summary{cursor:pointer;padding:9px 12px;border:1px solid #325173;border-radius:7px;background:#15253a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.artifact-popover{position:absolute;top:calc(100% + 8px);left:0;display:grid;grid-template-columns:minmax(200px,1fr) minmax(260px,1.3fr);width:min(720px,80vw);background:#0c1422;border:1px solid #39516e;border-radius:10px;box-shadow:0 18px 45px #0009;z-index:1000;overflow:hidden}#artifactMenu,#artifactSubmenu{max-height:65vh;overflow:auto;padding:8px}#artifactSubmenu{border-left:1px solid #26344d}.artifact-entry{display:flex;align-items:center;justify-content:space-between;gap:10px;width:100%;margin:3px 0;text-align:left;overflow-wrap:anywhere}.artifact-entry.selected{background:#204457;border-color:#22d3ee}.artifact-count{font-size:11px;color:#91a4c3;white-space:nowrap}
header h1{grid-column:1;grid-row:1;font-size:14px;padding:0 12px;white-space:nowrap}
.header-selection{grid-column:2;grid-row:1;display:flex;align-items:center;gap:20px;min-width:0;padding-right:16px}
#artifactPicker{flex:0 1 650px;min-width:0;width:min(650px,100%)}#artifactPicker summary{padding:6px 10px;font-size:13px}
.header-refresh{grid-column:3;grid-row:1;display:flex;justify-content:flex-end;gap:10px;padding:0 16px}.header-refresh button{padding:6px 10px;white-space:nowrap}
.header-status{grid-column:3;grid-row:2;padding:0 16px;text-align:right;font-size:12px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.header-help{grid-column:1;grid-row:2;padding-left:12px;font-size:11px;color:var(--muted)}
#replayName{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px}
.artifact-popover{grid-template-columns:minmax(0,1fr) 460px;width:min(950px,calc(100vw - 250px));max-width:none}
#artifactMenu .artifact-entry{padding:8px 10px}#artifactMenu .artifact-entry>span:first-child{min-width:0}#artifactSubmenu>.label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px 2px}
.artifact-navigation{grid-column:2;display:flex;justify-self:center;align-items:center;gap:8px;max-width:100%;min-width:0}.category-navigation{grid-row:2}.step-navigation{grid-row:3}.map-navigation{grid-row:4}.artifact-navigation button{padding:5px 12px}.artifact-navigation .navigation-arrow{min-width:38px;flex:none;margin:0 3px}.navigation-dot{width:7px;height:7px;flex:none;border-radius:50%;background:#52627b}.artifact-navigation[data-active="true"] .navigation-dot{background:var(--cyan);box-shadow:0 0 8px #22d3ee99}.navigation-choices{display:flex;align-items:center;gap:6px;flex:0 1 auto;min-width:0;overflow-x:auto;scrollbar-width:thin;scrollbar-color:#406383 transparent}.navigation-choices button{flex:none;max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.map-navigation-help{grid-column:1;grid-row:3;padding-left:12px;color:var(--muted);font-size:11px;white-space:nowrap}
button:disabled{opacity:.35;cursor:default}button:disabled:hover{border-color:#325173}.artifact-navigation button[aria-pressed="true"]{background:#204457;border-color:var(--cyan)}
.artifact-columns{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:4px 8px}.artifact-column-title{position:sticky;top:-8px;background:#0c1422;padding:7px 4px;margin-bottom:3px;z-index:1}
.artifact-cell{position:relative;display:flex;min-width:0;align-items:stretch}.artifact-cell .artifact-entry{flex:1;min-width:0;margin:0;font-size:12px;padding:7px 30px 7px 9px;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.artifact-map-cell{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:6px}.artifact-map-cell .artifact-entry{width:auto;min-width:52px;max-width:92px;padding:7px 9px;text-align:center;font-weight:650;font-variant-numeric:tabular-nums;background:#1b3045;border-color:#406383}.artifact-map-cell .artifact-entry.selected{background:#204457;border-color:var(--cyan)}.artifact-map-cell select{min-width:0;width:100%;padding:7px 8px}
.artifact-load-more{display:block;width:100%;margin:10px 0 2px;border-style:dashed;color:var(--cyan);font-size:12px}
.artifact-star{position:absolute;right:3px;top:50%;transform:translateY(-50%);padding:3px 5px;border:0;border-radius:4px;background:transparent;color:#91a4c3;font-size:16px;line-height:20px}.artifact-star:hover{background:#ffffff12;color:#fbbf24}.artifact-star:focus-visible{outline:1px solid var(--cyan)}.artifact-star.starred,.run-star{color:#fbbf24}.artifact-count{display:flex;gap:6px;align-items:center;flex-shrink:0}
@media(max-width:1200px){.header-selection{gap:10px}#replayName{display:none}.artifact-popover{grid-template-columns:minmax(0,1fr) 410px}.header-refresh{gap:6px;padding:0 12px}}
</style></head><body>
<header id="inspectorHeader">
<h1><span class="tag">SwarmEcho</span> Inspector</h1>
<div class="header-selection"><select id="replaySelect" hidden aria-hidden="true"></select><details id="artifactPicker"><summary id="artifactPickerButton">Choose a run…</summary><div class="artifact-popover"><div id="artifactMenu"></div><div id="artifactSubmenu"><p class="meta">Select a run to see its artifacts.</p></div></div></details><span id="replayName" class="meta"></span></div>
<div class="header-refresh"><button id="refreshRun" type="button" title="Look for new entries only in the selected run">Refresh run</button><button id="refreshReplays" type="button" title="Look for new runs and entries in all runs">Refresh all</button></div>
<span class="header-help">Drag to orbit · scroll to zoom</span>
<div id="categoryNavigation" class="artifact-navigation category-navigation" role="group" aria-label="Heatmap or replay; Up arrow selects this row" data-active="true"><span class="navigation-dot" aria-hidden="true"></span><button id="previousCategory" class="navigation-arrow" title="Heatmap (Left arrow when categories are active)" aria-label="Heatmap category" disabled>←</button><button id="heatmapMode" aria-pressed="false" disabled>Heatmap</button><button id="replayMode" aria-pressed="false" disabled>Replay</button><button id="nextCategory" class="navigation-arrow" title="Replay (Right arrow when categories are active)" aria-label="Replay category" disabled>→</button><span class="navigation-dot" aria-hidden="true"></span></div>
<span id="mapNavigationHelp" class="map-navigation-help">↑↓ row · ←→ item</span><div id="stepNavigation" class="artifact-navigation step-navigation" role="group" aria-label="Navigate steps; Up or Down arrow selects this row" data-active="false"><span class="navigation-dot" aria-hidden="true"></span><button id="previousArtifact" class="navigation-arrow" title="Previous listed step (Left arrow when steps are active)" aria-label="Previous listed step" disabled>←</button><div id="stepChoices" class="navigation-choices" role="group" aria-label="Steps for the selected category"></div><button id="nextArtifact" class="navigation-arrow" title="Next listed step (Right arrow when steps are active)" aria-label="Next listed step" disabled>→</button><span class="navigation-dot" aria-hidden="true"></span></div>
<div id="mapNavigation" class="artifact-navigation map-navigation hidden" role="group" aria-label="Navigate maps; Down arrow selects this row" data-active="false"><span class="navigation-dot" aria-hidden="true"></span><button id="previousMap" class="navigation-arrow" title="Previous map (Left arrow when maps are active)" aria-label="Previous map" disabled>←</button><div id="mapChoices" class="navigation-choices" role="group" aria-label="Maps for the selected step"></div><button id="nextMap" class="navigation-arrow" title="Next map (Right arrow when maps are active)" aria-label="Next map" disabled>→</button><span class="navigation-dot" aria-hidden="true"></span></div>
<div class="header-status"><span id="loading" class="loading hidden" role="status" aria-live="polite"><span class="spinner"></span><span id="loadingText">Loading...</span></span><span id="replayDiscovery" class="meta"></span></div>
</header>
<div id="layout"><aside id="buildingPanel" class="panel" style="border-left:0;border-right:1px solid var(--line)"><div class="card"><div class="label">Building visibility</div><div id="storeyVisibility"></div><hr><label class="layer-control"><span><input id="showOuterWalls" type="checkbox"> Render outer walls</span></label><label class="layer-control"><span><input id="showRoof" type="checkbox"> Render roof</span></label><hr><label class="layer-control"><span><input id="showCellGrid" type="checkbox" checked> Cell grid</span></label><label class="layer-control" style="margin-top:16px"><span>Building transparency</span><span id="wallTransparencyValue" class="opacity-value">0%</span></label><input id="wallTransparency" type="range" min="0" max="100" step="1" value="0" aria-label="Building transparency"><p id="buildingSource" class="meta"></p><label id="roadmapOrangeControl" class="layer-control hidden" style="margin-top:18px"><span><input id="roadmapOrangeGeometry" type="checkbox"> Orange obstacle rendering</span></label></div></aside><div id="scene"></div><aside class="panel">
<div id="heatmapControls" class="card hidden"><div class="label">Heatmap statistics</div><div class="stat-row"><span><i class="dot" style="background:var(--heat-chain)"></i>Chain success</span><b><span id="heatmapStatChain">—</span><span></span></b></div><div class="stat-row"><span><i class="dot" style="background:var(--heat-delivered)"></i>Found and delivered</span><b><span id="heatmapStatDelivered">—</span><span id="heatmapStatDeliveredDelta"></span></b></div><div class="stat-row"><span><i class="dot" style="background:var(--heat-visual)"></i>Visually found</span><b><span id="heatmapStatVisual">—</span><span id="heatmapStatVisualDelta"></span></b></div><div class="stat-row"><span><i class="dot" style="background:var(--heat-missed)"></i>Not found</span><b><span id="heatmapStatNotFound">—</span><span></span></b></div><hr class="heatmap-divider"><div class="label">Heatmap categories</div><label class="heatmap-category"><input id="heatmapShowChainSuccess" type="checkbox" checked><i class="dot" style="background:var(--heat-chain)"></i>Chain success</label><label class="heatmap-category"><input id="heatmapShowFoundDelivered" type="checkbox" checked><i class="dot" style="background:var(--heat-delivered)"></i>Found and delivered</label><label class="heatmap-category"><input id="heatmapShowVisuallyFound" type="checkbox" checked><i class="dot" style="background:var(--heat-visual)"></i>Visually found</label><label class="heatmap-category"><input id="heatmapShowNotFound" type="checkbox" checked><i class="dot" style="background:var(--heat-missed)"></i>Not found</label><div class="label" style="margin-top:16px">Required agreement</div><div id="heatmapAgreementButtons" class="agreement-buttons"></div></div>
<div class="card"><div class="label">Playback</div><div class="controls"><button id="play">▶ Play</button><button id="step">Step</button></div><input id="timeline" type="range" min="0" value="0"><div class="row"><span>Frame</span><b id="frame">0</b></div><div class="row"><span>Time</span><b id="time">0.0 s</b></div><select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option selected value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select><label style="display:block;margin-top:10px"><input id="loopReplay" type="checkbox"> Loop episode</label></div><div id="cameraControls" class="card"><div class="label">Camera</div><label class="layer-control"><span>Elevation</span><span id="cameraElevationValue" class="opacity-value">45°</span></label><input id="cameraElevation" type="range" min="-85" max="85" step="1" value="45" aria-label="Camera elevation in degrees"><label class="layer-control"><span><input id="autoRotate" type="checkbox"> Auto rotate</span><span id="rotateSpeedValue" class="opacity-value">10°/s</span></label><input id="rotateSpeed" type="range" min="-90" max="90" step="1" value="10" aria-label="Camera rotation speed in degrees per second"></div>
<div class="card"><div class="label">Episode</div><div id="status" class="value">Exploring</div><div class="row"><span>Reward</span><b id="reward">—</b></div><div class="row"><span>Coverage</span><b id="coverage">0%</b></div><div class="row"><span>Known agents</span><b id="known">0</b></div><div class="row"><span>Base connected</span><b id="baseConn">0</b></div><div class="label" style="margin-top:16px">Target known</div><div id="knownList" class="known-list"></div></div>
<div class="card"><div class="label">Layers</div><label class="layer-control"><span><input id="showCoverage" type="checkbox"> Coverage voxels</span><span id="coverageOpacityValue" class="opacity-value">2%</span></label><input id="coverageOpacity" class="opacity-control" type="range" min="0" max="1" step="0.001" value="0.02" aria-label="Coverage voxel opacity"><label class="layer-control"><span><input id="showVisualRange" type="checkbox"> Visual range</span><span id="visualOpacityValue" class="opacity-value">4.5%</span></label><input id="visualOpacity" class="opacity-control" type="range" min="0" max="1" step="0.001" value="0.045" aria-label="Visual range opacity"><label class="layer-control"><span><input id="showCommRange" type="checkbox"> Communication range</span><span id="commOpacityValue" class="opacity-value">2.5%</span></label><input id="commOpacity" class="opacity-control" type="range" min="0" max="1" step="0.001" value="0.025" aria-label="Communication range opacity"><label><input id="showLinks" type="checkbox" checked> Communication links</label><br><label><input id="showShell" type="checkbox" checked> Transparent shell</label></div>
<div id="replayLegend" class="card legend"><div class="label">Legend</div><span><i class="dot" style="background:#22d3ee"></i>Drone</span><span><i class="dot" style="background:#60a5fa"></i>Base</span><span><i class="dot" style="background:#fb7185"></i>Target</span><span><i class="dot" style="background:#fb7185"></i>Target-known drone / base</span><span><i class="dot" style="background:#a3e635"></i>Known area highlight</span></div>
<div id="selectedTarget" class="card hidden"><div class="label">Selected target</div><div id="selectedPoint" class="meta">Click a point to create a replay command.</div><input id="heatmapCommand" type="text" readonly style="margin-top:8px" value=""><button id="copyHeatmapCommand" type="button" style="margin-top:8px">Copy command</button></div>
</aside></div><script>
let D,H,mode='replay',frame=0,playing=false,timer=null,camera=null,heatmapAgreement=1,rotationFrame=null,rotationTime=null,rendering=false,renderQueue=Promise.resolve();const $=id=>document.getElementById(id);const DEFAULT_CAMERA={eye:{x:1.45,y:1.45,z:Math.hypot(1.45,1.45)*Math.tan(45*Math.PI/180)},projection:{type:'perspective'}};
function worldSize(){return D?.manifest?.world_size_m||H?.manifest?.world_size_m||[20,20,20]}
function sceneAxes(){let axis=title=>({title,gridcolor:'#22304a'});return {xaxis:axis('X'),yaxis:axis('Y'),zaxis:axis('Z')}}
function sceneLayout(revision){return {bgcolor:'#090e18',uirevision:revision,aspectmode:'data',...sceneAxes(),camera:camera||DEFAULT_CAMERA}}
function renderPlot(traces,revision,extra={}){renderQueue=renderQueue.catch(()=>{}).then(()=>{if(camera)rememberCamera();rendering=true;return Plotly.react('scene',traces,{margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'#090e18',scene:sceneLayout(revision),...extra},{responsive:true,displaylogo:false}).then(captureCamera).finally(()=>{rendering=false;rotationTime=null})});return renderQueue}
function setLoading(active,message){$('loading').classList.toggle('hidden',!active);$('replayDiscovery').classList.toggle('hidden',active);if(active)$('loadingText').textContent=message}
function setMode(nextMode){mode=nextMode;$('buildingPanel').style.visibility='visible';$('roadmapOrangeControl').classList.toggle('hidden',mode!=='roadmap');playing=false;clearTimeout(timer);$('play').textContent='▶ Play';let replayCards=[$('timeline').closest('.card'),$('status').closest('.card'),$('showCoverage').closest('.card')];replayCards.forEach(card=>card.classList.toggle('hidden',mode!=='replay'));$('replayLegend').classList.toggle('hidden',mode!=='replay');$('heatmapControls').classList.toggle('hidden',mode!=='heatmap');$('selectedTarget').classList.toggle('hidden',mode!=='heatmap');if($('roadmapControls'))$('roadmapControls').classList.toggle('hidden',mode!=='roadmap');if(mode==='heatmap'){$('selectedPoint').textContent='Click a point to create a replay command.';$('heatmapCommand').value='';renderHeatmapAgreementButtons();updateHeatmapStats()}}
function loadReplay(id,keepCamera=false){setLoading(true,'Loading artifact...');let url='/api/replay?id='+encodeURIComponent(id)+'&_='+Date.now();return fetch(url,{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Artifact failed to load');return r.json()}).then(d=>{if(!keepCamera)camera=null;if(d.kind==='roadmap'){D=d;H=null;setupBuildingVisibility(keepCamera);$('roadmapOrangeGeometry').checked=false;setMode('roadmap');$('replayName').textContent=(d.manifest.map_name||'Roadmap')+' · '+d.layouts.length+' layouts';return drawRoadmap()}if(d.kind==='heatmap'){H=d;D=null;setupBuildingVisibility(keepCamera);setMode('heatmap');$('replayName').textContent=(d.manifest.map_name||'evaluation')+' · '+d.positions.length+' targets';return drawHeatmap()}D=d;H=null;setupBuildingVisibility(keepCamera);setMode('replay');frame=0;$('timeline').max=d.manifest.frames-1;$('replayName').textContent=d.manifest.map_name+' · '+d.manifest.frames+' frames';renderKnownList(d.manifest.agents||d.position[0].length);return draw(0)}).finally(()=>setLoading(false))}
let artifactItems=[],artifactBusy=false,keyboardTrack='types',showingAllRuns=false,moreRunsAvailable=false;
let artifactStars=new Set();try{artifactStars=new Set(JSON.parse(localStorage.getItem('swarmecho.inspector.stars')||'[]'))}catch{}
function artifactParts(label){const start=label.indexOf('_[');return {run:start<0?label:label.slice(0,start),detail:start<0?'Open artifact':label.slice(start+2).replace(/\]_\[/g,' · ').replace(/\]$/,'')}}
function artifactDetail(item){if(item.random_eval_map_id)return artifactParts(item.label).detail+' · '+item.random_eval_map_id;const duplicates=artifactItems.filter(other=>artifactRun(other)===artifactRun(item)&&other.label===item.label);let detail=artifactParts(item.label).detail;if(artifactKind(item)==='roadmap'&&item.label.startsWith('random_buildings_'))detail=detail.replace(/ · Roadmap$/,'');return detail+(duplicates.length>1?' · '+(duplicates.indexOf(item)+1):'')}
function artifactKind(item){return item.kind||(/Heatmap/i.test(item.label)?'heatmap':'replay')}
function artifactRun(item){return item.run_id||artifactParts(item.label).run}
function artifactSteps(item){const match=artifactParts(item.label).detail.match(/^(\d+(?:\.\d+)?)\s*([MkG])?(?:\s|$)/i);return match?Number(match[1])*({m:1e6,k:1e3,g:1e9}[(match[2]||'').toLowerCase()]||1):null}
function currentArtifact(){return artifactItems.find(item=>String(item.id)===$('replaySelect').value)}
function runItems(item){return item?artifactItems.filter(other=>artifactRun(other)===artifactRun(item)):[]}
function columnItems(group,kind){return group.filter(item=>artifactKind(item)===kind).sort((a,b)=>(artifactSteps(b)??-1)-(artifactSteps(a)??-1)||(kind==='heatmap'?Number(/EVAL Heatmap/i.test(b.label))-Number(/EVAL Heatmap/i.test(a.label)):0))}
function navigationKey(item){return item.random_eval_group?'maps:'+item.random_eval_group:'item:'+item.id}
function navigationGroups(item){const groups=new Map();for(const entry of columnItems(runItems(item),artifactKind(item))){const key=navigationKey(entry);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(entry)}return [...groups.values()].reverse()}
function mapItems(item){return item?.random_eval_group?runItems(item).filter(entry=>artifactKind(entry)===artifactKind(item)&&entry.random_eval_group===item.random_eval_group).sort((a,b)=>String(a.random_eval_map_id).localeCompare(String(b.random_eval_map_id))):[]}
function categoryTarget(item,kind){const entries=columnItems(runItems(item),kind);if(!entries.length)return null;const latest=entries.filter(entry=>navigationKey(entry)===navigationKey(entries[0]));return latest.find(entry=>entry.random_eval_map_id&&entry.random_eval_map_id===item.random_eval_map_id)||latest[0]}
function setNavigationTrack(track){keyboardTrack=track==='maps'&&$('mapNavigation').classList.contains('hidden')?'steps':track;for(const name of ['types','steps','maps'])$(name==='types'?'categoryNavigation':name==='steps'?'stepNavigation':'mapNavigation').dataset.active=String(keyboardTrack===name)}
function updateArtifactNavigation(){
    const item=currentArtifact(),groups=item?navigationGroups(item):[],index=groups.findIndex(group=>group.includes(item));
    $('previousArtifact').disabled=artifactBusy||index<=0;
    $('nextArtifact').disabled=artifactBusy||index<0||index>=groups.length-1;
    const stepChoices=$('stepChoices');
    const sameSteps=stepChoices.children.length===groups.length&&groups.every((group,i)=>stepChoices.children[i].dataset.stepKey===navigationKey(group[0]));
    if(!sameSteps){stepChoices.replaceChildren();for(const group of groups){const button=document.createElement('button');button.type='button';button.dataset.stepKey=navigationKey(group[0]);stepChoices.appendChild(button)}}
    groups.forEach((group,i)=>{const button=stepChoices.children[i],entry=group[0],detail=artifactParts(entry.label).detail;button.textContent=detail.split(' · ')[0];button.title=detail;button.setAttribute('aria-label','Step '+detail);button.setAttribute('aria-pressed',String(group.includes(item)));button.disabled=artifactBusy;button.onclick=()=>{setNavigationTrack('steps');chooseArtifact(group.find(candidate=>candidate.random_eval_map_id&&candidate.random_eval_map_id===item.random_eval_map_id)||entry,true)};if(group.includes(item)&&button.scrollIntoView)button.scrollIntoView({block:'nearest',inline:'nearest'})});
    for(const kind of ['heatmap','replay']){const button=$(kind+'Mode');button.setAttribute('aria-pressed',String(!!item&&artifactKind(item)===kind));button.disabled=artifactBusy||!item||!categoryTarget(item,kind)}
    $('previousCategory').disabled=artifactBusy||!item||artifactKind(item)!=='replay'||!categoryTarget(item,'heatmap');
    $('nextCategory').disabled=artifactBusy||!item||artifactKind(item)!=='heatmap'||!categoryTarget(item,'replay');
    const maps=mapItems(item),mapIndex=maps.indexOf(item),showMaps=maps.length>1;
    $('inspectorHeader').classList.toggle('has-map-navigation',showMaps);
    $('mapNavigation').classList.toggle('hidden',!showMaps);
    if(!showMaps&&keyboardTrack==='maps')keyboardTrack='steps';
    setNavigationTrack(keyboardTrack);
    $('previousMap').disabled=artifactBusy||mapIndex<=0;
    $('nextMap').disabled=artifactBusy||mapIndex<0||mapIndex>=maps.length-1;
    const choices=$('mapChoices');
    const sameMaps=choices.children.length===maps.length&&maps.every((map,i)=>choices.children[i].dataset.mapId===String(map.id));
    if(!sameMaps){choices.replaceChildren();for(const map of maps){const button=document.createElement('button');button.type='button';button.dataset.mapId=String(map.id);choices.appendChild(button)}}
    maps.forEach((map,i)=>{const button=choices.children[i];button.textContent=String(map.random_eval_map_id).replace(/^random_eval_/,'Map ');button.title=map.random_eval_map_id;button.setAttribute('aria-label','Map '+map.random_eval_map_id);button.setAttribute('aria-pressed',String(map===item));button.disabled=artifactBusy;button.onclick=()=>{setNavigationTrack('maps');chooseArtifact(map,true)};if(map===item&&button.scrollIntoView)button.scrollIntoView({block:'nearest',inline:'nearest'})});
    $('refreshRun').disabled=artifactBusy||!item;$('refreshReplays').disabled=artifactBusy;
    const loadMore=$('artifactMenu').querySelector?.('.artifact-load-more');if(loadMore)loadMore.disabled=artifactBusy;
    if(item){$('artifactPickerButton').textContent=artifactParts(item.label).run+' · '+artifactDetail(item);$('artifactPickerButton').title=item.label}else $('artifactPickerButton').textContent='No artifacts found';
}
async function chooseArtifact(item,keepCamera=false){if(artifactBusy)return;const previous=currentArtifact();$('replaySelect').value=item.id;$('artifactPicker').open=false;artifactBusy=true;updateArtifactNavigation();try{await loadReplay(item.id,keepCamera)}catch(error){if(previous)$('replaySelect').value=previous.id;$('replayDiscovery').textContent=error.message}finally{artifactBusy=false;renderReplayList(artifactItems,$('replaySelect').value)}}
function moveCategory(direction){const item=currentArtifact();if(!item||artifactBusy)return;const kind=artifactKind(item),nextKind=direction<0&&kind==='replay'?'heatmap':direction>0&&kind==='heatmap'?'replay':null;if(nextKind){const next=categoryTarget(item,nextKind);if(next)chooseArtifact(next,true)}}
function moveArtifact(direction){const item=currentArtifact();if(!item||artifactBusy)return;const groups=navigationGroups(item),index=groups.findIndex(group=>group.includes(item)),next=groups[index+direction];if(next)chooseArtifact(next.find(entry=>entry.random_eval_map_id&&entry.random_eval_map_id===item.random_eval_map_id)||next[0],true)}
function moveMap(direction){const item=currentArtifact();if(!item||artifactBusy)return;const maps=mapItems(item),next=maps[maps.indexOf(item)+direction];if(next)chooseArtifact(next,true)}
function artifactRows(group){const left=columnItems(group,'heatmap'),right=columnItems(group,'replay'),rows=left.map(item=>[item,null]);let cursor=0;for(const item of right){const step=artifactSteps(item),match=step===null?-1:left.findIndex(other=>artifactSteps(other)===step);const row=Math.max(cursor,match<0?cursor:match);while(rows.length<=row)rows.push([null,null]);rows[row][1]=item;cursor=row+1}return rows}
function openArtifactSubmenu(group,anchor){
    const panel=$('artifactSubmenu');panel.replaceChildren();
    $('artifactMenu').querySelectorAll('button').forEach(b=>b.classList.toggle('selected',b===anchor));
    const heading=document.createElement('div');heading.className='label';heading.textContent=artifactParts(group[0].label).run;panel.appendChild(heading);
    if(group.every(item=>artifactKind(item)==='roadmap')){
        group.forEach(item=>{const button=document.createElement('button');button.type='button';button.className='artifact-entry';button.textContent=artifactDetail(item);button.title=item.label;button.classList.toggle('selected',String(item.id)===$('replaySelect').value);button.onclick=()=>chooseArtifact(item);panel.appendChild(button)});
        return;
    }
    const columns=document.createElement('div');columns.className='artifact-columns';panel.appendChild(columns);
    for(const title of ['Heatmaps','Replays']){const label=document.createElement('div');label.className='label artifact-column-title';label.textContent=title;columns.appendChild(label)}
    const addItem=(item,parent)=>{const cell=document.createElement('div');cell.className='artifact-cell';parent.appendChild(cell);if(!item)return;const button=document.createElement('button');button.type='button';button.className='artifact-entry';button.textContent=artifactDetail(item);button.title=item.label;button.classList.toggle('selected',String(item.id)===$('replaySelect').value);button.onclick=()=>chooseArtifact(item);cell.appendChild(button);if(['heatmap','replay'].includes(artifactKind(item))){const star=document.createElement('button');star.type='button';const starred=artifactStars.has(item.id);star.className='artifact-star'+(starred?' starred':'');star.textContent=starred?'★':'☆';star.title=starred?'Unstar item':'Star item';star.setAttribute('aria-label',star.title+': '+button.textContent);star.setAttribute('aria-pressed',String(starred));star.onclick=()=>{if(starred)artifactStars.delete(item.id);else artifactStars.add(item.id);try{localStorage.setItem('swarmecho.inspector.stars',JSON.stringify([...artifactStars]))}catch{$('replayDiscovery').textContent='Stars saved for this session only'}renderReplayList(artifactItems,$('replaySelect').value,artifactRun(item))};cell.appendChild(star)}};
    const collapsed=[],seen=new Set();for(const item of group){const key=item.random_eval_group?artifactKind(item)+':'+item.random_eval_group:null;if(key&&seen.has(key))continue;if(key)seen.add(key);collapsed.push(item)}
    artifactRows(collapsed).forEach(row=>row.forEach(item=>{
        if(!item?.random_eval_group){addItem(item,columns);return}
        const siblings=group.filter(other=>other.random_eval_group===item.random_eval_group&&artifactKind(other)===artifactKind(item));
        const cell=document.createElement('div');cell.className='artifact-cell artifact-map-cell';
        const detail=artifactParts(item.label).detail;
        const select=document.createElement('select');select.setAttribute('aria-label','Random evaluation map for '+detail);
        siblings.sort((a,b)=>a.random_eval_map_id.localeCompare(b.random_eval_map_id));
        for(const entry of siblings)select.add(new Option(entry.random_eval_map_id,entry.id));
        const current=currentArtifact();if(siblings.includes(current))select.value=current.id;
        const button=document.createElement('button');button.type='button';button.className='artifact-entry';
        button.classList.toggle('selected',siblings.includes(current));
        button.textContent=detail.split(' · ')[0];
        button.title=detail+' · Random maps ('+siblings.length+')';button.setAttribute('aria-label',button.title);
        button.onclick=()=>chooseArtifact(siblings.find(v=>String(v.id)===select.value)||item);
        select.onchange=()=>chooseArtifact(siblings.find(v=>String(v.id)===select.value));
        cell.append(button,select);columns.appendChild(cell);
    }));
    group.filter(item=>!['heatmap','replay'].includes(artifactKind(item))).forEach(item=>addItem(item,panel));
}
function renderReplayList(items,preferredId,openRun){
    artifactItems=items;const s=$('replaySelect'),menu=$('artifactMenu'),groups=new Map();s.replaceChildren();menu.replaceChildren();$('artifactSubmenu').replaceChildren();
    items.forEach(item=>{const option=document.createElement('option');option.value=item.id;option.textContent=item.label;s.appendChild(option);const key=artifactRun(item);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(item)});
    const first=items.find(x=>x.selected)||items[0];
    const choice=items.find(x=>String(x.id)===String(preferredId))||(first&&columnItems(runItems(first),artifactKind(first))[0])||first;
    if(choice)s.value=choice.id;
    groups.forEach((group,key)=>{const button=document.createElement('button');button.type='button';button.className='artifact-entry'+(group.length>1?' group':'');const label=document.createElement('span'),count=document.createElement('span');label.textContent=artifactParts(group[0].label).run;count.className='artifact-count';count.textContent=group.length;if(group.some(item=>artifactStars.has(item.id))){const star=document.createElement('span');star.className='run-star';star.textContent='★';star.title='Contains starred items';count.prepend(star)}button.append(label,count);button.onclick=()=>openArtifactSubmenu(group,button);menu.appendChild(button);if(openRun?key===openRun:choice&&group.includes(choice))openArtifactSubmenu(group,button)});
    if(moreRunsAvailable){const button=document.createElement('button');button.type='button';button.className='artifact-load-more';button.textContent='Load more runs and tests';button.disabled=artifactBusy;button.onclick=()=>refreshArtifacts(false,true);menu.appendChild(button)}
    updateArtifactNavigation();return Promise.resolve();
}
document.addEventListener('click',event=>{if(!$('artifactPicker').contains(event.target))$('artifactPicker').open=false});
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&$('artifactPicker').open){$('artifactPicker').open=false;$('artifactPickerButton').focus()}if(event.defaultPrevented||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||event.target.closest('input,select,textarea,[contenteditable="true"]')||$('artifactPicker').open)return;if(event.key==='ArrowUp'||event.key==='ArrowDown'){const tracks=$('mapNavigation').classList.contains('hidden')?['types','steps']:['types','steps','maps'],index=tracks.indexOf(keyboardTrack),next=tracks[Math.max(0,Math.min(tracks.length-1,index+(event.key==='ArrowUp'?-1:1)))];event.preventDefault();setNavigationTrack(next)}else if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();(keyboardTrack==='types'?moveCategory:keyboardTrack==='maps'?moveMap:moveArtifact)(event.key==='ArrowLeft'?-1:1)}});
async function refreshArtifacts(onlyRun=false,allRuns=false){
    if(artifactBusy)return;
    const selected=currentArtifact();if(onlyRun&&!selected)return;
    artifactBusy=true;updateArtifactNavigation();setLoading(true,onlyRun?'Finding run entries...':allRuns||showingAllRuns?'Finding all entries...':'Finding recent runs...');
    try{
        const url=onlyRun?'/api/replays?run='+encodeURIComponent(selected.run_id):allRuns||showingAllRuns?'/api/replays':'/api/replays?recent=3';
        const response=await fetch(url,{cache:'no-store'});
        if(!response.ok)throw Error('Artifact list failed to load');
        const items=await response.json();let merged=items;
        if(!onlyRun){showingAllRuns=allRuns||showingAllRuns;moreRunsAvailable=!showingAllRuns&&response.headers.get('X-Inspector-Has-More')==='true'}
        if(onlyRun){
            merged=[];let inserted=false;
            for(const item of artifactItems){
                if(artifactRun(item)===artifactRun(selected)){if(!inserted){merged.push(...items);inserted=true}}
                else merged.push(item);
            }
        }
        const preferred=items.find(item=>item.id===selected?.id)||(onlyRun?items[0]:selected);
        await renderReplayList(merged,preferred?.id);
        if(onlyRun&&!items.length){$('replaySelect').value='';updateArtifactNavigation();$('replayDiscovery').textContent='No entries remain in this run';return}
        const choice=currentArtifact();
        // Discovery preserves the current scene, camera and playback position.
        // Load only on initial discovery or when the previous item disappeared.
        if(choice&&(!(D||H)||choice.id!==selected?.id))await loadReplay(choice.id,!!selected);
        $('replayDiscovery').textContent=items.length+' entries'+(onlyRun?' in current run':showingAllRuns?' across all runs':' in recent runs');
    }catch(error){$('replayDiscovery').textContent=error.message}
    finally{artifactBusy=false;setLoading(false);updateArtifactNavigation()}
}
function refreshReplays(){return refreshArtifacts(false,true)}
function loadSelectedReplay(){const item=currentArtifact();if(item)return chooseArtifact(item)}
$('replaySelect').onchange=loadSelectedReplay;$('refreshRun').onclick=()=>refreshArtifacts(true);$('refreshReplays').onclick=refreshReplays;$('categoryNavigation').onfocusin=()=>setNavigationTrack('types');$('stepNavigation').onfocusin=()=>setNavigationTrack('steps');$('mapNavigation').onfocusin=()=>setNavigationTrack('maps');$('previousCategory').onclick=()=>{setNavigationTrack('types');moveCategory(-1)};$('nextCategory').onclick=()=>{setNavigationTrack('types');moveCategory(1)};$('previousArtifact').onclick=()=>{setNavigationTrack('steps');moveArtifact(-1)};$('nextArtifact').onclick=()=>{setNavigationTrack('steps');moveArtifact(1)};$('previousMap').onclick=()=>{setNavigationTrack('maps');moveMap(-1)};$('nextMap').onclick=()=>{setNavigationTrack('maps');moveMap(1)};for(const kind of ['heatmap','replay'])$(kind+'Mode').onclick=()=>{setNavigationTrack('types');const item=currentArtifact();if(item&&artifactKind(item)!==kind){const next=categoryTarget(item,kind);if(next)chooseArtifact(next,true)}};refreshArtifacts();
function lineTrace(points,color,width=3){return {type:'scatter3d',mode:'lines',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),line:{color,width},hoverinfo:'skip'}}
function boxTrace(s){let [x,y,z]=s,p=[[0,0,0],[x,0,0],[x,y,0],[0,y,0],[0,0,0],[0,0,z],[x,0,z],[x,y,z],[0,y,z],[0,0,z],[null,null,null],[x,0,0],[x,0,z],[null,null,null],[x,y,0],[x,y,z],[null,null,null],[0,y,0],[0,y,z]];return lineTrace(p,'rgba(120,155,205,.48)',2)}
const CUBE_FACES=[[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]];
function coverageTrace(coverage,cellSize,opacity){let x=[],y=[],z=[],i=[],j=[],k=[];for(let a=0;a<coverage.length;a++)for(let b=0;b<coverage[a].length;b++)for(let c=0;c<coverage[a][b].length;c++)if(coverage[a][b][c]){let o=x.length,px=a*cellSize,py=b*cellSize,pz=c*cellSize;[[px,py,pz],[px+cellSize,py,pz],[px+cellSize,py+cellSize,pz],[px,py+cellSize,pz],[px,py,pz+cellSize],[px+cellSize,py,pz+cellSize],[px+cellSize,py+cellSize,pz+cellSize],[px,py+cellSize,pz+cellSize]].forEach(v=>{x.push(v[0]);y.push(v[1]);z.push(v[2])});CUBE_FACES.forEach(q=>{i.push(o+q[0]);j.push(o+q[1]);k.push(o+q[2])})}return x.length?{type:'mesh3d',x,y,z,i,j,k,color:'#38bdf8',opacity,flatshading:true,hoverinfo:'skip',lighting:{ambient:.8,diffuse:.25,specular:0}}:null}
function sphereTrace(centres,radius,color,opacity){let x=[],y=[],z=[],i=[],j=[],k=[],lon=12,lat=8;centres.forEach(centre=>{let o=x.length;for(let row=0;row<=lat;row++)for(let col=0;col<=lon;col++){let theta=Math.PI*row/lat,phi=2*Math.PI*col/lon;x.push(centre[0]+radius*Math.sin(theta)*Math.cos(phi));y.push(centre[1]+radius*Math.sin(theta)*Math.sin(phi));z.push(centre[2]+radius*Math.cos(theta))}for(let row=0;row<lat;row++)for(let col=0;col<lon;col++){let a=o+row*(lon+1)+col,b=a+lon+1;i.push(a,b,a+1);j.push(b,b+1,b);k.push(a+1,a+1,b+1)}});return {type:'mesh3d',x,y,z,i,j,k,color,opacity,flatshading:true,hoverinfo:'skip',lighting:{ambient:1,diffuse:0,specular:0}}}
const HEATMAP_COLORS={chain_success:'#4ade80',found_and_delivered:'#60a5fa',visually_found:'#fcd34d',not_found:'#fb7185'};
const HEATMAP_CATEGORY_CONTROLS={chain_success:'heatmapShowChainSuccess',found_and_delivered:'heatmapShowFoundDelivered',visually_found:'heatmapShowVisuallyFound',not_found:'heatmapShowNotFound'};
let heatmapAgreementData=null;
function renderHeatmapAgreementButtons(){let buttons=$('heatmapAgreementButtons');buttons.replaceChildren();let total=Number(H?.manifest?.robustness_runs)||1;if(heatmapAgreementData!==H){heatmapAgreementData=H;heatmapAgreement=total}for(let agreement=1;agreement<=total;agreement++){let button=document.createElement('button');button.type='button';button.textContent=agreement+'/'+total;button.classList.toggle('selected',agreement===heatmapAgreement);button.onclick=()=>{heatmapAgreement=agreement;renderHeatmapAgreementButtons();updateHeatmapStats();drawHeatmap()};buttons.appendChild(button)}}
function updateHeatmapStats(){let counts={chain_success:0,found_and_delivered:0,visually_found:0,not_found:0};H.positions.forEach((_,index)=>counts[heatmapStage(index)]++);let total=H.positions.length||1,chain=counts.chain_success/total,delivered=(counts.chain_success+counts.found_and_delivered)/total,visual=(counts.chain_success+counts.found_and_delivered+counts.visually_found)/total,percent=value=>Math.round(value*100)+'%',delta=value=>(value>=0?'+':'')+Math.round(value*100)+'%';$('heatmapStatChain').textContent=percent(chain);$('heatmapStatDelivered').textContent=percent(delivered);$('heatmapStatDeliveredDelta').textContent='('+delta(delivered-chain)+')';$('heatmapStatVisual').textContent=percent(visual);$('heatmapStatVisualDelta').textContent='('+delta(visual-chain)+')';$('heatmapStatNotFound').textContent=percent(counts.not_found/total)}
function showHeatmapCommand(point,stage){let checkpoint=H.manifest.checkpoint||'<checkpoint-path>';let coords=point.map(value=>Number(value).toPrecision(9)).join(',');let command='uv run swarmecho-evaluate checkpoint='+checkpoint+' mode=selective_manual_pick target_position='+coords+(H.manifest.random_eval_map_id?' random_eval_map_id='+H.manifest.random_eval_map_id:'');$('selectedPoint').textContent=stage+' · ('+coords+')';$('heatmapCommand').value=command}
function heatmapStage(index){if(!H.stage_rates)return H.stages[index];let total=Number(H.manifest.robustness_runs)||1,confidence=Math.min(total,Math.max(1,heatmapAgreement))/total,epsilon=1e-9;if(H.stage_rates.chain_success[index]+epsilon>=confidence)return 'chain_success';if(H.stage_rates.found_and_delivered[index]+epsilon>=confidence)return 'found_and_delivered';if(H.stage_rates.visually_found[index]+epsilon>=confidence)return 'visually_found';return 'not_found'}
function drawHeatmap(){let traces=buildingTraces(),groups={chain_success:[],found_and_delivered:[],visually_found:[],not_found:[]};H.positions.forEach((point,index)=>{let stage=heatmapStage(index),rates=H.stage_rates?{chain:H.stage_rates.chain_success[index],delivered:H.stage_rates.found_and_delivered[index],visual:H.stage_rates.visually_found[index]}:{chain:Number(stage==='chain_success'),delivered:Number(stage==='chain_success'||stage==='found_and_delivered'),visual:Number(stage!=='not_found')};(groups[stage]||groups.not_found).push({point,stage,rates})});if(H.manifest.world_size_m)traces.push(boxTrace(H.manifest.world_size_m));if(H.obstacle_min?.length&&H.manifest.obstacle_layout_mode==='fixed')H.obstacle_min[0].forEach((lo,i)=>{traces.push(cuboidMesh(lo,H.obstacle_max[0][i]));traces.push(cuboidLines(lo,H.obstacle_max[0][i],'rgba(255,190,100,.8)'))});Object.entries(groups).forEach(([stage,entries])=>{if(!entries.length||!$(HEATMAP_CATEGORY_CONTROLS[stage]).checked)return;let points=entries.map(entry=>entry.point),color=HEATMAP_COLORS[stage];traces.push({type:'scatter3d',mode:'markers',x:points.map(point=>point[0]),y:points.map(point=>point[1]),z:points.map(point=>point[2]),customdata:entries.map(entry=>[entry.point,entry.stage,entry.rates.chain,entry.rates.delivered,entry.rates.visual,(H.final_chain_length||[])[H.positions.indexOf(entry.point)]||0]),marker:{size:8,color,opacity:.55,line:{color:'#e5eefc',width:1}},hovertemplate:stage+'<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>chain success=%{customdata[2]:.0%}<br>found and delivered=%{customdata[3]:.0%}<br>visually found=%{customdata[4]:.0%}<br>final chain length=%{customdata[5]:.2f}m<extra>Click to create replay command</extra>'})});return renderPlot(traces,'heatmap-camera',{uirevision:'heatmap-camera',showlegend:false})}
function updateElevation(){let eye=(camera||DEFAULT_CAMERA).eye,angle=Math.round(Math.atan2(eye.z,Math.hypot(eye.x,eye.y))*180/Math.PI);$('cameraElevation').value=angle;$('cameraElevationValue').textContent=angle+'°'}
function captureCamera(){rememberCamera();updateElevation();let scene=$('scene');if(!scene.on)return;if(!scene.__cameraListener){scene.on('plotly_relayout',event=>{if(event['scene.camera']){camera=event['scene.camera'];updateElevation()}});scene.__cameraListener=true}if(!scene.__heatmapClickListener){scene.on('plotly_click',event=>{if(mode!=='heatmap')return;let point=event.points.find(item=>item.customdata);if(point)showHeatmapCommand(point.customdata[0],point.customdata[1])});scene.__heatmapClickListener=true}updateAutoRotate()}
function rememberCamera(){let current=$('scene')._fullLayout?.scene?.camera;if(current)camera=structuredClone(current)}
function rotateCamera(timestamp){if(!$('autoRotate').checked){rotationFrame=null;rotationTime=null;return}if(rendering){rotationTime=null;rotationFrame=requestAnimationFrame(rotateCamera);return}if(rotationTime===null)rotationTime=timestamp;let elapsed=Math.min((timestamp-rotationTime)/1000,.1),speed=+$('rotateSpeed').value*Math.PI/180,current=camera||structuredClone(DEFAULT_CAMERA),eye=current.eye||DEFAULT_CAMERA.eye,angle=speed*elapsed,cos=Math.cos(angle),sin=Math.sin(angle);camera={...current,eye:{x:eye.x*cos-eye.y*sin,y:eye.x*sin+eye.y*cos,z:eye.z}};rotationTime=timestamp;Plotly.relayout('scene',{'scene.camera':camera});rotationFrame=requestAnimationFrame(rotateCamera)}
function updateAutoRotate(){if($('autoRotate').checked&&!rotationFrame){rotationTime=null;rotationFrame=requestAnimationFrame(rotateCamera)}else if(!$('autoRotate').checked&&rotationFrame){cancelAnimationFrame(rotationFrame);rotationFrame=null;rotationTime=null}}
function distance3(a,b){return Math.hypot(...a.map((v,k)=>v-b[k]))}
function segmentBlocked(a,b,mins,maxs){return mins.some((lo,q)=>{let hi=maxs[q],near=0,far=1;for(let k=0;k<3;k++){let d=b[k]-a[k];if(Math.abs(d)<1e-9){if(a[k]<lo[k]||a[k]>hi[k])return false;continue}let x=(lo[k]-a[k])/d,y=(hi[k]-a[k])/d;near=Math.max(near,Math.min(x,y));far=Math.min(far,Math.max(x,y));if(far<near)return false}return far>=near&&far>1e-6&&near<1-1e-6})}
let roadmapUiData=null,roadmapLayoutIndex=null,roadmapPaths=[],roadmapPairMode=false;
function updateRoadmapRanges(){
    const comm=Number($('roadmapCommRange').value);
    $('roadmapRoutes').querySelectorAll('[data-distance]').forEach(cell=>{cell.textContent=Number.isFinite(comm)&&comm>0?'['+(Number(cell.dataset.distance)/comm).toFixed(2)+']':'[—]'});
}
function ensureRoadmapUi(){
    if(!$('roadmapControls')){const card=document.createElement('div');card.id='roadmapControls';card.className='card';card.innerHTML='<div class="label">Roadmap paths</div><label>Layout <select id="roadmapLayout"></select></label><div class="row" style="margin:14px 0"><span>Roadmap nodes</span><b id="roadmapNodeCount" style="font-size:24px">—</b></div><label><input id="showRoadmapEdges" type="checkbox"> Visibility edges</label><br><label><input id="showRoadmapNodes" type="checkbox" checked> Vertices / nodes</label><p class="meta">Ranked by length in metres (lower is better). Differences are relative to the shortest path, not episode rewards.</p><div id="roadmapRoutes"></div>';document.querySelector('#scene + .panel').prepend(card);$('roadmapLayout').onchange=drawRoadmap;['showRoadmapEdges','showRoadmapNodes'].forEach(id=>$(id).onchange=drawRoadmap)}
    if(!$('roadmapCommRange')){const label=document.createElement('label');label.style.display='block';label.textContent='Communication range (m) ';const input=document.createElement('input');input.id='roadmapCommRange';input.type='number';input.min='0.01';input.step='any';input.value='50';input.style.width='80px';input.oninput=updateRoadmapRanges;label.appendChild(input);$('roadmapControls').insertBefore(label,$('roadmapControls').querySelector('p.meta'))}
    if(roadmapUiData!==D){roadmapUiData=D;roadmapLayoutIndex=null;$('roadmapLayout').replaceChildren();D.layouts.forEach((_,i)=>$('roadmapLayout').add(new Option('Layout '+(i+1),i)));$('roadmapLayout').value='0'}
    const index=+$('roadmapLayout').value||0;
    if(roadmapLayoutIndex!==index){
        roadmapLayoutIndex=index;const layout=D.layouts[index];$('roadmapNodeCount').textContent=layout.vertices.length.toLocaleString();roadmapPairMode=Array.isArray(layout.pairs);
        roadmapPaths=roadmapPairMode?layout.pairs.map((pair,i)=>{const path=pair.path,physical=path.slice(1).reduce((sum,p,j)=>sum+distance3(p,path[j]),0);return {path,physical,length:pair.path_length_m??physical,corners:pair.path_corner_count??null,base:pair.base,target:pair.target,rank:pair.rank??i+1}}):layout.paths.map((path,i)=>{const physical=path.slice(1).reduce((sum,p,j)=>sum+distance3(p,path[j]),0),length=layout.path_lengths_m?.[i]??physical,bonus=Number(D.manifest.roadmap_corner_bonus_m),corners=layout.path_corner_counts?.[i]??(bonus>0?Math.max(0,Math.round((length-physical)/bonus)):null);return {path,physical,length,corners}}).sort((a,b)=>a.length-b.length);
        $('roadmapControls').querySelector('p.meta').textContent=roadmapPairMode?'Longest farthest-point pairs, ranked by distance. Each checkbox shows one path and its B/T endpoints. '+(D.manifest.path_metric||'Distance in metres'):(D.manifest.path_metric||'Visibility-graph length (m); lower is better')+'. Differences use this score, including any configured corner allowance.';
        const panel=$('roadmapRoutes');panel.replaceChildren();
        if(!roadmapPaths.length)panel.textContent='No route found in this visibility graph.';
        roadmapPaths.forEach(({length,physical,corners},i)=>{const label=document.createElement('label'),input=document.createElement('input'),text=document.createElement('span');label.style.display='block';label.style.margin='10px 0';label.style.color='hsl('+(70+i*55)+' 85% 70%)';input.type='checkbox';input.id='showRoute'+i;input.checked=true;input.onchange=drawRoadmap;
            const best=roadmapPaths[0].length,delta=length-best; text.textContent=roadmapPairMode?' #'+roadmapPaths[i].rank+' · '+length.toFixed(2)+' m':' #'+(i+1)+(i?' · +'+delta.toFixed(2)+' m vs best':' · shortest');label.append(input,text);
            if(roadmapPairMode){const pair=roadmapPaths[i],where=document.createElement('span'),coords=p=>p.map(v=>Number(v).toFixed(1)).join(', ');where.style.cssText='display:block;margin:4px 0 0 20px;font-size:12px';where.textContent='B ('+coords(pair.base)+') → T ('+coords(pair.target)+')';label.appendChild(where)}
            const breakdown=document.createElement('span');breakdown.style.cssText='display:grid;grid-template-columns:minmax(0,1fr) auto auto;column-gap:8px;row-gap:5px;margin-top:8px;font-size:12px;font-variant-numeric:tabular-nums';
            const allowance=length-physical,rate=Number(D.manifest.roadmap_corner_bonus_m);
            [['raw distance',physical.toFixed(2)+' m',physical],['+corner bonus',(corners!==null&&Number.isFinite(rate)?corners+'×'+rate+'m = ':'')+allowance.toFixed(2)+' m',allowance],['total',length.toFixed(2)+' m',length]].forEach(([name,amount,value],row)=>{
                [name,amount,''].forEach((display,column)=>{const cell=document.createElement('span');cell.textContent=display;cell.style.whiteSpace='nowrap';if(column)cell.style.textAlign='right';if(row===2){cell.style.borderTop='1px solid currentColor';cell.style.paddingTop='6px';cell.style.fontWeight='600'}if(column===2&&row!==1){cell.dataset.distance=value;cell.title='Drone communication ranges'}breakdown.appendChild(cell)});
            });label.appendChild(breakdown);panel.appendChild(label)});
        updateRoadmapRanges();
    }
}
function cuboidLines(lo,hi,color){let c=[[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],[hi[0],hi[1],lo[2]],[lo[0],hi[1],lo[2]],[lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],[hi[0],hi[1],hi[2]],[lo[0],hi[1],hi[2]]],e=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]],p=[];e.forEach(q=>p.push(c[q[0]],c[q[1]],null));return {type:'scatter3d',mode:'lines',x:p.map(v=>v&&v[0]),y:p.map(v=>v&&v[1]),z:p.map(v=>v&&v[2]),line:{color,width:7},hoverinfo:'skip'}}
function cuboidMesh(lo,hi){let p=[[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],[hi[0],hi[1],lo[2]],[lo[0],hi[1],lo[2]],[lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],[hi[0],hi[1],hi[2]],[lo[0],hi[1],hi[2]]];return {type:'mesh3d',x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),i:CUBE_FACES.map(v=>v[0]),j:CUBE_FACES.map(v=>v[1]),k:CUBE_FACES.map(v=>v[2]),color:'#f97316',opacity:.5,flatshading:true,hoverinfo:'skip',lighting:{ambient:.45,diffuse:.75,specular:.25,roughness:.65,fresnel:.08},lightposition:{x:80,y:40,z:120}}}
function roadmapBuildingCount(layout,g){
    if(!g?.solid_min||g.solid_min.length>layout.obstacle_min.length)return null;
    const key=(lo,hi)=>[...lo,...hi].map(value=>Math.round(value*1000)).join(',');
    const expected=g.solid_min.map((lo,i)=>key(lo,g.solid_max[i])).sort();
    const actual=layout.obstacle_min.slice(0,expected.length).map((lo,i)=>key(lo,layout.obstacle_max[i])).sort();
    return expected.every((value,i)=>value===actual[i])?expected.length:null;
}
function roadmapOrangeTraces(layout,start=0,buildingCount=0){
    const tr=[],g=D.building,cell=g?.cell_size_m,opacity=1-(+$('wallTransparency').value/100);
    if(opacity<=0)return tr;
    const boxKey=(lo,hi)=>[...lo,...hi].map(value=>Math.round(value*1000)).join(',');
    const boxStoreys=new Map([...(g?.walls||[]),...(g?.floors||[])].map(({lo,hi,storey})=>[boxKey(lo,hi),storey]));
    const add=(lo,hi)=>{const mesh=cuboidMesh(lo,hi);mesh.opacity=opacity;tr.push(mesh);if($('showCellGrid').checked){const lines=cuboidLines(lo,hi,'rgba(255,190,100,.8)');lines.opacity=opacity;tr.push(lines)}};
    layout.obstacle_min.forEach((lo,i)=>{
        if(i<start)return;
        const hi=layout.obstacle_max[i],key=boxKey(lo,hi),storey=boxStoreys.get(key)??(cell?Math.max(0,Math.min(g.layers-1,Math.floor((lo[2]+hi[2])/(2*cell)))):0);
        if(g&&!visibleStoreys[storey])return;
        add(lo,hi);
    });
    if(start===0&&buildingCount&&g){
        if($('showOuterWalls').checked)g.walls.forEach(({lo,hi,storey,outer})=>{if(outer&&visibleStoreys[storey])add(lo,hi)});
        if($('showRoof').checked)g.roofs.forEach(({lo,hi,storey})=>{if(visibleStoreys[storey])add(lo,hi)});
    }
    return tr;
}
function drawRoadmap(){
    ensureRoadmapUi();const index=+$('roadmapLayout').value||0,layout=D.layouts[index],tr=[boxTrace(D.manifest.world_size_m)];
    const buildingCount=roadmapBuildingCount(layout,D.building);
    const orangeControl=$('roadmapOrangeGeometry');
    if(buildingCount===null){orangeControl.checked=true;orangeControl.disabled=true}
    else if(orangeControl.disabled){orangeControl.checked=false;orangeControl.disabled=false}
    $('buildingSource').textContent=buildingCount===null?'Building geometry unavailable or does not match this roadmap; showing orange obstacle boxes.':'Geometry: '+D.building.source+'. Visibility controls do not affect communication checks.';
    if($('roadmapOrangeGeometry').checked||buildingCount===null)tr.push(...roadmapOrangeTraces(layout,0,buildingCount||0));
    else{tr.push(...buildingTraces());tr.push(...roadmapOrangeTraces(layout,buildingCount,buildingCount))}
    if($('showRoadmapEdges').checked){const points=[];layout.edges.forEach(e=>points.push(layout.vertices[e[0]],layout.vertices[e[1]],[null,null,null]));tr.push(lineTrace(points,'rgba(34,211,238,.2)',2))}
    if($('showRoadmapNodes').checked)tr.push({type:'scatter3d',mode:'markers',x:layout.vertices.map(v=>v[0]),y:layout.vertices.map(v=>v[1]),z:layout.vertices.map(v=>v[2]),marker:{size:4,color:'#22d3ee'}});
    if(!roadmapPairMode)tr.push({type:'scatter3d',mode:'markers+text',x:[layout.base[0],layout.target[0]],y:[layout.base[1],layout.target[1]],z:[layout.base[2],layout.target[2]],text:['START','TARGET'],marker:{size:9,color:['#60a5fa','#fb7185']}});
    roadmapPaths.forEach(({path,length,base,target,rank},i)=>{if($('showRoute'+i).checked){const color='hsl('+(70+i*55)+' 85% 60%)',trace=lineTrace(path,color,7);trace.name='Path '+(rank??i+1)+' · '+length.toFixed(2)+' m';tr.push(trace);if(roadmapPairMode)for(const [point,label,symbol] of [[base,'B'+rank,'circle'],[target,'T'+rank,'diamond']])tr.push({type:'scatter3d',mode:'markers+text',x:[point[0]],y:[point[1]],z:[point[2]],text:[label],textposition:'top center',textfont:{color,size:11},marker:{size:9,color,symbol},hovertemplate:label+'<extra></extra>'})}});
    return renderPlot(tr,'roadmap-camera',{showlegend:false});
}
let visibleStoreys=[],buildingVisibilityKey=null;
function setupBuildingVisibility(keep=false){
    const artifact=D||H,g=artifact.building,key=artifact.manifest.map_name+':'+(g?.layers||0),panel=$('storeyVisibility');
    if(keep&&key===buildingVisibilityKey)return;
    buildingVisibilityKey=key;visibleStoreys=Array.from({length:g?.layers||0},()=>true);panel.replaceChildren();
    visibleStoreys.forEach((checked,z)=>{const label=document.createElement('label'),input=document.createElement('input');label.className='layer-control';input.type='checkbox';input.checked=checked;input.onchange=()=>{visibleStoreys[z]=input.checked;redrawBuildingScene()};label.append(input,document.createTextNode('Storey '+(z+1)));panel.appendChild(label)});
    $('showOuterWalls').checked=false;$('showRoof').checked=false;
    $('showOuterWalls').disabled=!g;$('showRoof').disabled=!g;
    $('buildingSource').textContent=g?'Geometry: '+g.source+'. Visibility controls do not affect communication checks.':'Map geometry unavailable; roadmap uses orange obstacle boxes.';
}
function buildingTraces(){
    const g=(D||H)?.building;if(!g)return [];
    const walls=g.walls.filter(w=>visibleStoreys[w.storey]&&(!w.outer||$('showOuterWalls').checked)),buckets=new Map();
    const solidOpacity=1-(+$('wallTransparency').value/100);
    // Batch pieces of equal opacity rather than adding a trace per tile.
    const bucket=opacity=>{const key=opacity.toFixed(4);if(!buckets.has(key))buckets.set(key,{
        mesh:{type:'mesh3d',x:[],y:[],z:[],i:[],j:[],k:[],facecolor:[],opacity,flatshading:true,hoverinfo:'skip',lighting:{ambient:.65,diffuse:.7,specular:.12,roughness:.85},lightposition:{x:80,y:40,z:120}},
        grid:{type:'scatter3d',mode:'lines',x:[],y:[],z:[],opacity,line:{color:'#73818d',width:2},hoverinfo:'skip'}});return buckets.get(key)};
    const add=(lo,hi,color,stripAxis=-1,strip=0)=>{
        const opacity=solidOpacity;
        if(opacity<=0)return;const mesh=bucket(opacity).mesh,cube=cuboidMesh(lo,hi),offset=mesh.x.length;
        for(const axis of ['x','y','z'])mesh[axis].push(...cube[axis]);
        CUBE_FACES.forEach((face,index)=>{
            // Omit internal strip faces so transparency does not expose false partitions.
            if(stripAxis>=0){const lower=stripAxis===0?10:4,upper=stripAxis===0?6:8;if((strip>0&&index>=lower&&index<lower+2)||(strip<5&&index>=upper&&index<upper+2))return}
            mesh.i.push(offset+face[0]);mesh.j.push(offset+face[1]);mesh.k.push(offset+face[2]);mesh.facecolor.push(color);
        });
    };
    const outline=(lo,hi)=>{
        if(!$('showCellGrid').checked||solidOpacity<=0)return;
        const lines=cuboidLines(lo,hi,''),grid=bucket(solidOpacity).grid;
        for(const axis of ['x','y','z'])grid[axis].push(...lines[axis]);
    };
    for(const {lo,hi} of walls){
        const axis=hi[0]-lo[0]>hi[1]-lo[1]?0:1;
        for(let i=0;i<6;i++){
            const a=[...lo],b=[...hi];a[axis]=lo[axis]+(hi[axis]-lo[axis])*i/6;b[axis]=lo[axis]+(hi[axis]-lo[axis])*(i+1)/6;
            add(a,b,['#959b9c','#9da2a2','#979d9d'][i%3],axis,i);
        }
        outline(lo,hi);
    }
    for(const {lo,hi,storey} of g.floors||[]){if(!visibleStoreys[storey])continue;const shade=(Math.round(lo[0]+lo[1])%3+3)%3;add(lo,hi,['#252e39','#2c3540','#303946'][shade]);outline(lo,hi)}
    if($('showRoof').checked)for(const {lo,hi,storey} of g.roofs){if(!visibleStoreys[storey])continue;add(lo,hi,'#78868c');outline(lo,hi)}
    return [...buckets.values()].flatMap(({mesh,grid})=>[...(mesh.x.length?[mesh]:[]),...(grid.x.length?[grid]:[])]);
}
function redrawBuildingScene(){if(mode==='heatmap'&&H)return drawHeatmap();if(mode==='roadmap'&&D)return drawRoadmap();if(mode==='replay'&&D)return draw(frame)}
function updateWallAppearance(){
    $('wallTransparencyValue').textContent=$('wallTransparency').value+'%';
    redrawBuildingScene();
}
for(const id of ['showOuterWalls','showRoof','showCellGrid'])$(id).onchange=updateWallAppearance;
$('wallTransparency').oninput=updateWallAppearance;
$('roadmapOrangeGeometry').onchange=redrawBuildingScene;
function replayLinkTraces(f,p,active,b,t,commRadius,baseCommRadius,visualRadius){
    const n=p.length,nodes=[...p,b,t],base=n,target=n+1,adj=Array.from({length:n+2},()=>Array(n+2).fill(false));
    const omin=[...(D.building?.solid_min||[]),...(D.obstacle_min?.[f]||[])],omax=[...(D.building?.solid_max||[]),...(D.obstacle_max?.[f]||[])];
    const connect=(i,j)=>{adj[i][j]=adj[j][i]=true};
    for(let i=0;i<n;i++)if(active[i]){
        for(let j=i+1;j<n;j++)if(active[j]&&distance3(p[i],p[j])<=commRadius&&!segmentBlocked(p[i],p[j],omin,omax))connect(i,j);
        if(distance3(p[i],b)<=baseCommRadius&&!segmentBlocked(b,p[i],omin,omax))connect(i,base);
        if(D.directly_sees_target?D.directly_sees_target[f][i]:distance3(p[i],t)<=visualRadius&&!segmentBlocked(p[i],t,omin,omax))connect(i,target);
    }
    const hopsFrom=start=>{const hops=Array(n+2).fill(Infinity),queue=[start];hops[start]=0;for(let k=0;k<queue.length;k++){const node=queue[k];adj[node].forEach((edge,next)=>{if(edge&&hops[next]===Infinity){hops[next]=hops[node]+1;queue.push(next)}})}return hops};
    const baseHops=hopsFrom(base),targetHops=hopsFrom(target),cb=D.connected_to_base?.[f]||baseHops.slice(0,n).map(Number.isFinite),ct=D.connected_to_target?.[f]||targetHops.slice(0,n).map(Number.isFinite);
    const complete=D.fully_connected?Boolean(D.fully_connected[f]):Number.isFinite(baseHops[target]),green=new Set(),edgeKey=(i,j)=>Math.min(i,j)+':'+Math.max(i,j);
    // Mirror the reward selector: minimum-hop paths to its two Euclidean tips,
    // with the lowest node index breaking equal-hop ties.
    const traceTip=(start,connected,goal)=>{
        let tip=-1,best=Infinity;for(let i=0;i<n;i++)if(active[i]&&connected[i]){const distance=distance3(p[i],goal);if(distance<best){best=distance;tip=i}}
        if(tip<0)return;const hops=hopsFrom(tip);let current=start;
        for(let k=0;k<n+2&&current!==tip;k++){const next=adj[current].findIndex((edge,j)=>edge&&hops[j]===hops[current]-1);if(next<0)break;green.add(edgeKey(current,next));current=next}
    };
    if(complete){
        if(D.manifest.allow_redundancy_reward){
            const visit=(node,path,seen)=>{if(node===target){for(let i=1;i<path.length;i++)green.add(edgeKey(path[i-1],path[i]));return}for(let next=0;next<n+2;next++)if(adj[node][next]&&!seen.has(next)){seen.add(next);path.push(next);visit(next,path,seen);path.pop();seen.delete(next)}};
            visit(base,[base],new Set([base]));
        }else{traceTip(base,cb,t);traceTip(target,ct,b)}
    }
    const blue='rgba(34,211,238,.35)',purple='rgba(167,139,250,.55)',pink='rgba(251,113,133,.65)',traces=[];
    for(let i=0;i<n+2;i++)for(let j=i+1;j<n+2;j++)if(adj[i][j]){
        const selected=complete&&green.has(edgeKey(i,j));let color=blue;
        if(selected)color='#a3e635';else if(!complete){if(j===target||(i<n&&j<n&&ct[i]&&ct[j]))color=pink;else if(j===base||(i<n&&j<n&&cb[i]&&cb[j]))color=purple}
        traces.push(lineTrace([nodes[i],nodes[j]],color,selected?5:3));
    }
    return traces;
}
function baseKnowsTarget(f,base){if(D.base_target_known)return Boolean(D.base_target_known[f]);for(let q=0;q<=f;q++){if(D.fully_connected?.[q])return true;if(!D.fully_connected&&D.connected_to_base&&D.connected_to_target&&D.active[q].some((on,i)=>on&&D.connected_to_base[q][i]&&D.connected_to_target[q][i]))return true}return false}
function renderKnownList(agentCount){let list=$('knownList');list.replaceChildren();['BASE',...Array.from({length:agentCount},(_,i)=>'D'+(i+1))].forEach(name=>{let item=document.createElement('button');item.type='button';item.className='known-item';item.textContent=name;list.appendChild(item)})}
function updateKnownList(agentKnown,baseKnown){let statuses=[baseKnown,...agentKnown];$('knownList').querySelectorAll('.known-item').forEach((item,i)=>item.classList.toggle('known',Boolean(statuses[i])))}
function draw(f){frame=+f;let p=D.position[f],active=D.active[f],known=D.target_known[f],tr=[],agents=p.filter((_,i)=>active[i]),b=D.base_position[f],t=D.target_position[f],visualRadius=D.manifest.visual_radius_m||4,commRadius=D.manifest.comm_radius_m||5,baseCommRadius=D.manifest.comm_radius_base_m||6,baseKnown=baseKnowsTarget(f,b);if($('showShell').checked)tr.push(boxTrace(D.manifest.world_size_m||[20,20,20]));tr.push(...buildingTraces());if(D.obstacle_min)D.obstacle_min[f].forEach((lo,i)=>{tr.push(cuboidMesh(lo,D.obstacle_max[f][i]));tr.push(cuboidLines(lo,D.obstacle_max[f][i],'rgba(255,190,100,.8)'))});if($('showVisualRange').checked&&agents.length)tr.push(sphereTrace(agents,visualRadius,'#fbbf24',+$('visualOpacity').value));if($('showCommRange').checked&&agents.length)tr.push(sphereTrace(agents,commRadius,'#a78bfa',+$('commOpacity').value));if($('showCommRange').checked)tr.push(sphereTrace([b],baseCommRadius,'#a78bfa',+$('commOpacity').value));updateKnownList(known,baseKnown);
let colors=p.map((_,i)=>known[i]?'#fb7185':'#22d3ee');tr.push({type:'scatter3d',mode:'markers+text',x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),text:p.map((_,i)=>'D'+(i+1)),textposition:'top center',marker:{size:7,color:colors,line:{color:'#e5eefc',width:1}},hovertemplate:'%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>'});
let targetPresent=D.target_present?Boolean(D.target_present[f]):true,landmarks=targetPresent?[b,t]:[b];tr.push({type:'scatter3d',mode:'markers+text',x:landmarks.map(v=>v[0]),y:landmarks.map(v=>v[1]),z:landmarks.map(v=>v[2]),text:targetPresent?['BASE','TARGET']:['BASE'],textposition:'top center',marker:{size:targetPresent?[11,9]:[11],color:targetPresent?[baseKnown?'#fb7185':'#60a5fa','#fb7185']:[baseKnown?'#fb7185':'#60a5fa'],symbol:targetPresent?['diamond','circle']:['diamond']}});
if($('showLinks').checked)tr.push(...replayLinkTraces(f,p,active,b,t,commRadius,baseCommRadius,visualRadius));
if($('showCoverage').checked){let coverage=coverageTrace(D.coverage[f],D.manifest.coverage_voxel_size_m||D.manifest.cell_size_m||5,+$('coverageOpacity').value);if(coverage)tr.push(coverage)}
const renderPromise=renderPlot(tr,'replay-camera',{uirevision:'replay-camera',showlegend:false});
$('timeline').value=f;$('frame').textContent=f+'/'+(D.manifest.frames-1);$('time').textContent=(f*D.manifest.dt).toFixed(1)+' s';let covered=D.coverage[f].flat(2),eligible=D.coverage_eligible?.flat(2),total=eligible?eligible.filter(Boolean).length:covered.length,cov=eligible?covered.filter((value,index)=>value&&eligible[index]).length:covered.filter(Boolean).length;$('coverage').textContent=(100*cov/Math.max(total,1)).toFixed(1)+'%';$('known').textContent=known.filter(Boolean).length;$('baseConn').textContent=D.connected_to_base[f].filter(Boolean).length;let successLabels={coverage:'SUCCESS: COVERAGE',discovery:'SUCCESS: VISUALLY FOUND',delivery:'SUCCESS: DELIVERED',chain_held:'SUCCESS: CHAIN HELD'};$('status').textContent=D.success[f]?(successLabels[D.manifest.success_condition]||'SUCCESS'):D.fully_connected[f]?'CHAIN HELD':known.some(Boolean)?'TARGET KNOWN':'EXPLORING';$('status').style.color=D.success[f]?'#a3e635':'#e5eefc';$('reward').textContent=D.reward_terms?D.reward_terms[f].flat().reduce((a,b)=>a+b,0).toFixed(2):'—';return renderPromise}
function updateOpacityValue(input,output){$(output).textContent=(Math.round(1000*+$(input).value)/10)+'%'}
$('timeline').oninput=e=>draw(e.target.value);['showCoverage','showVisualRange','showCommRange','showLinks','showShell'].forEach(id=>$(id).onchange=()=>draw(frame));[['coverageOpacity','coverageOpacityValue'],['visualOpacity','visualOpacityValue'],['commOpacity','commOpacityValue']].forEach(([input,output])=>{updateOpacityValue(input,output);$(input).oninput=()=>{updateOpacityValue(input,output);draw(frame)}});Object.values(HEATMAP_CATEGORY_CONTROLS).forEach(id=>$(id).onchange=()=>{if(mode==='heatmap')drawHeatmap()});$('copyHeatmapCommand').onclick=()=>{let command=$('heatmapCommand').value;if(command)navigator.clipboard.writeText(command).then(()=>{$('copyHeatmapCommand').textContent='Copied';setTimeout(()=>{$('copyHeatmapCommand').textContent='Copy command'},1200)})};$('cameraElevation').oninput=()=>{rememberCamera();let current=camera||structuredClone(DEFAULT_CAMERA),eye=current.eye||DEFAULT_CAMERA.eye,radius=Math.hypot(eye.x,eye.y,eye.z),azimuth=Math.atan2(eye.y,eye.x),elevation=+$('cameraElevation').value*Math.PI/180,planar=radius*Math.cos(elevation);camera={...current,eye:{x:planar*Math.cos(azimuth),y:planar*Math.sin(azimuth),z:radius*Math.sin(elevation)}};$('cameraElevationValue').textContent=$('cameraElevation').value+'°';Plotly.relayout('scene',{'scene.camera':camera})};$('autoRotate').onchange=updateAutoRotate;$('rotateSpeed').oninput=()=>{$('rotateSpeedValue').textContent=$('rotateSpeed').value+'°/s'};$('step').onclick=()=>draw(Math.min(frame+1,D.manifest.frames-1));$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'❚❚ Pause':'▶ Play';if(playing)tick();else clearTimeout(timer)};function tick(){if(!playing)return;let last=D.manifest.frames-1;if(frame>=last&&!$('loopReplay').checked){playing=false;$('play').textContent='▶ Play';timer=null;return}draw(frame>=last?0:frame+1).finally(()=>{if(playing)timer=setTimeout(tick,1000*D.manifest.dt/+$('speed').value)})}
</script></body></html>"""


def building_geometry(manifest: dict) -> dict | None:
    """Read authored walls/roof omitted from the replay's dynamic obstacles."""
    from swarmecho.core.config import MAP_DIR

    name = Path(str(manifest.get("map_name", ""))).stem
    path = resolve_map_file(MAP_DIR / f"{name}.yaml", MAP_DIR)
    data = manifest.get("building_snapshot")
    source = "replay snapshot" if data is not None else "current map (legacy artifact)"
    if data is None:
        if not path.is_file():
            return None
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("format") != "swarmecho-map/v1":
        return None
    grid = data["building_cell_grid"]
    cols, rows, layers = (grid[key] for key in ("cols", "rows", "layers"))
    cell = data["cell_size_m"]
    wall_half, tile_half = data["wall_thickness_m"] / 2, data["tile_thickness_m"] / 2
    interior = {tuple(c) for c in data.get("interior_cells", [
        [x, y, z] for x in range(cols) for y in range(rows) for z in range(layers)
    ])}
    walls, roofs, floors, solid_min, solid_max = [], [], [], [], []
    special = data["geometry"]
    openings = {axis: {tuple(v) for suffix in ("doors", "windows") for v in special.get(f"{axis}_{suffix}", [])}
                for axis in ("x", "y")}
    for axis, key in ((0, "x_walls"), (1, "y_walls")):
        for x, y, z in data["geometry"][key]:
            if (x, y, z) in openings["xy"[axis]]:
                continue
            lo = [x * cell, y * cell, z * cell]
            hi = [(x + 1) * cell, (y + 1) * cell, (z + 1) * cell]
            lo[axis] -= wall_half
            hi[axis] = lo[axis] + 2 * wall_half
            neighbor = [x, y, z]
            neighbor[axis] -= 1
            walls.append({"lo": lo, "hi": hi, "storey": z,
                          "outer": (x, y, z) not in interior or tuple(neighbor) not in interior})
            if (x, y)[axis] not in (0, (cols, rows)[axis]):
                solid_min.append(lo)
                solid_max.append(hi)
    for x, y, z in data["geometry"]["tiles"]:
        if z not in (0, layers):
            solid_min.append([x * cell, y * cell, z * cell - tile_half])
            solid_max.append([(x + 1) * cell, (y + 1) * cell, z * cell + tile_half])
        if (x, y, z) in interior:
            floors.append({"lo": [x * cell, y * cell, z * cell - tile_half],
                           "hi": [(x + 1) * cell, (y + 1) * cell, z * cell + tile_half],
                           "storey": z})
        if (x, y, z - 1) in interior and (x, y, z) not in interior:
            roofs.append({"lo": [x * cell, y * cell, z * cell - tile_half],
                          "hi": [(x + 1) * cell, (y + 1) * cell, z * cell + tile_half],
                          "storey": z - 1})
    if any(openings.values()) or special.get("stairs"):
        from swarmecho.env.buildings import compile_building
        compiled = compile_building({**data, "width": cols * cell, "height": rows * cell, "depth": layers * cell})
        box_key = lambda lo, hi: tuple(np.round(np.r_[lo, hi], 4))
        existing = {box_key(lo, hi) for lo, hi in zip(solid_min, solid_max)}
        for lo, hi in zip(compiled.solid_min_m.tolist(), compiled.solid_max_m.tolist()):
            if box_key(lo, hi) not in existing:
                walls.append({"lo": lo, "hi": hi, "storey": max(0, int(lo[2] / cell)), "outer": False})
        solid_min, solid_max = compiled.solid_min_m.tolist(), compiled.solid_max_m.tolist()
    return {"layers": layers, "cell_size_m": cell, "walls": walls, "roofs": roofs, "floors": floors,
            "source": source,
            "solid_min": data.get("solid_min_m", solid_min),
            "solid_max": data.get("solid_max_m", solid_max)}


def replay_payload(manifest_path: str | Path) -> dict:
    manifest, arrays = load_replay(manifest_path)
    return {
        "kind": "replay",
        "manifest": manifest,
        "building": building_geometry(manifest),
        **{name: value.tolist() for name, value in arrays.items()},
    }


def roadmap_payload(path: str | Path) -> dict:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"kind": "roadmap", **artifact,
            "building": building_geometry(artifact.get("manifest", {}))}


def heatmap_payload(info_path: str | Path) -> dict:
    """Load an evaluation CSV and its optional inspector metadata sidecar."""
    path = Path(info_path)
    records = load_eval_info_csv(path)
    if records["positions"].shape[-1] != 3:
        raise ValueError(f"heatmaps require x,y,z coordinates: {path}")
    manifest = {
        "format": "swarmecho-eval-heatmap/v1",
        "data_file": path.name,
        "map_name": path.stem,
    }
    sidecar = path.with_suffix(".heatmap.json")
    if sidecar.exists():
        try:
            manifest.update(json.loads(sidecar.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    manifest["format"] = canonical_marker(manifest["format"])
    checkpoint = _nearest_heatmap_checkpoint(path, manifest)
    if checkpoint is not None:
        manifest["checkpoint"] = str(checkpoint)
    obstacle_min = records["obstacle_min"]
    obstacle_max = records["obstacle_max"]
    layout_file = manifest.get("layout_file")
    if layout_file:
        from swarmecho.training.artifacts import load_eval_layout
        lower, upper = load_eval_layout(path.parent / layout_file)
        obstacle_min = np.broadcast_to(lower, (len(records["positions"]), *lower.shape))
        obstacle_max = np.broadcast_to(upper, (len(records["positions"]), *upper.shape))
    return {
        "kind": "heatmap",
        "manifest": manifest,
        "building": building_geometry(manifest),
        "positions": records["positions"].tolist(),
        "stages": records["stages"].tolist(),
        "stage_rates": {
            "chain_success": records["chain_success_rate"].tolist(),
            "found_and_delivered": records[
                "found_and_delivered_rate"
            ].tolist(),
            "visually_found": records["visually_found_rate"].tolist(),
        },
        "distance_to_base": records["distance_to_base"].tolist(),
        "final_chain_length": records["final_chain_length"].tolist(),
        "obstacle_min": obstacle_min.tolist(),
        "obstacle_max": obstacle_max.tolist(),
    }


def _nearest_heatmap_checkpoint(info_path: Path, manifest: dict) -> Path | None:
    """Resolve a heatmap to its closest saved checkpoint in the same run."""
    configured = manifest.get("checkpoint")
    if configured:
        return Path(str(configured))

    artifacts_dir = next(
        (parent for parent in info_path.parents if parent.name == "artifacts"),
        None,
    )
    training_update = manifest.get("training_update")
    if artifacts_dir is None or training_update is None:
        return None

    checkpoints_dir = artifacts_dir.parent / "checkpoints"
    candidates = [
        path
        for path in checkpoints_dir.iterdir()
        if path.is_dir() and parse_checkpoint_update(path) is not None
    ] if checkpoints_dir.is_dir() else []
    if not candidates:
        return None
    target_update = int(training_update)
    return min(
        candidates,
        key=lambda path: abs(parse_checkpoint_update(path) - target_update),
    )


def inspector_html() -> str:
    """Embed the installed Plotly runtime so inspection also works offline."""
    from plotly.offline import get_plotlyjs

    external = '<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>'
    return HTML.replace(external, f"<script>{get_plotlyjs()}</script>")


def discover_replays(root: str | Path = "outputs", *, single_run: bool = False) -> list[Path]:
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
        "*/artifacts/eval/*/eval_*/replays",
    ):
        replay_dirs.update(path for path in root.glob(pattern.removeprefix("*/") if single_run else pattern) if path.is_dir())
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


def discover_heatmaps(root: str | Path = "outputs", *, single_run: bool = False) -> list[Path]:
    """Find single-pass and robust heatmaps, excluding action-capture CSVs."""
    root = Path(root)
    if not root.exists():
        return []
    data_dirs: set[Path] = set()
    for pattern in (
        "*/artifacts/train/data",
        "*/artifacts/eval/*/data",
        "*/artifacts/eval/*/eval_*/data",
    ):
        data_dirs.update(path for path in root.glob(pattern.removeprefix("*/") if single_run else pattern) if path.is_dir())
    heatmaps: list[Path] = []
    for data_dir in data_dirs:
        for filename_pattern in ("eval_info_*.csv",):
            for path in data_dir.glob(filename_pattern):
                try:
                    manifest = json.loads(path.with_suffix(".heatmap.json").read_text(encoding="utf-8"))
                    if int(manifest.get("robustness_runs", 1)) < 1:
                        continue
                    if load_eval_info_csv(path)["positions"].shape[-1] == 3:
                        heatmaps.append(path.resolve())
                except (OSError, ValueError):
                    continue
    return sorted(set(heatmaps), key=lambda path: path.stat().st_mtime, reverse=True)


def discover_roadmap_tests(root: str | Path = "outputs") -> list[Path]:
    """Discover inspectable obstacle/roadmap integration-test artifacts."""
    root = Path(root)
    if not root.exists():
        return []
    results = []
    for path in root.glob("testresults/**/*.roadmap.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("format") == "swarmecho-roadmap-test/v1":
                results.append(path.resolve())
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(results, key=lambda path: path.stat().st_mtime, reverse=True)


def roadmap_label(path: Path) -> str:
    """Give generated suites a compact run label and per-map route label."""
    stem = path.name.removesuffix(".roadmap.json")
    try:
        name = json.loads(path.read_text(encoding="utf-8")).get("manifest", {}).get("map_name")
    except (OSError, ValueError):
        name = None
    if not name:
        return f"{stem}_[Roadmap]"
    if path.parent.name.startswith("random_buildings_"):
        run_name = re.sub(r"_seed\d+$", "", path.parent.name)
        extreme = re.search(r"_(shortest|longest|top\d+_longest)_farthest$", stem)
        detail = f"{name}_{extreme.group(1)}_farthest" if extreme else stem
        return f"{run_name}_[{detail}]_[Roadmap]"
    display_name = name
    if stem.startswith(name + "_["):
        return f"{stem}_[Roadmap]"
    detail = stem[len(name):].lstrip("_") if stem.startswith(name + "_") else stem
    return f"{display_name}_[{detail}]_[Roadmap]" if detail and detail != name else f"{display_name}_[Roadmap]"


def random_eval_metadata(kind, path):
    if kind not in {"heatmap", "replay"}:
        return {}
    sidecar = path.with_suffix(".heatmap.json") if kind == "heatmap" else path
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {key: data[key] for key in ("random_eval_map_id", "random_eval_group") if key in data}


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
    run_name = _artifact_run_name(path, fallback=path.stem)
    kind = "Target-Replay" if _is_target_replay(path) else "Replay"
    return _artifact_label(
        run_name,
        _replay_steps_label(path),
        kind,
        eval_name=_artifact_eval_name(path),
    )


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
    """Return a compact heatmap label that exposes train/eval provenance."""
    path = Path(info_path)
    run_name = _artifact_run_name(path, fallback=path.stem)
    scope = _heatmap_scope(path)
    return _artifact_label(
        run_name,
        _replay_steps_label(path),
        f"{scope.upper()} Heatmap",
        eval_name=_artifact_eval_name(path.with_suffix(".heatmap.json")),
    )


def _heatmap_scope(info_path: Path) -> str:
    """Read artifact scope, falling back to the canonical folder layout."""
    sidecar = info_path.with_suffix(".heatmap.json")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        scope = metadata.get("artifact_scope")
        if scope in {"train", "eval"}:
            return scope
    except (OSError, json.JSONDecodeError):
        pass

    parts = info_path.parts
    try:
        scope = parts[parts.index("artifacts") + 1]
        if scope in {"train", "eval"}:
            return scope
    except (ValueError, IndexError):
        pass
    return "eval"


def _artifact_run_name(path: Path, *, fallback: str) -> str:
    parts = path.parts
    try:
        return parts[parts.index("artifacts") - 1]
    except ValueError:
        return path.parent.parent.name if path.parent.name == "replays" else fallback


def _artifact_eval_name(path: Path) -> str | None:
    """Read the optional human evaluation label without exposing timestamps."""
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = metadata.get("eval_name") if isinstance(metadata, dict) else None
    return value if isinstance(value, str) and value else None


def _artifact_label(
    run_name: str,
    steps: str | None,
    kind: str,
    *,
    eval_name: str | None = None,
) -> str:
    label = f"{run_name}_[{steps or 'unknown'}]_[{kind}]"
    return f"{label}_[{eval_name}]" if eval_name else label


def make_handler(
    manifests: list[Path], initial: Path | None = None, root: str | Path | None = None
):
    resolved = [path.resolve() for path in manifests]
    initial = initial.resolve() if initial is not None else None
    replay_root = Path(root).resolve() if root is not None else None
    encoded_html = inspector_html().encode()
    manifests_lock = threading.Lock()
    cached_sources: list[tuple[str, Path]] = [("replay", path) for path in resolved]

    def source_id(kind: str, path: Path) -> str:
        return hashlib.sha256(f"{kind}:{path}".encode()).hexdigest()

    def run_directory(path: Path) -> Path:
        for parent in path.parents:
            if parent.name == "artifacts":
                return parent.parent
        return path.parent.parent if path.parent.name == "replays" else path.parent

    def run_id(kind: str, path: Path) -> str:
        # Generated building suites share one selector entry; other roadmaps
        # retain their historical map-based grouping.
        if kind == "roadmap":
            key = str(path.parent)
            if not path.parent.name.startswith("random_buildings_"):
                key += ":" + roadmap_label(path).split("_[", 1)[0]
        else:
            key = str(run_directory(path))
        return hashlib.sha256(key.encode()).hexdigest()

    def refresh_run_sources(identifier: str) -> list[tuple[str, Path]]:
        nonlocal cached_sources
        with manifests_lock:
            previous = [source for source in cached_sources if run_id(*source) == identifier]
        if not previous:
            raise ValueError("Unknown run")
        kind, path = previous[0]
        if kind == "roadmap":
            current = [(kind, item) for item in discover_roadmap_tests(replay_root or path.parent.parent)
                       if run_id(kind, item) == identifier]
        else:
            directory = run_directory(path)
            current = [("replay", item) for item in discover_replays(directory, single_run=True)]
            current += [("heatmap", item) for item in discover_heatmaps(directory, single_run=True)]
            # Explicitly supplied artifacts may live outside the standard folders.
            current += [source for source in previous
                        if source[1].is_file() and source not in current
                        and source[1].parent.name != "replays"
                        and "artifacts" not in source[1].parts]
        current.sort(key=lambda source: source[1].stat().st_mtime_ns, reverse=True)
        with manifests_lock:
            cached_sources = [source for source in cached_sources if run_id(*source) != identifier] + current
        return current

    def refresh_sources() -> list[tuple[str, Path]]:
        nonlocal cached_sources
        replays = discover_replays(replay_root) if replay_root is not None else list(resolved)
        heatmaps = discover_heatmaps(replay_root) if replay_root is not None else []
        roadmaps = discover_roadmap_tests(replay_root) if replay_root is not None else []
        sources = [("replay", path) for path in replays] + [
            ("heatmap", path) for path in heatmaps
        ] + [("roadmap", path) for path in roadmaps]
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

    def refresh_recent_sources() -> tuple[list[tuple[str, Path]], bool]:
        """Read recent run artifacts first; leave archived runs and tests for Load more."""
        nonlocal cached_sources
        if replay_root is None or not replay_root.is_dir():
            return refresh_sources(), False
        patterns = (
            "replays/*.json",
            "artifacts/train/replays/*.json",
            "artifacts/eval/replays/*.json",
            "artifacts/eval/*/replays/*.json",
            "artifacts/eval/*/eval_*/replays/*.json",
            "artifacts/train/data/eval_info_*.csv",
            "artifacts/eval/*/data/eval_info_*.csv",
            "artifacts/eval/*/eval_*/data/eval_info_*.csv",
        )
        candidates: list[tuple[int, Path]] = []
        for directory in replay_root.iterdir():
            if not directory.is_dir() or directory.name == "testresults":
                continue
            newest = 0
            for pattern in patterns:
                for path in directory.glob(pattern):
                    try:
                        newest = max(newest, path.stat().st_mtime_ns)
                    except OSError:
                        continue
            if newest:
                candidates.append((newest, directory))
        candidates.sort(key=lambda entry: entry[0], reverse=True)
        current: list[tuple[str, Path]] = []
        loaded_runs = 0
        for _, directory in candidates:
            run_sources = [("replay", path) for path in discover_replays(directory, single_run=True)]
            run_sources += [("heatmap", path) for path in discover_heatmaps(directory, single_run=True)]
            if not run_sources:
                continue
            current.extend(run_sources)
            loaded_runs += 1
            if loaded_runs == 3:
                break
        if initial is not None and initial.is_file() and ("replay", initial) not in current:
            current.append(("replay", initial))
        current.sort(key=lambda source: source[1].stat().st_mtime_ns, reverse=True)
        has_more = len(candidates) > loaded_runs or any(
            replay_root.glob("testresults/**/*.roadmap.json")
        )
        with manifests_lock:
            cached_sources = current
            return list(cached_sources), has_more

    def available_sources() -> list[tuple[str, Path]]:
        with manifests_lock:
            return list(cached_sources)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            has_more = False
            if parsed.path == "/api/replays":
                query = urllib.parse.parse_qs(parsed.query)
                try:
                    if "run" in query:
                        current = refresh_run_sources(query["run"][0])
                    elif query.get("recent") == ["3"]:
                        current, has_more = refresh_recent_sources()
                    else:
                        current = refresh_sources()
                except (ValueError, OSError):
                    self.send_error(404, "Run not found")
                    return
                items = [
                    {
                        "id": source_id(kind, path),
                        "run_id": run_id(kind, path),
                        "kind": kind,
                        **random_eval_metadata(kind, path),
                        "label": replay_label(path) if kind == "replay" else (
                            heatmap_label(path) if kind == "heatmap" else roadmap_label(path)
                        ),
                        "selected": path == initial if initial is not None else index == 0,
                    }
                    for index, (kind, path) in enumerate(current)
                ]
                body, content_type = json.dumps(items).encode(), "application/json"
            elif parsed.path == "/api/replay":
                query = urllib.parse.parse_qs(parsed.query)
                try:
                    identifier = query.get("id", [""])[0]
                    source = next((source for source in available_sources()
                                   if source_id(*source) == identifier), None)
                    if source is None:
                        raise ValueError("Unknown artifact")
                    kind, path = source
                    if kind == "replay":
                        payload = replay_payload(path)
                    elif kind == "heatmap":
                        payload = heatmap_payload(path)
                    else:
                        payload = roadmap_payload(path)
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
            if parsed.path == "/api/replays":
                self.send_header("X-Inspector-Has-More", "true" if has_more else "false")
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
    print(f"SwarmEcho Inspector: {url}")
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
