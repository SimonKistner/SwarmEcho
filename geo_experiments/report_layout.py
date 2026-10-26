"""Shared dark target-report shell and interactive count distributions."""
from __future__ import annotations

import html

from geo_experiments.dashboard import safe_json

DISTRIBUTION_STYLE = """
.distributions{display:grid;grid-template-columns:1fr 2fr;gap:18px}.distribution-card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 12px 6px;min-width:0}.distribution-card h3{margin:0 10px 8px}
.distribution-card svg{width:100%;display:block}@media(max-width:800px){.distributions{grid-template-columns:1fr}}
"""

STYLE = """
:root{color-scheme:dark;--bg:#0b1220;--panel:#111c2e;--text:#e4edf7;--muted:#9aacc3;--line:#2b3b53}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}
main{max-width:1480px;margin:auto;padding:36px 44px 70px}header{padding-bottom:24px;border-bottom:1px solid var(--line)}
.eyebrow{color:#68b9f2;font-size:12px;letter-spacing:.14em;text-transform:uppercase}h1{font-size:32px;margin:8px 0 12px;line-height:1.2}
h2{font-size:22px;margin:0 0 12px}h3{font-size:17px;margin:20px 0 8px}p{margin:8px 0 16px}section{margin-top:34px}
.muted,.note{color:var(--muted)}.note{font-size:13px;max-width:1120px}.pills{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}
.pill{background:var(--panel);border:1px solid var(--line);padding:8px 14px;border-radius:8px;color:var(--muted)}.pill b{color:var(--text);margin-right:5px}
.controls{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin-bottom:14px}label{color:var(--muted);font-size:13px}
select{margin-left:7px;background:var(--panel);color:var(--text);border:1px solid #425571;padding:7px 10px;border-radius:6px}
.legend{display:flex;gap:22px;flex-wrap:wrap;margin:10px 0 16px;color:var(--muted);font-size:13px}.legend span{display:flex;gap:7px;align-items:center}
.legend i,.source i{display:inline-block;width:10px;height:10px;border-radius:3px;flex-shrink:0}.chart{display:block;width:100%;height:auto;border-radius:10px;background:var(--panel)}[hidden]{display:none!important}
.table-scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel)}table{border-collapse:collapse;width:100%;min-width:660px;table-layout:fixed}
th{text-align:left;color:var(--muted);font-size:12px;font-weight:600;background:#18263c;padding:13px 20px}th:first-child{width:31%}th:nth-child(2){width:16%}
td{padding:12px 20px;vertical-align:middle}.overall td{border-top:1px solid var(--line);padding-top:20px}.overall:first-child td{border-top:0}
.source{display:flex;gap:8px;align-items:center;margin-top:7px;font-size:13px;color:var(--muted)}.accuracy{font-size:25px;font-weight:650;font-variant-numeric:tabular-nums}
.histogram{display:block;width:100%;height:auto;max-width:490px}.perfect{color:#6ed1b5;font-size:13px}.detail-row>td{padding-top:0;padding-bottom:14px}
summary{cursor:pointer;color:var(--muted);font-size:13px;width:fit-content}summary:hover{color:#68b9f2}.secondary{background:#142138;border-left:2px solid #425571;border-radius:0 6px 6px 0;margin:12px 0 3px;padding:5px 0}
.secondary table{min-width:620px}.secondary td:first-child{width:31%;color:var(--muted);font-size:13px}.secondary td:nth-child(2){width:16%;font-size:20px}.secondary tr+tr td{border-top:1px solid var(--line)}
.report-controls{margin-top:16px}a{color:#68b9f2}.metric-delta{font-size:13px;font-weight:500;color:#a7c5ee;white-space:nowrap}.metric-delta.positive{color:#6ed1b5}.metric-delta.negative{color:#f0838a}
.audit-table{min-width:750px;table-layout:auto}.audit-table th:first-child{width:34%}.audit-table th:nth-child(2){width:auto}.audit-table td{font-variant-numeric:tabular-nums;border-top:1px solid var(--line)}
.matrix-scroll{overflow:auto;border:1px solid var(--line);border-radius:10px}.matrix-large{min-width:1500px}.matrix-large img{width:100%;max-width:none}details.matrix{margin-top:18px}footer{margin-top:32px;border-top:1px solid var(--line);padding-top:16px}
@media(max-width:800px){main{padding:24px 14px 40px}h1{font-size:26px}td,th{padding-left:12px;padding-right:12px}.legend{gap:12px}.distributions{grid-template-columns:1fr}}
""" + DISTRIBUTION_STYLE


def document(title, body):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="color-scheme" content="dark"><title>' + html.escape(title) + '</title><style>'
            + STYLE + '</style></head><body><main>' + body + '</main></body></html>')


def distribution_panel(data, *, single_map=False, eligibility="delivered"):
    left_selector = ('<label>Left chart<select id="distribution-left" onchange="updateDistribution()">'
                     '<option value="storey">Target storeys</option><option value="cell">Target cells</option></select></label>'
                     if single_map else '<input type="hidden" id="distribution-left" value="storey">')
    rule = "Delivery to the base determines collection eligibility" if eligibility == "delivered" else "This historical protocol requires chain success for collection eligibility"
    return '''<section><h2>Sample distributions</h2><div class="controls">''' + left_selector + '''
<label>Compare<select id="distribution-category" onchange="updateDistribution()"><option value="outcomes">Spawns, delivered &amp; collected</option><option value="splits">Training, validation &amp; evaluation</option></select></label>
<label>Measure<select id="distribution-measure" onchange="updateDistribution()"><option value="count">Episode counts</option><option value="share">Share (%)</option></select></label>
<label><input id="distribution-success" type="checkbox" onchange="updateDistribution()"> Highlight chain successes</label></div>
<div id="distribution-legend" class="legend"></div><div class="distributions">
<div class="distribution-card"><h3 id="distribution-left-title">Target storeys</h3><svg id="distribution-left-chart" role="img" aria-label="Target distribution"></svg></div>
<div class="distribution-card"><h3>Target-required agents</h3><svg id="distribution-right-chart" role="img" aria-label="Distribution by target-required agents"></svg></div></div>
<p class="note">Target-required agents is the episode target's cached relay estimator count, not the number of deployed drones. “Not estimated” means no count was available.
''' + rule + '''; chain success is retained separately. Highlighting adds a hatched successful portion to each bar without changing its total.
Shares normalize each series separately; highlighted shares use that same denominator. Final-batch spawns/deliveries include episodes beyond the retained quota.</p></section>
<script id="distribution-data" type="application/json">''' + safe_json(data) + '''</script>
<script>
function updateDistribution(){
 const data=JSON.parse(document.getElementById('distribution-data').textContent);
 const category=document.getElementById('distribution-category').value;
 const measure=document.getElementById('distribution-measure').value;
 const highlight=document.getElementById('distribution-success').checked;
 const left=document.getElementById('distribution-left').value;
 const series=data[category];
 const legend=document.getElementById('distribution-legend');legend.replaceChildren();
 for(const row of series){const span=document.createElement('span');const dot=document.createElement('i');dot.style.background=row.color;
 span.append(dot,document.createTextNode(`${row.name} · ${row.total.toLocaleString()}`));legend.append(span);}
 if(highlight){const span=document.createElement('span');span.textContent='Hatched portion: chain success';legend.append(span);}
 document.getElementById('distribution-left-title').textContent=left==='cell'?'Target cells':'Target storeys';
 function draw(id,kind){
  const svg=document.getElementById(id),ns='http://www.w3.org/2000/svg';svg.replaceChildren();
  const names=data.labels[kind],w=kind==='required'?760:Math.max(450,names.length*30),h=335,L=58,R=12,T=15,B=kind==='cell'?100:65;
  svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
  function el(tag,attrs,text,parent=svg){const node=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))node.setAttribute(k,v);if(text!==undefined)node.textContent=text;parent.append(node);return node;}
  const patternId=id+'-hatch';const defs=el('defs',{});const pattern=el('pattern',{id:patternId,patternUnits:'userSpaceOnUse',width:6,height:6},undefined,defs);
  el('path',{d:'M-1,1 l2,-2 M0,6 l6,-6 M5,7 l2,-2',stroke:'#e4edf7','stroke-width':1.2},undefined,pattern);
  const heights=series.map(row=>row[kind].map(n=>measure==='share'?100*n/Math.max(1,row.total):n));
  const max=measure==='share'?100:Math.max(1,...heights.flat())*1.12;
  for(let i=0;i<=4;i++){const y=T+(h-T-B)*i/4;el('line',{x1:L,x2:w-R,y1:y,y2:y,stroke:'#2b3b53'});el('text',{x:L-7,y:y+4,fill:'#9aacc3','font-size':11,'text-anchor':'end'},measure==='share'?`${Math.round(max*(1-i/4))}%`:Math.round(max*(1-i/4)).toLocaleString());}
  const step=(w-L-R)/names.length,bw=Math.min(30,step*.8/series.length);
  names.forEach((name,c)=>{const center=L+step*(c+.5);el('text',{x:center,y:h-B+19,fill:'#9aacc3','font-size':kind==='cell'?10:12,'text-anchor':kind==='cell'?'end':'middle',...(kind==='cell'?{transform:`rotate(-65 ${center} ${h-B+19})`}:{})},name);
   series.forEach((row,s)=>{const value=heights[s][c],barH=(h-T-B)*value/max,x=center+(s-series.length/2)*bw,y=h-B-barH;
    const rect=el('rect',{x,y,width:bw*.92,height:barH,fill:row.color,rx:2});el('title',{},`${row.name}: ${row[kind][c].toLocaleString()} episodes; ${row['success_'+kind][c].toLocaleString()} chain successes`,rect);
    if(highlight){const n=row['success_'+kind][c],v=measure==='share'?100*n/Math.max(1,row.total):n,sh=(h-T-B)*v/max;el('rect',{x,y:h-B-sh,width:bw*.92,height:sh,fill:`url(#${patternId})`,'pointer-events':'none'});}
   });
  });
  el('text',{x:(L+w-R)/2,y:h-7,fill:'#9aacc3','font-size':12,'text-anchor':'middle'},kind==='required'?'Agents required by target (estimator)':kind==='cell'?'Absolute cell XYZ':'Building storey');
 }
 draw('distribution-left-chart',left);draw('distribution-right-chart','required');
 document.querySelectorAll('.error-histogram-view').forEach(view=>view.hidden=view.dataset.measure!==measure);
}
updateDistribution();
</script>'''
