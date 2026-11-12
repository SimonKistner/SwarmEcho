(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('path-examples').textContent);
  const thresholdSelector = document.getElementById('held-out-threshold');
  const populationSelector = document.getElementById('held-out-population');
  function updateResults() {
    data.threshold = +thresholdSelector.value / 100;
    for (const cell of document.querySelectorAll('[data-evaluation-threshold]')) {
      cell.hidden = cell.dataset.evaluationThreshold !== thresholdSelector.value ||
        cell.dataset.evaluationPopulation !== populationSelector.value;
    }
  }
  if (thresholdSelector && populationSelector) {
    thresholdSelector.onchange = populationSelector.onchange = updateResults;
    updateResults();
  }

  const rotations = [[[1,0,0],[0,1,0],[0,0,1]],[[0,-1,0],[1,0,0],[0,0,1]],[[-1,0,0],[0,-1,0],[0,0,1]],[[0,1,0],[-1,0,0],[0,0,1]]];
  // Same vertex/face ordering, initial elevation and camera distance as inspector.py.
  const faces = [[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]];
  const elevation = 10 * Math.PI / 180, distance = 2.9 * Math.pow(.85, 2);
  const planar = distance * Math.cos(elevation) / Math.SQRT2;
  const defaultCamera = {eye:{x:planar,y:planar,z:distance*Math.sin(elevation)},center:{x:0,y:0,z:0},up:{x:0,y:0,z:1},projection:{type:'perspective'}};
  function cubeVertices(lo,hi) {
    return [[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],[hi[0],hi[1],lo[2]],[lo[0],hi[1],lo[2]],
      [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],[hi[0],hi[1],hi[2]],[lo[0],hi[1],hi[2]]];
  }
  function mesh(cubes,color,opacity) {
    const trace={type:'mesh3d',x:[],y:[],z:[],i:[],j:[],k:[],text:[],color,opacity,flatshading:true,
      hovertemplate:'%{text}<extra></extra>',lighting:{ambient:.8,diffuse:.25,specular:0},showscale:false};
    for (const {vertices,text} of cubes) {
      const n=trace.x.length;
      for (const p of vertices) {trace.x.push(p[0]);trace.y.push(p[1]);trace.z.push(p[2]);trace.text.push(text||'Building geometry');}
      for (const f of faces) {trace.i.push(n+f[0]);trace.j.push(n+f[1]);trace.k.push(n+f[2]);}
    }
    return trace;
  }
  function line(points,color,width) {
    return {type:'scatter3d',mode:'lines',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),
      line:{color,width},hoverinfo:'skip',showlegend:false};
  }
  function boxLines(lo,hi) {
    const v=cubeVertices(lo,hi),points=[];
    for (const [a,b] of [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]) points.push(v[a],v[b],[null,null,null]);
    return points;
  }

  // Ported from inspector.configureUprightCamera. Merely specifying camera.up
  // does not stop Plotly's orbit controller from rolling. Keep its turntable
  // controller and suppress the horizontal wheel/touch roll component instead.
  function configureUprightCamera(graph) {
    const live=graph._fullLayout?.scene?._scene;
    if (!live?.camera?.view?.setMode) return;
    if (!live.__uprightControls) {
      const updateFx=live.updateFx;
      live.updateFx=function(dragmode,hovermode) {
        const view=this.camera.view;
        if (this.__uprightView!==view) {
          const rotate=view.rotate;
          view.rotate=function(timestamp,horizontal,vertical) {return rotate.call(this,timestamp,horizontal,vertical,0);};
          this.__uprightView=view;
        }
        if (dragmode==='orbit'||dragmode==='turntable') {
          if (this.camera.view.getMode()!=='turntable') {
            const pose=this.getCamera();
            this.camera.view.setMode('turntable');
            this.camera.lookAt([pose.eye.x,pose.eye.y,pose.eye.z],[pose.center.x,pose.center.y,pose.center.z],[0,0,1]);
          }
          this.camera.keyBindingMode='rotate';
          this.fullSceneLayout.hovermode=hovermode;
        } else updateFx.call(this,dragmode,hovermode);
      };
      live.__uprightControls=true;
    }
    live.updateFx(live.fullSceneLayout.dragmode,live.fullSceneLayout.hovermode);
  }

  // Inspector buildingTraces: same six wall strips, floor shades, outlines,
  // material lighting and omitted internal strip faces. World axes never change.
  function buildingTraces(building,q) {
    if (!building) return [];
    const opacity=1-(+q('walls').value/100);
    if (opacity<=0) return [];
    const solid={type:'mesh3d',x:[],y:[],z:[],i:[],j:[],k:[],facecolor:[],opacity,flatshading:true,hoverinfo:'skip',
      lighting:{ambient:.65,diffuse:.7,specular:.12,roughness:.85},lightposition:{x:80,y:40,z:120},showlegend:false};
    const grid=line([],'#73818d',2);grid.opacity=opacity;
    function add(lo,hi,color,stripAxis=-1,strip=0) {
      const n=solid.x.length;
      for (const p of cubeVertices(lo,hi)) {solid.x.push(p[0]);solid.y.push(p[1]);solid.z.push(p[2]);}
      faces.forEach((face,index)=>{
        if (stripAxis>=0) {
          const lower=stripAxis===0?10:4,upper=stripAxis===0?6:8;
          if ((strip>0&&index>=lower&&index<lower+2)||(strip<5&&index>=upper&&index<upper+2)) return;
        }
        solid.i.push(n+face[0]);solid.j.push(n+face[1]);solid.k.push(n+face[2]);solid.facecolor.push(color);
      });
    }
    function outline(lo,hi) {
      if (!q('grid').checked) return;
      for (const p of boxLines(lo,hi)) {grid.x.push(p[0]);grid.y.push(p[1]);grid.z.push(p[2]);}
    }
    for (const {lo,hi,outer} of building.walls||[]) {
      if (outer&&!q('outer').checked) continue;
      const axis=hi[0]-lo[0]>hi[1]-lo[1]?0:1;
      for (let strip=0;strip<6;strip++) {
        const a=[...lo],b=[...hi];
        a[axis]=lo[axis]+(hi[axis]-lo[axis])*strip/6;b[axis]=lo[axis]+(hi[axis]-lo[axis])*(strip+1)/6;
        add(a,b,['#959b9c','#9da2a2','#979d9d'][strip%3],axis,strip);
      }
      outline(lo,hi);
    }
    for (const {lo,hi} of building.floors||[]) {
      const shade=(Math.round(lo[0]+lo[1])%3+3)%3;
      add(lo,hi,['#252e39','#2c3540','#303946'][shade]);outline(lo,hi);
    }
    if (q('roof').checked) for (const {lo,hi} of building.roofs||[]) {add(lo,hi,'#78868c');outline(lo,hi);}
    return [...(solid.x.length?[solid]:[]),...(grid.x.length?[grid]:[])];
  }

  class Viewer {
    constructor(details) {
      this.details=details;this.example=data.examples[details.dataset.example];this.source=details.dataset.source;
      this.camera=null;this.drawSerial=0;this.host=details.querySelector('.example-content');
      this.host.innerHTML='<div class="example-controls"><label>Input <select class="input"></select></label><label>Seed <select class="seed"></select></label><label class="agent-control">Agent <select class="agent"></select></label><label class="snapshot-control">Snapshot <select class="snapshot"></select></label></div>'+
        '<div class="example-controls"><label>View <select class="mode"><option value="probability">Selected cells · blue probability opacity</option><option value="errors">Correct / extra / missed cells</option></select></label><label>Select cells ≥ <input class="threshold" type="range" min="0" max="100"><span class="threshold-value"></span></label><button class="reset" type="button">Reset to report threshold</button><label>Maximum cell opacity <input class="opacity" type="range" min="0.05" max="1" step="0.05" value="0.65"></label><label><input class="truth" type="checkbox"> True path cells</label></div>'+
        '<div class="example-controls"><label>Building transparency <input class="walls" type="range" min="0" max="100" step="1" value="0"><span class="walls-value">0%</span></label><label><input class="outer" type="checkbox"> Outer walls</label><label><input class="roof" type="checkbox"> Roof</label><label><input class="grid" type="checkbox" checked> Cell grid</label><span class="note">Drag to orbit · scroll to zoom · vertical stays upright</span></div><p class="example-status"></p><div class="path-scene" role="img" aria-label="Rotatable original building with reference route and predicted path cells"></div>';
      this.q=name=>this.host.querySelector('.'+name);this.scene=this.q('path-scene');
      this.q('threshold').value=(data.threshold??.8)*100;
      this.fill(this.q('input'),Object.keys(data.labels).map(k=>[k,data.labels[k]]),this.source);
      this.q('input').onchange=()=>{this.source=this.q('input').value;this.selectSeed();};
      this.q('seed').onchange=()=>this.selectAgent();this.q('agent').onchange=()=>this.selectSnapshot();this.q('snapshot').onchange=()=>this.draw();
      for (const name of ['mode','truth','outer','roof','grid']) this.q(name).onchange=()=>this.draw();
      for (const name of ['threshold','opacity','walls']) this.q(name).oninput=()=>this.draw();
      this.q('reset').onclick=()=>{this.q('threshold').value=(data.threshold??.8)*100;this.draw();};this.selectSeed();
    }
    fill(select,values,preferred) {
      select.replaceChildren();
      for (const [value,label] of values) {const option=document.createElement('option');option.value=value;option.textContent=label;select.append(option);}
      if (values.some(v=>String(v[0])===String(preferred))) select.value=preferred;
    }
    selectSeed() {const seeds=this.example.sources[this.source]||{};this.fill(this.q('seed'),Object.keys(seeds).map(k=>[k,k]),this.q('seed').value);this.selectAgent();}
    entries() {return (this.example.sources[this.source]||{})[this.q('seed').value]||[];}
    selectAgent() {
      const agents=[...new Set(this.entries().map(e=>e.agent))];
      this.fill(this.q('agent'),agents.map(a=>[a,a<0?'Base':'Agent '+a]),this.q('agent').value);
      this.q('agent-control').hidden=agents.length<=1;
      this.selectSnapshot();
    }
    selectSnapshot() {
      this.rows=this.entries().filter(e=>String(e.agent)===this.q('agent').value).sort((a,b)=>a.step-b.step);
      this.fill(this.q('snapshot'),this.rows.map((r,i)=>[i,'Step '+r.step+(r.first>=0?' (+'+(r.step-r.first)+' since info)':'')]),0);
      this.q('snapshot-control').hidden=this.rows.length<=1;
      this.draw();
    }
    toWorld(p) {const r=rotations[this.example.rotation];return [0,1,2].map(i=>this.example.anchor[i]+r.reduce((s,row,j)=>s+row[i]*p[j],0));}
    isVisible() {if(!this.details.open)return false;for(let p=this.details.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS'&&!p.open)return false;return true;}
    rememberCamera() {
      const layout=this.scene._fullLayout?.scene,live=layout?._scene;
      const pose=live?.getCamera?live.getCamera():layout?.camera;
      if (pose) this.camera={...structuredClone(pose),up:{x:0,y:0,z:1}};
    }
    captureCamera() {
      configureUprightCamera(this.scene);this.rememberCamera();
      for (const event of ['plotly_relayout','plotly_relayouting']) {
        this.scene.removeAllListeners(event);
        this.scene.on(event,changed=>{
          if (Object.keys(changed).some(key=>key.startsWith('scene.camera'))) {configureUprightCamera(this.scene);this.rememberCamera();}
        });
      }
    }
    draw() {
      const e=this.example,row=this.rows[+this.q('snapshot').value],threshold=+this.q('threshold').value/100;
      this.q('threshold-value').textContent=Math.round(threshold*100)+'%';this.q('walls-value').textContent=this.q('walls').value+'%';
      if (!this.isVisible()) return;
      const traces=buildingTraces(e.building,this.q);
      traces.push(line(e.path,'#e4f563',7));
      const points=[e.base,e.target];
      traces.push({type:'scatter3d',mode:'markers+text',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),
        text:['BASE','TARGET'],textposition:'top center',marker:{size:[7,6],color:['#60a5fa','#fb7185']},showlegend:false});
      const bins=new Map();let predicted=0,actual=0,shared=0;
      const errors=this.q('mode').value==='errors',maximumOpacity=+this.q('opacity').value;
      for (let i=0;i<125;i++) {
        // Preserve saved probability precision. Match NumPy's float32 cutoff
        // conversion at exact slider boundaries instead of rounding beliefs.
        const cutoff=row?.float32?Math.fround(threshold):threshold;
        const truth=Boolean(e.truth[i]),p=row?row.p[i]:0,selected=Boolean(row)&&p>=cutoff;
        predicted+=selected?1:0;actual+=truth?1:0;shared+=selected&&truth?1:0;
        let color=null,opacity=maximumOpacity;
        if (row&&errors) color=selected&&truth?'#6ed1b5':selected?'#f3bb65':truth?'#eb7faa':null;
        else if (selected) {color='#38bdf8';opacity=p*maximumOpacity;}
        if (!color&&this.q('truth').checked&&truth) color='#d9ed65';
        if (!color||opacity<=0) continue;
        const lo=data.offsets[i].map(v=>(v-.5)*e.cell),hi=lo.map(v=>v+e.cell);
        const vertices=cubeVertices(lo,hi).map(p=>this.toWorld(p));
        const key=color+'/'+opacity.toFixed(4);
        if (!bins.has(key)) bins.set(key,{color,opacity,cubes:[]});
        bins.get(key).cubes.push({vertices,text:'Relative cell '+data.offsets[i].join(', ')+'<br>Path belief: '+(p*100).toFixed(1)+'%<br>Reference: '+(truth?'on path':'off path')});
      }
      for (const {color,opacity,cubes} of bins.values()) traces.push(mesh(cubes,color,opacity));
      const similarity=predicted+actual?200*shared/(predicted+actual):100,exact=predicted===actual&&shared===actual;
      this.q('example-status').textContent=e.map+' · fixed held-out episode '+e.id+' · '+
        (row?(row.agent<0?'Base':'Agent '+row.agent)+', step '+row.step+' · Exact match: '+(exact?'yes':'no')+' · Path similarity: '+similarity.toFixed(1)+'%':
          'No snapshot available for this input/seed in this episode')+
        ' · Yellow line: shortest reference path. '+(errors?'Green: correct; orange: extra; pink: missed. ':'Blue: predicted path cells; opacity = probability × maximum opacity. ')+
        'Original building coordinates. Slider affects this example only.';
      const serial=++this.drawSerial;
      this.queue=(this.queue||Promise.resolve()).catch(()=>{}).then(()=>{
        if (!this.isVisible()||serial!==this.drawSerial) return;
        this.rememberCamera();
        return Plotly.react(this.scene,traces,{paper_bgcolor:'#090e18',font:{color:'#9aacc3'},margin:{l:0,r:0,t:0,b:0},showlegend:false,
          uirevision:'episode-'+e.id,scene:{bgcolor:'#090e18',aspectmode:'data',dragmode:'orbit',camera:structuredClone(this.camera||defaultCamera),
            xaxis:{title:{text:'X (m)'},gridcolor:'#22304a'},yaxis:{title:{text:'Y (m)'},gridcolor:'#22304a'},zaxis:{title:{text:'Z (m)'},gridcolor:'#22304a'}}},
          {responsive:true,displaylogo:false}).then(()=>{if(!this.isVisible()){Plotly.purge(this.scene);return;}this.captureCamera();});
      });
    }
    close() {this.drawSerial++;this.rememberCamera();Plotly.purge(this.scene);}
  }
  for (const details of document.querySelectorAll('details.path-example')) {
    let viewer=null;
    details.addEventListener('toggle',()=>{
      if (details.open) {
        if (!viewer) viewer=new Viewer(details);else viewer.draw();
        requestAnimationFrame(()=>{if(viewer.scene._fullLayout)Plotly.Plots.resize(viewer.scene);});
      } else if (viewer) viewer.close();
    });
    for (let parent=details.parentElement;parent;parent=parent.parentElement) if (parent.tagName==='DETAILS') parent.addEventListener('toggle',()=>{
      if (!viewer) return;if(!parent.open)viewer.close();else if(details.open)viewer.draw();
    });
  }
})();
