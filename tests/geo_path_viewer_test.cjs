// DOM/Plotly contract test. This runs project JS against test doubles, without
// browser access, network, a GPU scene or any user's browser state.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');

class Element {
  constructor(value=''){this._value=String(value);this.options=[];this.listeners={};this.checked=false;this.textContent='';}
  get value(){return this._value;} set value(v){this._value=String(v);}
  replaceChildren(){this.options=[];this.value='';}
  append(option){this.options.push(option);if(this.options.length===1)this.value=option.value;}
  addEventListener(name, fn){this.listeners[name]=fn;}
  removeAllListeners(name){delete this.listeners[name];} on(name,fn){this.listeners[name]=fn;}
}
const controls=Object.fromEntries(['input','seed','agent','agent-control','snapshot','snapshot-control','mode','threshold','threshold-value','reset','opacity','walls','walls-value','outer','roof','grid','truth','example-status','path-scene'].map(k=>[k,new Element()]));
Object.assign(controls.threshold,{value:80});Object.assign(controls.opacity,{value:.65});Object.assign(controls.walls,{value:0});controls.mode.value='probability';controls.grid.checked=true;
const host={querySelector(selector){return controls[selector.slice(1)];},set innerHTML(value){this.markup=value;}};
const details=new Element();Object.assign(details,{open:false,dataset:{example:'0',source:'finder_gru'},parentElement:null,querySelector:()=>host});
const truth=Array(125).fill(0);truth[62]=truth[63]=1;
const good=truth.map(v=>v?.9:.1),partial=good.map((v,i)=>i===63?.2:v);
good[63]=.8;
const entry=(p,step=2)=>({agent:0,step,first:2,p});
const boundary=truth.map(v=>Math.fround(v?.95:.1));
const boundaryEntry={...entry(boundary,22),float32:true};
const example={id:0,map:'fixture',cell:5,rotation:1,anchor:[7.5,7.5,2.5],base:[7.5,7.5,2.5],target:[7.5,2.5,2.5],path:[[7.5,7.5,2.5],[7.5,2.5,2.5]],
  building:{walls:[{lo:[5,0,0],hi:[5.25,5,5],outer:false},{lo:[0,0,0],hi:[.25,5,5],outer:true}],floors:[{lo:[0,0,-.25],hi:[5,5,.25]}],roofs:[{lo:[0,0,5],hi:[5,5,5.25]}]},
  truth,sources:{finder_gru:{0:[entry(good)]},all_knowing_till_end:{0:[entry(partial),entry(good,12),boundaryEntry,{...entry(good,4),agent:1}]},base:{0:[]}}};
const offsets=[];for(let z=-2;z<=2;z++)for(let y=-2;y<=2;y++)for(let x=-2;x<=2;x++)offsets.push([x,y,z]);
const payload={examples:{0:example},labels:{finder_gru:'Finder GRU',all_knowing_till_end:'All knowing',base:'Base'},offsets,threshold:.8};
const plots=[];let purged=0;
let modeChanges=0,rotationArguments;
const Plotly={react(scene,traces,layout){
  plots.push({traces,layout});
  let live=scene._fullLayout?.scene?._scene;
  if(!live){
    let mode='orbit';
    const view={getMode:()=>mode,setMode(next){modeChanges++;mode=next;},rotate(...args){rotationArguments=args;}};
    live={camera:{view,lookAt(eye,center,up){live.pose={eye:{x:eye[0],y:eye[1],z:eye[2]},center:{x:center[0],y:center[1],z:center[2]},up:{x:up[0],y:up[1],z:up[2]}};}},
      getCamera(){return live.pose;},updateFx(){}};
  }
  live.pose=structuredClone(layout.scene.camera);live.fullSceneLayout={dragmode:'orbit',hovermode:'closest'};
  scene._fullLayout={scene:{camera:layout.scene.camera,_scene:live}};return Promise.resolve();
},purge(scene){purged++;delete scene._fullLayout;},Plots:{resize(){}}};
const reportThreshold=new Element(80),population=new Element('common');
const scoreCells=[];
for(const threshold of [50,75,80,90,95])for(const source of ['common','available'])scoreCells.push({dataset:{evaluationThreshold:String(threshold),evaluationPopulation:source}});
const document={getElementById(id){return id==='path-examples'?{textContent:JSON.stringify(payload)}:id==='held-out-threshold'?reportThreshold:population;},
  querySelectorAll(selector){return selector==='details.path-example'?[details]:scoreCells;},createElement:()=>new Element()};
const context=vm.createContext({document,Plotly,structuredClone,requestAnimationFrame:fn=>fn(),console});
vm.runInContext(fs.readFileSync(path.resolve(__dirname,'../geo_experiments/shared/reports/path_viewer.js'),'utf8'),context);
const flush=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
  assert.equal(plots.length,0,'closed viewers must allocate no scene');
  assert.equal(scoreCells.filter(cell=>!cell.hidden).length,1);
  reportThreshold.value=95;reportThreshold.onchange();
  assert.equal(scoreCells.find(cell=>!cell.hidden).dataset.evaluationThreshold,'95');
  population.value='available';population.onchange();
  assert.equal(scoreCells.find(cell=>!cell.hidden).dataset.evaluationPopulation,'available');
  reportThreshold.value=80;population.value='common';reportThreshold.onchange();
  details.open=true;details.listeners.toggle();await flush();
  assert(!host.markup.includes('class="coordinates"'),'only original building coordinates are offered');
  assert(controls['example-status'].textContent.includes('Exact match: yes'));
  assert(controls['agent-control'].hidden&&controls['snapshot-control'].hidden,'single-entry finder selectors are hidden');
  assert(plots.at(-1).traces.some(t=>t.type==='scatter3d'&&t.line?.width===7),'reference line stays visible');
  assert.equal(plots.at(-1).traces.find(t=>t.line?.width===7).x[0],7.5,'route stays in world coordinates');
  const blue=plots.at(-1).traces.filter(t=>t.color==='#38bdf8');
  assert.equal(blue.length,2,'distinct probabilities get distinct opacity, with the same blue');
  assert.deepEqual(Array.from(blue,t=>t.opacity).sort(),[.8*.65,.9*.65].sort());
  const second=blue.find(t=>t.text[0].includes('1, 0, 0'));
  assert.equal(Math.min(...second.x),5);assert.equal(Math.max(...second.x),10);
  assert.equal(Math.min(...second.y),0);assert.equal(Math.max(...second.y),5,'canonical cells map back into the original building');
  const solid=plots.at(-1).traces.find(t=>t.facecolor);
  assert(solid.facecolor.includes('#959b9c')&&solid.facecolor.includes('#252e39'),'inspector wall and floor materials');
  assert.equal(solid.opacity,1);
  const live=controls['path-scene']._fullLayout.scene._scene;
  assert.equal(live.camera.view.getMode(),'turntable');
  live.camera.view.rotate(1,.2,.3,.7);assert.equal(rotationArguments[3],0,'mouse/touch roll is suppressed');
  assert.equal(modeChanges,1);
  live.pose.eye={x:2,y:-1,z:1};
  controls['path-scene'].listeners.plotly_relayouting({'scene.camera.eye.x':2});
  controls.threshold.value=95;controls.threshold.oninput();await flush();
  assert.equal(plots.at(-1).layout.scene.camera.eye.x,2,'drag pose survives threshold redraw');
  assert.equal(plots.at(-1).layout.scene.camera.up.z,1);
  assert.equal(modeChanges,1,'redraw does not restart the turntable tween');
  assert(controls['example-status'].textContent.includes('Path similarity: 0.0%'));
  controls.reset.onclick();await flush();assert.equal(controls.threshold.value,'80');
  controls.input.value='all_knowing_till_end';controls.input.onchange();await flush();
  assert(!controls['agent-control'].hidden&&!controls['snapshot-control'].hidden,'multiple agents and snapshots offer real choices');
  assert(controls['example-status'].textContent.includes('66.7%'));
  controls.agent.value=1;controls.agent.onchange();await flush();
  assert(!controls['agent-control'].hidden&&controls['snapshot-control'].hidden,'a one-snapshot agent hides just its snapshot selector');
  controls.agent.value=0;controls.agent.onchange();await flush();
  assert(!controls['snapshot-control'].hidden);
  controls.snapshot.value=1;controls.snapshot.onchange();await flush();
  assert(controls['example-status'].textContent.includes('step 12'));
  assert(controls['example-status'].textContent.includes('Exact match: yes'));
  controls.snapshot.value=2;controls.snapshot.onchange();await flush();
  controls.threshold.value=95;controls.threshold.oninput();await flush();
  assert(controls['example-status'].textContent.includes('Exact match: yes'),'float32 threshold boundaries match aggregate scoring');
  controls.walls.value=50;controls.walls.oninput();await flush();
  assert.equal(plots.at(-1).traces.find(t=>t.facecolor).opacity,.5);
  const wallCount=plots.at(-1).traces.find(t=>t.facecolor).x.length;
  controls.outer.checked=true;controls.outer.onchange();await flush();
  assert(plots.at(-1).traces.find(t=>t.facecolor).x.length>wallCount);
  controls.input.value='base';controls.input.onchange();await flush();
  assert(controls['agent-control'].hidden&&controls['snapshot-control'].hidden,'unavailable inputs expose no empty choices');
  assert(controls['example-status'].textContent.includes('No snapshot available'));
  details.open=false;details.listeners.toggle();assert.equal(purged,1);
  details.open=true;details.listeners.toggle();await flush();assert(plots.length>=6);
  console.log('Viewer contract passed: aggregate thresholds/populations, inspector upright controls/materials, blue probability opacity, original coordinates, camera persistence, lazy cleanup/reopen.');
})().catch(error=>{console.error(error);process.exitCode=1;});
