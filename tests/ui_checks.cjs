'use strict';

// Opt-in browser checks. Run with node --test tests/ui_checks.cjs.

// inspector_adapt_size
{
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {createContext, runInContext} = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');

const source = readFileSync(join(__dirname, '../src/swarmecho/visualize/inspector.py'), 'utf8');
const script = source.split('</aside></div><script>')[1].split('</script></body>')[0];
const statusFunctions = 'function renderKnownList' + script.split('function renderKnownList')[1].split('const rewardDebugPanel')[0];

test('adaptive replay inspector shows drone lifetimes and decommission dwell', () => {
    const elements = new Map();
    function element(id = '') {
        return {
            id, children: [], hidden: false, textContent: '', title: '',
            addEventListener() {},
            classList: {toggle() {}},
            replaceChildren(...items) {this.children = items;},
            appendChild(item) {this.children.push(item); if (item.id) elements.set(item.id, item);},
            insertBefore(item, before) {
                const index = this.children.indexOf(before);
                this.children.splice(index < 0 ? this.children.length : index, 0, item);
                if (item.id) elements.set(item.id, item);
            },
            querySelectorAll() {return this.children;},
        };
    }
    const context = createContext({
        document: {
            getElementById(id) {
                if (!elements.has(id)) elements.set(id, element(id));
                return elements.get(id) || null;
            },
            createElement: () => element(), addEventListener() {},
        },
        fetch: () => new Promise(() => {}),
    });
    runInContext('let D, frame=0; const $=id=>document.getElementById(id);' + statusFunctions, context);
    runInContext(`
        D = {
            manifest: {adapt_size: {enabled:true, decommission_hold_steps:15}},
            active: [[true,false,true],[true,true,true]],
            adapt_ids: [[0,1,2],[0,1,3]],
            adapt_dwell: [[4,0,0],[0,8,0]],
        };
        renderKnownList(3);
        frame=0;updateKnownList([true,false,false],true);
    `, context);
    const rows = elements.get('knownList').children;
    assert.match(rows[1].textContent, /Decommission 4\/15/);
    assert.equal(rows[2].hidden, true);
    assert.equal(rows[3].hidden, false);

    runInContext('frame=1;updateKnownList([true,true,false],true)', context);
    assert.equal(rows[2].hidden, false);
    assert.match(rows[2].textContent, /Decommission 8\/15/);
    assert.match(rows[3].title, /Lifetime ID 3/);
});

test('heatmap replay commands use authored paths or saved suite IDs', () => {
    const controls = {selectedPoint: {}, heatmapCommand: {}};
    const context = createContext({
        $: id => controls[id],
        PUBLIC_INSPECTOR: false,
        H: {manifest: {checkpoint: '/runs/my run/checkpoints/best', source_map_path: "/maps/team's map.yaml"}},
    });
    const functions = script.slice(script.indexOf('function quoteShellArg'), script.indexOf('\n}', script.indexOf('function showHeatmapCommand')) + 2);
    runInContext(functions + "\nshowHeatmapCommand([1, 2, 3], 'Delivered');", context);
    assert.ok(controls.heatmapCommand.value.includes("checkpoint='/runs/my run/checkpoints/best'"));
    assert.ok(controls.heatmapCommand.value.endsWith("eval_map='/maps/team'\\''s map.yaml'"));

    context.H.manifest = {checkpoint: '/runs/best', map_suite_map_id: 'static_eval_0000'};
    runInContext("showHeatmapCommand([1, 2, 3], 'Delivered');", context);
    assert.ok(controls.heatmapCommand.value.endsWith("map_suite_map_id='static_eval_0000'"));
});
}

// inspector_navigation
{
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {Script, createContext, runInContext} = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');

const source = readFileSync(join(__dirname, '../src/swarmecho/visualize/inspector.py'), 'utf8');
const script = source.split('</aside></div><script>')[1].split('</script></body>')[0];

test('category, step and map rows each handle their own arrow navigation', () => {
    const elements = new Map(), listeners = new Map();
    function element(id = '') {
        const classes = new Set(id === 'mapNavigation' || id === 'mapNavigationHelp' ? ['hidden'] : []);
        return {
            id, children: [], dataset: {}, style: {}, attributes: {}, value: '', disabled: false,
            classList: {
                toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); },
                contains(name) { return classes.has(name); },
            },
            replaceChildren(...children) { this.children = children; },
            appendChild(child) { this.children.push(child); },
            setAttribute(name, value) { this.attributes[name] = String(value); },
        };
    }
    const context = createContext({
        document: {
            getElementById(id) { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); },
            createElement: () => element(),
            addEventListener(name, listener) { listeners.set(name, listener); },
        },
        fetch: () => new Promise(() => {}),
    });
    new Script(script).runInContext(context);
    context.fixture = [
        {id:'h8a', run_id:'run', kind:'heatmap', label:'run_[8M]_[Heatmap]', random_eval_group:'step8', random_eval_map_id:'random_eval_0000'},
        {id:'h8b', run_id:'run', kind:'heatmap', label:'run_[8M]_[Heatmap]', random_eval_group:'step8', random_eval_map_id:'random_eval_0001'},
        {id:'h15a', run_id:'run', kind:'heatmap', label:'run_[15M]_[Heatmap]', random_eval_group:'step15', random_eval_map_id:'random_eval_0000'},
        {id:'h15b', run_id:'run', kind:'heatmap', label:'run_[15M]_[Heatmap]', random_eval_group:'step15', random_eval_map_id:'random_eval_0001'},
        {id:'r8a', run_id:'run', kind:'replay', label:'run_[8M]_[Replay]', random_eval_group:'step8', random_eval_map_id:'random_eval_0000'},
        {id:'r8b', run_id:'run', kind:'replay', label:'run_[8M]_[Replay]', random_eval_group:'step8', random_eval_map_id:'random_eval_0001'},
        {id:'single', run_id:'run', kind:'replay', label:'run_[20M]_[Replay]'},
    ];
    runInContext("artifactItems=fixture;artifactBusy=false;$('replaySelect').value='h8b';chooseArtifact=item=>{$('replaySelect').value=item.id;updateArtifactNavigation()};updateArtifactNavigation()", context);
    const get = id => elements.get(id);
    assert.equal(get('mapNavigation').classList.contains('hidden'), false);
    assert.equal(get('inspectorHeader').classList.contains('has-map-navigation'), true);
    assert.deepEqual(get('mapChoices').children.map(button => button.textContent), ['Map 0000', 'Map 0001']);
    assert.deepEqual(get('stepChoices').children.map(button => button.textContent), ['8M', '15M']);
    assert.equal(get('mapChoices').children[1].attributes['aria-pressed'], 'true');
    assert.equal(get('heatmapMode').attributes['aria-pressed'], 'true');
    assert.equal(get('categoryNavigation').dataset.active, 'true');

    const key = name => listeners.get('keydown')({key:name, target:{closest:() => false}, preventDefault(){}, defaultPrevented:false});
    key('ArrowRight');
    assert.equal(get('replaySelect').value, 'single'); // Newest replay step.
    assert.equal(get('replayMode').attributes['aria-pressed'], 'true');
    key('ArrowLeft');
    assert.equal(get('replaySelect').value, 'h15a'); // Newest heatmap step.
    assert.equal(get('stepChoices').children[1].attributes['aria-pressed'], 'true');
    key('ArrowDown');
    assert.equal(get('stepNavigation').dataset.active, 'true');
    key('ArrowLeft');
    assert.equal(get('replaySelect').value, 'h8a');
    assert.equal(get('previousArtifact').disabled, true);
    key('ArrowRight');
    assert.equal(get('replaySelect').value, 'h15a');
    key('ArrowDown');
    assert.equal(get('mapNavigation').dataset.active, 'true');
    key('ArrowRight');
    assert.equal(get('replaySelect').value, 'h15b');
    assert.equal(get('mapChoices').children[1].attributes['aria-pressed'], 'true');
    key('ArrowUp');
    key('ArrowLeft');
    assert.equal(get('replaySelect').value, 'h8b'); // Preserve map ID across steps.
    assert.equal(get('heatmapMode').attributes['aria-pressed'], 'true');

    get('replayMode').onclick();
    assert.equal(get('replaySelect').value, 'single');
    assert.equal(get('replayMode').attributes['aria-pressed'], 'true');
    key('ArrowDown');
    key('ArrowLeft');
    assert.equal(get('replaySelect').value, 'r8a');
    key('ArrowDown');
    key('ArrowRight');
    assert.equal(get('replaySelect').value, 'r8b');
    assert.equal(get('mapChoices').children[1].attributes['aria-pressed'], 'true');

    runInContext("$('replaySelect').value='single';updateArtifactNavigation()", context);
    assert.equal(get('mapNavigation').classList.contains('hidden'), true);
    assert.equal(get('inspectorHeader').classList.contains('has-map-navigation'), false);
    assert.equal(get('stepNavigation').dataset.active, 'true');
});

test('startup requests recent runs and load more requests the complete list', async () => {
    const elements = new Map(), pending = [];
    function element() {
        return {
            children: [], dataset: {}, style: {}, value: '',
            classList: {toggle(){}, contains(){return true}},
            replaceChildren(){this.children=[]}, appendChild(child){this.children.push(child)},
            setAttribute(){},
        };
    }
    const context = createContext({
        document: {
            getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id)},
            createElement: element, addEventListener(){},
        },
        fetch(url){return new Promise(resolve=>pending.push({url,resolve}))},
    });
    new Script(script).runInContext(context);
    runInContext("renderReplayList=async items=>{artifactItems=items;$('replaySelect').value=items[0]?.id||''};loadReplay=async()=>{};updateArtifactNavigation=()=>{};setLoading=()=>{}", context);
    assert.equal(pending[0].url, '/api/replays?recent=3');
    pending[0].resolve({ok:true,headers:{get:()=> 'true'},json:async()=>[{id:'recent',label:'recent_[1M]_[Replay]'}]});
    await new Promise(setImmediate);
    assert.equal(runInContext('moreRunsAvailable', context), true);
    assert.equal(runInContext('showingAllRuns', context), false);

    const fullRequest = runInContext('refreshArtifacts(false,true)', context);
    assert.equal(pending[1].url, '/api/replays');
    pending[1].resolve({ok:true,headers:{get:()=> 'false'},json:async()=>[
        {id:'recent',label:'recent_[1M]_[Replay]'},
        {id:'archived',label:'archived_[1M]_[Replay]'},
    ]});
    await fullRequest;
    assert.equal(runInContext('showingAllRuns', context), true);
    assert.equal(runInContext('moreRunsAvailable', context), false);
    assert.equal(runInContext('artifactItems.length', context), 2);
});
}

// inspector_visibility
{
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const source = readFileSync(join(__dirname, '../src/swarmecho/visualize/inspector.py'), 'utf8');
const script = source.split('</aside></div><script>')[1].split('</script></body>')[0];

test('storey-filtered walls, transparency, height fade, floor grid and camera elevation', () => {
    const elements = new Map();
    function element() { return {checked:false,value:'0',options:[],style:{},children:[],classList:{toggle(){}},
        replaceChildren(){this.children=[]},append(...items){this.children.push(...items)},
        querySelectorAll(){return this.children},
        setAttribute(name,value){this[name]=String(value)},
        appendChild(item){this.children.push(item)}}; }
    const context = vm.createContext({assert, fetch:()=>new Promise(()=>{}),
        document:{getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id)},
            createElement:element,createTextNode:text=>text,addEventListener(){}},
    });
    vm.runInContext(script, context);
    vm.runInContext(`
        const wall=(storey,outer)=>({storey,outer,lo:[0,0,storey*5],hi:[.25,5,(storey+1)*5]});
        D={manifest:{map_name:'test'},building:{layers:2,walls:[wall(0,false),wall(1,false),wall(0,true),wall(1,true)],roofs:[{lo:[0,0,10],hi:[5,5,10.25]}]}};
        setupBuildingVisibility();
        $('wallTransparency').value='50';
        assert.equal(visibleStoreys.join(','),'true,false');assert.equal(Math.max(...buildingTraces()[0].z),5);
        assert.equal($('showOuterWalls').checked,false);assert.equal($('showRoof').checked,false);
        visibleStoreys=[false,false];$('showOuterWalls').checked=true;
        assert.equal(buildingTraces().length,0);
        visibleStoreys=[true,false];const count=buildingTraces()[0].x.length;
        $('showOuterWalls').checked=false;assert.equal(buildingTraces()[0].x.length,count/2);
        visibleStoreys=[false,false];
        $('showRoof').checked=true;assert.equal(buildingTraces()[0].x.length,8);
        $('showOuterWalls').checked=false;assert.equal(buildingTraces()[0].x.length,8);
        setupBuildingVisibility(true);assert.equal($('showRoof').checked,true);
        $('showRoof').checked=false;assert.equal(buildingTraces().length,0);
        D.building.floors=[{lo:[0,0,-.25],hi:[5,5,.25]}];$('showFloor').checked=true;$('showCellGrid').checked=true;
        assert.equal(buildingTraces().length,2);assert.equal(buildingTraces()[0].facecolor.length,12);
        $('showFloor').checked=false;visibleStoreys=[true,false];
        const mesh=buildingTraces()[0];assert.ok(mesh.x.length>8);assert.equal(mesh.facecolor.length,mesh.i.length);assert.equal(mesh.opacity,.5);
        $('heightFade').checked=true;$('heightFadeStart').value='0';
        assert.equal(wallTransparencyAt(0,0,10),0);assert.equal(wallTransparencyAt(5,0,10),.25);assert.equal(wallTransparencyAt(10,0,10),.5);
        $('heightFadeStart').value='50';assert.equal(wallTransparencyAt(5,0,10),0);assert.equal(wallTransparencyAt(7.5,0,10),.25);
        assert.ok(buildingTraces().every(trace=>trace.opacity>=.5&&trace.opacity<=1));
        $('heightFade').checked=false;$('wallTransparency').value='100';assert.equal(buildingTraces().length,0);
        renderReplayList([{id:'a',label:'run_[1M]_[Replay]'},{id:'b',label:'run_[2M]_[Replay]'},{id:'c',label:'other_[3M]_[Replay]'}]);
        assert.equal($('artifactMenu').children.length,2);
        const columns=$('artifactSubmenu').children.find(item=>item.className==='artifact-columns');
        assert.ok(columns);
        const replayButtons=columns.children.flatMap(item=>item.children).filter(item=>item.className==='artifact-entry');
        assert.equal(replayButtons.length,2);
        assert.equal(replayButtons.map(item=>item.textContent).sort().join('|'),'1M · Replay|2M · Replay');
        assert.equal(artifactParts('run_[2M]_[Replay]').detail,'2M · Replay');
        const e=DEFAULT_CAMERA.eye;
        assert.ok(Math.abs(Math.atan2(e.z,Math.hypot(e.x,e.y))*180/Math.PI-45)<1e-9);
        D.building=null;setupBuildingVisibility();assert.equal(buildingTraces().length,0);
    `, context);
});
}

// map_builder_ui
{
// Run with: node --test tests/ui_checks.cjs (no Python environment).
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const html = readFileSync(join(__dirname, '../src/swarmecho/curriculum_config/maps/scripts/maze_builder/index.html'), 'utf8');

function editor() {
    const noop = () => {};
    const gl = new Proxy({}, {get: (_, key) => /^[A-Z_]+$/.test(key) ? 1 : () => true});
    const ctx = new Proxy({}, {get: () => noop});
    function element() {
        return {
            dataset: {}, style: {}, options: [], children: [], checked: true,
            set innerHTML(value) { this.options = []; this.children = []; },
            add(option) { this.options.push(option); },
            append(child) { this.children.push(child); },
            getBoundingClientRect: () => ({left: 0, top: 0, width: 800, height: 600}),
            getContext: kind => kind === 'webgl' ? gl : ctx,
            addEventListener: noop, setPointerCapture: noop,
        };
    }
    const elements = new Map();
    const context = vm.createContext({
        assert, devicePixelRatio: 1,
        document: {
            getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
            createElement: element, querySelectorAll: () => [],
        },
        Option: function(text, value) { this.text = text; this.value = value; },
        ResizeObserver: class { observe() {} },
        fetch: () => new Promise(() => {}),
    });
    const run = source => vm.runInContext(source, context);
    run(html.split('<script>')[1].split('</script>')[0]);
    run(`doc={name:'test',cols:1,rows:1,layers:1,cell_size_m:5,tile_thickness_m:.25,wall_thickness_m:.25,
        interior_cells:[[0,0,0]],tiles:[],x_walls:[],y_walls:[],target_exclusion_cells:[],base_cell:null};`);
    return run;
}

test('empty volume has no opaque floor; exclusion display is independent of tiles', () => {
    editor()(`
        draw();assert.equal(faces.length,0);assert.ok(edges.length>0);
        doc.target_exclusion_cells=[[0,0,0]];draw();assert.ok(faces.length>0);
        $('showExclusions').checked=false;$('showExclusions').onchange();assert.equal(faces.length,0);
        assert.equal(doc.target_exclusion_cells.length,1);
        $('showExclusions').checked=true;draw();assert.ok(faces.length>0);
    `);
});

test('tile painting uses tile state and preserves volume, exclusions and base', () => {
    editor()(`
        doc.target_exclusion_cells=[[0,0,0]];doc.base_cell=[0,0,0];
        const before=JSON.stringify([doc.interior_cells,doc.target_exclusion_cells,doc.base_cell]);
        const p=project(.5,.5,0),event={clientX:p[0],clientY:p[1],pointerId:1};
        beginEdit(event);assert.equal(selection.add,true);endEdit();assert.equal(doc.tiles.length,1);
        beginEdit(event);assert.equal(selection.add,false);endEdit();assert.equal(doc.tiles.length,0);
        assert.equal(JSON.stringify([doc.interior_cells,doc.target_exclusion_cells,doc.base_cell]),before);
        mode='exclude';beginEdit(event);endEdit();assert.equal(doc.target_exclusion_cells.length,0);
        beginEdit(event);endEdit();assert.equal(doc.target_exclusion_cells.length,1);
    `);
});

test('layer selector replaces a former roof even when the option count is unchanged', () => {
    editor()(`
        doc.tiles=[[0,0,1]];updateUi();assert.equal($('layer').options[1].text,'Roof · inspection');
        doc.layers=2;doc.tiles=[];doc.interior_cells.push([0,0,1]);layer=1;draw();
        assert.equal($('layer').options.length,2);assert.equal($('layer').options[1].text,'Level 2');
        assert.equal(faces.length,0);assert.ok($('count').textContent.includes('1 volume cells'));
        doc.tiles=[[0,0,2]];updateUi();assert.equal($('layer').options[2].value,2);
        $('layer').onchange({target:{value:'2'}});assert.equal(layer,2);assert.ok(faces.length>0);
    `);
});

test('Add roof changes geometry without switching the current storey', async () => {
    await editor()(`
        fetch=async()=>({ok:true,json:async()=>({ok:true,document:{...doc,tiles:[[0,0,1]]}})});
        $('roofBtn').onclick().then(()=>{
            assert.equal(layer,0);assert.equal(doc.tiles[0][2],1);
            assert.equal($('layer').options[1].text,'Roof · inspection');
        });
    `);
});

test('wall drag previews a straight row and commits only on release, for add and erase', () => {
    for(const angle of [-.65,.65])editor()(`
        doc.cols=5;doc.rows=5;mode='walls';yaw=${angle};
        const p=project(.5,.5,0),event={clientX:p[0],clientY:p[1],pointerId:1};
        beginEdit(event);assert.equal(pointer.type,'walls');
        const kind=pointer.wall[0],key=kind==='x'?'x_walls':'y_walls',other=kind==='x'?'y_walls':'x_walls';
        const a=project(0,0,0),b=project(kind==='y'?1:0,kind==='x'?1:0,0),dx=b[0]-a[0],dy=b[1]-a[1];
        const drag=n=>({clientX:event.clientX+n*dx,clientY:event.clientY+n*dy});
        moveEdit(drag(3));assert.equal(selectedWalls().length,4);assert.equal(doc[key].length,0);
        moveEdit(drag(2));assert.equal(selectedWalls().length,3);assert.equal(doc[key].length,0);
        const preview=[];polygon=(points,fill)=>preview.push(fill);drawWallPreview();
        assert.equal(preview.length,3);assert.ok(preview.every(fill=>fill==='#45ddea99'));
        endEdit();assert.equal(doc[key].length,3);assert.equal(doc[other].length,0);
        beginEdit(event);assert.equal(pointer.add,false);moveEdit(drag(2));
        assert.equal(selectedWalls().length,3);assert.equal(doc[key].length,3);
        preview.length=0;drawWallPreview();assert.equal(preview.length,3);
        assert.ok(preview.every(fill=>fill==='#ef526d99'));
        endEdit();assert.equal(doc[key].length,0);
    `);
});

test('wall picking accepts cell centres again and cancelled previews change nothing', () => {
    editor()(`
        doc.cols=5;doc.rows=5;mode='walls';
        const p=project(2.5,2.5,0),event={clientX:p[0],clientY:p[1],pointerId:1};
        assert.ok(nearestWall(event.clientX,event.clientY));
        beginEdit(event);assert.equal(pointer.type,'walls');
        moveEdit({clientX:p[0]+80,clientY:p[1]+40});cancelEdit();
        assert.equal(pointer,null);assert.equal(doc.x_walls.length+doc.y_walls.length,0);
    `);
});
}

// geo_path_viewer_test
{
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
}
