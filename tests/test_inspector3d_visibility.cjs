const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const source = readFileSync(join(__dirname, '../src/swarmecho/visualize/inspector3d.py'), 'utf8');
const script = source.split('</aside></div><script>')[1].split('</script></body>')[0];

test('storey-filtered walls, transparency, height fade, floor grid and camera elevation', () => {
    const elements = new Map();
    function element() { return {checked:false,value:'0',options:[],style:{},children:[],classList:{toggle(){}},
        replaceChildren(){this.children=[]},append(...items){this.children.push(...items)},
        querySelectorAll(){return this.children},
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
        assert.equal($('artifactSubmenu').children.length,3);
        assert.equal(artifactParts('run_[2M]_[Replay]').detail,'2M · Replay');
        const e=DEFAULT_CAMERA.eye;
        assert.ok(Math.abs(Math.atan2(e.z,Math.hypot(e.x,e.y))*180/Math.PI-25)<1e-9);
        D.building=null;setupBuildingVisibility();assert.equal(buildingTraces().length,0);
    `, context);
});
