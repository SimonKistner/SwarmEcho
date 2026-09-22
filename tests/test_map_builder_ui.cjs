// Run with: node --test tests/test_map_builder_ui.cjs (no Python environment).
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
