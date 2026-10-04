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
