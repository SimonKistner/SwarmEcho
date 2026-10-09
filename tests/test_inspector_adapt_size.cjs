const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {createContext, runInContext} = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');

const source = readFileSync(join(__dirname, '../src/swarmecho/visualize/inspector.py'), 'utf8');
const script = source.split('</aside></div><script>')[1].split('</script></body>')[0];
const statusFunctions = 'function renderKnownList' + script.split('function renderKnownList')[1].split('const REWARD_COLORS')[0];

test('adaptive replay inspector shows current lifetimes, votes, dwell and base receipts', () => {
    const elements = new Map();
    function element(id = '') {
        return {
            id, children: [], hidden: false, textContent: '', title: '',
            addEventListener() {},
            classList: {toggle() {}},
            replaceChildren(...items) {this.children = items;},
            appendChild(item) {this.children.push(item); if (item.id) elements.set(item.id, item);},
            querySelectorAll() {return this.children;},
        };
    }
    const context = createContext({
        document: {
            getElementById(id) {
                if (!elements.has(id) && id !== 'adaptiveBaseCard') elements.set(id, element(id));
                return elements.get(id) || null;
            },
            createElement: () => element(), addEventListener() {},
        },
        fetch: () => new Promise(() => {}),
    });
    runInContext('let D, frame=0; const $=id=>document.getElementById(id);' + statusFunctions, context);
    runInContext(`
        D = {
            manifest: {adapt_size: {enabled:true, vote_holding:10, decommission_hold_steps:15}},
            active: [[true,false,true],[true,true,true]],
            adapt_vote: [[true,false,false],[false,true,false]],
            adapt_ids: [[0,1,2],[0,1,3]],
            adapt_dwell: [[4,0,0],[0,8,0]],
            adapt_round: [1,2],
            adapt_available_votes: [[7,0,0],[0,4,0]],
            adapt_base_votes: [[7,0,0],[0,14,0]],
            adapt_spent: [[0,0,0],[0,10,0]],
        };
        renderKnownList(3);
        frame=0;updateKnownList([true,false,false],true);
    `, context);
    const rows = elements.get('knownList').children;
    assert.match(rows[1].textContent, /YES.*4\/15/);
    assert.equal(rows[2].hidden, true);
    assert.equal(rows[3].hidden, false);
    let card = elements.get('adaptiveBaseCard');
    assert.match(card.children[0].textContent, /Round 1/);
    assert.equal(card.children.length, 3);
    assert.match(card.children[1].textContent, /ID 0.*7\/10/);

    runInContext('frame=1;updateKnownList([true,true,false],true)', context);
    card = elements.get('adaptiveBaseCard');
    assert.equal(rows[2].hidden, false);
    assert.match(rows[2].textContent, /YES.*8\/15/);
    assert.match(rows[3].title, /Lifetime ID 3/);
    assert.match(card.children[0].textContent, /Round 2/);
    assert.match(card.children[2].textContent, /ID 1.*4\/10/);
    assert.match(card.children[2].title, /consumed by rejected calls: 10/);
});
