import test from 'node:test';
import assert from 'node:assert/strict';
import { EvidenceShare, canShareEvidence } from '../vmd/static/evidence-share.mjs';
import { ApiClient } from '../vmd/static/ui.mjs';

const evidence = (id = 'saved') => ({id, camera_name: 'Saved camera', event_type: 'fight', mode: 'live',
  review: 'unreviewed', signals: {live_camera: false}, clip: 'evidence.avi'});
const configured = {configured: true, base_url: 'https://evidence.example', message: 'Ready'};
const empty = {active_count: 0, latest_expires_at: null};
const future = () => Date.now()/1000 + 3600;
const link = (id = 'saved') => ({incident_id: id, url: `https://evidence.example/e/${id}`, expires_at: future()});

async function withDialog(run) {
  const previous = globalThis.document, clipboard = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  const node = tag => ({tag, value: '', textContent: '', hidden: false, disabled: false, open: false,
    children: [], attributes: {}, listeners: {},
    append(...children) {this.children.push(...children);},
    setAttribute(key, value) {this.attributes[key] = value;}, removeAttribute(key) {delete this.attributes[key]; delete this[key];},
    addEventListener(key, fn) {this.listeners[key] = fn;},
    showModal() {this.open = true;}, close() {this.open = false; this.listeners.close?.();},
    focus() {this.focused = true;}, select() {this.selected = true;},
  });
  globalThis.document = {createElement: node, body: node('body')};
  const share = new EvidenceShare({incidents: [evidence()], api: {}});
  try { await run(share); }
  finally {
    share.nodes?.dialog.close(); globalThis.document = previous;
    if (clipboard) Object.defineProperty(globalThis, 'navigator', clipboard);
    else delete globalThis.navigator;
  }
}

test('sharing requires saved real evidence with a clip and excludes synthetic, unknown and dismissed incidents', () => {
  const event = evidence();
  assert.equal(canShareEvidence(event), true);
  assert.equal(canShareEvidence({...event, signals: {live_camera: true}}), true);
  for (const patch of [{mode: 'demo'}, {mode: 'presentation'}, {mode: undefined}, {signals: {}},
    {signals: {live_camera: 'false'}}, {signals: {live_camera: 1}}, {clip: null}, {clip: ''}, {review: 'false_positive'}])
    assert.equal(canShareEvidence({...event, ...patch}), false);
  assert.equal(canShareEvidence(null), false);
});

test('opening only reads status; explicit generation, session-only URL, and revocation target the saved incident', async () => withDialog(async share => {
  const requests = [];
  let active = 2;
  share.workspace.api.request = async (path, body, method, timeoutMs) => {
    requests.push({path, body, method, ...(timeoutMs === undefined ? {} : {timeoutMs})});
    if (path === '/evidence/status') return configured;
    if (method === 'DELETE') { const count = active; active = 0; return {revoked_count: count}; }
    if (body !== undefined) {active++; return link();}
    return {active_count: active, latest_expires_at: active ? future() : null};
  };
  await share.open(evidence());
  assert.deepEqual(requests, [{path: '/evidence/status', body: undefined, method: undefined},
    {path: '/incidents/saved/share', body: undefined, method: undefined}]);
  assert.equal(share.nodes.result.hidden, true);
  assert.equal(share.nodes.generate.disabled, false);
  assert.match(share.nodes.count.textContent, /2 active.*cannot be recovered/);
  share.workspace.selected = 'other-camera';
  await share.generate();
  assert.equal(requests[2].path, '/incidents/saved/share');
  assert.deepEqual(requests[2].body, {});
  assert.equal(requests[2].method, 'POST');
  assert.equal(requests[2].timeoutMs, 195000, 'conversion can finish within its 180-second server limit');
  assert.equal(share.nodes.input.readOnly, true);
  assert.equal(share.nodes.input.value, link().url);
  assert.equal(share.nodes.open.href, link().url);
  assert.equal(share.nodes.open.rel, 'noopener noreferrer');
  assert.match(share.nodes.expiry.textContent, /Expires:.*local time/);
  assert.match(share.nodes.message.textContent, /No message or call was sent/);
  share.nodes.dialog.close();
  assert.equal(share.nodes.input.value, '', 'a closed dialog retains no bearer URL in its input');
  assert.equal(share.nodes.open.href, undefined);
  await share.open(evidence());
  assert.equal(share.nodes.input.value, '', 'closing discards the bearer URL');
  assert.match(share.nodes.count.textContent, /3 active.*cannot be recovered/);
  await share.revoke();
  assert.deepEqual(requests.at(-1), {path: '/incidents/saved/share', body: {}, method: 'DELETE'});
  assert.match(share.nodes.message.textContent, /3 video links revoked.*Calls are unchanged/);
  assert.equal(share.nodes.revoke.disabled, true);
  assert.equal(share.nodes.open.href, undefined);
}));

test('offline/error states block generation; clipboard failure selects readonly URL and expiry disables copying/opening', async () => withDialog(async share => {
  let offline = true;
  const requests = [];
  share.workspace.api.request = async (path, body) => {
    requests.push({path, body});
    if (path === '/evidence/status') return {...configured, configured: !offline};
    return body === undefined ? empty : link();
  };
  await share.open(evidence());
  assert.equal(share.nodes.generate.disabled, true);
  assert.match(share.nodes.status.textContent, /offline.*Start the evidence tunnel/);
  await share.generate();
  assert.equal(requests.length, 2);
  offline = false; await share.refresh(); await share.generate();
  Object.defineProperty(globalThis, 'navigator', {configurable: true, value: {clipboard: {async writeText() {throw new Error('Denied');}}}});
  await share.copy();
  assert.equal(share.nodes.input.focused, true); assert.equal(share.nodes.input.selected, true);
  assert.match(share.nodes.message.textContent, /copy it manually/);
  let copied;
  navigator.clipboard.writeText = async value => {copied = value;};
  await share.copy(); assert.equal(copied, link().url);
  share.session.expires = Date.now()/1000 - 1; share.render();
  assert.equal(share.nodes.copy.disabled, true); assert.equal(share.nodes.open.hidden, true);
  assert.match(share.nodes.expiry.textContent, /Expired:/);
  copied = null; await share.copy(); assert.equal(copied, null);
  share.workspace.api.request = async () => {throw new Error('Service offline');};
  await share.refresh();
  assert.equal(share.nodes.generate.disabled, true); assert.match(share.nodes.error.textContent, /Service offline/);
}));

test('late reads and mutations cannot overwrite another incident or create duplicate requests while busy', async () => withDialog(async share => {
  const original = evidence(), next = evidence('next'), requests = [];
  let resolveRead, resolveGenerate;
  share.workspace.api.request = (path, body) => {
    requests.push({path, body});
    if (path === '/evidence/status') return Promise.resolve(configured);
    if (path.includes('/next/')) return Promise.resolve(empty);
    if (body !== undefined) return new Promise(resolve => {resolveGenerate = resolve;});
    return new Promise(resolve => {resolveRead = resolve;});
  };
  const first = share.open(original);
  await share.open(next);
  resolveRead({active_count: 99, latest_expires_at: future()}); await first;
  assert.equal(share.session.id, 'next'); assert.equal(share.session.links.active_count, 0);
  const second = share.open(original);
  resolveRead(empty); await second;
  const generating = share.generate(); await share.generate();
  assert.equal(requests.filter(row => row.body !== undefined).length, 1);
  share.nodes.dialog.close(); await share.open(next);
  resolveGenerate(link()); await generating;
  assert.equal(share.session.id, 'next'); assert.equal(share.nodes.input.value, '');
  assert.equal(share.session.links.active_count, 0);
}));

test('invalid link responses and failed revoke leave no unsafe open action or false success', async () => withDialog(async share => {
  share.workspace.api.request = async (path, body, method) => {
    if (method === 'DELETE') throw new Error('Revoke failed');
    if (path === '/evidence/status') return configured;
    return body === undefined ? {active_count: 1, latest_expires_at: future()} : {...link(), url: 'javascript:alert(1)'};
  };
  await share.open(evidence()); await share.generate();
  assert.equal(share.nodes.open.href, undefined); assert.match(share.nodes.error.textContent, /valid video link/);
  assert.match(share.nodes.error.textContent, /Check sharing status before retrying/);
  await share.revoke();
  assert.match(share.nodes.error.textContent, /Revoke failed/);
  assert.equal(share.session.links.active_count, 1); assert.equal(share.nodes.revoke.disabled, false);
}));

test('API requests keep twelve-second defaults and allow a bounded conversion timeout without waiting', async () => {
  const oldFetch = globalThis.fetch, oldTimeout = AbortSignal.timeout, durations = [], requests = [], signals = [];
  AbortSignal.timeout = milliseconds => {
    durations.push(milliseconds);
    const signal = new AbortController().signal; signals.push(signal); return signal;
  };
  globalThis.fetch = async (url, options) => {requests.push({url, options}); return {ok: true, async json() {return {ok: true};}};};
  try {
    const api = new ApiClient();
    await api.request('/evidence/status');
    await api.request('/incidents/saved/share', {}, 'POST', 195000);
    assert.deepEqual(durations, [12000, 195000]);
    assert.equal(requests[0].url, '/api/evidence/status');
    assert.equal(requests[0].options.method, undefined);
    assert.equal(requests[1].options.method, 'POST');
    assert.equal(requests[1].options.body, '{}');
    assert.equal(requests[1].options.headers['X-VMD-Client'], 'dashboard');
    requests.forEach((request, index) => assert.equal(request.options.signal, signals[index]));
  } finally {globalThis.fetch = oldFetch; AbortSignal.timeout = oldTimeout;}
});
