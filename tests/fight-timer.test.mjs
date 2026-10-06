import test from 'node:test';
import assert from 'node:assert/strict';
import {liveFightTimer, renderFightTimer} from '../vmd/static/camera-settings.mjs';

const camera = {status: 'running', settings: {fight_confirmation_seconds: 9},
  alert: {event_type: 'possible_fight', signals: {fight_timer_seconds: 99}},
  signals: {fight_timer_state: 'counting', fight_timer_seconds: 1.25,
    fight_timer_required_seconds: 2, fight_timer_reason: 'Supported striking continues'}};

test('live fight timer uses only backend evidence time and current configured requirement', () => {
  assert.deepEqual(liveFightTimer(camera), {state: 'counting', seconds: 1.25, required: 2,
    label: 'Possible fight', value: '1.3 / 2.0 s', reason: 'Supported striking continues'});
  const later = {...camera, signals: {...camera.signals, fight_timer_seconds: .125}};
  assert.equal(liveFightTimer(later).value, '0.1 / 2.0 s', 'backend reset must replace earlier progress');
  assert.equal(liveFightTimer({...camera, signals: {...camera.signals, fight_timer_seconds: 0}}).value, '0.0 / 2.0 s');
});

test('elapsed time never invents confirmation and paused or depth-waiting states retain backend values', () => {
  for (const state of ['counting', 'paused', 'waiting_depth', 'waiting_strikes']) {
    const timer = liveFightTimer({...camera, signals: {...camera.signals,
      fight_timer_state: state, fight_timer_seconds: 2.75, fight_timer_reason: 'Fresh matching depth is required'}});
    assert.equal(timer.label, 'Possible fight');
    assert.equal(timer.value, '2.8 / 2.0 s');
    assert.equal(timer.reason, 'Fresh matching depth is required');
  }
  assert.equal(liveFightTimer({...camera, signals: {...camera.signals, fight_timer_state: 'confirmed'}}).label, 'Fight detected');
});

test('pending contact depth is Checking interaction until the backend accepts a strike onset', () => {
  const signals = {...camera.signals, fight_timer_state: 'waiting_depth', fight_timer_seconds: 0,
    fight_timer_started: false, fight_timer_reason: 'Checking depth at the striking contact frame'};
  assert.equal(liveFightTimer({...camera, signals}).label, 'Checking interaction');
  assert.equal(liveFightTimer({...camera, signals}).value, '0.0 / 2.0 s');
  assert.equal(liveFightTimer({...camera, signals: {...signals, fight_timer_started: true}}).label, 'Possible fight');
  delete signals.fight_timer_started;
  assert.equal(liveFightTimer({...camera, signals}).label, 'Possible fight', 'older snapshots retain their prior behavior');
});

test('active backend fight alert remains visible through a short paused check without fabricating timer progress', () => {
  const signals = {...camera.signals, fight_timer_state: 'paused', fight_timer_started: true,
    fight_timer_reason: 'Waiting for supported movement to resume'};
  const active = {...camera, alert: {event_type: 'fight'}, signals};
  const timer = liveFightTimer(active);
  assert.equal(timer.label, 'Fight detected');
  assert.equal(timer.state, 'paused');
  assert.equal(timer.seconds, 1.25);
  assert.equal(timer.reason, 'Fight detected · latest check: Waiting for supported movement to resume');
  assert.equal(liveFightTimer({...active, alert: null}).label, 'Possible fight');
  assert.equal(liveFightTimer({...active, review: 'false_positive'}).label, 'Possible fight');
  assert.equal(liveFightTimer({...active, signals: {...signals, fight_timer_started: false}}).label, 'Checking interaction');
  assert.equal(liveFightTimer({...active, stale: true}), null);
  assert.equal(liveFightTimer({...active, status: 'finished'}), null);
});

test('historical, stopped, stale, idle and malformed data cannot display a live fight timer', () => {
  const overrides = [null, {status: 'idle'}, {status: 'finished'}, {status: 'error'}, {stale: true},
    {presentation: true}, {signals: {}},
    ...['idle', '', 'invented'].map(fight_timer_state => ({signals: {...camera.signals, fight_timer_state}})),
    ...[-1, NaN, Infinity, '1.2', null].map(fight_timer_seconds => ({signals: {...camera.signals, fight_timer_seconds}})),
    ...[0, -1, Infinity, '2', null].map(fight_timer_required_seconds => ({signals: {...camera.signals, fight_timer_required_seconds}})),
  ];
  for (const override of overrides)
    assert.equal(liveFightTimer(override === null ? null : {...camera, ...override}), null, JSON.stringify(override));
});

class Node {
  constructor(tag = 'div') {this.tag = tag; this.children = []; this.dataset = {}; this.attributes = {};}
  get childElementCount() {return this.children.length;}
  append(...nodes) {this.children.push(...nodes);}
  replaceChildren(...nodes) {this.children = nodes;}
  setAttribute(key, value) {this.attributes[key] = value;}
  querySelector(selector) {
    for (const node of this.children) {
      if (selector === node.tag || selector.startsWith('.') && node.className === selector.slice(1)) return node;
      const nested = node.querySelector(selector);
      if (nested) return nested;
    }
    return null;
  }
}

test('timer renders accessible clamped progress, preserves backend reason and removes stale progress', () => {
  const originalDocument = globalThis.document;
  globalThis.document = {createElement: tag => new Node(tag)};
  try {
    const node = new Node();
    renderFightTimer(node, camera);
    assert.equal(node.hidden, false);
    assert.equal(node.querySelector('progress').value, 1.25);
    assert.equal(node.querySelector('progress').max, 2);
    assert.match(node.querySelector('progress').attributes['aria-valuetext'], /Supported striking/);
    const reason = 'Waiting for <matching> depth';
    renderFightTimer(node, {...camera, signals: {...camera.signals, fight_timer_state: 'waiting_depth',
      fight_timer_seconds: 3, fight_timer_reason: reason}});
    assert.equal(node.querySelector('progress').value, 2, 'only the progress bar is capped');
    assert.equal(node.querySelector('.fight-timer-value').textContent, '3.0 / 2.0 s');
    assert.equal(node.querySelector('.fight-timer-title').textContent, 'Possible fight');
    assert.equal(node.querySelector('.fight-timer-reason').textContent, reason);
    assert.equal(node.dataset.state, 'waiting_depth');
    renderFightTimer(node, {...camera, stale: true});
    assert.equal(node.hidden, true);
    assert.equal(node.childElementCount, 0);
    assert.equal(node.dataset.state, undefined);
    renderFightTimer(node, {...camera, signals: {...camera.signals, fight_timer_state: 'confirmed'}});
    assert.equal(node.querySelector('.fight-timer-title').textContent, 'Fight detected');
  } finally {globalThis.document = originalDocument;}
});
