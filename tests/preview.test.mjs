import test from 'node:test';
import assert from 'node:assert/strict';
import {PollLoop, setPreviewImage} from '../vmd/static/ui.mjs';

test('frame cadence includes request time, never overlaps, and stops during an in-flight request', async () => {
  const originalTimeout = globalThis.setTimeout;
  const originalPerformance = globalThis.performance;
  let now = 0, finish;
  const scheduled = [];
  globalThis.performance = {now: () => now};
  globalThis.setTimeout = (callback, delay) => { scheduled.push({callback, delay}); return 0; };
  const loop = new PollLoop(() => new Promise(resolve => {finish = resolve;}), 100, {fixedRate: true});
  try {
    loop.active = true;
    let work = loop.tick();
    assert.equal(scheduled.length, 0, 'a pending request cannot launch another request');
    now = 40; finish(); await work;
    assert.equal(scheduled.pop().delay, 60, 'wait only the unused part of the frame interval');
    work = loop.tick(); now = 200; finish(); await work;
    assert.equal(scheduled.pop().delay, 0, 'slow requests never create catch-up request queues');
    work = loop.tick(); loop.stop(); finish(); await work;
    assert.equal(scheduled.length, 0, 'stopping an active request prevents rescheduling');
    const normal = new PollLoop(async () => {now += 40;}, 900);
    normal.active = true; await normal.tick(); normal.stop();
    assert.equal(scheduled.pop().delay, 900, 'other dashboard polling keeps its existing delay');
  } finally {
    loop.stop(); globalThis.setTimeout = originalTimeout; globalThis.performance = originalPerformance;
  }
});

test('unchanged previews avoid repeated image decoding and still clear stale views', () => {
  let assignments = 0;
  const image = {set src(value) {this.source = value; assignments++;}}, empty = {};
  setPreviewImage(image, empty, 'first');
  setPreviewImage(image, empty, 'first');
  assert.equal(assignments, 1);
  assert.equal(image.hidden, false); assert.equal(empty.hidden, true);
  setPreviewImage(image, empty, null);
  assert.equal(image.hidden, true); assert.equal(empty.hidden, false);
  setPreviewImage(image, empty, 'first');
  assert.equal(image.hidden, false); assert.equal(assignments, 1);
  setPreviewImage(image, empty, 'second');
  assert.equal(assignments, 2); assert.equal(image.source, 'data:image/jpeg;base64,second');
  const replacement = {set src(value) {assignments++;}};
  setPreviewImage(replacement, {}, 'second');
  assert.equal(assignments, 3, 'new and replaced camera elements get their own first image');
});
