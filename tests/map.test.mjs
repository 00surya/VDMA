import test from 'node:test';
import assert from 'node:assert/strict';
import {MapLocation, validCenter, isLiveCameraSource} from '../vmd/static/map-location.mjs';

test('location requests coalesce across maps and camera setup, preserving measured precision', async () => {
  let success, calls = 0;
  const location = new MapLocation({getCurrentPosition(ok, fail, options) {
    calls++; success = ok; assert.equal(options.maximumAge, 0);
  }});
  const observed = [];
  location.subscribe(state => observed.push(state));
  assert.equal(location.state.center, null);
  const first = location.locate();
  assert.equal(first, location.locate());
  await Promise.resolve();
  success({coords: {latitude: 19.07601, longitude: 72.87772, accuracy: 25}});
  await first;
  const saved = await location.cameraLocation(' Test entrance ');
  assert.equal(calls, 1);
  assert.deepEqual(location.state.center, [19.07601, 72.87772]);
  assert.equal(observed.at(-1).revision, 1);
  assert.deepEqual(saved, {place: 'Test entrance', latitude: 19.07601, longitude: 72.87772,
    source: 'browser', accuracy_meters: 25, captured_at: location.state.capturedAt / 1000});
});

test('denied, unavailable and timed-out locations never become fake coordinates; retry works', async () => {
  for (const code of [1, 2, 3]) {
    const location = new MapLocation({getCurrentPosition(ok, fail) { fail({code}); }});
    await assert.rejects(location.cameraLocation(), /denied|unavailable|timed out/);
    assert.equal(location.state.center, null);
    assert.equal(location.state.source, 'none');
    location.geolocation = {getCurrentPosition(ok) { ok({coords: {latitude: 0, longitude: 0, accuracy: 10}}); }};
    const saved = await location.cameraLocation();
    assert.equal(saved.latitude, 0);
    assert.equal(saved.longitude, 0);
    assert.equal(saved.place, 'Laptop location');
  }
  await assert.rejects(new MapLocation(null).cameraLocation(), /unavailable/);
});

test('expired or failed measurements cannot silently set a new camera location', async () => {
  const location = new MapLocation({getCurrentPosition(ok) { ok({coords: {latitude: 30, longitude: 78, accuracy: 9}}); }});
  await location.locate();
  location.state.capturedAt -= 61000;
  location.geolocation = {getCurrentPosition(ok, fail) { fail({code: 1}); }};
  await assert.rejects(location.cameraLocation(), /denied/);
  assert.deepEqual(location.state.center, [30, 78]); // map may retain a clearly labelled old fix
  location.geolocation = {getCurrentPosition(ok) { ok({coords: {latitude: NaN, longitude: 78}}); }};
  await assert.rejects(location.cameraLocation(), /unavailable/);
  assert.equal(validCenter([0, Infinity]), false);
  assert.equal(validCenter([91, 0]), false);
});

test('only webcams and network camera sources acquire laptop coordinates', () => {
  for (const source of ['0', '9', 'http://192.0.2.1:8080/video', 'rtsp://example.test/live'])
    assert.equal(isLiveCameraSource(source), true);
  for (const source of ['', '/tmp/recording.mp4', 'movie.avi', '10'])
    assert.equal(isLiveCameraSource(source), false);
});

test('camera setup can read an optional location immediately without prompting or waiting', async () => {
  let success, calls = 0;
  const location = new MapLocation({getCurrentPosition(ok) { calls++; success = ok; }});
  assert.equal(location.currentCameraLocation('Gate'), null);
  assert.equal(calls, 0);
  const pending = location.locate();
  await Promise.resolve();
  assert.equal(location.currentCameraLocation('Gate'), null, 'an unresolved GPS prompt cannot block adding a camera');
  assert.equal(calls, 1);
  success({coords: {latitude: 0, longitude: 0, accuracy: 12}});
  await pending;
  assert.equal(location.currentCameraLocation('Gate').place, 'Gate');
  assert.equal(location.currentCameraLocation().latitude, 0);
  location.state.capturedAt -= 61000;
  assert.equal(location.currentCameraLocation(), null, 'an old coordinate must not be reused');
  location.state.capturedAt = Date.now();
  location.state.status = 'error';
  assert.equal(location.currentCameraLocation(), null, 'a failed refresh must not reuse its old fix');
  assert.equal(calls, 1, 'reading optional location never retries geolocation');
});
