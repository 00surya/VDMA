import test from 'node:test';
import assert from 'node:assert/strict';
import { ResponseUI } from '../vmd/static/response.mjs';
import { sharedLocation } from '../vmd/static/map-location.mjs';

test('camera location saves the opening camera once and cancels a closed or denied location request', async () => {
  const previousDocument = globalThis.document;
  const previousCameraLocation = sharedLocation.cameraLocation;
  const nodes = new Map();
  globalThis.document = {getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, {disabled: false, textContent: '', open: true,
      elements: {place: {value: 'Entrance'}}, close() {this.open = false;}});
    return nodes.get(id);
  }};
  try {
    const requests = [];
    const response = Object.create(ResponseUI.prototype);
    response.workspace = {selectedState: () => ({camera_id: 'new-selection'}), async refreshCameras() {}, toast() {}};
    response.api = {async request(path, body) {requests.push({path, body});}};
    const location = {place: 'Entrance', latitude: 12, longitude: 34, source: 'browser', accuracy_meters: 15, captured_at: 100};
    let resolveLocation;
    sharedLocation.cameraLocation = place => {
      assert.equal(place, 'Entrance');
      return new Promise(resolve => {resolveLocation = resolve;});
    };
    response.locationSession = {cameraId: 'original-camera', saving: false};
    const save = response.saveLocation();
    await response.saveLocation();
    assert.equal(document.getElementById('location-save').disabled, true);
    resolveLocation(location);
    await save;
    assert.deepEqual(requests, [{path: '/cameras/original-camera/location', body: location}]);

    document.getElementById('location-dialog').open = true;
    response.locationSession = {cameraId: 'closed-camera', saving: false};
    const cancelled = response.saveLocation();
    document.getElementById('location-dialog').close();
    resolveLocation(location);
    await cancelled;
    assert.equal(requests.length, 1, 'closing during location acquisition must not save');

    document.getElementById('location-dialog').open = true;
    response.locationSession = {cameraId: 'denied-camera', saving: false};
    sharedLocation.cameraLocation = async () => {throw new Error('Location access denied');};
    await response.saveLocation();
    assert.equal(requests.length, 1, 'permission denial must not use fabricated coordinates');
    assert.match(document.getElementById('location-error').textContent, /denied/);
    assert.equal(document.getElementById('location-retry').disabled, false);
  } finally {
    globalThis.document = previousDocument;
    sharedLocation.cameraLocation = previousCameraLocation;
  }
});
