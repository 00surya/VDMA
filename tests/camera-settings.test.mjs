import test from 'node:test';
import assert from 'node:assert/strict';
import { CameraSettingsEditor, readCameraSettings, validateCameraSettings, depthStatusText, fightStatusText } from '../vmd/static/camera-settings.mjs';

const settings = {device: 'auto', depth: 'ZipDepth', depth_fps: 1, detection_mode: 'depth_confirmed',
  eco_mode: false, object_detection: true, object_fps: 1, threshold: .72, hold_seconds: .9, fight_confirmation_seconds: 1.2, target_fps: 8};
function formWith(values) {
  return {elements: Object.fromEntries(Object.entries(values).map(([key, value]) => [key, {
    value: String(value), error: '', setCustomValidity(message) {this.error = message;},
  }])), reportValidity() {return Object.values(this.elements).every(field => !field.error);}};
}

test('shared camera settings reject disabled or too-slow depth for confirmation, then recover on valid edits', () => {
  const form = formWith({...settings, depth: 'off', depth_fps: .1, source: 'private-source'});
  validateCameraSettings(form);
  assert.equal(form.reportValidity(), false);
  form.elements.detection_mode.value = 'responsive';
  validateCameraSettings(form);
  assert.equal(form.reportValidity(), true);
  const parsed = readCameraSettings(form);
  assert.equal(parsed.eco_mode, false);
  assert.equal(parsed.object_detection, true);
  assert.equal(parsed.threshold, .72);
  assert.equal(parsed.object_fps, 1);
  assert.equal(parsed.fight_confirmation_seconds, 1.2);
  assert.equal(parsed.source, undefined, 'editing must not send source or unrelated form values');
});

test('depth readiness distinguishes stopped, missing, stale and separated pair samples', () => {
  const camera = {status: 'running', settings, depth_meta: {status: 'ready', age_seconds: .4, confirmation_fresh: true}};
  assert.match(depthStatusText(camera), /waiting for a matching tracked pair/);
  assert.match(depthStatusText({...camera, status: 'idle'}), /paused/);
  assert.match(depthStatusText({...camera, depth_meta: {status: 'loading'}}), /loading/);
  assert.match(depthStatusText({...camera, depth_meta: {status: 'error'}}), /unavailable/);
  assert.match(depthStatusText({...camera, depth_meta: {...camera.depth_meta, age_seconds: 2.8, confirmation_fresh: false}}), /too old/);
  assert.match(depthStatusText({...camera, signals: {depth_status: 'separated'}}), /pair depth: separated/);
  assert.match(depthStatusText({...camera, signals: {depth_status: 'compatible', depth_samples: 3}}), /3 matching samples/);
  assert.match(depthStatusText({...camera, settings: {...settings, detection_mode: 'responsive'}}), /cannot confirm fights/);
});

test('live fight check explains independent blockers even when score clears the threshold', () => {
  const camera = {status: 'running', score: .8, people: 2, settings,
    signals: {blockers: ['Both people need a reliable upper-body pose', 'No supported body contact']},
    alert: {event_type: 'fight'}};
  const text = fightStatusText(camera);
  assert.match(text, /Score 80 \/ 100 · threshold 72/);
  assert.match(text, /reliable upper-body pose/);
  assert.match(text, /No supported body contact/);
  assert.doesNotMatch(text, /Fight detected/, 'a historical alert must not replace current blockers');
  assert.equal(fightStatusText({...camera, stale: true}), 'Live fight check paused');
  assert.match(fightStatusText({...camera, people: 1, signals: {}}), /two visible tracked people/);
});

test('settings save targets the opening camera once and preserves errors for correction', async () => {
  const oldDocument = globalThis.document;
  const nodes = new Map([['camera-settings-form', formWith(settings)]]);
  globalThis.document = {getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, {disabled: false, textContent: '', open: true, close() {this.open = false;}});
    return nodes.get(id);
  }};
  try {
    const editor = Object.create(CameraSettingsEditor.prototype), requests = [];
    let resolveRequest;
    editor.workspace = {busy: false, selectedState: () => ({camera_id: 'new-selection'}),
      api: {request(...args) {requests.push(args); return new Promise(resolve => {resolveRequest = resolve;});}},
      async refreshCameras() {}, renderCameras() {}, toast() {}};
    editor.session = {cameraId: 'original-camera', saving: false};
    const saving = editor.save();
    await editor.save();
    assert.equal(requests.length, 1);
    assert.deepEqual(requests[0], ['/cameras/original-camera/settings', settings, 'PATCH']);
    assert.equal(document.getElementById('camera-settings-inputs').disabled, true);
    resolveRequest({status: 'idle'});
    await saving;
    assert.equal(editor.workspace.busy, false);
    assert.equal(document.getElementById('camera-settings-dialog').open, false);

    document.getElementById('camera-settings-dialog').open = true;
    editor.workspace.api.request = async () => {throw new Error('Camera is still stopping');};
    await editor.save();
    assert.equal(document.getElementById('camera-settings-dialog').open, true);
    assert.match(document.getElementById('camera-settings-error').textContent, /still stopping/);
    assert.equal(document.getElementById('camera-settings-inputs').disabled, false);
  } finally {
    globalThis.document = oldDocument;
  }
});
