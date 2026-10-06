import test from 'node:test';
import assert from 'node:assert/strict';
import { alarmLevel, responseText, responseSummary, callingNotice, dispatchLocked, incidentHandledExpired, manualDispatchEligible, newIncidentCamera, ResponseUI } from '../vmd/static/response.mjs';

test('handled incidents release at exactly thirty seconds using persisted first handling time', () => {
  const event = {id: 'saved', handled_at: 100};
  const alert = {handled_at: 105, incident_saved: true};
  assert.equal(incidentHandledExpired(event, alert, 129.999), false);
  assert.equal(incidentHandledExpired(event, alert, 130), true);
  assert.equal(incidentHandledExpired({...event, handled_at: 105}, {...alert, handled_at: 100}, 130), true);
  assert.equal(incidentHandledExpired(null, {...alert, handled_at: 100}, 130), true);
  assert.equal(incidentHandledExpired(null, {...alert, handled_at: 100, incident_saved: false}, 130), false);
  assert.equal(incidentHandledExpired({created: 100, review: 'unreviewed'}, {updated: 100}, 500), false);
  assert.equal(incidentHandledExpired({handled_at: null}, {handled_at: '100'}, 500), false);
  assert.equal(incidentHandledExpired({handled_at: NaN}, {handled_at: -1}, 500), false);
  assert.equal(incidentHandledExpired({id: 'false-positive', review: 'false_positive', handled_at: 100}, null, 130), true);
});

test('alarm volume rises during ten-second countdown and stops on acknowledgement or provider acceptance', () => {
  const alert = {created: 100, deadline: 110, state: 'pending', event: {mode: 'live', signals: {live_camera: true}}};
  assert.ok(alarmLevel(alert, 100) < alarmLevel(alert, 105));
  assert.ok(alarmLevel(alert, 105) < alarmLevel(alert, 110));
  assert.equal(alarmLevel(alert, 110), alarmLevel(alert, 200));
  assert.equal(alarmLevel({...alert, state: 'acknowledged'}, 105), 0);
  assert.equal(alarmLevel({...alert, state: 'cancelled'}, 105), 0);
  assert.equal(alarmLevel({...alert, state: 'requested', deliveries: [{status: 'accepted'}]}, 115), 0);
  for (const event of [{mode: 'live', signals: {live_camera: false}}, {mode: 'demo', signals: {live_camera: true}}, {mode: 'live'}, undefined])
    assert.equal(alarmLevel({...alert, state: 'requested', event}, 115), 0, 'only physical live-camera events can sound an alarm');
  assert.match(responseText(alert, 103), /7s/);
  assert.match(responseText({...alert, state: 'requested', deliveries: [{name: 'Hospital', status: 'accepted'}]}, 115), /delivery unconfirmed/);
});

test('calling setup explicitly distinguishes disabled, unconfigured and trial demo calls', () => {
  assert.match(callingNotice({enabled: false, configured: true}), /Voice calls are off/);
  assert.match(callingNotice({enabled: true, configured: false}), /No calls can be placed/);
  assert.match(callingNotice({enabled: true, configured: true, mode: 'trial_template'}), /Incident and location are not spoken/);
  assert.match(callingNotice({enabled: true, configured: true, mode: 'incident'}), /read the incident/);
});

test('voice statuses distinguish a queued call from an answered call and failures keep the alarm active', () => {
  const alert = {created: 100, deadline: 110, updated: 111, state: 'requested',
    event: {mode: 'live', signals: {live_camera: true}},
    calls: [{name: 'Hospital', status: 'queued'}], calls_submitted: true};
  assert.equal(alarmLevel(alert, 115), 0);
  assert.match(responseText(alert, 115), /not answered yet/);
  const failed = {...alert, calls_submitted: false, calls: [{name: 'Hospital', status: 'no-answer'}]};
  assert.ok(alarmLevel(failed, 115) > 0);
  assert.match(responseText(failed, 115), /no answer/);
  assert.match(responseText({...alert, calls: [{name: 'Hospital', status: 'completed'}]}, 115), /human response unconfirmed/);
  assert.match(responseText({...alert, state: 'cancelled'}, 115), /cannot be recalled/);
  assert.match(responseText({...alert, calls: [], calls_submitted: false, call_issues: ['Voice calls are disabled']}, 115), /disabled/);
  const handled = {...failed, handled_at: 111, incident_saved: true};
  assert.ok(alarmLevel(handled, 140.999) > 0);
  assert.equal(alarmLevel(handled, 141), 0);
  assert.ok(alarmLevel({...handled, incident_saved: false}, 141) > 0, 'saving must finish before the alarm can expire');
  assert.match(responseSummary(handled, 141), /unsuccessful or unconfirmed/);
  assert.match(responseText(handled, 141), /no answer/);
});

test('new live incidents select a camera once without old history or recordings stealing selection', () => {
  const cameras = [{camera_id: 'north', mode: 'live'}, {camera_id: 'south', mode: 'live'}];
  const event = {id: 'first', camera_id: 'north', mode: 'live', created: 100, review: 'unreviewed', signals: {live_camera: true}};
  const seen = new Set();
  assert.equal(newIncidentCamera([event], [], seen, 105), null);
  assert.equal(newIncidentCamera([event], cameras, seen, 105), 'north');
  assert.equal(newIncidentCamera([event], cameras, seen, 106), null);
  const fresh = {...event, id: 'second', camera_id: 'south', created: 110};
  assert.equal(newIncidentCamera([fresh, event], cameras, seen, 112), 'south');
  assert.equal(newIncidentCamera([{...fresh, id: 'recording', signals: {live_camera: false}},
    {...event, id: 'old', created: 5}, {...fresh, id: 'reviewed', review: 'false_positive'}], cameras, seen, 112), null);
  assert.equal(newIncidentCamera([{...event, id: 'handled', handled_at: 100}], cameras, new Set(), 130), null);
  assert.equal(newIncidentCamera([{...event, id: 'new-same-camera', created: 131}], cameras, seen, 132), 'north');
});

test('accepted dispatch locks immediately, preserves truthful failures and permits an expired unsent request', () => {
  const alert = {state: 'requested', updated: 100};
  assert.equal(dispatchLocked(alert, 101), true);
  assert.equal(dispatchLocked(alert, 401), false);
  assert.equal(dispatchLocked({...alert, calls: [{status: 'failed'}]}, 401), true);
  assert.equal(dispatchLocked({...alert, state: 'acknowledged'}, 101), false);
  assert.equal(dispatchLocked({...alert, state: 'cancelled'}, 101), true);
  assert.equal(responseSummary({...alert, calls: [{name: 'Someone', status: 'completed'}]}, 101), 'Authorities requested');
  assert.match(responseSummary({...alert, calls: [{status: 'failed'}]}, 101), /unsuccessful.*Centre settings/);
  const expired = {...alert, handled_at: 100, incident_saved: true};
  assert.match(responseSummary(expired, 401), /expired without sending.*Recent response details/);
  assert.doesNotMatch(responseText(expired, 401), /slide again/);
});

test('manual dispatch asks for hospital afterwards, locks slider, and leaves a route to reopen the question', async () => {
  const previousDocument = globalThis.document;
  const nodes = new Map();
  globalThis.document = {getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, {value: 0, hidden: false, disabled: false, textContent: '', open: false,
      showModal() {this.open = true;}, close() {this.open = false;}});
    return nodes.get(id);
  }};
  try {
    const event = {id: 'incident', mode: 'live', camera_name: 'North', signals: {live_camera: true}, review: 'unreviewed'};
    const requested = {id: event.id, event, state: 'requested', updated: 100, handled_at: 100, incident_saved: true, hospital: false};
    let camera = {name: 'North', mode: 'live', status: 'running', stale: false};
    let now = 101;
    let selectedEvent = event;
    const requests = [];
    const response = Object.create(ResponseUI.prototype);
    Object.assign(response, {
      alerts: [], calling: {enabled: true}, messaging: {}, selectedKey: event.id, now: () => now,
      workspace: {selectedState: () => camera, selectedEvent: () => selectedEvent, toast() {}},
      api: {async request(path, body) {requests.push({path, body}); return {...requested, id: selectedEvent.id, event: selectedEvent};}},
      async refresh() {this.paint();}, paint() {this.renderDispatch(selectedEvent);},
    });
    const slider = document.getElementById('dispatch-slider');
    slider.value = 47;
    response.renderDispatch(event);
    assert.equal(slider.value, 47, 'polling must not reset a slider while it is being dragged');
    await response.dispatch(event);
    assert.equal(requests.length, 1);
    assert.deepEqual(requests[0].body, {action: 'dispatch', hospital: false});
    assert.equal(slider.disabled, true);
    assert.equal(slider.value, 100);
    assert.equal(document.getElementById('dispatch-label').textContent, 'DISPATCH REQUESTED');
    assert.equal(document.getElementById('hospital-dialog').open, true);
    assert.equal(response.hospitalIncident.id, event.id);
    assert.equal(document.getElementById('hospital-response').hidden, false);
    document.getElementById('hospital-dialog').close();
    response.openHospital(event);
    assert.equal(document.getElementById('hospital-dialog').open, true);
    assert.equal(requests.length, 1, 'reopening the hospital question must not redial authorities');
    await response.dispatch(event);
    assert.equal(requests.length, 1, 'a locked incident cannot submit a second dispatch');
    response.renderDispatch({...event, id: 'overlapping'});
    assert.equal(slider.disabled, false, 'the old incident cooldown must not block a distinct new incident');
    response.alerts[0].calls = [{status: 'failed'}];
    now = 129.999;
    response.renderDispatch(event);
    assert.match(document.getElementById('dispatch-state').textContent, /Monitoring resumes in 1s/);
    assert.match(document.getElementById('dispatch-state').textContent, /unsuccessful or unconfirmed/);
    assert.equal(slider.value, 100);
    response.alerts = JSON.parse(JSON.stringify(response.alerts));
    response.selectedKey = null;
    now = 130;
    response.renderDispatch(event);
    assert.equal(document.getElementById('dispatch-label').textContent, 'MONITORING');
    assert.equal(document.getElementById('dispatch-state').textContent, 'Ready for next incident.');
    assert.equal(slider.value, 0);
    assert.equal(slider.disabled, true, 'monitoring requires a new incident before dispatch');
    assert.equal(document.getElementById('hospital-response').hidden, true);
    assert.equal(response.hospitalIncident.id, event.id, 'an open hospital decision remains bound to its original incident');
    await response.dispatch(event);
    assert.equal(requests.length, 1, 'expiry resets the camera, never the saved incident dispatch lock');
    assert.equal(dispatchLocked(response.alerts[0], now), true);
    selectedEvent = {...event, id: 'next', created: 130};
    response.renderDispatch(selectedEvent);
    assert.equal(slider.disabled, false, 'another incident from the same camera is independently actionable');
    assert.equal(slider.value, 0);
    assert.equal(document.getElementById('dispatch-label').textContent, 'SLIDE TO DISPATCH');
    const reviewed = {...selectedEvent, review: 'false_positive', handled_at: 140};
    now = 169.999;
    response.renderDispatch(reviewed);
    assert.match(document.getElementById('dispatch-state').textContent, /Marked false positive.*resumes in 1s/);
    now = 170;
    response.renderDispatch(reviewed);
    assert.equal(document.getElementById('dispatch-label').textContent, 'MONITORING');
    assert.equal(requests.length, 1, 'polling and review expiry must not dispatch anything');
    camera = {...camera, status: 'stopped'};
    response.renderDispatch(null);
    assert.equal(document.getElementById('dispatch-label').textContent, 'CAMERA OFFLINE');
    assert.equal(document.getElementById('dispatch-state').textContent, 'Connect camera to monitor new incidents.');
    camera = {...camera, status: 'running', stale: true};
    response.renderDispatch(null);
    assert.equal(document.getElementById('dispatch-label').textContent, 'CAMERA OFFLINE');
  } finally {globalThis.document = previousDocument;}
});

test('an upgraded incident selects its camera once without a new incident ID', () => {
  const seen = new Set(), cameras = [{camera_id: 'north', mode: 'live'}];
  const first = {id: 'episode', camera_id: 'north', mode: 'live', created: 100,
    review: 'unreviewed', event_type: 'possible_snatching', signals: {live_camera: true}};
  assert.equal(newIncidentCamera([first], cameras, seen, 101), 'north');
  assert.equal(newIncidentCamera([first], cameras, seen, 102), null);
  const upgraded = {...first, event_type: 'snatching_detected', signals: {...first.signals, escalated_at: 104}};
  assert.equal(newIncidentCamera([upgraded], cameras, seen, 105), 'north');
  assert.equal(newIncidentCamera([upgraded], cameras, seen, 106), null);
});

test('manual dispatch requires saved real-source provenance, including recordings but excluding synthetic and unknown sources', () => {
  const event = {mode: 'live', review: 'unreviewed', signals: {live_camera: false}};
  assert.equal(manualDispatchEligible(event), true);
  assert.equal(manualDispatchEligible({...event, signals: {live_camera: true}}), true);
  assert.equal(manualDispatchEligible({...event, review: 'confirmed'}), true);
  for (const mode of ['demo', 'presentation', undefined])
    assert.equal(manualDispatchEligible({...event, mode}), false);
  for (const live_camera of [undefined, null, 0, 1, 'false', 'true'])
    assert.equal(manualDispatchEligible({...event, signals: {live_camera}}), false);
  assert.equal(manualDispatchEligible({...event, review: 'false_positive'}), false);
  assert.equal(manualDispatchEligible(null), false);
});

test('finished or removed recording evidence dispatches manually once and never claims live monitoring resumes', async () => {
  const previousDocument = globalThis.document, nodes = new Map();
  const node = () => ({value: 0, hidden: false, disabled: false, textContent: '', open: false,
    listeners: {}, children: [], setAttribute() {}, append(...items) {this.children.push(...items);},
    addEventListener(name, fn) {this.listeners[name] = fn;},
    showModal() {this.open = true;}, close() {this.open = false;}});
  globalThis.document = {getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, node());
    return nodes.get(id);
  }, createElement: node, body: node()};
  try {
    const event = {id: 'recorded', mode: 'live', camera_id: 'removed-session', camera_name: 'Saved recording',
      event_type: 'fight', signals: {live_camera: false}, review: 'unreviewed', clip: 'saved.avi'};
    let camera = {camera_id: event.camera_id, mode: 'live', status: 'finished', live_camera: false};
    let now = 100;
    const requests = [], response = Object.create(ResponseUI.prototype);
    Object.assign(response, {
      alerts: [], calling: {enabled: true}, messaging: {}, busy: false, now: () => now,
      workspace: {incidents: [{...event, id: 'newer-recording-event'}, event], selectedState: () => camera, selectedEvent: () => event, toast() {}},
      api: {async request(path, body) {
        requests.push({path, body});
        return {id: event.id, event, state: 'requested', updated: now, handled_at: 100,
          incident_saved: true, hospital: !!body.hospital};
      }},
      async refresh() {this.paint();},
      paint() {this.renderDispatch(event); this.renderEvidenceDispatch();},
    });
    response.renderDispatch(event);
    assert.equal(document.getElementById('dispatch-slider').disabled, false);
    assert.match(document.getElementById('dispatch-state').textContent, /Recording · manual dispatch only/);
    assert.equal(requests.length, 0, 'viewing finished analysis does not submit a response');
    camera = null;
    response.openEvidenceDispatch(event);
    assert.equal(response.evidenceEvent().id, 'recorded', 'library selection stays on its exact saved ID');
    assert.equal(response.evidenceDialog.dialog.open, true);
    assert.match(response.evidenceDialog.details.textContent, /Saved recording.*manual dispatch only/);
    assert.equal(response.evidenceDialog.slider.disabled, false, 'a removed camera cannot block saved evidence');
    response.evidenceDialog.slider.value = 47;
    response.renderEvidenceDispatch();
    assert.equal(response.evidenceDialog.slider.value, 47, 'polling preserves a partially moved slider');
    await response.dispatch(response.evidenceEvent());
    assert.deepEqual(requests[0], {path: '/incidents/recorded/response', body: {action: 'dispatch', hospital: false}});
    assert.equal(response.evidenceDialog.slider.disabled, true);
    assert.equal(response.evidenceDialog.slider.value, 100);
    assert.equal(document.getElementById('hospital-dialog').open, true);
    assert.equal(response.hospitalIncident.id, event.id);
    now = 131;
    response.paint();
    assert.equal(response.evidenceDialog.slider.disabled, true);
    assert.doesNotMatch(document.getElementById('dispatch-state').textContent, /Monitoring resumes|Ready for next/);
    assert.equal(document.getElementById('dispatch-label').textContent, 'DISPATCH REQUESTED');
    await response.dispatch(event);
    assert.equal(requests.length, 1);
    await response.act(event.id, 'call_hospital', true);
    assert.deepEqual(requests[1].body, {action: 'call_hospital', hospital: true});
    assert.equal(response.evidenceDialog.hospital.hidden, true);
    response.alerts[0].calls = [{status: 'failed'}];
    now = 1000;
    response.renderEvidenceDispatch();
    assert.equal(response.evidenceDialog.slider.disabled, true, 'old attempts remain locked');
    assert.match(response.evidenceDialog.status.textContent, /unsuccessful or unconfirmed/);
    response.workspace.incidents = [{...event, review: 'false_positive'}];
    response.renderEvidenceDispatch();
    assert.equal(response.evidenceDialog.slider.disabled, true);
    assert.match(response.evidenceDialog.status.textContent, /false positive/);
    await response.dispatch(response.evidenceEvent());
    assert.equal(requests.length, 2);
  } finally {globalThis.document = previousDocument;}
});
