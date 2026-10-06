import { $, el } from './ui.mjs';
import { centreFields, readCentre } from './centre.mjs';
import { sharedLocation } from './map-location.mjs';

export function callingNotice(calling) {
  if (!calling) return 'Checking voice calling setup…';
  if (!calling.enabled) return 'Voice calls are off. Enable Twilio voice calls in Centre settings and save.';
  if (!calling.configured) return 'Voice calling is not configured on the server. No calls can be placed yet.';
  if (calling.mode === 'gemini_live') return 'Gemini conversational calls configured · announcement fallback if the voice gateway is unavailable.';
  return calling.mode === 'trial_template'
    ? 'Trial test calls · Twilio demo prompt only. Incident and location are not spoken.'
    : 'Voice calls configured to read the incident and camera location aloud.';
}

function incidentResetAt(event, alert) {
  if (!event && alert?.incident_saved !== true) return null;
  const handled = [event?.handled_at, alert?.handled_at].filter(value =>
    typeof value === 'number' && Number.isFinite(value) && value >= 0);
  return handled.length ? Math.min(...handled) + 30 : null;
}

export function incidentHandledExpired(event, alert, now) {
  const resetAt = incidentResetAt(event, alert);
  return resetAt !== null && now >= resetAt;
}

export function alarmLevel(alert, now) {
  if (!alert || !['pending', 'requested'].includes(alert.state)) return 0;
  if (alert.event?.mode !== 'live' || alert.event.signals?.live_camera !== true) return 0;
  if (incidentHandledExpired(null, alert, now)) return 0;
  if (alert.calls_submitted) return 0;
  if (!alert.calls?.length && alert.deliveries?.length && alert.deliveries.every(row => row.status === 'accepted')) return 0;
  return .025 + .125 * Math.min(1, Math.max(0, (now - alert.created) / 10));
}

export function responseText(alert, now) {
  if (!alert) return 'Slide to call authorities. You can request a hospital next.';
  if (alert.state === 'pending') return `Automatic dispatch in ${Math.max(0, Math.ceil(alert.deadline-now))}s · acknowledge to stop`;
  const outcomes = (alert.calls || []).map(row => `${row.name}: ${
    ({queued: 'call queued (not answered yet)', ringing: 'ringing', 'in-progress': 'call connected',
      completed: 'call completed (human response unconfirmed)', 'no-answer': 'no answer'})[row.status] || row.status
    }${row.error ? ' · '+row.error : ''}`);
  outcomes.push(...(alert.deliveries || []).map(row =>
    `${row.name}: SMS ${row.status === 'accepted' ? 'accepted by Twilio (delivery unconfirmed)' : row.status}${row.error ? ' · '+row.error : ''}`));
  if (alert.state === 'acknowledged') return ['Acknowledged · further automatic dispatch stopped', ...outcomes].join('; ');
  if (alert.state === 'cancelled') return ['Cancelled · further dispatch stopped; submitted calls cannot be recalled here', ...outcomes].join('; ');
  if (!alert.calls_submitted) outcomes.push(...(alert.call_issues || []));
  if (now-(alert.voice_requested_at ?? alert.updated) > 300 && !alert.calls?.length && !alert.deliveries?.length)
    return 'Dispatch expired without sending. Check Centre settings → Recent response details.';
  return outcomes.length ? outcomes.join('; ') : 'Calls requested · waiting for submission';
}

export function dispatchLocked(alert, now) {
  if (alert?.calls?.length || alert?.deliveries?.length || alert?.state === 'cancelled') return true;
  return alert?.state === 'requested' && now-(alert.voice_requested_at ?? alert.updated) <= 300;
}

export function manualDispatchEligible(event) {
  return event?.mode === 'live' && typeof event.signals?.live_camera === 'boolean'
    && event.review !== 'false_positive';
}

export function responseSummary(alert, now) {
  if (!alert) return 'Slide to call authorities.';
  if (alert.state === 'pending') return `Automatic dispatch in ${Math.max(0, Math.ceil(alert.deadline-now))}s`;
  if (alert.state === 'acknowledged') return 'Acknowledged · automatic escalation stopped';
  if (alert.state === 'cancelled') return 'Marked false positive · further dispatch stopped';
  if ((alert.calls || []).some(row => ['failed', 'busy', 'no-answer', 'canceled', 'uncertain'].includes(row.status)))
    return 'A call was unsuccessful or unconfirmed. Check Centre settings → Recent response details.';
  if (!alert.calls_submitted && alert.call_issues?.length) return alert.call_issues.join(' · ');
  if (!dispatchLocked(alert, now)) return incidentHandledExpired(null, alert, now)
    ? 'Dispatch expired without sending. Check Centre settings → Recent response details.'
    : 'Dispatch expired without sending. Check settings, then slide again.';
  return alert.hospital ? 'Authorities and hospital requested' : 'Authorities requested';
}

export function newIncidentCamera(incidents, cameras, seen, now) {
  const available = new Set(cameras.filter(camera => !camera.presentation && camera.mode === 'live').map(camera => camera.camera_id));
  const version = event => `${event.id}:${event.event_type}`;
  const fresh = incidents.filter(event => available.has(event.camera_id) && !seen.has(version(event)));
  fresh.forEach(event => seen.add(version(event)));
  return fresh.filter(event => event.mode === 'live' && event.signals?.live_camera && event.review === 'unreviewed'
    && available.has(event.camera_id) && !incidentHandledExpired(event, null, now)
    && (event.signals?.escalated_at ?? event.created) >= now-30 && (event.signals?.escalated_at ?? event.created) <= now+5)
    .sort((a, b) => (b.signals?.escalated_at ?? b.created)-(a.signals?.escalated_at ?? a.created))[0]?.camera_id || null;
}

export class ResponseUI {
  constructor(workspace) {
    this.workspace = workspace;
    this.api = workspace.api;
    this.alerts = [];
    this.receivedAt = 0;
    this.offset = 0;
    this.audio = null;
    this.lastBeep = 0;
    this.selectedKey = null;
    this.busy = false;
    $('enable-alarm').addEventListener('click', async () => {
      try {
        this.audio ||= new AudioContext();
        await this.audio.resume();
        $('enable-alarm').textContent = 'Alarm sound enabled';
        this.beep(.025);
      } catch { workspace.toast('Sound could not start. Check browser and device audio settings.'); }
    });
    $('acknowledge-alert').addEventListener('click', () => {
      const event = workspace.selectedEvent();
      if (event) this.act(event.id, 'acknowledge');
    });
    $('hospital-no').addEventListener('click', () => $('hospital-dialog').close());
    $('hospital-response').addEventListener('click', () => {
      const event = workspace.selectedEvent();
      if (event) this.openHospital(event);
    });
    $('hospital-yes').addEventListener('click', async () => {
      if (!this.hospitalIncident || this.busy) return;
      $('hospital-yes').disabled = true;
      $('hospital-no').disabled = true;
      const accepted = await this.act(this.hospitalIncident.id, 'call_hospital', true);
      $('hospital-yes').disabled = false;
      $('hospital-no').disabled = false;
      if (accepted) $('hospital-dialog').close();
      else $('hospital-error').textContent = this.actionError || 'The hospital request was not accepted. Check Centre settings and try again.';
    });
    $('centre-settings').addEventListener('click', () => this.openCentre());
    $('close-centre').addEventListener('click', () => $('centre-dialog').close());
    $('centre-form').addEventListener('submit', async event => {
      event.preventDefault();
      try {
        const result = await this.api.request('/centre', {...readCentre(event.target),
          calling_enabled: $('calling-enabled').checked, messaging_enabled: $('messaging-enabled').checked});
        workspace.profile.name = result.name;
        workspace.renderProfile();
        $('centre-dialog').close();
        workspace.toast('Centre contacts saved.');
        await this.refresh();
      } catch (exc) { $('centre-error').textContent = exc.message; }
    });
    $('camera-location').addEventListener('click', () => this.openLocation());
    $('close-location').addEventListener('click', () => $('location-dialog').close());
    $('location-dialog').addEventListener('close', () => { this.locationSession = null; });
    $('location-retry').addEventListener('click', () => this.locateCamera());
    $('location-form').addEventListener('submit', event => {
      event.preventDefault();
      this.saveLocation();
    });
    this.timer = setInterval(() => this.paint(), 250);
  }
  now() { return Date.now()/1000 + this.offset; }
  async refresh() {
    try {
      const data = await this.api.request('/response-alerts');
      this.alerts = data.alerts;
      this.offset = data.server_time - Date.now()/1000;
      this.receivedAt = Date.now();
      this.messaging = data.messaging;
      this.calling = data.calling;
      this.error = data.error;
      this.paint();
    } catch {
      this.error = 'Response service unavailable. The server countdown may still be running.';
    }
  }
  beep(volume) {
    if (!this.audio || this.audio.state !== 'running') return;
    const tone = this.audio.createOscillator(), gain = this.audio.createGain();
    tone.frequency.value = 880;
    const now = this.audio.currentTime;
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(volume, now+.015);
    gain.gain.setValueAtTime(volume, now+.12);
    gain.gain.linearRampToValueAtTime(0, now+.18);
    tone.connect(gain); gain.connect(this.audio.destination);
    tone.start(); tone.stop(now+.2);
    tone.onended = () => { tone.disconnect(); gain.disconnect(); };
  }
  paint() {
    const fresh = Date.now()-this.receivedAt < 3000;
    const active = this.alerts.filter(item => alarmLevel(item, this.now()) > 0);
    const banner = $('response-banner');
    banner.hidden = !active.length && !this.error;
    const signature = JSON.stringify([active.map(item => [item.id, responseSummary(item, this.now()), item.hospital]), this.error, fresh, this.audio?.state]);
    if (signature !== this.bannerSignature) {
      this.bannerSignature = signature;
      banner.replaceChildren();
      if (this.error) banner.append(el('p', this.error));
      if (active.length && this.audio?.state !== 'running') banner.append(el('p', 'Sound is off. Choose Enable alarm sound on this device.'));
      for (const alert of active) {
        const row = el('div', null, 'response-row');
        row.append(el('strong', `${alert.event.camera_name} · ${alert.event.event_type.replaceAll('_', ' ')}`),
          el('span', fresh ? responseSummary(alert, this.now()) : 'Status is stale; reconnect to check dispatch.'));
        const view = el('button', 'View camera', 'button secondary');
        view.addEventListener('click', () => {
          this.workspace.setCameraMode('connected');
          this.workspace.selected = alert.event.camera_id;
          this.workspace.showPage('operations');
        });
        const ack = el('button', 'Acknowledge', 'button secondary');
        ack.addEventListener('click', () => this.act(alert.id, 'acknowledge'));
        row.append(view, ack);
        banner.append(row);
      }
    }
    const level = fresh ? Math.max(0, ...active.map(alert => alarmLevel(alert, this.now()))) : 0;
    if (level && Date.now()-this.lastBeep >= 900) { this.beep(level); this.lastBeep = Date.now(); }
    if (this.workspace.page === 'operations') this.renderDispatch(this.workspace.selectedEvent());
    this.renderEvidenceDispatch();
  }
  renderDispatch(event) {
    const camera = this.workspace.selectedState();
    const simulated = event?.mode === 'presentation' || event?.mode === 'demo' || camera?.presentation || camera?.mode === 'demo';
    $('acknowledge-alert').hidden = true;
    $('hospital-response').hidden = true;
    $('camera-location').hidden = !camera || camera.presentation || camera.mode === 'demo' || camera.live_camera === false;
    if (camera && !camera.presentation) {
      $('inspect-sector').textContent = camera.location
        ? `${camera.location.place} · ${camera.location.latitude}, ${camera.location.longitude}`
        : 'Camera place and coordinates not configured';
    }
    if (simulated) {
      this.selectedKey = null;
      this.lockedKey = null;
      return false;
    }
    let alert = this.alerts.find(item => item.id === event?.id);
    const now = this.now();
    if (event?.signals?.live_camera !== false && incidentHandledExpired(event, alert, now)) {
      event = null;
      alert = null;
    }
    if (this.selectedKey !== event?.id) {
      this.selectedKey = event?.id;
      $('dispatch-slider').value = 0;
    }
    const eligible = manualDispatchEligible(event);
    const locked = dispatchLocked(alert, now);
    $('dispatch-slider').disabled = !eligible || this.busy || locked;
    if (locked) $('dispatch-slider').value = 100;
    else if (this.lockedKey === event?.id) $('dispatch-slider').value = 0;
    this.lockedKey = locked ? event?.id : null;
    const idleCamera = !event && camera?.mode === 'live' && camera.live_camera !== false;
    const monitoring = idleCamera && camera.status === 'running' && !camera.stale;
    $('dispatch-label').textContent = monitoring ? 'MONITORING' : idleCamera ? 'CAMERA OFFLINE' : alert?.state === 'cancelled' ? 'DISPATCH CANCELLED' : locked ? 'DISPATCH REQUESTED' : this.busy ? 'REQUESTING…' : 'SLIDE TO DISPATCH';
    $('dispatch-next').hidden = true;
    $('hospital-response').hidden = !eligible || !locked || alert?.state === 'cancelled' || alert?.hospital;
    $('hospital-response').disabled = this.busy;
    const resetAt = event?.signals?.live_camera === false ? null : incidentResetAt(event, alert);
    const summary = eligible ? responseSummary(alert, now) : event?.review === 'false_positive'
      ? 'Marked false positive · incident saved' : monitoring ? 'Ready for next incident.'
      : idleCamera ? 'Connect camera to monitor new incidents.' : 'Select a saved camera or recording incident to dispatch.';
    $('dispatch-state').textContent = (eligible && !alert && event.signals.live_camera === false ? 'Recording · manual dispatch only. ' : '')
      + summary + (resetAt !== null ? ` · Monitoring resumes in ${Math.max(0, Math.ceil(resetAt-now))}s` : '');
    $('acknowledge-alert').hidden = !alert || !['pending', 'requested'].includes(alert.state) || alarmLevel(alert, now) === 0;
    $('false-positive').disabled = !event || event.review === 'false_positive' || this.workspace.busy;
    $('inspect-play').hidden = !event?.clip;
    return true;
  }
  async dispatch(event) {
    const alert = this.alerts.find(item => item.id === event?.id);
    if (!manualDispatchEligible(event) || this.busy
        || (event.signals.live_camera && incidentHandledExpired(event, alert, this.now()))
        || dispatchLocked(alert, this.now())) return;
    if (!await this.act(event.id, 'dispatch', false)) return;
    this.openHospital(event);
  }
  openEvidenceDispatch(event) {
    if (!manualDispatchEligible(event)) return;
    if (!this.evidenceDialog) {
      const dialog = el('dialog'), heading = el('div', null, 'dialog-heading'),
        title = el('h2', 'Review recording dispatch'), close = el('button', '×', 'icon-button'),
        details = el('p'), sliderBox = el('div', null, 'dispatch-slider'),
        label = el('span'), slider = el('input'), status = el('p', null, 'field-help'),
        hospital = el('button', 'Request hospital', 'button secondary');
      dialog.id = 'evidence-dispatch-dialog';
      title.id = 'evidence-dispatch-title';
      dialog.setAttribute('aria-labelledby', title.id);
      details.id = 'evidence-dispatch-context';
      dialog.setAttribute('aria-describedby', details.id);
      close.type = 'button'; close.autofocus = true;
      close.setAttribute('aria-label', 'Close recording dispatch');
      close.addEventListener('click', () => dialog.close());
      slider.type = 'range'; slider.min = 0; slider.max = 100; slider.value = 0;
      slider.setAttribute('aria-label', 'Slide to dispatch recording incident');
      slider.addEventListener('change', () => {
        if (!slider.disabled && Number(slider.value) >= 95) this.dispatch(this.evidenceEvent());
        else slider.value = 0;
      });
      hospital.type = 'button';
      hospital.addEventListener('click', () => this.openHospital(this.evidenceEvent()));
      heading.append(title, close); sliderBox.append(label, slider);
      dialog.append(heading, details, sliderBox, status, hospital);
      document.body.append(dialog);
      this.evidenceDialog = {dialog, details, label, slider, status, hospital};
    }
    this.evidenceIncident = event;
    this.actionError = '';
    this.evidenceDialog.slider.value = 0;
    this.renderEvidenceDispatch();
    if (!this.evidenceDialog.dialog.open) this.evidenceDialog.dialog.showModal();
  }
  evidenceEvent() {
    return this.workspace.incidents?.find(event => event.id === this.evidenceIncident?.id) || this.evidenceIncident;
  }
  renderEvidenceDispatch() {
    if (!this.evidenceDialog || !this.evidenceIncident) return;
    const {details, label, slider, status, hospital} = this.evidenceDialog,
      event = this.evidenceEvent(), alert = this.alerts.find(item => item.id === event?.id),
      locked = dispatchLocked(alert, this.now()), eligible = manualDispatchEligible(event);
    details.textContent = `${event.camera_name} · ${event.event_type.replaceAll('_', ' ')}. Recording analysis · manual dispatch only. Review the saved evidence before requesting a response.`;
    slider.disabled = !eligible || locked || this.busy;
    if (locked) slider.value = 100;
    label.textContent = alert?.state === 'cancelled' ? 'DISPATCH CANCELLED' : locked ? 'DISPATCH REQUESTED' : this.busy ? 'REQUESTING…' : 'SLIDE TO DISPATCH';
    status.textContent = event.review === 'false_positive' ? 'Marked false positive · further dispatch stopped'
      : this.actionError || responseSummary(alert, this.now());
    hospital.hidden = !eligible || !locked || alert?.state === 'cancelled' || alert?.hospital;
    hospital.disabled = this.busy;
  }
  openHospital(event) {
    this.hospitalIncident = event;
    $('hospital-context').textContent = `Authority calls requested for ${event.camera_name}. Should the saved hospital be called too?`;
    $('hospital-error').textContent = '';
    if (!$('hospital-dialog').open) $('hospital-dialog').showModal();
  }
  async act(id, action, hospital) {
    if (this.busy) return false;
    this.actionError = '';
    if (['dispatch', 'call_hospital'].includes(action) && !this.calling?.enabled
        && (action === 'call_hospital' || !this.messaging?.enabled)) {
      this.actionError = 'Voice calls are off. Enable them in Centre settings, save, then dispatch again.';
      $('dispatch-slider').value = 0;
      this.workspace.toast(this.actionError);
      if ($('hospital-dialog').open) $('hospital-dialog').close();
      await this.openCentre();
      return false;
    }
    this.busy = true;
    this.paint();
    try {
      const alert = await this.api.request(`/incidents/${encodeURIComponent(id)}/response`, {action, ...(hospital === undefined ? {} : {hospital})});
      const index = this.alerts.findIndex(item => item.id === id);
      if (index < 0) this.alerts.push(alert);
      else this.alerts[index] = {...this.alerts[index], ...alert};
      await this.refresh();
      if (action === 'call_hospital') this.workspace.toast('Hospital call requested.');
      return true;
    } catch (exc) {
      this.actionError = exc.message;
      $('dispatch-slider').value = 0;
      this.workspace.toast(exc.message);
      return false;
    }
    finally { this.busy = false; this.paint(); }
  }
  async openCentre() {
    try {
      const data = await this.api.request('/centre');
      centreFields($('centre-edit-fields'), data.centre);
      $('calling-enabled').checked = !!data.centre.calling_enabled;
      $('calling-status').textContent = data.calling.configured
        ? data.calling.mode === 'trial_template'
          ? 'Twilio trial template configured. Calls play the demo prompt only, without incident or location details. Trial recipients must be verified in Twilio.'
          : 'Calls read the centre, incident, location and original detection time using Twilio speech. Trial recipients must be verified in Twilio. Check Recent response details for acceptance or errors.'
        : 'Twilio calling credentials are missing on the server. Calls will wait until configured.';
      const history = $('centre-response-history');
      history.replaceChildren();
      const recent = [...this.alerts].filter(alert => alert.calls?.length || alert.deliveries?.length || alert.state === 'requested').slice(-10).reverse();
      if (!recent.length) history.append(el('p', 'No response requests yet.'));
      for (const alert of recent) {
        const row = el('div');
        row.append(el('strong', `${alert.event.camera_name} · ${alert.event.event_type.replaceAll('_', ' ')}`),
          el('p', responseText(alert, this.now())));
        history.append(row);
      }
      $('messaging-enabled').checked = data.centre.messaging_enabled;
      $('messaging-status').textContent = data.messaging.configured
        ? 'Twilio credentials and evidence URL configured. Keep the evidence gateway and HTTPS tunnel running.'
        : 'Setup needed: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER and VMD_EVIDENCE_BASE_URL on the server. No messages will send until configured.';
      $('centre-error').textContent = '';
      $('centre-dialog').showModal();
    } catch (exc) { this.workspace.toast(exc.message); }
  }
  openLocation() {
    const camera = this.workspace.selectedState();
    if (!camera || camera.presentation || camera.mode === 'demo' || camera.live_camera === false) return;
    this.locationSession = {cameraId: camera.camera_id, saving: false};
    $('location-form').elements.place.value = camera.location?.place ?? '';
    $('location-error').textContent = '';
    $('location-dialog').showModal();
    this.locateCamera();
  }
  async locateCamera() {
    const session = this.locationSession;
    if (!session || session.saving) return;
    $('location-save').disabled = true;
    $('location-retry').disabled = true;
    $('location-error').textContent = '';
    $('location-status').textContent = 'Getting this laptop’s location…';
    try {
      const state = await sharedLocation.locate();
      if (this.locationSession !== session || !$('location-dialog').open) return;
      $('location-status').textContent = state.message;
      $('location-save').disabled = state.status !== 'ready' || state.source !== 'device';
    } catch (exc) {
      if (this.locationSession === session) $('location-error').textContent = exc.message;
    } finally {
      if (this.locationSession === session) $('location-retry').disabled = false;
    }
  }
  async saveLocation() {
    const session = this.locationSession;
    if (!session || session.saving || !$('location-dialog').open) return;
    session.saving = true;
    $('location-save').disabled = true;
    $('location-retry').disabled = true;
    $('location-error').textContent = '';
    try {
      const location = await sharedLocation.cameraLocation($('location-form').elements.place.value);
      if (this.locationSession !== session || !$('location-dialog').open) return;
      await this.api.request(`/cameras/${encodeURIComponent(session.cameraId)}/location`, location);
      if (this.locationSession === session) $('location-dialog').close();
      await this.workspace.refreshCameras();
      this.workspace.toast('Laptop location saved for future camera incidents.');
    } catch (exc) {
      if (this.locationSession === session) $('location-error').textContent = exc.message;
    } finally {
      session.saving = false;
      if (this.locationSession === session) {
        $('location-save').disabled = false;
        $('location-retry').disabled = false;
      }
    }
  }
  close() { clearInterval(this.timer); this.audio?.close(); }
}
