import { $, el } from './ui.mjs';

const numericFields = ['threshold', 'hold_seconds', 'fight_confirmation_seconds', 'depth_fps', 'target_fps', 'object_fps'];
const booleanFields = ['eco_mode', 'object_detection'];
const stringFields = ['device', 'depth', 'detection_mode'];
export const settingFields = [...stringFields, ...numericFields, ...booleanFields];

export function readCameraSettings(form) {
  return Object.fromEntries(settingFields.map(key => [key,
    numericFields.includes(key) ? Number(form.elements[key].value)
      : booleanFields.includes(key) ? form.elements[key].value === 'true'
        : form.elements[key].value]));
}

export function validateCameraSettings(form) {
  const {depth, depth_fps: rate, detection_mode: mode} = form.elements;
  depth.setCustomValidity(mode.value === 'depth_confirmed' && depth.value === 'off'
    ? 'Choose a depth model for Depth-confirmed fight detection.' : '');
  rate.setCustomValidity(mode.value === 'depth_confirmed' && Number(rate.value) < .5
    ? 'Depth-confirmed mode needs at least 0.5 depth updates per second.' : '');
}

export function depthStatusText(camera) {
  const model = camera?.settings?.depth;
  if (!model || model === 'off') return 'Depth off · confirmed fights unavailable';
  if (camera.status !== 'running' || camera.stale)
    return `Depth ${camera.status === 'starting' ? 'starting' : 'paused'} · ${model} saved`;
  const meta = camera.depth_meta || {};
  if (meta.status === 'error') return `Depth unavailable · ${meta.error || 'worker error'}`;
  if (meta.status === 'loading') return `Depth loading · ${model}`;
  if (!['ready', 'simulated'].includes(meta.status) || meta.age_seconds == null)
    return `Depth waiting for a sample · ${model}`;
  const age = `${meta.age_seconds}s old`;
  if (!meta.confirmation_fresh || meta.stale) return `Depth too old for confirmation · ${age}`;
  if (camera.settings.detection_mode !== 'depth_confirmed')
    return `Depth ready · ${age} · Review-only mode cannot confirm fights`;
  const pairStatus = camera.signals?.depth_status;
  const pair = pairStatus === 'compatible' && camera.signals?.depth_samples > 0
    ? `pair depth compatible · ${camera.signals.depth_samples} matching samples`
    : pairStatus && pairStatus !== 'waiting'
      ? `pair depth: ${pairStatus.replaceAll('_', ' ')}`
      : 'waiting for a matching tracked pair';
  return `Depth ready · ${age} · ${pair}`;
}

export function fightStatusText(camera) {
  if (camera?.status !== 'running' || camera.stale) return 'Live fight check paused';
  const signals = camera.signals || {};
  const threshold = signals.threshold ?? camera.settings?.threshold;
  const score = Number.isFinite(camera.score) && Number.isFinite(threshold)
    ? `Score ${Math.round(camera.score * 100)} / 100 · threshold ${Math.round(threshold * 100)}` : 'Live fight check';
  const blockers = (signals.blockers || []).filter(value => typeof value === 'string' && value);
  const status = blockers.length ? blockers.slice(0, 2).join(' · ')
    : camera.people < 2 ? 'Needs two visible tracked people'
      : camera.assessment === 'fight_detected' ? 'Fight detected'
        : 'Watching for supported body contact and repeated interaction';
  return `${score} · ${status}`;
}

export function liveFightTimer(camera) {
  if (camera?.status !== 'running' || camera.stale || camera.presentation) return null;
  const signals = camera.signals || {}, state = signals.fight_timer_state;
  const seconds = signals.fight_timer_seconds, required = signals.fight_timer_required_seconds;
  if (!['counting', 'paused', 'waiting_depth', 'waiting_strikes', 'confirmed'].includes(state)
      || !Number.isFinite(seconds) || seconds < 0 || !Number.isFinite(required) || required <= 0) return null;
  const activeConfirmed = signals.fight_timer_started !== false
    && camera.alert?.event_type === 'fight' && camera.review !== 'false_positive';
  const reason = typeof signals.fight_timer_reason === 'string' ? signals.fight_timer_reason : '';
  return {
    state, seconds, required,
    label: signals.fight_timer_started === false ? 'Checking interaction'
      : state === 'confirmed' || activeConfirmed ? 'Fight detected' : 'Possible fight',
    value: `${seconds.toFixed(1)} / ${required.toFixed(1)} s`,
    reason: activeConfirmed && state !== 'confirmed'
      ? `Fight detected · latest check: ${reason || state.replaceAll('_', ' ')}` : reason,
  };
}

export function renderFightTimer(node, camera) {
  const timer = liveFightTimer(camera);
  node.hidden = !timer;
  if (!timer) {
    node.replaceChildren();
    delete node.dataset.state;
    return;
  }
  if (!node.childElementCount) {
    const heading = el('div', null, 'fight-timer-heading');
    heading.append(el('strong', null, 'fight-timer-title'), el('span', null, 'fight-timer-value'));
    const progress = el('progress');
    progress.setAttribute('aria-label', 'Fight confirmation evidence');
    node.append(heading, progress, el('small', null, 'fight-timer-reason'));
  }
  node.dataset.state = timer.state;
  node.querySelector('.fight-timer-title').textContent = timer.label;
  node.querySelector('.fight-timer-value').textContent = timer.value;
  const progress = node.querySelector('progress');
  progress.max = timer.required;
  progress.value = Math.min(timer.seconds, timer.required);
  progress.setAttribute('aria-valuetext', `${timer.value} · ${timer.reason || timer.state.replaceAll('_', ' ')}`);
  node.querySelector('.fight-timer-reason').textContent = timer.reason;
}

export class CameraSettingsEditor {
  constructor(workspace) {
    this.workspace = workspace;
    this.session = null;
    const fields = $('camera-setting-fields').cloneNode(true);
    fields.removeAttribute('id');
    const help = $('camera-settings-help').cloneNode(true);
    help.removeAttribute('id');
    $('camera-settings-inputs').append(fields, help);
    $('edit-camera-settings').addEventListener('click', () => this.open());
    $('close-camera-settings').addEventListener('click', () => $('camera-settings-dialog').close());
    $('camera-settings-form').addEventListener('submit', event => {
      event.preventDefault();
      this.save();
    });
    for (const id of ['source-form', 'camera-settings-form']) {
      $(id).addEventListener('input', () => validateCameraSettings($(id)));
      $(id).addEventListener('change', () => validateCameraSettings($(id)));
    }
    $('camera-settings-dialog').addEventListener('cancel', event => {
      if (this.session?.saving) event.preventDefault();
    });
  }

  open() {
    const camera = this.workspace.selectedState();
    if (this.workspace.busy || !camera?.settings || camera.presentation || camera.mode === 'demo') return;
    this.session = {cameraId: camera.camera_id, saving: false};
    const form = $('camera-settings-form');
    form.reset();
    for (const key of settingFields) {
      if (camera.settings[key] != null) form.elements[key].value = String(camera.settings[key]);
    }
    validateCameraSettings(form);
    $('camera-settings-title').textContent = `Settings · ${camera.name}`;
    $('camera-settings-context').textContent =
      'Apply updates this camera’s saved settings. A running or recovering camera restarts briefly; a stopped camera stays stopped. The source, location and saved incidents are kept.';
    $('camera-settings-error').textContent = '';
    $('camera-settings-inputs').disabled = false;
    $('camera-settings-save').disabled = false;
    $('close-camera-settings').disabled = false;
    $('camera-settings-dialog').showModal();
  }

  async save() {
    const session = this.session, form = $('camera-settings-form');
    if (!session || session.saving || this.workspace.busy) return;
    validateCameraSettings(form);
    if (!form.reportValidity()) return;
    const settings = readCameraSettings(form);
    session.saving = true;
    this.workspace.busy = true;
    $('camera-settings-inputs').disabled = true;
    $('camera-settings-save').disabled = true;
    $('close-camera-settings').disabled = true;
    $('camera-settings-error').textContent = '';
    try {
      await this.workspace.api.request(`/cameras/${encodeURIComponent(session.cameraId)}/settings`, settings, 'PATCH');
      await this.workspace.refreshCameras();
      $('camera-settings-dialog').close();
      this.workspace.toast('Camera settings saved.');
    } catch (error) {
      $('camera-settings-error').textContent = error.message;
    } finally {
      session.saving = false;
      this.workspace.busy = false;
      $('camera-settings-inputs').disabled = false;
      $('camera-settings-save').disabled = false;
      $('close-camera-settings').disabled = false;
      this.workspace.renderCameras();
    }
  }
}
