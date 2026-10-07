export function overlaysEnabled(preferences) {
  return preferences?.showDetectionOverlays !== false;
}

export function displayFramesPath(path, enabled) {
  return enabled ? path : `${path}${path.includes('?') ? '&' : '?'}overlays=false`;
}

export function renderOverlayButton(button, enabled, icon) {
  button.replaceChildren(icon(enabled ? 'eye' : 'eyeOff'),
    button.ownerDocument.createTextNode(enabled ? 'Hide pose & boxes' : 'Show pose & boxes'));
  button.setAttribute('aria-pressed', String(enabled));
  button.title = 'Display only. Detection continues in the background.';
}
