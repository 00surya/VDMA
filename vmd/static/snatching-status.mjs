function readable(value) {
  return typeof value === 'string' ? value.replaceAll('_', ' ') : 'waiting for observations';
}

export function snatchingStatusText(camera) {
  if (!camera || camera.presentation) return 'Snatching check unavailable';
  const enabled = camera.settings?.snatching_vehicles === true;
  if (camera.status !== 'running' || camera.stale)
    return `Snatching check paused${enabled ? ' · rider review enabled' : ''}`;
  const standing = camera.signals?.snatching;
  const vehicle = camera.signals?.vehicle_snatching;
  const parts = [];
  if (standing) {
    const detail = standing.blockers?.find(item => typeof item === 'string');
    parts.push(`On foot: ${readable(standing.phase)}${detail ? ' · ' + detail : ''}`);
  }
  if (enabled) {
    const meta = camera.object_meta || {};
    if (meta.status === 'error') parts.push('Rider review unavailable · check the object model');
    else if (meta.stale || meta.status !== 'ready') parts.push('Rider review waiting for a fresh vehicle sample');
    else {
      const candidate = vehicle?.candidates?.[0];
      const detail = candidate?.blockers?.find(item => typeof item === 'string')
        || vehicle?.blockers?.find(item => typeof item === 'string');
      parts.push(`Rider review: ${readable(candidate?.phase || vehicle?.phase)}${detail ? ' · ' + detail : ''}`);
    }
  } else parts.push('Rider review off · enable in camera settings');
  return parts.join(' | ');
}
