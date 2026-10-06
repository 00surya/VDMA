import { validCenter } from "./map-location.mjs";

function point(location) {
  const latlng = [location?.latitude, location?.longitude];
  // Web Mercator tiles cannot represent the poles; do not clamp saved coordinates.
  return validCenter(latlng) && Math.abs(latlng[0]) <= 85.0511287798 ? latlng : null;
}

function groups(rows, field) {
  const grouped = new Map();
  for (const row of rows) {
    const key = JSON.stringify(row.latlng);
    if (!grouped.has(key)) grouped.set(key, { latlng: row.latlng, [field]: [], count: 0 });
    const group = grouped.get(key);
    group[field].push(row);
    group.count++;
  }
  return [...grouped.values()];
}

/** Original recorded coordinates only: no demo sites or relocation of old incidents. */
export function buildMapData({ cameras = [], incidents = [], scope = "all" } = {}) {
  const inScope = row => scope === "all" || String(row.sector) === String(scope);
  const liveCameras = cameras.filter(camera => camera.mode !== "demo" && camera.mode !== "presentation" && !camera.presentation &&
    camera.live_camera === true && !camera.recording_id && inScope(camera));
  const realIncidents = incidents.filter(event => event.mode === "live" && !event.presentation &&
    event.signals?.live_camera === true && event.review !== "false_positive" && inScope(event));
  const locatedCameras = liveCameras.map(camera => ({ ...camera, latlng: point(camera.location) }))
    .filter(camera => camera.latlng);
  const locatedIncidents = realIncidents.map(event => ({ ...event, latlng: point(event.signals.location) }))
    .filter(event => event.latlng);
  return {
    cameras: locatedCameras,
    incidents: locatedIncidents,
    cameraGroups: groups(locatedCameras, "cameras"),
    hotspots: groups(locatedIncidents, "incidents"),
    missingCameras: liveCameras.length - locatedCameras.length,
    missingIncidents: realIncidents.length - locatedIncidents.length,
  };
}
