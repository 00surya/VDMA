import test from "node:test";
import assert from "node:assert/strict";
import { buildMapData } from "../vmd/static/map-data.mjs";

const location = { place: "Test site", latitude: 12, longitude: 34 };
const camera = { camera_id: "one", name: "Test camera", mode: "live", live_camera: true, sector: "4", location };
const incident = { id: "event", camera_id: "one", mode: "live", sector: "4", review: "unreviewed",
  signals: { live_camera: true, location } };

test("maps only real located cameras and original live incident locations, without moving history", () => {
  const data = buildMapData({ cameras: [camera, { ...camera, camera_id: "two" },
    { ...camera, camera_id: "missing", location: null },
    { ...camera, camera_id: "invalid", location: { ...location, longitude: Infinity } },
    { ...camera, mode: "demo" }, { ...camera, presentation: true },
    { ...camera, live_camera: false }, { ...camera, recording_id: "recording" }],
    incidents: [incident, { ...incident, id: "older", signals: { live_camera: true,
      location: { ...location, latitude: -12 } } },
    { ...incident, id: "no-location", signals: { live_camera: true } },
    { ...incident, id: "false-positive", review: "false_positive" },
    { ...incident, id: "demo", mode: "demo" },
    { ...incident, id: "recording", signals: { live_camera: false, location } }],
  });
  assert.equal(data.cameras.length, 2);
  assert.equal(data.cameraGroups.length, 1);
  assert.equal(data.cameraGroups[0].count, 2);
  assert.deepEqual(data.cameraGroups[0].latlng, [12, 34]);
  assert.equal(data.incidents.length, 2);
  assert.deepEqual(data.hotspots.map(h => h.latlng), [[12, 34], [-12, 34]]);
  assert.equal(data.missingCameras, 2);
  assert.equal(data.missingIncidents, 1);
  assert.deepEqual(incident.signals.location, location);
  assert.equal(camera.latlng, undefined);
});

test("scope, zero coordinates and duplicate locations retain exact counts; empty data has no invented points", () => {
  const data = buildMapData({ scope: "4", cameras: [camera, { ...camera, sector: "2" }],
    incidents: [incident, { ...incident, id: "second", review: "confirmed" }, { ...incident, sector: "2" }],
  });
  assert.equal(data.cameras.length, 1);
  assert.equal(data.hotspots.length, 1);
  assert.equal(data.hotspots[0].count, 2);
  assert.equal(buildMapData({ cameras: [{ ...camera, location: { latitude: 0, longitude: 0 } }] }).cameras.length, 1);
  assert.equal(buildMapData({ cameras: [{ ...camera, location: { latitude: "12", longitude: 34 } }] }).missingCameras, 1);
  assert.deepEqual(buildMapData(), { cameras: [], incidents: [], cameraGroups: [], hotspots: [], missingCameras: 0, missingIncidents: 0 });
});

test("restored stopped physical cameras retain their saved location before running again", () => {
  const restored = {...camera, mode: null, status: "idle"};
  const data = buildMapData({cameras: [restored, {...restored, live_camera: false}]});
  assert.equal(data.cameras.length, 1);
  assert.deepEqual(data.cameras[0].latlng, [12, 34]);
});

test("coordinates outside Web Mercator are unplotted and counted without relocating records", () => {
  const latitudes = [85.0511287798, -85.0511287798, 0, 85.0511287799, -85.0511287799, 88.2, -90];
  const cameras = latitudes.map((latitude, index) => ({ ...camera, camera_id: String(index),
    location: { latitude, longitude: 0 } }));
  const incidents = cameras.map((source, index) => ({ ...incident, id: String(index),
    signals: { live_camera: true, location: source.location } }));
  const data = buildMapData({ cameras, incidents });
  const expected = latitudes.slice(0, 3).map(latitude => [latitude, 0]);
  assert.deepEqual(data.cameras.map(row => row.latlng), expected);
  assert.deepEqual(data.cameraGroups.map(row => row.latlng), expected);
  assert.deepEqual(data.incidents.map(row => row.latlng), expected);
  assert.deepEqual(data.hotspots.map(row => row.latlng), expected);
  assert.equal(data.missingCameras, 4);
  assert.equal(data.missingIncidents, 4);
  assert.deepEqual(cameras.map(row => row.location.latitude), latitudes);
  assert.deepEqual(incidents.map(row => row.signals.location.latitude), latitudes);
});
