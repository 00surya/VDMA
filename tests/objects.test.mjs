import test from "node:test";
import assert from "node:assert/strict";
import { objectSummary, weaponAction } from "../vmd/static/ui.mjs";

test("knife labels stay visible for one person without requiring an incident", () => {
  const camera = { status: "running", people: 1, alert: null, object_meta: {status: "ready"}, scene_objects: [
    {label: "person", confidence: .98}, {label: "knife", confidence: .31},
    {label: "knife", confidence: .94}, {label: "toilet", confidence: .99},
  ]};
  assert.equal(objectSummary(camera), "Weapons: knife 94%");
  assert.equal(weaponAction(camera), "Knife detected");
  assert.equal(objectSummary({...camera, object_meta: {status: "ready", stale: true}}), "Weapon sample stale");
  assert.equal(objectSummary({...camera, object_meta: {status: "error"}}), "Weapon detector unavailable");
  assert.equal(objectSummary({...camera, object_meta: {status: "off"}}), "Weapon detection off");
  assert.equal(objectSummary({...camera, scene_objects: []}), "No knife or gun detected in this sample");
});

test("gun and combined actions require fresh successful detections", () => {
  const camera = {status: "running", people: 0, alert: null,
    object_meta: {status: "ready", knife_status: "ready"},
    scene_objects: [{label: "guns", confidence: .98}]};
  assert.equal(weaponAction(camera), "Gun detected");
  assert.equal(objectSummary(camera), "Weapons: gun 98%");
  assert.equal(weaponAction({...camera, scene_objects: [{label: "gun", confidence: .98},
    {label: "knife", confidence: .97}]}), "Knife and gun detected");
  for (const override of [
    {scene_objects: []}, {stale: true}, {presentation: true},
    ...["idle", "error", "finished", "stopping", "starting"].map(status => ({status})),
    ...["off", "error", "loading"].map(status => ({object_meta: {status}})),
    {object_meta: {status: "ready", stale: true}},
    {object_meta: {status: "ready", knife_status: "error"}},
    {object_meta: {status: "ready", knife_error: "Inference failed"}},
    {scene_objects: [{label: "gun", confidence: .49}, {label: "knife", confidence: NaN},
      {label: "gun", confidence: 2}, {label: "knife", confidence: "0.8"}, {label: "toilet", confidence: .99}]},
  ]) assert.equal(weaponAction({...camera, ...override}), null, JSON.stringify(override));
});

test("live weapon labels and actions require confidence strictly above 90%", () => {
  for (const label of ["knife", "gun", "guns"]) {
    for (const confidence of [.89, .90, .9001, 1]) {
      const camera = {status: "running", object_meta: {status: "ready"},
        scene_objects: [{label, confidence}]};
      const expected = label === "knife" ? "Knife detected" : "Gun detected";
      assert.equal(weaponAction(camera), confidence > .90 ? expected : null);
      assert.equal(objectSummary(camera).startsWith("Weapons:"), confidence > .90);
    }
  }
});
