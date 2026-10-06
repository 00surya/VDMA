import test from "node:test";
import assert from "node:assert/strict";
import {
  buildDataset,
  summarizeEvents,
  demoCameras,
  scoped,
  dispatchTransition,
  ZONES,
  percentageShares,
} from "../vmd/static/data.mjs";
import { validateProfile, readJSON } from "../vmd/static/session.mjs";

const now = new Date("2026-09-12T13:40:00+05:30");

test("one presentation ledger produces consistent line, pie, sector and total counts", () => {
  const dataset = buildDataset(now);
  for (const days of [7, 30]) {
    const report = summarizeEvents(dataset.events, now, days);
    assert.equal(
      report.total,
      report.trend.reduce((sum, row) => sum + row.value, 0),
    );
    assert.equal(
      report.total,
      report.byType.reduce((sum, row) => sum + row.value, 0),
    );
    assert.equal(
      report.total,
      report.bySector.reduce((sum, row) => sum + row.value, 0),
    );
    assert.equal(report.today, 14);
    assert.ok(
      report.events.every((event) => event.created <= now.getTime() / 1000),
    );
  }
  assert.equal(summarizeEvents(dataset.events, now, 7).weapon, 3);
  assert.equal(
    new Set(dataset.events.map((event) => event.id)).size,
    dataset.events.length,
  );
});

test("each operator sector only contributes its own presentation records", () => {
  const dataset = buildDataset(now);
  let sum = 0;
  for (const zone of ZONES) {
    const report = summarizeEvents(dataset.events, now, 7, zone.id);
    sum += report.total;
    assert.ok(report.events.every((event) => event.sector === zone.id));
    assert.equal(report.bySector.length, 1);
  }
  assert.equal(sum, summarizeEvents(dataset.events, now, 7).total);
});

test("fixtures are deterministic and do not create future records just after midnight", () => {
  assert.deepEqual(buildDataset(now), buildDataset(now));
  const midnight = new Date(now);
  midnight.setHours(0, 0, 0, 0);
  assert.ok(
    buildDataset(midnight).events.every(
      (event) => event.created <= midnight.getTime() / 1000,
    ),
  );
});

test("presentation camera IDs and events remain explicitly separate from live inputs", () => {
  const cameras = demoCameras(buildDataset(now));
  assert.equal(cameras.length, 6);
  assert.equal(new Set(cameras.map((camera) => camera.camera_id)).size, 6);
  assert.ok(
    cameras.every(
      (camera) => camera.presentation && camera.mode === "presentation",
    ),
  );
  assert.equal(scoped(cameras, "4").length, 1);
});

test("dispatch requires ordered transitions and cannot accidentally skip to resolved", () => {
  assert.equal(dispatchTransition("unassigned", "resolve"), "unassigned");
  assert.equal(dispatchTransition("unassigned", "dispatch"), "queued");
  assert.equal(dispatchTransition("queued", "dispatch"), "queued");
  assert.equal(dispatchTransition("queued", "arrive"), "on_scene");
  assert.equal(dispatchTransition("on_scene", "resolve"), "resolved");
  assert.equal(dispatchTransition("resolved", "dispatch"), "resolved");
});

test("presentation profiles validate roles and recover from unavailable local storage", () => {
  assert.equal(validateProfile({ role: "superuser" }), null);
  assert.equal(validateProfile(null), null);
  assert.deepEqual(validateProfile({ role: "operator", sector: "unlisted" }), {
    role: "operator",
    sector: "4",
    name: "Sector operator",
  });
  assert.deepEqual(readJSON({ getItem: () => "{broken" }, "profile", {}), {});
  assert.equal(
    readJSON(
      {
        getItem: () => {
          throw new Error("blocked");
        },
      },
      "profile",
      null,
    ),
    null,
  );
});

test("displayed chart percentages sum to 100 without changing underlying counts", () => {
  assert.deepEqual(percentageShares([1, 1, 1]), [34, 33, 33]);
  assert.equal(
    percentageShares([29, 14, 14, 43, 3]).reduce(
      (sum, value) => sum + value,
      0,
    ),
    100,
  );
  assert.deepEqual(percentageShares([0, 0]), [0, 0]);
});
