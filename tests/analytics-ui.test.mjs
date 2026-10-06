import test from "node:test";
import assert from "node:assert/strict";
import {renderAnalytics} from "../vmd/static/reports.mjs";

class Node {
  constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this.style = {}; this.clientWidth = 600; this.text = ""; }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return this.text + this.children.map(child => child.textContent).join(" "); }
  setAttribute(name, value) { this.attributes[name] = value; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; this.text = ""; }
}
function withDOM(run) {
  const previous = globalThis.document, nodes = new Map();
  globalThis.document = {
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, new Node("div")); return nodes.get(id); },
    createElement(tag) { return new Node(tag); },
    createElementNS(_namespace, tag) { return new Node(tag); },
  };
  try { run(id => document.getElementById(id)); } finally { globalThis.document = previous; }
}
function descendants(node, tag) { return node.children.flatMap(child => [...(child.tag === tag ? [child] : []), ...descendants(child, tag)]); }
function summary(overrides = {}) {
  return {days: 7, generated_at: 1790424000, start: 1789905600, timezone: "IST", today: 12,
    total: 151, weapon: 150, confirmed: 3, unreviewed: 148, false_positives: 2,
    trend: [{date: "2026-09-26", value: 151}],
    by_type: [{event_type: "gun_detected", value: 150}, {event_type: "fight", value: 1}],
    by_camera: [{camera_id: "cam-a", camera_name: "Main gate", value: 151}],
    crowd: [{hour: 1790420400, camera_id: "cam-a", camera_name: "Main gate", average: 2.5, peak: 5, samples: 20}],
    crowd_since: 1790337600, ...overrides};
}

test("analytics renders backend totals beyond the recent incident limit and labels actual cameras", () => withDOM($ => {
  renderAnalytics({summary: summary()});
  assert.equal($("analytics-total").textContent, "151 incidents");
  const metrics = $("analytics-metrics").children;
  assert.match(metrics[0].textContent, /INCIDENTS TODAY 12/);
  assert.match(metrics[1].textContent, /WEAPON DETECTIONS 150/);
  assert.match(metrics[2].textContent, /AWAITING REVIEW 148/);
  assert.match($("incident-types").textContent, /Gun detected/);
  assert.match($("incident-trend").textContent, /151 incidents/);
  assert.match($("incidents-sector").textContent, /Incidents by camera.*Main gate.*151/);
  assert.match($("crowd-table").textContent, /Main gate 2.5 5 20/);
  assert.match($("analytics-footnote").textContent, /2 marked false positive/);
  assert.match($("analytics-footnote").textContent, /3 confirmed by review/);
  assert.doesNotMatch($("analytics-metrics").textContent, /RESPONSE|fictional|Sector/);
}));

test("empty analytics shows zero real observations without sample data", () => withDOM($ => {
  renderAnalytics({summary: summary({days: 30, total: 0, today: 0, weapon: 0, confirmed: 0, unreviewed: 0,
    false_positives: 0, trend: [], by_type: [], by_camera: [], crowd: []})});
  assert.equal($("analytics-total").textContent, "0 incidents");
  assert.match($("trend-title").textContent, /30 Days/);
  assert.match($("incident-types").textContent, /No observations/);
  assert.match($("incidents-sector").textContent, /No incidents/);
  assert.match($("crowd-table").textContent, /No physical-camera people-count samples/);
}));

test("loading and failures never masquerade as zero observations; cached data is marked stale", () => withDOM($ => {
  renderAnalytics({loading: true});
  assert.match($("analytics-status").textContent, /Loading/);
  assert.equal($("analytics-total").textContent, "—");
  assert.doesNotMatch($("analytics-metrics").textContent, /\b0\b/);
  renderAnalytics({error: "Database unavailable"});
  assert.match($("analytics-status").textContent, /Analytics unavailable.*Database unavailable/);
  assert.match($("incident-trend").textContent, /Data unavailable/);
  assert.equal($("analytics-total").textContent, "—");
  renderAnalytics({summary: summary(), error: "Database unavailable"});
  assert.equal($("analytics-total").textContent, "151 incidents");
  assert.match($("analytics-provenance").textContent, /UPDATE FAILED/);
  assert.match($("analytics-status").textContent, /last saved snapshot/);
}));

test("camera grouping preserves every count and safe full-name tooltips", () => withDOM($ => {
  const unsafe = '<img src=x onerror="alert(1)">',
    cameras = Array.from({length: 10}, (_, i) => ({camera_id: `c${i}`, camera_name: i === 0 ? unsafe : `Camera ${i} with a long name`, value: 10 - i}));
  renderAnalytics({summary: summary({by_camera: cameras})});
  const bars = descendants($("incidents-sector"), "rect");
  assert.equal(bars.length, 8);
  assert.equal(bars.reduce((total, bar) => total + Number(bar.attributes["aria-label"].split(": ").at(-1)), 0), 55);
  assert.equal(bars.at(-1).attributes["aria-label"], "Other cameras (3): 6");
  assert.equal(bars[0].children[0].textContent, `${unsafe} · 10 incidents`);
  assert.equal(descendants($("incidents-sector"), "img").length, 0);
  assert.equal(cameras.length, 10);
}));
