/** Deterministic presentation fixtures. Never sent to the camera API or incident database. */
export const EVENT_NAMES = Object.freeze({
  fight: "Fight detected",
  possible_fight: "Possible fight",
  possible_snatching: "Possible snatching",
  snatching_detected: "Snatching detected",
  person_down: "Person down",
  possible_fall: "Possible fall",
  person_down_after_fight: "Person down after fight signal",
  hands_up: "Hands-up posture",
  crowd: "Crowd density",
  weapon: "Weapon flag",
  knife_detected: "Knife detected",
  gun_detected: "Gun detected",
  unattended_object: "Unattended item",
});
export const COLORS = [
  "#ef4444",
  "#f59e0b",
  "#eab308",
  "#6b7280",
  "#3b82f6",
  "#10b981",
];
export const ZONES = Object.freeze([
  {
    id: "1",
    name: "Central district",
    x: 410,
    y: 390,
    score: 52,
    unit: "DELTA-01",
    cameras: 12,
  },
  {
    id: "2",
    name: "Residential",
    x: 352,
    y: 560,
    score: 87,
    unit: "DELTA-02",
    cameras: 9,
  },
  {
    id: "3",
    name: "Civic centre",
    x: 402,
    y: 235,
    score: 68,
    unit: "DELTA-03",
    cameras: 8,
  },
  {
    id: "4",
    name: "Market",
    x: 545,
    y: 197,
    score: 31,
    unit: "DELTA-04",
    cameras: 14,
  },
  {
    id: "6",
    name: "Park",
    x: 525,
    y: 580,
    score: 84,
    unit: "DELTA-06",
    cameras: 6,
  },
  {
    id: "7",
    name: "Transit hub",
    x: 646,
    y: 347,
    score: 48,
    unit: "DELTA-07",
    cameras: 11,
  },
  {
    id: "8",
    name: "University",
    x: 721,
    y: 511,
    score: 72,
    unit: "DELTA-08",
    cameras: 10,
  },
  {
    id: "9",
    name: "Industrial",
    x: 238,
    y: 298,
    score: 70,
    unit: "DELTA-09",
    cameras: 7,
  },
]);
export function riskTone(score) {
  return score >= 80
    ? "green"
    : score >= 60
      ? "yellow"
      : score >= 40
        ? "orange"
        : "red";
}
export function scoped(items, sector = "all") {
  return sector === "all"
    ? items
    : items.filter((item) => String(item.sector ?? item.id) === String(sector));
}
export function buildDataset(now = new Date()) {
  const start = new Date(now);
  start.setHours(0, 0, 0, 0);
  const dailyCounts = [12, 19, 8, 15, 22, 13, 14];
  const events = [];
  for (let day = 0; day < 30; day++) {
    const count = day < 23 ? 9 + (day % 8) : dailyCounts[day - 23];
    const dayKey = start.getTime() - (29 - day) * 86400000;
    for (let index = 0; index < count; index++) {
      const serial = events.length;
      const zone = ZONES[(index * 3 + day) % ZONES.length];
      const type = [
        "crowd",
        "fight",
        "crowd",
        "possible_fall",
        "person_down",
        "fight",
        "crowd",
      ][serial % 7];
      // Today's fixtures occur before the current time, not later in the day.
      const availableSeconds =
        day === 29 ? Math.max(0, (now - start) / 1000) : 86400;
      const created =
        (start.getTime() - (29 - day) * 86400000) / 1000 +
        Math.floor(((index + 0.5) / count) * availableSeconds);
      events.push({
        id: `sample-${dayKey}-${index}`,
        created,
        mode: "presentation",
        event_type: type,
        sector: zone.id,
        camera_name: `CAM-${zone.id.padStart(2, "0")}-${String((index % 5) + 1).padStart(2, "0")}`,
        score: 0.74 + (serial % 20) / 100,
        review:
          serial % 11 === 0
            ? "false_positive"
            : day === 29 && index > count - 4
              ? "unreviewed"
              : "confirmed",
        response_seconds: 117 + (serial % 75),
        reasons: [`${EVENT_NAMES[type]} in ${zone.name.toLowerCase()}`],
      });
    }
  }
  // Three synthetic weapon flags in the presentation week.
  const weekEvents = events.filter(
    (event) => event.created >= start / 1000 - 6 * 86400,
  );
  for (const index of [12, 43, 77])
    if (weekEvents[index]) {
      weekEvents[index].event_type = "weapon";
      weekEvents[index].reasons = ["Weapon flag · presentation scenario"];
    }
  return Object.freeze({
    created: now.getTime() / 1000,
    events: Object.freeze(events),
  });
}
export function summarizeEvents(
  events,
  now = new Date(),
  days = 7,
  sector = "all",
) {
  const today = new Date(now);
  today.setHours(0, 0, 0, 0);
  const start = today.getTime() / 1000 - (days - 1) * 86400;
  const selected = scoped(events, sector).filter(
    (event) => event.created >= start && event.created <= now.getTime() / 1000,
  );
  const trend = Array.from({ length: days }, (_, index) => {
    const time = start + index * 86400;
    return {
      label: new Date(time * 1000).toLocaleDateString(
        "en-IN",
        days === 7 ? { weekday: "short" } : { day: "numeric", month: "short" },
      ),
      value: selected.filter(
        (event) => event.created >= time && event.created < time + 86400,
      ).length,
    };
  });
  const byType = Object.entries(EVENT_NAMES)
    .map(([key, label]) => ({
      label,
      value: selected.filter((event) => event.event_type === key).length,
    }))
    .filter((item) => item.value);
  const bySector = ZONES.map((zone) => ({
    label: `Sector ${zone.id}`,
    value: selected.filter((event) => event.sector === zone.id).length,
  })).filter((item) => sector === "all" || item.label === `Sector ${sector}`);
  const responses = selected
    .map((event) => event.response_seconds)
    .filter(Number.isFinite);
  return {
    events: selected,
    trend,
    byType,
    bySector,
    total: selected.length,
    today: trend.at(-1)?.value || 0,
    previous: trend.at(-2)?.value || 0,
    weapon: selected.filter((event) => ["weapon", "knife_detected", "gun_detected"].includes(event.event_type)).length,
    response: responses.length
      ? responses.reduce((a, b) => a + b, 0) / responses.length
      : null,
  };
}
export function demoCameras(dataset) {
  return [
    ["CAM-ALPHA-04", "4", 6, "fight", 0.91, "Striking arm movement"],
    ["CAM-BRAVO-12", "7", 48, "crowd", 0.84, "Sustained crowding"],
    ["CAM-CHARLIE-08", "2", 9, null, 0, "Normal activity"],
    ["CAM-DELTA-15", "9", 0, null, 0, "Normal activity"],
    ["CAM-ECHO-03", "1", 12, null, 0, "Normal activity"],
    ["CAM-FOXTROT-21", "6", 4, null, 0, "Normal activity"],
  ].map(([name, sector, people, type, score, action], index) => ({
    camera_id: `presentation-${index}`,
    name,
    sector,
    people,
    score,
    action,
    presentation: true,
    mode: "presentation",
    status: "running",
    fps: 24,
    latency_ms: 39 + index * 2,
    sequence: 1,
    alert: type
      ? {
          event_type: type,
          label: EVENT_NAMES[type].toUpperCase(),
          reasons: [`${action} · presentation scenario`],
        }
      : null,
    event_id: `presentation-alert-${new Date(dataset.created * 1000).setHours(0, 0, 0, 0)}-${index}`,
    last_frame_at: dataset.created - index * 3,
  }));
}
export const INFRASTRUCTURE = Object.freeze([
  ["AI inference engine", 12, 99.98],
  ["Database cluster", 8, 99.99],
  ["Camera feed gateway", 24, 99.95],
  ["Object detection service", 45, 99.92],
  ["Alert notification service", 6, 100],
  ["Video stream processor", 18, 99.96],
  ["Authentication service", 5, 99.99],
  ["Load balancer", 3, 100],
  ["Cache layer (Redis)", 2, 99.97],
  ["Evidence storage", 11, 99.98],
  ["Telemetry collector", 9, 99.99],
  ["Dispatch coordinator", 14, 100],
]);
export function dispatchTransition(current, action) {
  const next = {
    unassigned: { dispatch: "queued" },
    queued: { arrive: "on_scene" },
    on_scene: { resolve: "resolved" },
  };
  return next[current]?.[action] || current;
}

/** Largest-remainder allocation keeps displayed whole percentages summing to 100. */
export function percentageShares(values) {
  const total = values.reduce((sum, value) => sum + value, 0);
  if (!total) return values.map(() => 0);
  const exact = values.map((value) => (value / total) * 100),
    result = exact.map(Math.floor);
  const order = exact
    .map((value, index) => ({ index, remainder: value - result[index] }))
    .sort((a, b) => b.remainder - a.remainder);
  const remaining = 100 - result.reduce((sum, value) => sum + value, 0);
  for (let i = 0; i < remaining; i++) result[order[i].index]++;
  return result;
}
