import { $, el, icon, badge, formatTime, secondsLabel } from "./ui.mjs";
import {
  summarizeEvents,
  ZONES,
  INFRASTRUCTURE,
  riskTone,
  scoped,
  EVENT_NAMES,
} from "./data.mjs";
import { lineChart, barChart, donutChart } from "./charts.mjs";

export function statCard(
  label,
  value,
  subtitle,
  symbol,
  change = "",
  tone = "",
) {
  const card = el("article", null, `stat-card ${tone}`),
    main = el("div", null, "stat-main"),
    copy = el("div"),
    glyph = el("span", null, "stat-icon");
  copy.append(
    el("span", label, "stat-label"),
    el("strong", value),
    el("p", subtitle),
  );
  glyph.append(icon(symbol));
  main.append(copy, glyph);
  card.append(main);
  if (change) card.append(el("p", change, "stat-change"));
  return card;
}
function compactCard(label, value, symbol, tone = "") {
  const card = el("article", null, `stat-card compact ${tone}`),
    glyph = el("span", null, "stat-icon"),
    copy = el("div");
  glyph.append(icon(symbol));
  copy.append(el("strong", value), el("p", label));
  card.append(glyph, copy);
  return card;
}
export function renderOverview(dataset, scope) {
  const zones = scoped(ZONES, scope),
    summary = summarizeEvents(
      dataset.events,
      new Date(dataset.created * 1000),
      7,
      scope,
    );
  $("overview-metrics").replaceChildren(
    statCard(
      "MONITORED CAMERAS",
      zones.reduce((sum, zone) => sum + zone.cameras, 0),
      "Across the demo network",
      "camera",
    ),
    statCard("INCIDENTS TODAY", summary.today, "Presentation ledger", "alert"),
    statCard(
      "RESPONSE UNITS",
      zones.length,
      "Available in this scenario",
      "shield",
    ),
    statCard(
      "AVG. RESPONSE",
      secondsLabel(summary.response),
      "Selected sectors · 7 days",
      "clock",
    ),
  );
  $("sector-overview").replaceChildren(
    ...zones.map((zone) => {
      const row = el("div", null, "sector-row"),
        name = el("div");
      name.append(el("span", `Sector ${zone.id}`), el("small", zone.name));
      row.append(
        name,
        el("strong", `${zone.cameras} cameras`),
        badge(zone.score, riskTone(zone.score)),
      );
      return row;
    }),
  );
  $("overview-events").replaceChildren(
    ...summary.events
      .slice(-5)
      .reverse()
      .map((event) => {
        const row = el("div", null, "activity-item"),
          copy = el("div");
        copy.append(
          el("strong", EVENT_NAMES[event.event_type]),
          el("small", `Sector ${event.sector} · ${formatTime(event.created)}`),
        );
        row.append(icon("alert"), copy);
        return row;
      }),
  );
}
export function renderAnalytics({summary = null, loading = false, error = null, days = 7}) {
  const period = summary?.days ?? days;
  $("analytics-provenance").textContent = error && summary
    ? "SAVED SNAPSHOT · UPDATE FAILED" : "LOCAL OBSERVATIONS";
  $("analytics-status").textContent = error
    ? `${summary ? "Update failed; showing the last saved snapshot. " : "Analytics unavailable. "}${error}`
    : loading || !summary ? "Loading local analytics…"
      : `Updated ${formatTime(summary?.generated_at)} · ${summary?.timezone || "server local time"}`;
  $("trend-title").textContent = `Incident Trends (Last ${period} Days)`;
  $("crowd-panel").hidden = false;
  if (!summary) {
    $("analytics-metrics").replaceChildren(
      statCard("INCIDENTS TODAY", "—", "Awaiting stored observations", "alert"),
      statCard("WEAPON DETECTIONS", "—", `Last ${period} days`, "target"),
      statCard("AWAITING REVIEW", "—", `Last ${period} days`, "check"),
    );
    $("analytics-total").textContent = "—";
    for (const id of ["incident-trend", "incident-types", "incidents-sector", "crowd-table"])
      $(id).replaceChildren(el("p", error ? "Data unavailable." : "Loading observations…", "page-footnote"));
    $("analytics-footnote").textContent = "Analytics uses stored camera incidents and people-count samples.";
    return;
  }
  $("analytics-metrics").replaceChildren(
    statCard("INCIDENTS TODAY", summary.today, "Since server local midnight", "alert"),
    statCard("WEAPON DETECTIONS", summary.weapon, `Last ${period} days · gun and knife incidents`, "target"),
    statCard("AWAITING REVIEW", summary.unreviewed, `Last ${period} days · saved incidents`, "check"),
  );
  $("analytics-total").textContent = `${summary.total} incident${summary.total === 1 ? "" : "s"}`;
  const trend = summary.trend.map((row) => ({
    ...row,
    label: new Date(`${row.date}T12:00:00`).toLocaleDateString("en-IN", {day: "2-digit", month: "short"}),
  }));
  lineChart($("incident-trend"), trend);
  donutChart($("incident-types"), summary.by_type.map((row) => ({
    label: EVENT_NAMES[row.event_type] || row.event_type.replaceAll("_", " "), value: row.value,
  })));
  const cameras = [...summary.by_camera].sort((a, b) => b.value - a.value);
  const cameraRows = cameras.slice(0, cameras.length > 8 ? 7 : 8).map((row) => {
    const label = row.camera_name || row.camera_id || "Unknown camera";
    return {label, shortLabel: label.length > 13 ? `${label.slice(0, 12)}…` : label, value: row.value};
  });
  if (cameras.length > 8) cameraRows.push({
    label: `Other cameras (${cameras.length - 7})`, shortLabel: "Other cameras",
    value: cameras.slice(7).reduce((sum, row) => sum + row.value, 0),
  });
  if (cameraRows.length) barChart($("incidents-sector"), cameraRows, "Incidents by camera");
  else $("incidents-sector").replaceChildren(el("p", "No incidents for this period.", "page-footnote"));
  $("analytics-footnote").textContent =
    `All stored physical-camera incidents in the selected period; synthetic tests and recordings are excluded. ` +
    `${summary.false_positives} marked false positive and excluded from incident totals and charts. ` +
    `${summary.confirmed} confirmed by review; remaining detections may be unverified. ` +
    `People counts use available hourly camera samples${summary.crowd_since ? ` since ${formatTime(summary.crowd_since)}` : ""}; older samples without camera-source metadata are excluded.`;
  const table = el("table"), head = el("thead"), header = el("tr");
  ["Hour", "Camera", "Average people", "Peak", "Samples"].forEach((label) => header.append(el("th", label)));
  head.append(header);
  table.append(head);
  const body = el("tbody");
  for (const row of summary.crowd) {
    const tr = el("tr");
    [formatTime(row.hour), row.camera_name || row.camera_id || "Unknown camera", row.average, row.peak, row.samples]
      .forEach((value) => tr.append(el("td", value)));
    body.append(tr);
  }
  table.append(body);
  $("crowd-table").replaceChildren(summary.crowd.length ? table
    : el("p", "No physical-camera people-count samples in the last 24 hours.", "page-footnote"));
}
export function renderHealth(cameras, connection, dataset) {
  const sample = $("health-dataset").value === "sample",
    now = Date.now() / 1000;
  const rows = sample
    ? INFRASTRUCTURE.map(([name, latency, uptime]) => ({
        name,
        latency,
        uptime,
        status: "online",
        checked: dataset.created,
      }))
    : [
        {
          name: "Local API",
          status: connection ? "online" : "offline",
          latency: null,
          checked: connection ? now : null,
        },
        {
          name: "SQLite evidence writer",
          status: !connection
            ? "unknown"
            : cameras.some((camera) => camera.storage_error)
              ? "degraded"
              : "online",
          latency: null,
          checked: connection ? now : null,
        },
        ...cameras.flatMap((camera) => [
          {
            name: `${camera.name} · pose & motion`,
            status: !connection
              ? "unknown"
              : camera.stale
                ? "degraded"
                : camera.status === "running"
                  ? "online"
                  : camera.status === "error"
                    ? "offline"
                    : camera.status === "starting"
                      ? "degraded"
                      : "idle",
            latency: camera.status === "running" ? camera.latency_ms : null,
            checked: camera.last_frame_at,
          },
          ...(camera.depth && camera.depth !== "off"
            ? [
                {
                  name: `${camera.name} · sampled depth`,
                  status: !connection
                    ? "unknown"
                    : ["idle", "finished", "stopping"].includes(camera.status)
                      ? "idle"
                      : camera.status === "error"
                        ? "offline"
                        : camera.depth_meta?.status === "error"
                          ? "offline"
                          : camera.depth_meta?.stale
                            ? "degraded"
                            : ["ready", "simulated"].includes(
                                  camera.depth_meta?.status,
                                )
                              ? "online"
                              : camera.depth_meta?.status === "loading"
                                ? "degraded"
                                : "idle",
                  latency: camera.depth_meta?.latency_ms,
                  checked: camera.depth_submitted_at,
                },
              ]
            : []),
        ]),
      ];
  const online = rows.filter((row) => row.status === "online").length,
    offline = rows.filter((row) => row.status === "offline").length,
    degraded = rows.filter((row) => row.status === "degraded").length;
  const latency = rows.filter((row) => Number.isFinite(row.latency));
  $("health-percent").textContent =
    `${((online / rows.length) * 100).toFixed(1)}%`;
  $("health-provenance").textContent = sample
    ? "PRESENTATION INFRASTRUCTURE"
    : "LOCAL ENGINE";
  $("health-description").textContent = sample
    ? "A demonstration of a city-scale component inventory."
    : "Measured status from this laptop and its connected cameras.";
  $("health-metrics").replaceChildren(
    compactCard("Online", online, "check"),
    compactCard("Offline", offline, "alert", "red"),
    compactCard("Degraded", degraded, "activity", "yellow"),
    compactCard(
      "Avg. latency",
      latency.length
        ? `${Math.round(latency.reduce((sum, row) => sum + row.latency, 0) / latency.length)}ms`
        : "—",
      "clock",
      "blue",
    ),
  );
  $("health-components").replaceChildren(
    ...rows.map((row) => {
      const tr = el("tr"),
        status = el("td"),
        uptime = el("td");
      status.append(
        badge(
          row.status.toUpperCase(),
          { online: "green", offline: "red", degraded: "yellow" }[row.status] ||
            "neutral",
        ),
      );
      if (sample) {
        const wrap = el("div", null, "uptime"),
          bar = el("progress");
        bar.max = 100;
        bar.value = row.uptime;
        bar.setAttribute("aria-label", `${row.name} sample uptime`);
        wrap.append(bar, el("span", `${row.uptime.toFixed(2)}%`));
        uptime.append(wrap);
      } else uptime.textContent = "Not measured";
      tr.append(
        el("td", row.name),
        status,
        el("td", row.latency == null ? "—" : `${row.latency}ms`),
        uptime,
        el("td", row.checked ? formatTime(row.checked) : "—", "mono"),
      );
      return tr;
    }),
  );
  $("health-footnote").textContent = sample
    ? "Sample infrastructure only. Local centre sign-in and Twilio configuration are separate from this illustrative remote inventory."
    : "Idle cameras are stopped, not failed. Uptime percentages are not fabricated. Head blurring is best effort; depth has its own sampling time.";
}
