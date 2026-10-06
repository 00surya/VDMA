/** Small DOM primitives shared by the workspace views. Dynamic copy always uses textContent. */
export const $ = (id) => document.getElementById(id);
export function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}
export function svg(tag, attributes = {}, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes))
    node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = String(text);
  return node;
}
const paths = {
  home: "M3 10 12 3l9 7v11h-6v-7H9v7H3Z",
  activity: "M2 12h4l3-9 5 18 3-9h5",
  map: "m3 5 6-3 6 3 6-3v17l-6 3-6-3-6 3Zm6-3v17m6-14v17",
  trend: "m3 17 6-6 4 3 8-10m-6 0h6v6",
  chart: "M4 3v18h18M8 16v-5m5 5V6m5 10V9",
  chip: "M6 6h12v12H6ZM9 9h6v6H9ZM9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4",
  camera: "M4 6h12v12H4a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2Zm12 4 6-4v12l-6-4",
  alert: "m12 3 10 18H2Zm0 6v5m0 3v.1",
  pin: "M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 1 1 16 0ZM15 10a3 3 0 1 1-6 0 3 3 0 0 1 6 0",
  clock: "M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0ZM12 6v6l4 3",
  check: "m5 12 4 4L20 5",
  target:
    "M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0ZM18 12a6 6 0 1 1-12 0 6 6 0 0 1 12 0ZM14 12a2 2 0 1 1-4 0 2 2 0 0 1 4 0",
  user: "M16 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0ZM4 22v-3a8 8 0 0 1 16 0v3",
  grid: "M3 3h6v6H3Zm12 0h6v6h-6ZM3 15h6v6H3Zm12 0h6v6h-6Z",
  list: "M8 6h14M8 12h14M8 18h14M2 6h1m-1 6h1m-1 6h1",
  shield: "m12 2 9 4v7c0 6-9 9-9 9S3 19 3 13V6Z",
  arrow: "m9 5 7 7-7 7",
  logout: "M9 3H3v18h6m-1-9h14m-5-5 5 5-5 5",
};
export function icon(name, className = "") {
  const node = svg("svg", {
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    "stroke-width": 1.7,
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    "aria-hidden": "true",
    class: `icon ${className}`,
  });
  node.append(svg("path", { d: paths[name] || paths.shield }));
  return node;
}
export function setText(id, value) {
  $(id).textContent = value ?? "—";
}
function visibleWeapons(camera) {
  const meta = camera?.object_meta || {};
  if (camera?.presentation || camera?.stale || meta.stale || meta.status !== "ready" ||
      meta.knife_status === "error" || meta.knife_error ||
      (camera?.status && camera.status !== "running")) return [];
  const labels = new Map();
  for (const item of camera?.scene_objects || []) {
    const label = item.label === "guns" ? "gun" : item.label;
    if (["knife", "gun"].includes(label) && Number.isFinite(item.confidence) &&
        item.confidence > .90 && item.confidence <= 1)
      labels.set(label, Math.max(labels.get(label) || 0, item.confidence));
  }
  return [...labels].sort((a, b) => a[0] === b[0] ? 0 : a[0] === "knife" ? -1 : 1);
}
export function weaponAction(camera) {
  if (camera?.status !== "running") return null;
  const labels = visibleWeapons(camera).map(([label]) => label);
  if (!labels.length) return null;
  const label = labels.join(" and ");
  return `${label[0].toUpperCase()}${label.slice(1)} detected`;
}
export function objectSummary(camera) {
  const meta = camera?.object_meta || {};
  if (meta.status === "off" || !camera?.object_detection && !meta.status) return "Weapon detection off";
  if (meta.status === "error" || meta.knife_status === "error" || meta.knife_error) return "Weapon detector unavailable";
  if (camera?.status && camera.status !== "running") return "Weapon detection inactive";
  if (camera?.stale || meta.stale) return "Weapon sample stale";
  if (meta.status !== "ready") return "Loading weapon detector…";
  const items = visibleWeapons(camera);
  return items.length
    ? `Weapons: ${items.map(([label, confidence]) => `${label} ${Math.round(confidence * 100)}%`).join(" · ")}`
    : "No knife or gun detected in this sample";
}
export function formatTime(seconds) {
  return new Date(seconds * 1000).toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}
export function secondsLabel(value) {
  const total = Math.round(value);
  return value == null
    ? "—"
    : `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, "0")}s`;
}
export function badge(text, tone = "neutral") {
  return el("span", text, `badge ${tone}`);
}
export function hydrateIcons(root = document) {
  root
    .querySelectorAll("[data-icon]")
    .forEach((node) => node.replaceChildren(icon(node.dataset.icon)));
}

export class ApiClient {
  async request(path, body, method = "POST", timeoutMs = 12000) {
    const response = await fetch(`/api${path}`, {
      ...(body === undefined
        ? {}
        : {
            method,
            headers: {
              "Content-Type": "application/json",
              "X-VMD-Client": "dashboard",
            },
            body: JSON.stringify(body),
          }),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok) {
      if (response.status === 401 && !path.startsWith('/auth/')) location.replace('/login.html');
      const value = await response.json().catch(() => ({}));
      throw new Error(
        typeof value.detail === "string"
          ? value.detail
          : value.detail?.[0]?.msg || `Request failed (${response.status})`,
      );
    }
    return response.json();
  }
}

/** Each loop waits for its last request: slow servers cannot accumulate overlapping polls. */
export class PollLoop {
  constructor(task, interval, {fixedRate = false} = {}) {
    this.task = task;
    this.interval = interval;
    this.active = false;
    this.fixedRate = fixedRate;
  }
  start() {
    if (!this.active) {
      this.active = true;
      this.tick();
    }
  }
  stop() {
    this.active = false;
    clearTimeout(this.timer);
  }
  async tick() {
    const started = performance.now();
    try {
      await this.task();
    } finally {
      if (this.active)
        this.timer = setTimeout(() => this.tick(), this.fixedRate
          ? Math.max(0, this.interval - (performance.now() - started)) : this.interval);
    }
  }
}

// Keep the last encoded image per element without retaining removed camera cards.
const displayedFrames = new WeakMap();
export function setPreviewImage(image, empty, jpeg) {
  image.hidden = !jpeg;
  empty.hidden = !!jpeg;
  if (jpeg && displayedFrames.get(image) !== jpeg) {
    image.src = `data:image/jpeg;base64,${jpeg}`;
    displayedFrames.set(image, jpeg);
  }
}
