import "./leaflet.js";
import { el, icon, formatTime } from "./ui.mjs";
import { EVENT_NAMES } from "./data.mjs";
import { buildMapData } from "./map-data.mjs";
import { sharedLocation } from "./map-location.mjs";

/** Lazily mounted map of saved camera and incident coordinates. */
export class CoverageMap {
  constructor(target, { onSelect = () => {}, location = sharedLocation } = {}) {
    this.target = target;
    this.onSelect = onSelect;
    this.location = location;
    this.selected = null;
    this.heat = false;
    this.scope = "all";
    this.cameras = [];
    this.incidents = [];
    this.data = buildMapData();
    this.appliedRevision = -1;
    this.markerNodes = new Map();
    const page = target.closest(".map-page");
    this.status = page.querySelector("[data-map-status]");
    this.locateButton = page.querySelector("[data-map-locate]");
    this.locateButton.addEventListener("click", () => this.locate());
    this.unsubscribe = location.subscribe(state => {
      this.status.textContent = state.message;
      this.status.dataset.state = state.status;
      this.locateButton.disabled = state.status === "locating";
      this.locateButton.textContent = state.status === "locating" ? "Locating…" : "Use my location";
      if (this.visible()) this.syncLocation();
    });
  }
  visible() {
    return !this.target.closest(".page").hidden;
  }
  activate() {
    if (!this.map) this.mount();
    if (!this.map) return;
    this.map.invalidateSize({ pan: false });
    this.syncLocation();
    if (!this.location.attempted) this.location.locate();
  }
  mount() {
    const L = globalThis.L;
    if (!L) {
      this.status.textContent = "Map library could not load. Reload this page to retry.";
      return;
    }
    this.node = el("div", null, "coverage-map");
    this.node.setAttribute("aria-label", "Interactive street map with real camera locations and saved incidents");
    this.target.append(this.node);
    this.map = L.map(this.node, {
      zoomControl: false, minZoom: 2, maxZoom: 19, worldCopyJump: true,
      scrollWheelZoom: true, attributionControl: true,
    });
    this.map.attributionControl.setPrefix('<a href="https://leafletjs.com">Leaflet</a>');
    this.tiles = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19, updateWhenIdle: true, keepBuffer: 1,
      // Override the app's no-referrer policy only for map tile requests.
      referrerPolicy: "strict-origin-when-cross-origin",
    });
    this.tileNotice = el("div", null, "map-tile-notice");
    this.tileNotice.setAttribute("role", "status");
    this.tileNotice.hidden = true;
    this.target.append(this.tileNotice);
    this.tiles.on("loading", () => { this.tileErrors = 0; });
    this.tiles.on("load", () => { if (!this.tileErrors) this.tileNotice.hidden = true; });
    const retry = el("button", "Retry map", "text-button");
    retry.addEventListener("click", () => this.tiles.redraw());
    const noticeText = el("span");
    this.tileNotice.replaceChildren(noticeText, retry);
    this.tiles.on("tileerror", () => {
      this.tileErrors++;
      noticeText.textContent = "Street tiles unavailable. Check your connection.";
      this.tileNotice.hidden = false;
    });
    this.zoneLayer = L.layerGroup().addTo(this.map);
    this.positionLayer = L.layerGroup().addTo(this.map);
    const center = this.location.state.center;
    // A world overview is a viewport, never a claimed camera/device location.
    this.map.setView(center || [0, 0], center ? 14 : 2);
    this.tiles.addTo(this.map);
    L.control.scale({ imperial: false, position: "bottomleft" }).addTo(this.map);
    this.resizeObserver = new ResizeObserver(() => {
      if (this.visible()) this.map.invalidateSize({ pan: false });
    });
    this.resizeObserver.observe(this.target);
  }
  syncLocation() {
    if (!this.map || this.appliedRevision === this.location.state.revision) return;
    this.appliedRevision = this.location.state.revision;
    this.renderMarkers();
    this.reset();
  }
  update({ cameras = this.cameras, incidents = this.incidents, heat = this.heat,
    scope = this.scope, selected = this.selected } = {}) {
    const changedScope = scope !== this.scope;
    const hadPoints = this.data.cameras.length + this.data.incidents.length;
    this.heat = heat;
    this.scope = scope;
    this.selected = selected;
    this.cameras = cameras;
    this.incidents = incidents;
    this.data = buildMapData({ cameras, incidents, scope });
    // Ignore changing FPS/sequence values so polling does not close open popups.
    const signature = JSON.stringify([heat, scope,
      this.data.cameras.map(c => [c.camera_id, c.name, c.status, c.stale, c.alert?.event_type, c.location]),
      this.data.incidents.map(e => [e.id, e.review, e.created, e.event_type, e.camera_name, e.signals.location]),
    ]);
    if (this.map && signature !== this.signature) {
      this.renderMarkers();
      if (changedScope || !hadPoints) this.reset();
    }
    this.signature = signature;
    this.highlight();
  }
  renderMarkers() {
    if (!this.map) return;
    const L = globalThis.L;
    this.zoneLayer.clearLayers();
    this.positionLayer.clearLayers();
    this.markerNodes.clear();
    const state = this.location.state;
    if (state.source === "device" && state.center) {
      if (state.accuracy !== null)
        L.circle(state.center, { radius: state.accuracy, color: "#60a5fa", weight: 1,
          fillColor: "#3b82f6", fillOpacity: 0.08, interactive: false }).addTo(this.positionLayer);
      L.circleMarker(state.center, { radius: 7, color: "white", weight: 3,
        fillColor: "#3b82f6", fillOpacity: 1 })
        .bindTooltip("Laptop location reported by your browser").addTo(this.positionLayer);
    }
    if (this.heat) {
      for (const hotspot of this.data.hotspots) {
        const popup = el("div", null, "street-popup");
        popup.append(el("h3", `${hotspot.count} saved ${hotspot.count === 1 ? "incident" : "incidents"}`),
          el("p", hotspot.incidents[0].signals.location.place),
          el("small", hotspot.incidents.some(event => event.signals.location_correction)
            ? "Location corrected by request · false positives excluded"
            : "Original detection locations · false positives excluded"));
        for (const event of hotspot.incidents.slice(0, 8))
          popup.append(el("p", `${EVENT_NAMES[event.event_type] || event.event_type} · ${event.camera_name} · ${formatTime(event.created)}`));
        if (hotspot.count > 8) popup.append(el("small", `${hotspot.count - 8} more in the Incident Library`));
        L.circleMarker(hotspot.latlng, {
          radius: Math.min(55, 18 + 8 * Math.log2(hotspot.count)),
          color: "#f0444f", weight: 1, fillColor: "#f0444f",
          fillOpacity: Math.min(.7, .2 + .08 * Math.log2(hotspot.count + 1)),
        }).bindTooltip(`${hotspot.count} saved ${hotspot.count === 1 ? "incident" : "incidents"}`)
          .bindPopup(popup).addTo(this.zoneLayer);
      }
    }
    for (const group of this.data.cameraGroups) {
      const active = group.cameras.some(c => c.status === "running" && !c.stale);
      const alert = group.cameras.some(c => c.status === "running" && !c.stale && c.alert);
      const tone = alert ? "red" : active ? "green" : "yellow";
      const pin = el("span", null, `street-pin ${tone}`);
      pin.append(icon(alert ? "alert" : "camera"));
      const title = group.cameras.map(c => c.name).join(" · ");
      const marker = L.marker(group.latlng, {
        icon: L.divIcon({ html: pin, className: "street-marker", iconSize: [34, 34], iconAnchor: [17, 17] }),
        title, alt: title, keyboard: true, riseOnHover: true,
      }).addTo(this.zoneLayer);
      marker.bindTooltip(el("span", group.cameras.length > 1 ? `${group.cameras.length} cameras at this location` : title),
        { direction: "bottom", offset: [0, 18], className: "street-label" });
      const popup = el("div", null, "street-popup");
      popup.append(el("span", "SAVED CAMERA LOCATION", "section-label"));
      for (const camera of group.cameras) {
        popup.append(el("h3", camera.name), el("p", camera.location.place),
          el("p", camera.stale ? "Feed is stale" : String(camera.status || "Unknown")));
        const select = el("button", `Select ${camera.name}`, "button secondary");
        select.type = "button";
        select.addEventListener("click", () => this.select(camera));
        popup.append(select);
        this.markerNodes.set(camera.camera_id, marker);
      }
      marker.bindPopup(popup, { maxWidth: 300, minWidth: 180, className: "street-popup-shell" });
      marker.on("click", () => { if (group.cameras.length === 1) this.select(group.cameras[0]); });
    }
    this.highlight();
  }
  select(camera) {
    this.selected = camera.camera_id;
    this.highlight();
    this.onSelect(camera);
  }
  highlight() {
    const selectedMarker = this.markerNodes.get(this.selected);
    for (const marker of new Set(this.markerNodes.values())) {
      const selected = marker === selectedMarker;
      marker.getElement()?.classList.toggle("selected", selected);
      marker.getElement()?.setAttribute("aria-pressed", String(selected));
      marker.setZIndexOffset(selected ? 400 : 0);
    }
  }
  changeZoom(delta) {
    if (this.map) this.map.setZoom(this.map.getZoom() + Math.sign(delta));
  }
  reset() {
    if (!this.map) return;
    const points = this.data.cameras.map(camera => camera.latlng);
    if (this.heat) points.push(...this.data.hotspots.map(hotspot => hotspot.latlng));
    if (this.location.state.center) points.push(this.location.state.center);
    if (points.length) this.map.fitBounds(points, { padding: [45, 55], maxZoom: 15, animate: false });
    else this.map.setView([0, 0], 2);
  }
  async locate() {
    await this.location.locate();
    if (this.location.state.center) this.reset();
  }
}
