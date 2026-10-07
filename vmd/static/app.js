import {
  $,
  el,
  icon,
  badge,
  hydrateIcons,
  formatTime,
  ApiClient,
  PollLoop,
  setPreviewImage,
  objectSummary,
  weaponAction,
} from "./ui.mjs";
import { IncidentAIUI, canAnalyze } from './incident-ai.mjs';
import { MediaPlayer, timecode } from './player.mjs';
import { EvidenceShare, canShareEvidence } from './evidence-share.mjs';
import { ResponseUI, newIncidentCamera, incidentHandledExpired, manualDispatchEligible } from './response.mjs';
import {
  requireSession,
  signOut,
  SECTORS,
  sectorName,
  readPreferences,
  savePreferences,
} from "./session.mjs";
import {
  buildDataset,
  demoCameras,
  EVENT_NAMES,
  scoped,
  dispatchTransition,
} from "./data.mjs";
import { CoverageMap } from "./map.mjs";
import { buildMapData } from "./map-data.mjs";
import { sharedLocation, isLiveCameraSource } from "./map-location.mjs";
import { renderOverview, renderAnalytics, renderHealth } from "./reports.mjs";
import { CameraSettingsEditor, readCameraSettings, validateCameraSettings, depthStatusText, fightStatusText, unattendedStatusText, liveFightTimer, renderFightTimer } from './camera-settings.mjs';
import { snatchingStatusText } from './snatching-status.mjs';
import { overlaysEnabled, displayFramesPath, renderOverlayButton } from './overlay-display.mjs';

const PAGES = {
  home: ["Home", "Citywide awareness and coordinated response"],
  operations: [
    "Live Operations",
    "Real-time video monitoring and incident response",
  ],
  map: ["Map View", "Saved cameras and this laptop’s browser location"],
  heatmap: ["Safety Heatmap", "Saved incidents at their original locations"],
  analytics: ["Analytics", "Performance metrics and incident trends"],
  health: ["System Health", "Monitoring of local and demonstration components"],
  incidents: ["Incident Library", "Review evidence and record your assessment"],
};

class Workspace {
  constructor(profile) {
    this.profile = profile;
    this.api = new ApiClient();
    this.dataset = buildDataset();
    const stored = readPreferences();
    this.preferences = {
      cameraSectors: {},
      reviews: {},
      dispatches: {},
      ...stored,
    };
    this.showDetectionOverlays = overlaysEnabled(this.preferences);
    for (const key of ["cameraSectors", "reviews", "dispatches"])
      if (
        !this.preferences[key] ||
        typeof this.preferences[key] !== "object" ||
        Array.isArray(this.preferences[key])
      )
        this.preferences[key] = {};
    this.cameras = [];
    this.cameraLimit = 4;
    this.incidents = [];
    this.seenIncidents = new Set();
    this.recordings = [];
    this.player = new MediaPlayer();
    this.analytics = null;
    this.analyticsError = null;
    this.analyticsLoading = true;
    this.analyticsPending = false;
    this.connected = false;
    this.cameraMode = "connected";
    this.selected = null;
    this.previewSelection = null;
    this.view = "grid";
    this.page = "operations";
    this.scope = profile.role === "operator" ? profile.sector : "all";
    this.selectedMapCamera = null;
    this.locationAttempts = new Set();
    this.pendingCameraPlaces = new Map();
    this.busy = false;
    this.dispatchKey = null;
    this.sampleCameras = demoCameras(this.dataset);
    this.cardNodes = new Map();
    hydrateIcons();
    this.cameraSettings = new CameraSettingsEditor(this);
    this.bind();
    this.renderOverlayControls();
    this.response = new ResponseUI(this);
    this.evidenceShare = new EvidenceShare(this);
    this.incidentAI = new IncidentAIUI(this);
    this.renderProfile();
    this.deploymentMap = new CoverageMap($("deployment-map"), {
      onSelect: (camera) => this.selectMapCamera(camera),
    });
    this.safetyMap = new CoverageMap($("safety-map"), {
      onSelect: (camera) => this.selectMapCamera(camera),
    });
    this.unsubscribeLocation = sharedLocation.subscribe(state => {
      $('source-location-status').textContent = state.message + (state.status === 'ready'
        ? '' : ' You can connect the camera now without a location.');
      $('source-location-retry').disabled = state.status === 'locating';
      if (state.status === 'ready') this.locateMissingCameras();
    });
    this.showPage(
      location.hash.slice(1) in PAGES ? location.hash.slice(1) : "operations",
    );
    this.loops = [
      new PollLoop(() => this.refreshCameras(), 900),
      new PollLoop(() => this.refreshFrames(), 1000/8, {fixedRate: true}),
      new PollLoop(() => this.refreshEvidence(), 3500),
      new PollLoop(() => this.response.refresh(), 500),
      new PollLoop(() => this.refreshAnalytics(), 5000),
    ];
    this.loops.forEach((loop) => loop.start());
    this.clock = setInterval(() => {
      $("clock").textContent = new Date().toLocaleTimeString("en-GB");
    }, 1000);
    window.addEventListener("pagehide", () => {
      this.loops.forEach((loop) => loop.stop());
      clearInterval(this.clock);
      this.response.close();
      this.unsubscribeLocation();
    });
    window.addEventListener("pageshow", (event) => {
      if (event.persisted) location.reload();
    });
    window.addEventListener("hashchange", () =>
      this.showPage(location.hash.slice(1), false),
    );
    let resizeTimer;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (this.page === "analytics") this.renderReport();
      }, 200);
    });
  }
  bind() {
    document
      .querySelectorAll("[data-page]")
      .forEach((button) =>
        button.addEventListener("click", () =>
          this.showPage(button.dataset.page),
        ),
      );
    $("sign-out").addEventListener("click", signOut);
    $("mobile-sign-out").addEventListener("click", signOut);
    document
      .querySelectorAll(".nav")
      .forEach((button) =>
        button.setAttribute(
          "aria-label",
          button.querySelector("span:nth-child(2)").textContent.trim(),
        ),
      );
    for (const option of [
      new Option("All sectors", "all"),
      ...SECTORS.map(
        (item) => new Option(`Sector ${item.id} · ${item.name}`, item.id),
      ),
    ])
      $("scope-filter").add(option);
    for (const item of SECTORS)
      $("source-sector").add(
        new Option(`Sector ${item.id} · ${item.name}`, item.id),
      );
    $("scope-filter").value = this.scope;
    $("source-sector").value = this.profile.sector;
    $("scope-filter").addEventListener("change", () => {
      this.scope = $("scope-filter").value;
      this.renderCurrentPage();
    });
    $("connected-mode").addEventListener("click", () =>
      this.setCameraMode("connected"),
    );
    $("network-mode").addEventListener("click", () =>
      this.setCameraMode("sample"),
    );
    $("grid-button").addEventListener("click", () => this.setView("grid"));
    $("feed-button").addEventListener("click", () => this.setView("feed"));
    for (const id of ['overlay-toggle', 'viewer-overlay-toggle'])
      $(id).addEventListener('click', () => this.toggleOverlays());
    $("close-camera-viewer").addEventListener("click", () =>
      $("camera-viewer").close(),
    );
    $("camera-viewer").addEventListener("close", () => {
      this.setView("grid");
      this.cardNodes.get(this.selected)?.focus({ preventScroll: true });
    });
    $("viewer-camera").addEventListener("change", () => {
      this.selected = $("viewer-camera").value;
      this.clearFrames();
      this.renderCameras();
    });
    $("source-button").addEventListener("click", () => this.openCameraForm());
    $("manage-cameras").addEventListener("click", () =>
      this.openCameraManager(),
    );
    $("close-camera-manager").addEventListener("click", () =>
      $("camera-manager").close(),
    );
    $("manager-add-camera").addEventListener("click", () => {
      $("camera-manager").close();
      this.openCameraForm();
    });
    $("close-dialog").addEventListener("click", () =>
      $("source-dialog").close(),
    );
    $('source-location-retry').addEventListener('click', () => sharedLocation.locate());
    $("source-form").addEventListener("submit", (event) =>
      this.addCamera(event),
    );
    $("demo-button").addEventListener("click", () => this.addPipelineTest());
    $("eco-button").addEventListener("click", () => this.toggleEco());
    $("reset-demo").addEventListener("click", () => this.resetDemo());
    for (const action of ["stop", "restart", "remove"])
      $(`${action}-button`).addEventListener("click", () =>
        this.cameraAction(action),
      );
    document
      .querySelectorAll("[data-scenario]")
      .forEach((button) =>
        button.addEventListener("click", () =>
          this.changeScenario(button.dataset.scenario),
        ),
      );
    $("dispatch-slider").addEventListener("change", () => this.dispatch());
    $("dispatch-next").addEventListener("click", () => this.advanceDispatch());
    $("false-positive").addEventListener("click", () => this.reviewSelected());
    $("inspect-play").addEventListener("click", () => this.playIncident(this.selectedEvent()));
    $("recording-upload").addEventListener("submit", event => this.uploadRecording(event));
    $('analytics-period').addEventListener('change', () => {
      this.analytics = null;
      this.analyticsError = null;
      this.analyticsLoading = true;
      this.renderReport();
      this.refreshAnalytics();
    });
    $('analytics-retry').addEventListener('click', () => this.refreshAnalytics());
    $("health-dataset").addEventListener("change", () =>
      renderHealth(this.cameras, this.connected, this.dataset),
    );
    for (const id of ["library-dataset", "incident-filter"])
      $(id).addEventListener("change", () => this.renderIncidents());
    for (const [prefix, key] of [
      ["map", "deploymentMap"],
      ["heat", "safetyMap"],
    ]) {
      $(`${prefix}-zoom-in`).addEventListener("click", () =>
        this[key].changeZoom(0.2),
      );
      $(`${prefix}-zoom-out`).addEventListener("click", () =>
        this[key].changeZoom(-0.2),
      );
      $(`${prefix}-reset`).addEventListener("click", () => this[key].reset());
    }
    $("map-open-camera").addEventListener("click", () => {
      if (!this.selectedMapCamera) return;
      this.setCameraMode("connected");
      this.selected = this.selectedMapCamera;
      this.showPage("operations");
    });
  }
  renderProfile() {
    $("profile-title").textContent = this.profile.name;
    $("profile-scope").textContent =
      this.profile.role === "admin"
        ? "Centre administrator"
        : `Operator · Sector ${this.profile.sector}`;
    $("welcome-title").textContent =
      this.profile.role === "admin"
        ? "Centre overview"
        : `${sectorName(this.profile.sector)} overview`;
  }
  showPage(name, updateHash = true) {
    if (!(name in PAGES)) name = "operations";
    if (name !== "operations" && $("camera-viewer").open)
      $("camera-viewer").close();
    const changed = this.page !== name;
    this.page = name;
    $('scope-filter').hidden = ['map', 'heatmap', 'analytics'].includes(name);
    document
      .querySelectorAll(".page")
      .forEach((node) => (node.hidden = node.id !== name));
    document.querySelectorAll(".nav").forEach((node) => {
      node.classList.toggle("active", node.dataset.page === name);
      if (node.dataset.page === name) node.setAttribute("aria-current", "page");
      else node.removeAttribute("aria-current");
    });
    [$("page-title").textContent, $("page-subtitle").textContent] = PAGES[name];
    document.title = `${PAGES[name][0]} · VMD Shield`;
    if (updateHash) history.replaceState(null, "", `#${name}`);
    this.renderCurrentPage();
    if (name === "analytics") this.refreshAnalytics();
    if (changed) window.scrollTo(0, 0);
  }
  renderCurrentPage() {
    if (this.page === "operations") this.renderCameras();
    else if (this.page === "home") renderOverview(this.dataset, this.scope);
    else if (this.page === "analytics") this.renderReport();
    else if (this.page === "health")
      renderHealth(this.cameras, this.connected, this.dataset);
    else if (this.page === "incidents") this.renderIncidents();
    else this.renderMaps();
  }
  toast(message) {
    $("toast").textContent = message;
    $("toast").hidden = false;
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => ($("toast").hidden = true), 4500);
  }
  persist() {
    if (!savePreferences(this.preferences))
      this.toast(
        "This browser could not save the demo state. It remains available until you reload.",
      );
  }
  async refreshCameras() {
    try {
      const data = await this.api.request("/cameras");
      this.cameras = data.cameras;
      if (Number.isInteger(data.limit) && data.limit > 0)
        this.cameraLimit = data.limit;
      this.recordingSlotAvailable = data.recording_slot_available ?? this.cameras.length < this.cameraLimit;
      this.connected = true;
      $("notice").textContent = data.settings_error || '';
      $("notice").hidden = !data.settings_error;
    } catch {
      this.connected = false;
      $("notice").textContent =
        "The local camera service is unavailable. Presentation screens remain available.";
      $("notice").hidden = false;
    }
    const running = this.cameras.filter(
      (camera) => camera.status === "running" && !camera.stale,
    ).length;
    $("sidebar-status").textContent = this.connected
      ? `${running} / ${this.cameras.length} RUNNING`
      : "UNAVAILABLE";
    if (this.page === "operations") this.renderCameras();
    if (this.page === "health")
      renderHealth(this.cameras, this.connected, this.dataset);
    this.renderCameraManager();
    if (this.page === 'map' || this.page === 'heatmap') this.renderMaps();
    this.locateMissingCameras();
  }
  async refreshEvidence() {
    try {
      const [incidents, recordings] = await Promise.all([
        this.api.request("/incidents"),
        this.api.request("/recordings"),
      ]);
      this.incidents = incidents;
      const detectedCamera = newIncidentCamera(incidents, this.cameras, this.seenIncidents, this.response.now());
      if (detectedCamera && this.cameraMode === 'connected') this.selected = detectedCamera;
      this.recordings = recordings;
      if (this.page === 'map' || this.page === 'heatmap') this.renderMaps();
      $("incident-count").textContent = incidents.filter(
        (event) => event.review === "unreviewed",
      ).length;
      if (this.page === "incidents") this.renderIncidents();
      if (this.page === "operations") this.renderCameras();
    } catch {
      /* Camera polling owns service-error messaging. Retain the last observed evidence. */
    }
  }
  displayCameras() {
    if (this.cameraMode === "sample")
      return scoped(this.sampleCameras, this.scope).map((camera) => ({
        ...camera,
        review: this.preferences.reviews[camera.event_id] || "unreviewed",
      }));
    return this.cameras.map((camera) => ({
      ...camera,
      sector: this.preferences.cameraSectors[camera.camera_id] || "4",
      ...(this.connected
        ? {}
        : {
            status: "error",
            stale: true,
            message: "Local service unavailable",
          }),
    }));
  }
  selectedState() {
    return this.displayCameras().find(
      (camera) => camera.camera_id === this.selected,
    );
  }
  setCameraMode(mode) {
    this.cameraMode = mode;
    this.selected = null;
    this.cardNodes.clear();
    $("camera-grid").replaceChildren();
    this.renderCameras();
    this.clearFrames();
  }
  setView(view) {
    if (view === "feed" && !this.selectedState()) {
      this.toast("Select or add a camera to open its analysis view.");
      return;
    }
    this.view = view;
    $("camera-grid").hidden = view !== "grid";
    $("selected-feeds").hidden = view !== "feed";
    for (const name of ["grid", "feed"]) {
      $(`${name}-button`).classList.toggle("selected", name === view);
      $(`${name}-button`).setAttribute("aria-pressed", String(name === view));
    }
    this.clearFrames();
    if (view === "feed") {
      this.renderViewer();
      if (!$("camera-viewer").open) $("camera-viewer").showModal();
    } else if ($("camera-viewer").open) $("camera-viewer").close();
  }
  renderCameras() {
    const cameras = this.displayCameras(),
      sample = this.cameraMode === "sample";
    if (!cameras.some((camera) => camera.camera_id === this.selected))
      this.selected = cameras[0]?.camera_id || null;
    if (this.previewSelection !== this.selected) {
      this.clearFrames();
      this.previewSelection = this.selected;
    }
    $("camera-count").textContent =
      `${this.cameras.length} / ${this.cameraLimit}`;
    $("connected-mode").classList.toggle("selected", !sample);
    $("network-mode").classList.toggle("selected", sample);
    $("reset-demo").hidden = !sample;
    $("network-note").hidden = !sample;
    $("source-button").disabled = this.busy;
    $("demo-button").disabled =
      this.busy || !this.connected || this.cameras.length >= this.cameraLimit;
    $("demo-button").title =
      this.cameras.length >= this.cameraLimit
        ? "All camera slots are occupied. Use Manage cameras to free a slot."
        : "";
    $("demo-button").hidden = sample;
    $("workspace-caption").textContent = sample
      ? "Fictional camera records · independent of live inputs"
      : `${this.cameras.length} / ${this.cameraLimit} camera slots used${this.cameras.length >= this.cameraLimit
        ? this.recordingSlotAvailable ? " · Analyse can reuse the completed recording slot" : " · stop and remove a session to free a slot" : ""}`;
    const keep = new Set(cameras.map((camera) => camera.camera_id));
    for (const [id, node] of this.cardNodes)
      if (!keep.has(id)) {
        node.remove();
        this.cardNodes.delete(id);
      }
    if (!cameras.length) {
      const empty = el("div", null, "empty-state");
      empty.append(
        icon("camera"),
        el(
          "p",
          sample
            ? "No presentation cameras assigned to this sector."
            : "Connect a phone, webcam or CCTV stream to start monitoring.",
        ),
      );
      $("camera-grid").replaceChildren(empty);
    } else {
      $("camera-grid")
        .querySelectorAll(".empty-state")
        .forEach((node) => node.remove());
      for (const camera of cameras) {
        let card = this.cardNodes.get(camera.camera_id);
        if (!card) {
          card = this.makeCameraCard(camera);
          this.cardNodes.set(camera.camera_id, card);
          $("camera-grid").append(card);
        }
        this.updateCameraCard(card, camera);
      }
    }
    this.renderInspector();
  }
  makeCameraCard(camera) {
    const card = el("button", null, "camera-tile");
    card.type = "button";
    card.dataset.camera = camera.camera_id;
    const stage = el("div", null, "tile-stage"),
      image = el("img"),
      empty = el("span", null, "tile-empty"),
      status = el("span", null, "tile-status"),
      warning = el("span", null, "tile-warning");
    image.alt = `Processed feed from ${camera.name}`;
    image.hidden = true;
    empty.append(icon("camera"));
    warning.append(icon("alert"));
    stage.append(image, empty, status, warning);
    const info = el("div", null, "tile-info"),
      heading = el("div", null, "tile-heading"),
      location = el("p"),
      locationText = el("span");
    heading.append(el("h3", camera.name), badge("LIVE"));
    location.append(icon("pin"), locationText);
    info.append(
      heading,
      location,
      el("strong", null, "tile-alert"),
      el("div", null, "fight-timer compact"),
      el("small", null, "tile-note"),
      el("small", null, "tile-objects"),
    );
    card.append(stage, info);
    card.title = 'Click to select. Double-click or press Enter to open the feed.';
    card.addEventListener("click", () => {
      this.selected = camera.camera_id;
      this.renderCameras();
    });
    card.addEventListener("dblclick", () => {
      this.setView("feed");
    });
    card.addEventListener('keydown', event => {
      if (event.key !== 'Enter') return;
      event.preventDefault();
      this.selected = camera.camera_id;
      this.renderCameras();
      this.setView('feed');
    });
    return card;
  }
  updateCameraCard(card, camera) {
    const weapon = weaponAction(camera);
    const alert =
      camera.status === "running" &&
      !camera.stale &&
      camera.review !== "false_positive"
        ? camera.alert
        : null;
    const timer = liveFightTimer(camera);
    const fight = !weapon && (!alert || ['possible_fight', 'fight'].includes(alert.event_type)) ? timer : null;
    const critical =
      fight ? fight.label === 'Fight detected' : alert &&
      ["fight", "snatching_detected", "weapon", "knife_detected", "gun_detected", "person_down_after_fight"].includes(alert.event_type);
    const evaluating =
      !alert && !weapon && !fight &&
      !camera.stale &&
      camera.status === "running" &&
      ["evaluating", "checking_interaction"].includes(camera.assessment);
    card.classList.toggle("selected", camera.camera_id === this.selected);
    card.setAttribute(
      "aria-pressed",
      String(camera.camera_id === this.selected),
    );
    card.classList.toggle("critical", !!critical);
    card.classList.toggle("warning", !critical && (!!alert || !!weapon || !!fight || evaluating));
    card.querySelector(".tile-warning").hidden = !alert && !weapon && !fight;
    card.querySelector("h3").textContent = camera.name;
    card.querySelector(".tile-status").textContent = camera.presentation
      ? "SAMPLE"
      : camera.stale
        ? "STALE"
        : camera.status === "running"
          ? camera.mode === "demo"
            ? "DEMO"
            : "LIVE"
          : camera.status.toUpperCase();
    const tag = card.querySelector(".badge");
    tag.textContent = fight?.label === 'Checking interaction' ? 'CHECKING' : alert || fight
      ? critical
        ? "CRITICAL"
        : "WARNING"
      : weapon ? "DETECTED" : evaluating
        ? "CHECKING"
        : camera.status === "running"
          ? camera.presentation
            ? "READY"
            : camera.mode === "demo"
              ? "DEMO"
              : "LIVE"
          : camera.status.toUpperCase();
    tag.className = `badge ${alert || fight ? (critical ? "red" : "orange") : weapon || evaluating ? "orange" : camera.status === "running" ? "green" : "neutral"}`;
    card.querySelector(".tile-info p span").textContent = sectorName(
      camera.sector,
    );
    const recording = camera.recording;
    card.querySelector(".tile-alert").textContent = recording && camera.status === 'finished'
      ? `Analysis complete${alert ? ` · ${alert.label}` : ''}`
      : fight ? fight.label : alert
      ? `${alert.label}${camera.presentation ? ` (${Math.round(camera.score * 100)} / 100)` : ""}`
      : weapon || (camera.stale
        ? "Waiting for fresh frames"
        : camera.status === "error"
          ? camera.message
          : camera.status === "starting"
            ? camera.message || "Loading analysis pipeline"
          : camera.status === "running"
            ? evaluating
              ? "Checking repeated interaction"
              : camera.assessment === "camera_moving"
                ? "Camera moving · fight checks paused"
                : camera.eco_state === "quiet"
                  ? "Eco quiet · periodic analysis"
                  : camera.eco_mode ? "Eco active · full analysis" : "Monitoring"
            : "Camera stopped");
    renderFightTimer(card.querySelector('.fight-timer'), camera);
    card.querySelector(".tile-note").textContent = camera.presentation
      ? "Presentation scenario"
      : `${camera.people || 0} people · ${camera.signals?.people ?? camera.people ?? 0} tracked · ${camera.fps || 0} fps${camera.sampling_limited ? " · limited sampling" : ""}`;
    if (recording) {
      const percentage = Number.isFinite(recording.progress)
        ? `${Math.round(Math.min(1, Math.max(0, recording.progress)) * 100)}% · ` : '';
      const position = camera.status === 'finished' ? recording.duration_seconds : recording.position_seconds;
      const duration = Number.isFinite(recording.duration_seconds) ? timecode(recording.duration_seconds) : 'unknown duration';
      card.querySelector('.tile-note').textContent = `${percentage}${timecode(position)} / ${duration} · ${camera.people || 0} people`;
    }
    if (camera.auto_reconnect && (camera.status === 'error' || camera.stale))
      card.querySelector('.tile-note').textContent = `Auto reconnect in ${camera.retry_in_seconds || 0}s · saved settings`;
    card.querySelector(".tile-objects").textContent = camera.presentation ? "" : objectSummary(camera);
  }
  selectedEvent() {
    const camera = this.selectedState();
    if (!camera) return null;
    if (camera.presentation)
      return camera.alert
        ? {
            id: camera.event_id,
            event_type: camera.alert.event_type,
            created: camera.last_frame_at,
            camera_id: camera.camera_id,
            camera_name: camera.name,
            score: camera.score,
            sector: camera.sector,
            mode: "presentation",
            review: this.preferences.reviews[camera.event_id] || "unreviewed",
            reasons: camera.alert.reasons,
          }
        : null;
    // A live weapon observation is not the camera's older saved incident.
    if (!camera.alert && weaponAction(camera)) return null;
    const latest =
      this.incidents.find((event) => event.camera_id === camera.camera_id) ||
      null;
    // Retire the handled selection, not its evidence or the detector's current episode.
    const response = this.response.alerts.find((alert) => alert.id === latest?.id);
    if (latest?.signals?.live_camera !== false && incidentHandledExpired(latest, response, this.response.now())) return null;
    return camera.status === 'running' && camera.alert && latest?.event_type !== camera.alert.event_type
      ? null
      : latest;
  }
  renderViewer() {
    const cameras = this.displayCameras(),
      camera = this.selectedState(),
      weapon = weaponAction(camera),
      timer = liveFightTimer(camera),
      fight = !weapon && (!camera?.alert || ['possible_fight', 'fight'].includes(camera.alert.event_type)) ? timer : null;
    const signature = JSON.stringify(
      cameras.map((item) => [item.camera_id, item.name]),
    );
    if (signature !== this.viewerCameraSignature) {
      this.viewerCameraSignature = signature;
      $("viewer-camera").replaceChildren(
        ...cameras.map((item) => new Option(item.name, item.camera_id)),
      );
    }
    $("viewer-camera").value = this.selected || "";
    $("viewer-title").textContent = camera?.name || "Camera unavailable";
    $("viewer-assessment").textContent =
      camera?.status === "error"
        ? camera.auto_reconnect ? "Camera reconnecting · saved settings" : "Camera disconnected"
        : fight?.label || camera?.alert?.label || weapon ||
          (camera?.presentation
            ? "Presentation camera"
            : ["evaluating", "checking_interaction"].includes(camera?.assessment)
              ? "Checking interaction"
              : "No active alert");
    $("viewer-reason").textContent = !camera
      ? "This camera was removed. Choose another camera or return to the grid."
      : camera.presentation
        ? "Sample camera cards have no real video or depth stream. Choose a connected input for analysis."
        : camera.status !== "running"
          ? camera.message
          : weapon && !camera.alert
            ? "Detected in the latest sampled frame. Review the highlighted object below."
          : fight ? fight.reason
          : camera.detection_warning
            ? camera.detection_warning
          : camera.signals?.blockers?.length
            ? `${camera.alert ? "Recent alert remains visible. Current frame: " : ""}${camera.signals.blockers.join(" · ")}`
            : camera.people < 2
              ? "Fight detection needs two visible tracked people. One-person punches are not a fight signal."
              : camera.sampling_limited
                ? "Processing is too slow to maintain the current motion history."
                : camera.reasons?.join(" · ") ||
                  "Watching pose and image motion for sustained interaction.";
    renderFightTimer($('viewer-fight-timer'), camera);
  }
  renderInspector() {
    const camera = this.selectedState(),
      event = this.selectedEvent(),
      weapon = weaponAction(camera),
      running = camera?.status === "running" && !camera.stale;
    this.incidentAI?.select(event);
    this.renderViewer();
    $("inspect-objects").textContent = camera?.presentation ? "Sample camera" : objectSummary(camera);
    const alert = running ? camera.alert : null,
      reviewed = event?.review === "false_positive",
      timer = liveFightTimer(camera),
      fight = !weapon && (!alert || ['possible_fight', 'fight'].includes(alert.event_type)) ? timer : null;
    renderFightTimer($('inspect-fight-timer'), camera);
    $("inspector-camera").textContent = camera?.name || "None";
    $("inspect-name").textContent = camera?.name || "No camera selected";
    $("inspect-location").textContent = camera
      ? sectorName(camera.sector)
      : "Select a camera to begin";
    $("inspect-sector").textContent = camera ? sectorName(camera.sector) : "—";
    $("inspect-alert").hidden = !alert && !event && !weapon && !fight;
    $("inspect-time").textContent = event
      ? formatTime(event.created)
      : camera?.last_frame_at
        ? formatTime(camera.last_frame_at)
        : "—";
    $('assessment-context').textContent = fight ? 'Current detection' : 'Detected event';
    $("assessment-title").textContent = fight ? fight.label : reviewed
      ? "Marked false positive"
      : alert
        ? EVENT_NAMES[alert.event_type] || alert.label
        : event
          ? EVENT_NAMES[event.event_type] || event.event_type
          : camera?.status === "error"
            ? "Camera unavailable"
            : camera
              ? "No active incident"
              : "Awaiting input";
    $("assessment-score").textContent =
      fight ? `Current heuristic score: ${Math.round((camera.score ?? 0) * 100)} / 100`
      : event || alert
        ? `${camera?.presentation ? "Sample score" : event?.signals?.object_detection || ['knife_detected', 'gun_detected'].includes(alert?.event_type) ? "Model score" : "Heuristic score"}: ${Math.round((event?.score ?? alert?.score ?? camera?.score ?? 0) * 100)} / 100${event ? " · " + event.review.replaceAll("_", " ") : ""}`
        : running && camera?.signals?.threshold != null
          ? `Current score: ${Math.round(camera.score * 100)} / 100 · threshold ${Math.round(camera.signals.threshold * 100)}`
          : "No incident selected";
    $("inspect-action").textContent = weapon || fight?.label || (reviewed
      ? "Dismissed after review"
      : camera?.action ||
        (alert || event
          ? camera?.signals?.pattern || "Review the evidence"
          : "Monitoring"));
    $("assessment-text").textContent = camera?.presentation
      ? "Presentation scenario. Scores and incidents are fictional."
      : camera?.status === "error"
        ? camera.message
        : fight ? fight.reason : event
          ? "A saved detection is selected. Review its clip before drawing conclusions."
          : running && camera?.signals?.blockers?.length
            ? camera.signals.blockers.join(" · ")
            : "Head blur is best effort. Signals are not a determination of intent or injury.";
    if (weapon && !alert) {
      $("assessment-title").textContent = weapon;
      $("assessment-score").textContent = objectSummary(camera);
      $("assessment-text").textContent = "Detected in the latest sampled frame. Review the highlighted object in the camera view.";
    }
    for (const [id, value] of Object.entries({
      people: camera?.people,
      fps: camera?.fps,
      score: camera ? Math.round(camera.score * 100) : null,
      latency: camera?.latency_ms == null ? null : `${camera.latency_ms}ms`,
    }))
      $(id).textContent = value ?? "—";
    const signals = camera?.signals || {};
    $("signal-details").replaceChildren(
      ...[
        [
          "Wrist / limb speed",
          signals.wrist_speed == null
            ? "—"
            : `${signals.wrist_speed} body scales/s`,
        ],
        ["Image proximity", signals.proximity ?? "—"],
        ["Local motion", signals.pair_flow ?? signals.local_flow ?? "—"],
        ["Striking-limb image motion", signals.contact_motion ?? "—"],
        ["Fight threshold", signals.threshold ?? camera?.settings?.threshold ?? "—"],
        ["Automatic recovery", camera?.auto_reconnect ? `Enabled · ${camera.reconnect_attempts || 0} retries` : 'Stopped / manual playback'],
        ["Saved compute device", camera?.settings?.device || '—'],
        ["Saved depth model", camera?.settings?.depth || '—'],
        ["Saved grappling review window", camera?.settings?.hold_seconds == null ? '—' : `${camera.settings.hold_seconds}s`],
        ["Sustained interaction observation", signals.review_seconds == null ? '—' : `${signals.review_seconds} / ${signals.possible_required_seconds ?? camera?.settings?.hold_seconds ?? '—'}s`],
        ["Supported contact motion", signals.motion_supported_seconds == null ? '—' : `${signals.motion_supported_seconds}s`],
        [
          "Fight escalation timer",
          signals.fight_timer_seconds == null ? '—' : `${signals.fight_timer_seconds} / ${signals.fight_timer_required_seconds ?? '—'}s`,
        ],
        ["Supported strike bursts", signals.strike_bursts ?? 0],
        ["Distinct strike bouts", signals.strike_bouts ?? 0],
        ["Detection mode", camera?.detection_mode === "depth_confirmed" ? "Depth-confirmed" : "Review only"],
        ["Matching depth span", signals.confirmation_seconds == null ? "—" : `${signals.confirmation_seconds}s`],
        ["Matching depth samples", signals.depth_samples ?? "—"],
        ["Depth worker", camera?.depth_meta?.status || "off"],
        ["Eco mode", camera?.eco_mode ? camera.eco_state : "off"],
        ["Pose checks skipped by eco", camera?.eco_skipped_frames ?? 0],
        ["Pose frames processed", camera?.processed_frames ?? 0],
        ["Object detector", camera?.object_meta?.status || "off"],
        ["Weapon detector", camera?.object_meta?.knife_error || camera?.object_meta?.knife_status || "off"],
        ["Knife / gun", objectSummary(camera)],
        ["Unattended items", unattendedStatusText(camera)],
        ["Snatching checks", snatchingStatusText(camera)],
        ["Person-down dwell", `${camera?.settings?.person_down_seconds ?? 3}s`],
      ].flatMap(([key, value]) => [el("dt", key), el("dd", value)]),
    );
    $("reasons").replaceChildren(
      ...(event?.reasons || alert?.reasons || camera?.reasons || []).map(
        (reason) => el("li", reason),
      ),
    );
    $("buffer-value").textContent =
      `Pre-event buffer: ${camera?.buffer_seconds || 0} / 10s`;
    $("stop-button").disabled =
      this.busy ||
      !camera ||
      camera.presentation ||
      !["running", "starting", "stopping"].includes(camera.status);
    $("restart-button").disabled =
      this.busy ||
      !camera ||
      camera.presentation ||
      !["idle", "error", "finished"].includes(camera.status);
    $("remove-button").disabled = this.busy || !camera || camera.presentation;
    $("selected-camera-controls").hidden = !camera || camera.presentation;
    $('edit-camera-settings').disabled = this.busy || !camera?.settings || camera.presentation || camera.mode === 'demo';
    $('camera-depth-status').textContent = depthStatusText(camera);
    $('camera-fight-status').textContent = fightStatusText(camera);
    $('camera-unattended-status').textContent = unattendedStatusText(camera);
    $('camera-snatching-status').textContent = snatchingStatusText(camera);
    $("eco-button").disabled = this.busy || !camera || camera.presentation || camera.mode !== "live";
    $("eco-button").textContent = camera?.eco_mode ? "Disable eco mode" : "Enable eco mode";
    $("eco-button").setAttribute("aria-pressed", String(!!camera?.eco_mode));
    $("demo-controls").hidden = !(camera?.mode === "demo" && running);
    document
      .querySelectorAll("[data-scenario]")
      .forEach((button) =>
        button.classList.toggle(
          "selected",
          camera?.scenario === button.dataset.scenario,
        ),
      );
    $("feed-source").textContent = camera?.name || "No camera selected";
    $("feed-status").textContent = camera?.presentation
      ? "SAMPLE"
      : camera?.stale
        ? "STALE"
        : camera?.mode === "demo" && running
          ? "SYNTHETIC TEST"
          : camera?.status?.toUpperCase() || "IDLE";
    $("frame-number").textContent = camera?.presentation
      ? "Presentation camera"
      : camera?.sequence
        ? `FRAME ${camera.sequence}`
        : "Awaiting input";
    this.renderDispatch(event);
  }
  renderDispatch(event) {
    if (this.response?.renderDispatch(event)) return;
    const key = event?.id || null,
      state = this.preferences.dispatches[key]?.state || "unassigned";
    if (this.dispatchKey !== key) {
      $("dispatch-slider").value = 0;
      this.dispatchKey = key;
    }
    $("dispatch-slider").disabled =
      !event || event.review === "false_positive" || state !== "unassigned";
    if (state !== "unassigned") $("dispatch-slider").value = 100;
    $("dispatch-label").textContent =
      {
        unassigned: "SLIDE TO DISPATCH",
        queued: "DISPATCH QUEUED",
        on_scene: "UNIT ON SCENE",
        resolved: "INCIDENT RESOLVED",
      }[state] || "SLIDE TO DISPATCH";
    $("dispatch-next").hidden = !["queued", "on_scene"].includes(state);
    $("dispatch-next").disabled = event?.review === "false_positive";
    $("dispatch-next").textContent =
      state === "queued" ? "Mark unit on scene" : "Resolve demo dispatch";
    $("dispatch-state").textContent =
      state === "unassigned"
        ? "Demo dispatch · no external messages"
        : `${this.preferences.dispatches[key]?.unit || "Response unit"} · ${state.replaceAll("_", " ")} · simulated`;
    $("false-positive").disabled =
      !event || event.review === "false_positive" || this.busy;
    $("inspect-play").hidden = !event?.clip;
  }
  dispatch() {
    const slider = $("dispatch-slider"),
      event = this.selectedEvent();
    if (slider.disabled) return;
    if (
      Number(slider.value) < 95 ||
      !event ||
      event.review === "false_positive"
    ) {
      slider.value = 0;
      return;
    }
    if (event.mode !== 'presentation' && event.mode !== 'demo') {
      this.response.dispatch(event);
      return;
    }
    const current =
      this.preferences.dispatches[event.id]?.state || "unassigned";
    if (current !== "unassigned") return;
    this.preferences.dispatches[event.id] = {
      state: dispatchTransition(current, "dispatch"),
      unit: `DELTA-${String(event.sector || this.selectedState()?.sector || "4").padStart(2, "0")}`,
      updated: Date.now(),
    };
    this.persist();
    this.renderDispatch(event);
    this.toast("Demo dispatch queued. No external notification was sent.");
  }
  advanceDispatch() {
    const event = this.selectedEvent();
    if (!event) return;
    const current = this.preferences.dispatches[event.id];
    if (!current) return;
    current.state = dispatchTransition(
      current.state,
      current.state === "queued" ? "arrive" : "resolve",
    );
    current.updated = Date.now();
    this.persist();
    this.renderDispatch(event);
  }
  resetDemo() {
    for (const key of ["reviews", "dispatches"])
      this.preferences[key] = Object.fromEntries(
        Object.entries(this.preferences[key]).filter(
          ([id]) =>
            !id.startsWith("sample-") && !id.startsWith("presentation-"),
        ),
      );
    this.dispatchKey = null;
    this.persist();
    this.renderCameras();
    this.toast(
      "Presentation reviews and dispatches reset. Local evidence is unchanged.",
    );
  }
  async reviewSelected() {
    const event = this.selectedEvent();
    if (event) await this.review(event, "false_positive");
  }
  async review(event, decision) {
    try {
      this.busy = true;
      if (event.mode === "presentation") {
        this.preferences.reviews[event.id] = decision;
        this.persist();
      } else {
        await this.api.request(
          `/incidents/${encodeURIComponent(event.id)}/review`,
          { decision },
        );
        await this.refreshEvidence();
      }
      this.toast(
        event.mode === "presentation"
          ? "Demo review updated."
          : "Incident review saved.",
      );
    } catch (error) {
      this.toast(error.message);
    } finally {
      this.busy = false;
      this.renderInspector();
      if (this.page === "incidents") this.renderIncidents();
      if (this.page === "operations") this.renderCameras();
    }
  }
  clearFrames() {
    for (const view of ["pose", "depth", "object"]) {
      $(`${view}-image`).hidden = true;
      $(`${view}-empty`).hidden = false;
    }
    $("inspect-image").hidden = true;
    $("inspect-placeholder").hidden = false;
    $("object-list").textContent = "";
    $("object-label").textContent = "WAITING";
  }
  setImage(image, empty, jpeg) {
    setPreviewImage(image, empty, jpeg);
  }
  renderOverlayControls() {
    for (const id of ['overlay-toggle', 'viewer-overlay-toggle'])
      renderOverlayButton($(id), this.showDetectionOverlays, icon);
    $('pose-view-title').textContent = this.showDetectionOverlays ? 'Video & pose' : 'Video';
    $('pose-image').alt = this.showDetectionOverlays
      ? 'Selected camera with pose and incident overlays' : 'Selected camera without pose lines or bounding boxes';
  }
  toggleOverlays() {
    this.showDetectionOverlays = !this.showDetectionOverlays;
    this.preferences.showDetectionOverlays = this.showDetectionOverlays;
    if (!savePreferences(this.preferences))
      this.toast('This browser could not save the display option. It still works for this session.');
    this.renderOverlayControls();
    this.clearFrames();
    for (const card of this.cardNodes.values())
      this.setImage(card.querySelector('img'), card.querySelector('.tile-empty'), null);
  }
  async refreshFrames() {
    if (document.hidden || this.page !== "operations") return;
    const selected = this.selected,
      mode = this.cameraMode,
      view = this.view,
      overlays = this.showDetectionOverlays;
    if (mode === "sample") {
      this.clearFrames();
      $("pose-empty").querySelector("p").textContent =
        "Presentation camera · no live video";
      $("depth-label").textContent = "Presentation";
      $("depth-message").textContent =
        "Sample cameras have no live image or depth stream.";
      return;
    }
    try {
      $("pose-empty").querySelector("p").textContent = "No image available";
      let bundle;
      if (view === "grid") {
        const previews = await this.api.request(displayFramesPath('/camera-frames', overlays));
        if (mode !== this.cameraMode || view !== this.view || overlays !== this.showDetectionOverlays) return;
        for (const [id, card] of this.cardNodes)
          this.setImage(
            card.querySelector("img"),
            card.querySelector(".tile-empty"),
            previews[id]?.pose,
          );
        bundle = previews[selected];
      } else if (selected)
        bundle = await this.api.request(
          displayFramesPath(`/cameras/${encodeURIComponent(selected)}/frames`, overlays),
        );
      if (
        selected !== this.selected ||
        mode !== this.cameraMode ||
        view !== this.view ||
        overlays !== this.showDetectionOverlays
      )
        return;
      this.setImage($("inspect-image"), $("inspect-placeholder"), bundle?.pose);
      if (view === "feed") {
        this.setImage($("pose-image"), $("pose-empty"), bundle?.pose);
        this.setImage($("depth-image"), $("depth-empty"), bundle?.depth);
        this.setImage($("object-image"), $("object-empty"), bundle?.objects);
        const objects = bundle?.object_meta || {};
        $("object-label").textContent = objects.status === "error" ? "UNAVAILABLE" : objects.stale ? "STALE · HIDDEN"
          : objects.sequence != null ? `FRAME ${objects.sequence} · ${objects.age_seconds}s old`
            : (objects.status || "off").toUpperCase();
        $("object-message").textContent = objects.error || objects.knife_error || (objects.status === "ready"
          ? `Independent object sample · up to ${objects.target_fps} updates/second · weapon detector ${objects.knife_status || "off"}`
          : objects.status === "loading" ? "Loading object checks in the background. Pose continues independently."
            : "Enable weapon or unattended-item checks in camera settings.");
        $("object-list").textContent = [objectSummary({object_meta: objects, scene_objects: bundle?.scene_objects}),
          unattendedStatusText(this.selectedState())].join(' · ');
        const meta = bundle?.depth_meta || {};
        const camera = this.selectedState();
        if (camera?.status === "error" || camera?.status === "idle") {
          $("depth-label").textContent = "WAITING FOR CAMERA";
          $("depth-message").textContent =
            "Reconnect this camera to resume its depth analysis.";
          return;
        }
        $("depth-label").textContent = meta.stale
          ? "STALE · HIDDEN"
          : meta.sequence != null
            ? `FRAME ${meta.sequence} · ${meta.age_seconds}s old`
            : meta.status === "loading"
              ? "LOADING IN BACKGROUND"
              : meta.status === "error"
                ? "UNAVAILABLE"
                : "NOT ENABLED";
        $("depth-message").textContent =
          meta.error ||
          (meta.stale
            ? "Old depth frames are hidden while pose continues."
            : meta.status === "loading"
              ? "The depth model is loading in the background. Video and pose continue independently."
              : meta.status === "ready"
                ? meta.fight_confirmation_support
                  ? "Depth supports only the tracked pair from its original frame. Relative depth does not measure metres."
                  : "Fight confirmation is disabled in Review-only mode. Snatching escalation still requires matched grab depth."
                : "Enable sampled depth when connecting a camera. Relative depth does not measure metres.");
      }
    } catch {
      /* Keep the last frame; source/service status is shown independently. */
    }
  }
  async openCameraForm(recording = null) {
    if (this.busy) return;
    // Completed file sessions can release a slot between normal camera polls.
    if (recording) await this.refreshCameras();
    if (!this.connected) {
      this.toast(
        "The camera service is unavailable. Wait for it to reconnect, then try again.",
      );
      return;
    }
    if (this.cameras.length >= this.cameraLimit && !(recording && this.recordingSlotAvailable)) {
      this.openCameraManager();
      return;
    }
    $("form-error").textContent = "";
    $('source-form').reset();
    validateCameraSettings($('source-form'));
    $('source-sector').value = this.profile.sector;
    this.recordingForAnalysis = recording;
    const source = $('source-form').elements.source;
    source.disabled = !!recording;
    source.placeholder = recording ? `Uploaded video: ${recording.name}` : 'http://PHONE_IP:8080/video';
    $('source-dialog-title').textContent = recording ? 'Analyse a recording' : 'Connect a camera';
    if (recording) $('source-form').elements.name.value = recording.name.slice(0, 64);
    $('source-location-panel').hidden = !!recording;
    $("source-dialog").showModal();
    if (!recording) sharedLocation.locate();
  }
  openCameraManager() {
    $("camera-manager-error").textContent = "";
    $("camera-manager-status").textContent = "";
    $("camera-manager").showModal();
    this.managerSignature = null;
    this.renderCameraManager();
  }
  renderCameraManager() {
    if (!$("camera-manager").open) return;
    const full = this.cameras.length >= this.cameraLimit;
    $("camera-capacity").textContent = !this.connected
      ? "Camera service unavailable. Reconnect the service to manage inputs."
      : full
        ? `All ${this.cameraLimit} slots are occupied. Remove a camera below to add another.`
        : `${this.cameras.length} of ${this.cameraLimit} slots used · ${this.cameraLimit - this.cameras.length} available`;
    $("manager-add-camera").disabled = this.busy || !this.connected || full;
    const signature = JSON.stringify([
      this.connected,
      this.busy,
      this.cameras.map(({ camera_id, name, status, mode }) => [
        camera_id,
        name,
        status,
        mode,
      ]),
    ]);
    if (signature === this.managerSignature) return;
    this.managerSignature = signature;
    const rows = this.cameras.map((camera) => {
      const row = el("article", null, "managed-camera");
      row.setAttribute("aria-label", camera.name);
      const details = el("div", null, "managed-camera-details");
      details.append(
        el("strong", camera.name),
        el(
          "span",
          `${camera.status.toUpperCase()} · ${camera.mode === "demo" ? "Pipeline test" : "Camera input"}`,
        ),
      );
      const remove = el("button", "Remove", "button outline remove-camera");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove ${camera.name}`);
      remove.disabled = this.busy || !this.connected;
      remove.addEventListener("click", () =>
        this.cameraAction("remove", camera.camera_id),
      );
      row.append(icon("camera"), details, remove);
      return row;
    });
    $("managed-cameras").replaceChildren(
      ...(rows.length
        ? rows
        : [
            el(
              "p",
              "No cameras configured. Add an input to start.",
              "field-help",
            ),
          ]),
    );
  }
  async addCamera(event) {
    event.preventDefault();
    if (this.busy) return;
    const button = $("source-submit"),
      values = Object.fromEntries(new FormData(event.target)),
      sector = values.sector;
    delete values.sector;
    validateCameraSettings(event.target);
    if (!event.target.reportValidity()) return;
    Object.assign(values, readCameraSettings(event.target));
    values.mode = "live";
    const place = values.location_place?.trim();
    delete values.location_place;
    if (this.recordingForAnalysis) values.recording_id = this.recordingForAnalysis.id;
    this.busy = true;
    button.disabled = true;
    $("form-error").textContent = "";
    try {
      if (!this.recordingForAnalysis && isLiveCameraSource(values.source))
        values.location = sharedLocation.currentCameraLocation(place);
      if (!$('source-dialog').open) return;
      const camera = await this.api.request("/cameras", values);
      if (place && values.location === null) this.pendingCameraPlaces.set(camera.camera_id, place);
      this.preferences.cameraSectors[camera.camera_id] = sector;
      this.persist();
      this.cameraMode = "connected";
      this.selected = camera.camera_id;
      await this.refreshCameras();
      $("source-dialog").close();
      if (this.recordingForAnalysis) {
        this.showPage('operations');
        this.toast('Recording analysis started. The selected feed shows its progress and detections.');
      } else {
        this.toast(values.location === null
          ? 'Camera pipeline starting. Location is unavailable; retry from Camera location when ready.'
          : 'Camera pipeline starting.');
      }
    } catch (error) {
      $("form-error").textContent = error.message;
    } finally {
      this.busy = false;
      button.disabled = false;
      this.renderCameras();
    }
  }
  async toggleEco() {
    const camera = this.selectedState();
    if (this.busy || !camera || camera.presentation || camera.mode !== "live") return;
    this.busy = true;
    this.renderInspector();
    try {
      await this.api.request(`/cameras/${encodeURIComponent(camera.camera_id)}/eco`, { enabled: !camera.eco_mode });
      await this.refreshCameras();
      this.toast(camera.eco_mode ? "Continuous analysis enabled." : "Eco mode enabled. Motion restores full analysis.");
    } catch (error) {
      this.toast(error.message);
    } finally {
      this.busy = false;
      this.renderCameras();
    }
  }
  async addPipelineTest() {
    if (this.busy) return;
    this.busy = true;
    try {
      const camera = await this.api.request("/cameras", {
        mode: "demo",
        name: `Pipeline test ${this.cameras.length + 1}`,
      });
      this.cameraMode = "connected";
      this.selected = camera.camera_id;
      await this.refreshCameras();
    } catch (error) {
      this.toast(error.message);
    } finally {
      this.busy = false;
      this.renderCameras();
    }
  }
  async cameraAction(action, cameraId = this.selected) {
    const camera = this.cameras.find((item) => item.camera_id === cameraId);
    if (!camera || this.busy) return;
    this.busy = true;
    $("camera-manager-error").textContent = "";
    $("camera-manager-status").textContent = "";
    this.renderInspector();
    this.renderCameraManager();
    try {
      await this.api.request(
        `/cameras/${encodeURIComponent(camera.camera_id)}/${action}`,
        {},
      );
      if (action === "remove" && this.selected === camera.camera_id)
        this.selected = null;
      await this.refreshCameras();
      if (action === "remove") {
        const message = `${camera.name} removed. Its saved incidents and clips are kept.`;
        $("camera-manager-status").textContent = message;
        if (!$("camera-manager").open) this.toast(message);
      }
    } catch (error) {
      $("camera-manager-error").textContent = error.message;
      this.toast(error.message);
    } finally {
      this.busy = false;
      this.renderCameras();
      this.renderCameraManager();
      if (action === "remove" && $("camera-manager").open)
        $("camera-manager")
          .querySelector(
            ".remove-camera:not(:disabled), #manager-add-camera:not(:disabled)",
          )
          ?.focus();
    }
  }
  async changeScenario(scenario) {
    const camera = this.selectedState();
    if (camera?.mode !== "demo") return;
    try {
      await this.api.request(
        `/cameras/${encodeURIComponent(camera.camera_id)}/scenario`,
        { scenario },
      );
    } catch (error) {
      this.toast(error.message);
    }
  }
  async refreshAnalytics() {
    if (this.page !== 'analytics' || this.analyticsPending) return;
    const days = Number($('analytics-period').value);
    this.analyticsPending = true;
    this.analyticsLoading = true;
    this.renderReport();
    try {
      const summary = await this.api.request(`/analytics/summary?days=${days}`);
      if (Number($('analytics-period').value) === days) {
        this.analytics = summary;
        this.analyticsError = null;
      }
    } catch (error) {
      if (Number($('analytics-period').value) === days) this.analyticsError = error.message;
    } finally {
      this.analyticsPending = false;
      this.analyticsLoading = false;
      if (this.page === 'analytics') {
        this.renderReport();
        if (Number($('analytics-period').value) !== days) this.refreshAnalytics();
      }
    }
  }
  renderReport() {
    const days = Number($('analytics-period').value);
    renderAnalytics({summary: this.analytics?.days === days ? this.analytics : null,
      days, loading: this.analyticsLoading, error: this.analyticsError});
    $('analytics-retry').hidden = !this.analyticsError;
    $('analytics-retry').disabled = this.analyticsPending;
  }
  async locateMissingCameras() {
    if (!this.connected || sharedLocation.state.status !== 'ready' ||
        Date.now() - sharedLocation.state.capturedAt > 60000) return;
    for (const camera of this.cameras) {
      if (!camera.live_camera || camera.location || this.locationAttempts.has(camera.camera_id)) continue;
      this.locationAttempts.add(camera.camera_id);
      try {
        const location = await sharedLocation.cameraLocation(this.pendingCameraPlaces.get(camera.camera_id));
        await this.api.request(`/cameras/${encodeURIComponent(camera.camera_id)}/location`, location);
        camera.location = location;
        this.pendingCameraPlaces.delete(camera.camera_id);
      } catch (error) {
        this.toast(`Could not save laptop location for ${camera.name}: ${error.message}`);
      }
    }
    if (this.page === 'map' || this.page === 'heatmap') this.renderMaps();
  }
  mapInputs() {
    return {cameras: this.cameras, incidents: this.incidents, scope: 'all'};
  }
  renderMaps() {
    const inputs = this.mapInputs(), data = buildMapData(inputs);
    const camera = data.cameras.find(item => item.camera_id === this.selectedMapCamera) || data.cameras[0];
    this.selectedMapCamera = camera?.camera_id || null;
    this.deploymentMap.update({...inputs, selected: this.selectedMapCamera});
    this.safetyMap.update({...inputs, heat: true, selected: this.selectedMapCamera});
    (this.page === 'heatmap' ? this.safetyMap : this.deploymentMap).activate();
    this.selectMapCamera(camera);
    $('safety-statistics').replaceChildren(...[
      ['Located cameras', data.cameras.length],
      ['Cameras without location', data.missingCameras],
      ['Mapped incidents', data.incidents.length],
      ['Incidents without location', data.missingIncidents],
    ].flatMap(([name, value]) => [el('dt', name), el('dd', value)]));
    $('safety-updated').textContent = `${this.connected ? 'Local records' : 'Service offline · last loaded records'} · latest 100 incidents, excluding false positives and recordings`;
  }
  selectMapCamera(camera) {
    this.selectedMapCamera = camera?.camera_id || null;
    $('map-sector-title').textContent = camera?.name || 'No located camera';
    $('map-sector-name').textContent = camera?.location?.place || 'Allow laptop location access to place connected cameras on the map.';
    $('map-open-camera').disabled = !camera;
    const dl = el('dl');
    if (camera) for (const [key, value] of [
      ['Status', this.connected ? camera.status : 'Service offline'],
      ['Location source', camera.location.source === 'browser' ? 'Laptop browser' : 'Saved location'],
      ['Accuracy', camera.location.accuracy_meters == null ? 'Not recorded' : `±${Math.round(camera.location.accuracy_meters)} m`],
    ]) dl.append(el('dt', key), el('dd', value));
    $('map-sector-stats').replaceChildren(dl);
    $('heat-selection').replaceChildren(
      el('strong', camera?.name || 'No located cameras'),
      el('div', camera?.location?.place || 'No synthetic markers are shown.'),
      el('small', 'Incident locations are captured when detected. Unlocated history is not moved to this laptop.'),
    );
    this.deploymentMap.update({selected: this.selectedMapCamera});
    this.safetyMap.update({selected: this.selectedMapCamera});
  }
  renderIncidents() {
    const sample = $("library-dataset").value === "sample",
      filter = $("incident-filter").value;
    $('recordings-panel').hidden = sample;
    this.renderRecordings();
    let rows = sample
      ? this.dataset.events
          .slice()
          .reverse()
          .map((event) => ({
            ...event,
            review: this.preferences.reviews[event.id] || event.review,
          }))
      : this.incidents;
    if (sample) rows = scoped(rows, this.scope);
    rows = rows.filter((event) => filter === "all" || event.review === filter);
    $("library-summary").textContent =
      `${rows.length} ${sample ? "presentation records" : "local records"} · ${sample ? "fictional event ledger" : "latest 100 incidents"}`;
    $("incident-library").replaceChildren(
      ...(rows.length
        ? rows.map((event) => this.incidentNode(event))
        : [el("div", "No incidents match this view.", "empty-state")]),
    );
  }
  incidentNode(event) {
    const node = el(
        "article",
        null,
        `incident ${event.review !== "unreviewed" ? "reviewed" : ""}`,
      ),
      copy = el("div"),
      meta = el("div", null, "incident-meta");
    node.append(el("div", null, "incident-accent"));
    copy.append(
      el(
        "h3",
        `${EVENT_NAMES[event.event_type] || event.event_type} · ${event.camera_name || "Camera"}`,
      ),
    );
    meta.append(
      badge(
        event.mode === "presentation"
          ? "SAMPLE"
          : event.mode === "demo"
            ? "PIPELINE TEST"
            : "LOCAL",
      ),
      el("span", formatTime(event.created)),
      el("span", `${Math.round(event.score * 100)} / 100`),
      el("span", event.review.replaceAll("_", " ").toUpperCase()),
    );
    copy.append(meta, el("p", event.reasons.join(" · ")));
    if (event.clip_error) copy.append(el("p", event.clip_error));
    const actions = el("div", null, "incident-actions");
    if (event.clip) {
      const play = el('button', 'Watch evidence', 'button primary');
      play.addEventListener('click', () => this.playIncident(event));
      const link = el("a", "Download clip (AVI)", "button secondary");
      link.href = `/api/incidents/${encodeURIComponent(event.id)}/clip`;
      actions.append(play, link);
    }
    if (canAnalyze(event)) {
      const analyze = el('button', 'Gemini analysis & questions', 'button secondary');
      analyze.addEventListener('click', () => this.incidentAI.open(event));
      actions.append(analyze);
    }
    if (canShareEvidence(event)) {
      const share = el('button', 'Video link', 'button secondary');
      share.addEventListener('click', () => this.evidenceShare.open(event));
      actions.append(share);
    }
    if (event.signals?.live_camera === false && manualDispatchEligible(event)) {
      const dispatch = el('button', 'Review dispatch', 'button secondary');
      dispatch.addEventListener('click', () => this.response.openEvidenceDispatch(event));
      actions.append(dispatch);
    }
    for (const [decision, label] of [
      ["confirmed", "Confirm after review"],
      ["false_positive", "Mark false positive"],
      ["unreviewed", "Reset review"],
    ]) {
      if (
        decision === event.review ||
        (decision === "unreviewed" && event.review === "unreviewed")
      )
        continue;
      const button = el("button", label, "button secondary");
      button.addEventListener("click", async () => {
        button.disabled = true;
        await this.review(event, decision);
        if (button.isConnected) button.disabled = false;
      });
      actions.append(button);
    }
    node.append(copy, actions);
    return node;
  }
  playIncident(event) {
    if (!event?.clip) return;
    const base = `/api/incidents/${encodeURIComponent(event.id)}`;
    this.player.open({title: `${EVENT_NAMES[event.event_type] || event.event_type} · ${event.camera_name}`,
      url: `${base}/play`, download: `${base}/clip`,
      recordedAt: event.created - (event.signals?.buffer_seconds || 0), eventAt: event.signals?.buffer_seconds ?? null});
  }
  renderRecordings() {
    const signature = JSON.stringify(this.recordings);
    if (signature === this.recordingSignature) return;
    this.recordingSignature = signature;
    $('recording-list').replaceChildren(...(this.recordings.length ? this.recordings.map(recording => {
      const row = el('article', null, 'recording-row'), copy = el('div'), actions = el('div', null, 'recording-actions');
      copy.append(el('h3', recording.name), el('p', `${timecode(recording.duration)} · ${(recording.size / 1048576).toFixed(1)} MB · uploaded ${formatTime(recording.created)}`));
      const play = el('button', 'Watch video', 'button secondary');
      play.addEventListener('click', () => this.player.open({title: recording.name,
        url: `/api/recordings/${recording.id}/play`, download: `/api/recordings/${recording.id}/download`}));
      const analyze = el('button', 'Analyse', 'button primary');
      analyze.addEventListener('click', () => this.openCameraForm(recording));
      actions.append(play, analyze); row.append(copy, actions); return row;
    }) : [el('p', 'No recordings uploaded yet.', 'field-help')]));
  }
  async uploadRecording(event) {
    event.preventDefault();
    if (this.uploadingRecording) return;
    const file = $('recording-file').files[0];
    if (!file) return;
    if (file.size > 1024 ** 3) { $('recording-status').textContent = 'Videos must be 1 GB or smaller.'; return; }
    this.uploadingRecording = true;
    $('recording-file').disabled = true;
    $('recording-submit').disabled = true;
    $('recording-status').textContent = 'Uploading and checking the video…';
    try {
      const response = await fetch('/api/recordings', {method: 'POST', body: file,
        headers: {'X-VMD-Client': 'dashboard', 'X-File-Name': encodeURIComponent(file.name), 'Content-Type': 'application/octet-stream'}});
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'Video upload failed');
      $('recording-file').value = '';
      this.recordings = [result, ...this.recordings.filter(item => item.id !== result.id)];
      this.renderRecordings();
      $('recording-status').textContent = `${result.name} saved. Watch it or choose Analyse to configure detection.`;
      await this.refreshEvidence();
    } catch (error) { $('recording-status').textContent = error.message; }
    finally {
      this.uploadingRecording = false;
      $('recording-file').disabled = false;
      $('recording-submit').disabled = false;
    }
  }
}

const profile = await requireSession();
if (profile) new Workspace(profile);
