/** Browser coordinates are shared by maps and camera forms; never invented. */
export function validCenter(center) {
  return Array.isArray(center) && center.length === 2 && center.every(Number.isFinite) &&
    Math.abs(center[0]) <= 90 && Math.abs(center[1]) <= 180;
}

export function isLiveCameraSource(source) {
  return /^(?:[0-9]|https?:\/\/.+|rtsps?:\/\/.+)$/i.test((source || '').trim());
}

export class MapLocation {
  constructor(geolocation = globalThis.navigator?.geolocation) {
    this.geolocation = geolocation;
    this.listeners = new Set();
    this.attempted = false;
    this.pending = null;
    this.state = {center: null, source: 'none', accuracy: null, capturedAt: null,
      status: 'idle', message: 'Laptop location not acquired', revision: 0};
  }
  subscribe(listener) {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }
  publish(changes) {
    this.state = {...this.state, ...changes};
    this.listeners.forEach(listener => listener(this.state));
  }
  locate() {
    if (this.pending) return this.pending;
    this.attempted = true;
    this.publish({status: 'locating', message: 'Finding laptop location… Allow browser location access when prompted.'});
    this.pending = Promise.resolve().then(() => new Promise(resolve => {
      let finished = false;
      const done = changes => {
        if (finished) return;
        finished = true;
        clearTimeout(timer);
        this.publish(changes);
        resolve(this.state);
      };
      const fail = error => {
        const reason = error?.code === 1
          ? 'Location access denied. Allow location in your browser and system settings, then retry.'
          : error?.code === 3 ? 'Location timed out. Retry when location services are available.'
            : 'Laptop location unavailable. Enable location services and retry.';
        done({status: 'error', message: reason + (this.state.center ? ' Showing the last measured position.' : ' No location has been assigned.')});
      };
      const timer = setTimeout(() => fail({code: 3}), 18000);
      if (!this.geolocation) return fail();
      try {
        this.geolocation.getCurrentPosition(position => {
          const center = [position.coords.latitude, position.coords.longitude];
          if (!validCenter(center)) return fail();
          const rawAccuracy = position.coords.accuracy;
          const accuracy = Number.isFinite(rawAccuracy) && rawAccuracy >= 0 ? rawAccuracy : null;
          done({center, accuracy, capturedAt: Date.now(), source: 'device', status: 'ready',
            message: `Laptop location${accuracy === null ? '' : ` · accuracy ±${Math.round(accuracy).toLocaleString()} m`}`,
            revision: this.state.revision + 1});
        }, fail, {enableHighAccuracy: true, timeout: 15000, maximumAge: 0});
      } catch { fail(); }
    })).finally(() => { this.pending = null; });
    return this.pending;
  }
  currentCameraLocation(place = '') {
    if (this.state.status !== 'ready' || !Number.isFinite(this.state.capturedAt) ||
        Date.now() - this.state.capturedAt > 60000 || !validCenter(this.state.center)) return null;
    return {place: place.trim() || 'Laptop location', latitude: this.state.center[0], longitude: this.state.center[1],
      source: 'browser', accuracy_meters: this.state.accuracy, captured_at: this.state.capturedAt / 1000};
  }
  async cameraLocation(place = '') {
    if (!this.currentCameraLocation(place)) await this.locate();
    const location = this.currentCameraLocation(place);
    if (!location) throw new Error(this.state.message);
    return location;
  }
}

export const sharedLocation = new MapLocation();
