/** Backend cookie sessions authenticate the centre; helpers below retain presentation preferences. */
const SESSION_KEY = "vmd.presentation-session.v1";
const PREFS_KEY = "vmd.presentation-preferences.v1";

export const SECTORS = [
  { id: "1", name: "Central district" },
  { id: "2", name: "Residential" },
  { id: "3", name: "Civic centre" },
  { id: "4", name: "Market" },
  { id: "6", name: "Park" },
  { id: "7", name: "Transit hub" },
  { id: "8", name: "University" },
  { id: "9", name: "Industrial" },
];

export function validateProfile(value) {
  if (!value || !["admin", "operator"].includes(value.role)) return null;
  const sector = SECTORS.some((item) => item.id === value.sector)
    ? value.sector
    : "4";
  const fallback =
    value.role === "admin" ? "City administrator" : "Sector operator";
  const name =
    typeof value.name === "string" ? value.name.trim().slice(0, 64) : "";
  return { role: value.role, sector, name: name || fallback };
}

export function readJSON(storage, key, fallback) {
  try {
    return JSON.parse(storage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}

export function getSession() {
  try {
    return validateProfile(readJSON(sessionStorage, SESSION_KEY, null));
  } catch {
    return null;
  }
}
export function saveSession(value) {
  const session = validateProfile(value);
  if (!session) throw new Error("Choose an available workspace role.");
  sessionStorage.setItem(SESSION_KEY, JSON.stringify(session));
  return session;
}
export async function signOut() {
  try {
    const response = await fetch('/api/auth/logout', {method: 'POST', headers: {'X-VMD-Client': 'dashboard'}});
    if (!response.ok && response.status !== 401) throw new Error('Sign out failed');
  } catch { window.alert('Cannot reach the server to sign out. Please try again.'); return; }
  sessionStorage.removeItem(SESSION_KEY);
  location.assign("/login.html");
}
export async function requireSession() {
  try {
    const response = await fetch('/api/auth/status');
    const status = await response.json();
    if (status.authenticated) return {name: status.centre.name, role: 'admin', sector: '4'};
  } catch { /* The sign-in page reports connection errors. */ }
  location.replace('/login.html');
  return null;
}
export function readPreferences() {
  try {
    return readJSON(localStorage, PREFS_KEY, {});
  } catch {
    return {};
  }
}
export function savePreferences(value) {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}
export function sectorName(id) {
  const sector = SECTORS.find((item) => item.id === String(id));
  return sector ? `Sector ${sector.id} · ${sector.name}` : "Unassigned sector";
}
