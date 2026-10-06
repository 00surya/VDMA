import { ApiClient } from './ui.mjs';
import { centreFields, readCentre } from './centre.mjs';
const api = new ApiClient();
const form = document.getElementById('login-form');
const error = document.getElementById('login-error');
const button = form.querySelector('button[type=submit]');
let registered = true;
try {
  const status = await api.request('/auth/status');
  if (status.authenticated) location.replace('/');
  registered = status.registered;
  document.getElementById('login-title').textContent = registered ? 'Centre sign in' : 'Register your centre';
  document.getElementById('login-description').textContent = registered
    ? 'Sign in to manage your cameras and response contacts.'
    : 'Set up this installation with at least one authority and a hospital contact.';
  if (!registered) centreFields(document.getElementById('centre-fields'));
  document.getElementById('password').autocomplete = registered ? 'current-password' : 'new-password';
  button.textContent = registered ? 'Sign in →' : 'Create centre →';
  button.disabled = false;
} catch (exc) { error.textContent = exc.message; }
form.addEventListener('submit', async event => {
  event.preventDefault();
  button.disabled = true;
  error.textContent = '';
  try {
    const password = form.elements.password.value;
    await api.request(registered ? '/auth/login' : '/auth/signup',
      registered ? {password} : {...readCentre(form), password});
    location.assign('/');
  } catch (exc) { error.textContent = exc.message; }
  finally { button.disabled = false; }
});
