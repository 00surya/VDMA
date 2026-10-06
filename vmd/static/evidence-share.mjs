import { el } from './ui.mjs';

export function canShareEvidence(event) {
  return event?.mode === 'live' && typeof event.signals?.live_camera === 'boolean'
    && event.review !== 'false_positive' && typeof event.clip === 'string' && event.clip.length > 0;
}

function expiryTime(seconds) {
  return new Date(seconds * 1000).toLocaleString();
}

export class EvidenceShare {
  constructor(workspace) {
    this.workspace = workspace;
    this.session = null;
  }
  createDialog() {
    if (this.nodes) return;
    const dialog = el('dialog'), heading = el('div', null, 'dialog-heading'),
      title = el('h2', 'Share evidence'), close = el('button', '×', 'icon-button'),
      context = el('p'), status = el('p', null, 'field-help'), count = el('p', null, 'field-help'),
      warning = el('p', 'Anyone with this link can view until it expires. This laptop and the evidence tunnel must stay online.', 'field-help'),
      actions = el('div', null, 'evidence-share-actions'), generate = el('button', 'Generate video link', 'button primary'),
      refresh = el('button', 'Check sharing status', 'button secondary'), result = el('div'),
      input = el('input'), expiry = el('p', null, 'field-help'), linkActions = el('div', null, 'evidence-share-actions'),
      copy = el('button', 'Copy link', 'button secondary'), open = el('a', 'Open link', 'button secondary'),
      revoke = el('button', 'Revoke all links', 'button outline'),
      revokeHelp = el('p', 'Revoking disables every existing video link for this incident, including links already sent. It does not cancel calls.', 'field-help'),
      message = el('p', null, 'field-help'), error = el('p', null, 'error-text');
    dialog.id = 'evidence-share-dialog'; title.id = 'evidence-share-title'; context.id = 'evidence-share-context';
    dialog.setAttribute('aria-labelledby', title.id); dialog.setAttribute('aria-describedby', context.id);
    close.setAttribute('aria-label', 'Close evidence sharing'); close.autofocus = true;
    for (const button of [close, generate, refresh, copy, revoke]) button.type = 'button';
    input.type = 'text'; input.readOnly = true; input.setAttribute('aria-label', 'Evidence video link');
    open.target = '_blank'; open.rel = 'noopener noreferrer';
    status.setAttribute('role', 'status'); message.setAttribute('role', 'status'); error.setAttribute('role', 'alert');
    close.addEventListener('click', () => dialog.close());
    dialog.addEventListener('close', () => {
      this.session = null; clearInterval(this.timer);
      input.value = ''; open.removeAttribute('href'); result.hidden = true;
    });
    generate.addEventListener('click', () => this.generate());
    refresh.addEventListener('click', () => this.refresh());
    copy.addEventListener('click', () => this.copy());
    revoke.addEventListener('click', () => this.revoke());
    heading.append(title, close); actions.append(generate, refresh); linkActions.append(copy, open);
    result.append(input, expiry, linkActions);
    dialog.append(heading, context, status, count, warning, actions, result, revokeHelp, revoke, message, error);
    document.body.append(dialog);
    this.nodes = {dialog, context, status, count, generate, refresh, result, input, expiry, copy, open, revoke, message, error};
  }
  async open(event) {
    if (!canShareEvidence(event)) return;
    this.createDialog();
    clearInterval(this.timer);
    this.session = {id: event.id, event, loading: false, busy: false, config: null, links: null, url: null, expires: null, error: '', message: ''};
    if (!this.nodes.dialog.open) this.nodes.dialog.showModal();
    this.timer = setInterval(() => this.render(), 1000);
    await this.refresh();
  }
  current(session) { return this.session === session && this.nodes.dialog.open; }
  event(session) { return this.workspace.incidents?.find(event => event.id === session.id) || session.event; }
  path(session) { return `/incidents/${encodeURIComponent(session.id)}/share`; }
  async refresh() {
    const session = this.session;
    if (!session || session.loading || session.busy) return;
    session.loading = true; session.error = ''; this.render();
    try {
      const [config, links] = await Promise.all([
        this.workspace.api.request('/evidence/status'), this.workspace.api.request(this.path(session)),
      ]);
      if (!this.current(session)) return;
      session.config = config; session.links = links;
      if (links.active_count === 0 && session.url) {
        session.url = null; session.expires = null;
        session.message = 'No active links remain for this incident.';
      }
    } catch (error) {
      if (this.current(session)) { session.config = null; session.links = null; session.error = error.message; }
    } finally {
      if (this.current(session)) { session.loading = false; this.render(); }
    }
  }
  async generate() {
    const session = this.session;
    if (!session || session.busy || session.loading || !session.config?.configured || !canShareEvidence(this.event(session))) return;
    session.busy = true; session.error = ''; session.message = ''; this.render();
    try {
      const value = await this.workspace.api.request(this.path(session), {}, 'POST', 195000);
      if (!this.current(session)) return;
      const url = new URL(value.url);
      if (value.incident_id !== session.id || url.protocol !== 'https:' || url.username || url.password
          || !Number.isFinite(value.expires_at) || value.expires_at <= Date.now()/1000)
        throw new Error('The server did not return a valid video link. Check sharing status before trying again.');
      session.url = url.href; session.expires = value.expires_at;
      session.message = 'Video link generated. No message or call was sent.';
      try { session.links = await this.workspace.api.request(this.path(session)); }
      catch { session.links = null; session.error = 'The link was generated, but its active-link count could not be refreshed.'; }
    } catch (error) {
      if (this.current(session)) session.error = `${error.message} Check sharing status before retrying; the link may have been generated.`;
    } finally {
      if (this.current(session)) { session.busy = false; this.render(); }
    }
  }
  async revoke() {
    const session = this.session;
    if (!session || session.busy || session.loading) return;
    session.busy = true; session.error = ''; session.message = ''; this.render();
    try {
      const value = await this.workspace.api.request(this.path(session), {}, 'DELETE');
      if (!this.current(session)) return;
      session.url = null; session.expires = null;
      session.links = {active_count: 0, latest_expires_at: null};
      session.message = `${value.revoked_count} video link${value.revoked_count === 1 ? '' : 's'} revoked. Calls are unchanged.`;
    } catch (error) {
      if (this.current(session)) session.error = error.message;
    } finally {
      if (this.current(session)) { session.busy = false; this.render(); }
    }
  }
  async copy() {
    const session = this.session;
    if (!session?.url || session.expires <= Date.now()/1000 || session.busy) return;
    try {
      await navigator.clipboard.writeText(session.url);
      if (this.current(session)) session.message = 'Video link copied.';
    } catch {
      if (this.current(session)) {
        this.nodes.input.focus(); this.nodes.input.select();
        session.message = 'Copy is unavailable. The link is selected; copy it manually.';
      }
    }
    if (this.current(session)) this.render();
  }
  render() {
    const session = this.session;
    if (!session || !this.nodes) return;
    const node = this.nodes, event = this.event(session), expired = session.expires <= Date.now()/1000,
      busy = session.busy || session.loading, active = session.links?.active_count;
    node.context.textContent = `${event.camera_name} · ${event.event_type.replaceAll('_', ' ')} · ${event.signals.live_camera ? 'Camera evidence' : 'Recording evidence'}`;
    node.status.textContent = session.loading ? 'Checking evidence sharing…'
      : session.config?.configured ? 'Sharing is configured. Keep the evidence tunnel running.'
      : session.config ? 'Evidence sharing is offline. Start the evidence tunnel, then check again.' : 'Sharing status is unavailable. Check again.';
    node.count.textContent = Number.isInteger(active) ? `${active} active video link${active === 1 ? '' : 's'} at last check.`
      + (active && session.links.latest_expires_at ? ` Last expiry: ${expiryTime(session.links.latest_expires_at)}.` : '')
      + (active && !session.url ? ' Existing link addresses cannot be recovered; generate a new link if needed.' : '') : '';
    node.generate.disabled = busy || !session.config?.configured || !canShareEvidence(event);
    node.generate.textContent = session.busy ? 'Working…' : session.url ? 'Generate another link' : 'Generate video link';
    node.refresh.disabled = busy;
    node.result.hidden = !session.url;
    node.input.value = session.url || '';
    node.expiry.textContent = session.url ? `${expired ? 'Expired' : 'Expires'}: ${expiryTime(session.expires)} (local time).` : '';
    node.copy.disabled = busy || !session.url || expired;
    node.open.hidden = busy || !session.url || expired;
    if (session.url && !expired) node.open.href = session.url;
    else node.open.removeAttribute('href');
    node.revoke.disabled = busy || !(active > 0 || session.url);
    node.message.textContent = session.message;
    node.error.textContent = session.error;
  }
}
