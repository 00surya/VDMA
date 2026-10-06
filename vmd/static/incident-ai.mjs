import { el } from './ui.mjs';

export function canAnalyze(event) {
  return event?.mode === 'live' && typeof event.signals?.live_camera === 'boolean'
    && event.review !== 'false_positive' && Boolean(event.clip);
}

export class IncidentAIUI {
  constructor(workspace) {
    this.workspace = workspace;
    this.session = null;
    this.inspect = el('button', 'Gemini incident analyst', 'button secondary');
    this.inspect.type = 'button'; this.inspect.hidden = true;
    document.getElementById('inspect-play')?.after(this.inspect);
    this.inspect.addEventListener('click', () => this.open(workspace.selectedEvent()));
  }
  select(event) { this.inspect.hidden = !canAnalyze(event); }
  path(session, suffix='') { return `/incidents/${encodeURIComponent(session.id)}/ai${suffix}`; }
  current(session) { return this.session === session && this.dialog.open; }
  create() {
    if (this.dialog) return;
    this.dialog = el('dialog', null, 'incident-ai-dialog');
    const heading = el('div', null, 'dialog-heading'), close = el('button', '×', 'icon-button');
    close.type = 'button'; close.setAttribute('aria-label', 'Close Gemini incident analyst');
    close.addEventListener('click', () => this.dialog.close());
    heading.append(el('h2', 'Gemini incident analyst'), close);
    this.context = el('p'); this.status = el('p', null, 'field-help'); this.status.setAttribute('role', 'status');
    const notice = el('p', 'New incidents are analyzed automatically in the background using selected evidence frames sent to Google. Reports are AI observations requiring review. Calls and local alerts continue independently.', 'field-help');
    const actions = el('div', null, 'incident-actions');
    this.analyze = el('button', 'Analyze clip', 'button primary');
    this.quality = el('select'); this.quality.setAttribute('aria-label', 'Analysis model');
    for (const [value, label] of [['quality', 'Quality model'], ['fast', 'Fast model']]) {
      const option = el('option', label); option.value = value; this.quality.append(option);
    }
    this.follow = el('button', 'Follow aftermath · 2 minutes', 'button secondary');
    this.approve = el('button', 'Approve briefing for future calls', 'button secondary');
    this.download = el('a', 'Download PDF report', 'button secondary');
    this.jsonDownload = el('a', 'Download JSON', 'button secondary');
    actions.append(this.quality, this.analyze, this.follow, this.approve, this.download, this.jsonDownload);
    this.output = el('div', null, 'incident-ai-output');
    const form = el('form'), label = el('label', 'Ask about this incident'), input = el('input');
    input.maxLength = 600; input.required = true; input.placeholder = 'What happens before the person falls?';
    label.append(input); this.askButton = el('button', 'Ask Gemini', 'button secondary'); this.askButton.type = 'submit';
    form.append(label, this.askButton); this.answer = el('p'); this.answer.setAttribute('role', 'status');
    form.addEventListener('submit', async event => {
      event.preventDefault(); const session = this.session;
      if (!session || session.busy) return;
      this.answer.textContent = 'Reviewing evidence…';
      const value = await this.action('/ask', {question: input.value}, 60000);
      if (value && this.current(session)) this.answer.textContent = `${value.answer}\nEvidence times: ${value.evidence_seconds.join(', ')}s. ${value.limitations}`;
    });
    this.analyze.addEventListener('click', () => this.action('', {quality: this.quality.value === 'quality'}));
    this.follow.addEventListener('click', () => this.action('/follow', {enabled: !this.session?.data?.following}));
    this.approve.addEventListener('click', () => this.action('/approve', {enabled: !this.session?.data?.approved}));
    this.dialog.append(heading, this.context, notice, this.status, actions, this.output, form, this.answer);
    this.dialog.addEventListener('close', () => {
      clearTimeout(this.timer); this.session = null; this.output.replaceChildren(); this.answer.textContent = ''; input.value = '';
    });
    document.body.append(this.dialog);
  }
  open(event) {
    if (!canAnalyze(event)) return;
    this.create(); clearTimeout(this.timer);
    this.session = {id: event.id, event, data: null, config: null, busy: false, error: ''};
    this.context.textContent = `${event.camera_name} · ${event.event_type.replaceAll('_', ' ')} · ${new Date(event.created*1000).toLocaleString()}`;
    this.answer.textContent = ''; this.output.replaceChildren();
    this.download.href = `/api/incidents/${encodeURIComponent(event.id)}/ai/report.pdf`;
    this.jsonDownload.href = `/api/incidents/${encodeURIComponent(event.id)}/ai/report`;
    if (!this.dialog.open) this.dialog.showModal();
    this.refresh();
  }
  async refresh() {
    const session = this.session;
    if (!session || !this.current(session)) return;
    try {
      const [config, data] = await Promise.all([
        this.workspace.api.request('/ai/status'), this.workspace.api.request(this.path(session))]);
      if (!this.current(session)) return;
      session.config = config; session.data = data; session.error = '';
    } catch (error) { if (this.current(session)) session.error = error.message; }
    if (!this.current(session)) return;
    this.render(); clearTimeout(this.timer);
    this.timer = setTimeout(() => this.refresh(), 2500);
  }
  async action(suffix, body, timeout=12000) {
    const session = this.session;
    if (!session || session.busy) return null;
    session.busy = true; session.error = ''; this.render();
    try {
      const value = await this.workspace.api.request(this.path(session, suffix), body, 'POST', timeout);
      if (!this.current(session)) return null;
      if (suffix !== '/ask') session.data = value;
      return value;
    } catch (error) { if (this.current(session)) session.error = error.message; return null; }
    finally { if (this.current(session)) { session.busy = false; this.render(); } }
  }
  render() {
    const session = this.session;
    if (!session) return;
    const data = session.data, ready = session.config?.configured, pending = ['queued', 'running'].includes(data?.status);
    this.status.textContent = session.error || session.config?.error || data?.error ||
      `${data?.status?.replaceAll('_', ' ') || 'Loading setup…'}${data?.following ? ` · following for ${data.follow_seconds_remaining}s` : ''}`;
    this.download.hidden = data?.status !== 'ready';
    this.analyze.disabled = session.busy || !ready || pending;
    this.follow.disabled = session.busy || (!ready && !data?.following) || session.event.signals.live_camera !== true;
    this.follow.textContent = data?.following ? 'Stop aftermath updates' : 'Follow aftermath · 2 minutes';
    this.approve.disabled = session.busy || data?.status !== 'ready';
    this.approve.textContent = data?.approved ? 'Withdraw briefing approval' : 'Approve briefing for future calls';
    this.askButton.disabled = session.busy || !ready || pending;
    this.output.replaceChildren();
    if (data?.report) {
      const report = data.report;
      this.output.append(el('h3', 'AI observations'), el('p', report.summary),
        el('p', `Local alert assessment: ${report.assessment.replaceAll('_', ' ')} · ${data.approved ? 'Briefing approved' : 'Draft briefing'}`, 'field-help'));
      const subjects = el('ul'); report.subjects.forEach(text => subjects.append(el('li', text))); this.output.append(subjects);
      const timeline = el('ol'); report.timeline.forEach(row => timeline.append(el('li', `${row.seconds.toFixed(1)}s · ${row.observation}`))); this.output.append(timeline);
      if (report.uncertainties.length) this.output.append(el('p', `Unclear: ${report.uncertainties.join(' · ')}`));
      this.output.append(el('h3', 'Call briefing draft'), el('p', report.briefing));
    }
    for (const item of data?.observations || []) this.output.append(
      el('h3', `Later observation · ${new Date(item.observed_at*1000).toLocaleTimeString()}`), el('p', item.report.summary));
  }
}
