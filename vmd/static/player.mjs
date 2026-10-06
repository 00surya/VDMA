import { $, formatTime } from './ui.mjs';

export function timecode(seconds) {
  const value = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  return [Math.floor(value / 3600), Math.floor(value / 60) % 60, value % 60]
    .map(part => String(part).padStart(2, '0')).join(':');
}
export function seekTime(current, delta, duration) {
  return Math.max(0, Math.min(Number.isFinite(duration) ? duration : 0, current + delta));
}

export class MediaPlayer {
  constructor() {
    this.dialog = $('media-player');
    this.video = $('playback-video');
    this.video.muted = true;
    $('close-player').addEventListener('click', () => this.dialog.close());
    this.dialog.addEventListener('close', () => {
      this.video.pause();
      this.video.removeAttribute('src');
      this.video.load();
    });
    $('playback-toggle').addEventListener('click', () => this.toggle());
    for (const button of this.dialog.querySelectorAll('[data-seek]'))
      button.addEventListener('click', () => this.seek(Number(button.dataset.seek)));
    $('playback-start').addEventListener('click', () => this.seek(-this.video.duration));
    $('playback-end').addEventListener('click', () => this.seek(this.video.duration));
    $('playback-event').addEventListener('click', () => {
      this.video.currentTime = seekTime(0, this.eventAt, this.video.duration);
    });
    $('playback-timeline').addEventListener('input', event => {
      this.video.currentTime = Number(event.target.value);
      this.render();
    });
    $('playback-speed').addEventListener('change', event => {
      this.video.playbackRate = Number(event.target.value);
    });
    $('playback-fullscreen').addEventListener('click', () => {
      this.dialog.requestFullscreen?.().catch(() => {
        $('playback-message').textContent = 'Fullscreen is unavailable in this browser.';
      });
    });
    for (const event of ['loadedmetadata', 'durationchange', 'timeupdate', 'play', 'pause', 'ended', 'ratechange'])
      this.video.addEventListener(event, () => this.render());
    this.video.addEventListener('waiting', () => { $('playback-message').textContent = 'Buffering…'; });
    this.video.addEventListener('playing', () => { $('playback-message').textContent = ''; });
    this.video.addEventListener('error', () => {
      if (this.dialog.open) $('playback-message').textContent = 'Playback unavailable. Use Download original, or retry after the video finishes preparing.';
      this.setEnabled(false);
    });
    this.dialog.addEventListener('keydown', event => {
      if (['INPUT', 'SELECT', 'BUTTON', 'A'].includes(event.target.tagName)) return;
      if ([' ', 'ArrowLeft', 'ArrowRight', 'j', 'k', 'l'].includes(event.key)) {
        event.preventDefault();
        if ([' ', 'k'].includes(event.key)) this.toggle();
        else this.seek(['ArrowLeft', 'j'].includes(event.key) ? -10 : 10);
      }
    });
  }
  setEnabled(enabled) {
    for (const control of this.dialog.querySelectorAll('[data-playback-control]')) control.disabled = !enabled;
  }
  open({title, url, download, recordedAt = null, eventAt = null}) {
    this.video.pause();
    this.recordedAt = recordedAt;
    this.eventAt = eventAt;
    $('playback-title').textContent = title;
    $('playback-download').href = download;
    $('playback-speed').value = '1';
    $('playback-event').hidden = !Number.isFinite(eventAt);
    $('playback-message').textContent = 'Preparing browser playback… The first play may take longer.';
    $('playback-position').textContent = '00:00:00 / 00:00:00';
    $('playback-duration').textContent = '00:00:00';
    $('playback-clock').textContent = recordedAt ? formatTime(recordedAt) : 'Recording time';
    $('playback-state').textContent = 'LOADING';
    $('playback-toggle').textContent = 'Play';
    $('playback-timeline').value = '0';
    $('playback-timeline').max = '0';
    this.setEnabled(false);
    this.video.src = url;
    this.video.playbackRate = 1;
    this.video.load();
    if (!this.dialog.open) this.dialog.showModal();
  }
  toggle() {
    if (!Number.isFinite(this.video.duration)) return;
    if (this.video.paused) this.video.play().catch(() => {
      $('playback-message').textContent = 'Could not start playback. Try Play again.';
    });
    else this.video.pause();
  }
  seek(delta) {
    if (Number.isFinite(this.video.duration)) {
      this.video.currentTime = seekTime(this.video.currentTime, delta, this.video.duration);
      this.render();
    }
  }
  render() {
    const {duration, currentTime, paused, ended} = this.video;
    if (!this.dialog.open || !Number.isFinite(duration)) return;
    this.setEnabled(true);
    $('playback-message').textContent = '';
    $('playback-toggle').textContent = paused ? 'Play' : 'Pause';
    $('playback-state').textContent = ended ? 'END' : paused ? 'PAUSED' : 'PLAYING';
    $('playback-position').textContent = `${timecode(currentTime)} / ${timecode(duration)}`;
    $('playback-clock').textContent = this.recordedAt
      ? formatTime(this.recordedAt + currentTime) : `RECORDING · ${timecode(currentTime)}`;
    $('playback-timeline').max = String(duration);
    $('playback-timeline').value = String(currentTime);
    $('playback-timeline').setAttribute('aria-valuetext', `${timecode(currentTime)} of ${timecode(duration)}`);
    $('playback-duration').textContent = timecode(duration);
  }
}
