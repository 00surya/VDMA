const $ = (id) => document.getElementById(id);
const presets = {
  support: {state: 'My package arrived damaged and I want a refund.', question: 'Which queue should handle this?', options: ['billing', 'shipping', 'technical', 'general']},
  review: {state: 'A recorded clip shows two people standing apart. One raises a hand to wave. No physical contact is visible. The depth measurement is unavailable.', question: 'Which review label best matches the described evidence?', options: ['ordinary interaction', 'possible physical conflict', 'insufficient evidence']},
  passage: {state: 'The treaty was signed in Paris in 1992. It entered into force the following year, after the last signatory ratified it.', question: 'Did the treaty enter into force in 1992?', options: ['yes', 'no', 'unknown']},
};
let options = [...presets.support.options];
let busy = false;
let modelReady = false;
let modelLoading = true;
let generation = 0;
let lastResult = null;

function clearResult() {
  generation++;
  lastResult = null;
  $('answer-result').hidden = true;
  $('answer-empty').hidden = false;
  $('raw-details').hidden = true;
  $('raw-output').textContent = '';
  $('announcement').textContent = '';
  $('latency').textContent = '—';
  $('result-badge').textContent = busy ? 'Inputs changed' : 'Awaiting a run';
  $('state-count').textContent = `${$('state').value.length} / 6000`;
  $('error').hidden = true;
  document.querySelectorAll('[data-preset]').forEach((button) => button.classList.remove('selected'));
}

function renderOptions() {
  $('options').replaceChildren();
  options.forEach((value, index) => {
    const chip = document.createElement('div');
    chip.className = 'option-chip';
    const input = document.createElement('input');
    input.value = value;
    input.maxLength = 80;
    input.required = true;
    input.setAttribute('aria-label', `Option ${index + 1}`);
    input.style.width = `${Math.min(Math.max(value.length + 1, 5), 25)}ch`;
    input.addEventListener('input', () => {
      options[index] = input.value;
      input.style.width = `${Math.min(Math.max(input.value.length + 1, 5), 25)}ch`;
      clearResult();
    });
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.textContent = '×';
    remove.setAttribute('aria-label', `Remove option ${index + 1}`);
    remove.addEventListener('click', () => {
      options.splice(index, 1);
      renderOptions();
      clearResult();
      $('new-option').focus();
    });
    chip.append(input, remove);
    $('options').append(chip);
  });
  $('options-count').textContent = `${options.length} choices`;
  $('add-option').disabled = options.length >= 12;
}

function showError(message) {
  $('error').textContent = message;
  $('error').hidden = false;
}

function addOption() {
  const value = $('new-option').value.trim();
  if (!value) return;
  if (options.length >= 12) return showError('Use at most 12 options.');
  if (options.some((item) => item.trim().toLowerCase() === value.toLowerCase())) return showError('Each option must be distinct.');
  options.push(value);
  $('new-option').value = '';
  renderOptions();
  clearResult();
  $('new-option').focus();
}
$('add-option').addEventListener('click', addOption);
$('new-option').addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); addOption(); } });
['state', 'question'].forEach((id) => $(id).addEventListener('input', clearResult));
document.querySelectorAll('[data-preset]').forEach((button) => button.addEventListener('click', () => {
  const preset = presets[button.dataset.preset];
  $('state').value = preset.state;
  $('question').value = preset.question;
  $('new-option').value = '';
  options = [...preset.options];
  renderOptions(); clearResult(); button.classList.add('selected');
}));

function renderResult(result) {
  lastResult = result;
  $('answer-empty').hidden = true;
  $('answer-result').hidden = false;
  $('winner').textContent = result.answer;
  $('result-badge').textContent = 'Live local result';
  $('latency').textContent = result.latency_ms >= 1000 ? `${(result.latency_ms / 1000).toFixed(2)} s` : `${Math.round(result.latency_ms)} ms`;
  $('scores').replaceChildren();
  result.scores.forEach(({label, score}) => {
    const row = document.createElement('div'); row.className = 'score-row';
    const header = document.createElement('div'); header.className = 'score-header';
    const name = document.createElement('span'); name.textContent = label;
    const number = document.createElement('span'); number.textContent = `${(score * 100).toFixed(1)}%`;
    header.append(name, number);
    const track = document.createElement('div'); track.className = 'score-track'; track.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('div'); fill.className = 'score-fill'; fill.style.width = `${score * 100}%`;
    track.append(fill); row.append(header, track); $('scores').append(row);
  });
  $('margin').textContent = `${(result.margin * 100).toFixed(1)} pp lead`;
  $('tokens').textContent = `${result.input_tokens} / 512 tokens`;
  $('raw-output').textContent = JSON.stringify(result, null, 2);
  $('raw-details').hidden = false;
  $('announcement').textContent = `Top choice: ${result.answer}, ${(result.scores[0].score * 100).toFixed(1)} percent. All scores are displayed.`;
}

function updateRunButton() {
  $('run').disabled = busy || !modelReady;
  $('run').textContent = busy ? 'Deciding…' : modelReady ? 'Run decision ↗' : modelLoading ? 'Loading model…' : 'Model unavailable';
}

$('decision-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (busy || !modelReady) return;
  if ($('new-option').value.trim()) {
    addOption();
    if ($('new-option').value.trim()) return;
  }
  const cleaned = options.map((x) => x.trim());
  if (cleaned.length < 2 || cleaned.some((x) => !x) || new Set(cleaned.map((x) => x.toLowerCase())).size !== cleaned.length) return showError('Add 2–12 distinct, nonempty options before running.');
  if (!$('state').value.trim() || !$('question').value.trim()) return showError('Enter both a state and a question.');
  clearResult(); const runGeneration = generation;
  busy = true; updateRunButton();
  $('model-card').classList.add('running');
  $('result-badge').textContent = 'Scoring options…';
  $('run-hint').textContent = 'Running on your CPU…';
  $('answer-card')?.setAttribute('aria-busy', 'true');
  try {
    const response = await fetch('/api/decide', {method:'POST', headers:{'Content-Type':'application/json','X-Decision-Client':'local-ui'}, body:JSON.stringify({state:$('state').value, question:$('question').value, options:cleaned}), signal:AbortSignal.timeout(120000)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'The decision could not be completed.');
    if (runGeneration === generation) { renderResult(result); $('run-hint').textContent = 'Decision complete.'; }
    else { $('run-hint').textContent = 'Inputs changed. Run again for a fresh result.'; $('result-badge').textContent = 'Run updated inputs'; }
  } catch (error) {
    if (runGeneration === generation) showError(error.name === 'TimeoutError' ? 'The request timed out. The CPU may still be finishing this call; retry shortly.' : error.message);
    $('result-badge').textContent = 'No result';
    $('run-hint').textContent = 'Check the message and try again.';
  } finally {
    busy = false; updateRunButton(); $('model-card').classList.remove('running');
    $('answer-card')?.removeAttribute('aria-busy');
  }
});

$('copy-json').addEventListener('click', async () => {
  if (!lastResult) return;
  try { await navigator.clipboard.writeText(JSON.stringify(lastResult, null, 2)); $('copy-json').textContent = 'Copied'; }
  catch { const selection = getSelection(); const range = document.createRange(); range.selectNodeContents($('raw-output')); selection.removeAllRanges(); selection.addRange(range); $('copy-json').textContent = 'Selected — press copy'; }
  setTimeout(() => { $('copy-json').textContent = 'Copy JSON'; }, 2200);
});

async function checkStatus() {
  try {
    const response = await fetch('/api/status', {signal:AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error('Local server unavailable');
    const info = await response.json(); modelReady = info.status === 'ready'; modelLoading = info.status === 'loading';
    $('runtime').dataset.status = info.status;
    $('runtime').lastElementChild.textContent = modelReady ? 'Local model · CPU' : info.status === 'loading' ? 'Loading local model' : 'Model unavailable';
    if (!modelReady && info.error) showError(info.error);
    if (info.status === 'loading') { $('run-hint').textContent = 'Loading weights into memory…'; setTimeout(checkStatus, 2000); }
    else if (modelReady) $('run-hint').textContent = 'Ready when you are.';
  } catch { modelReady = false; modelLoading = false; $('runtime').dataset.status = 'error'; $('runtime').lastElementChild.textContent = 'Server offline'; showError('Local server unavailable. Start decision-lab/run.sh and refresh this page.'); }
  updateRunButton();
}
renderOptions();
$('state-count').textContent = `${$('state').value.length} / 6000`;
checkStatus();
