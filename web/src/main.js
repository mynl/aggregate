// SPA entry point. Loaded by index.html as a module.
//
// Responsibilities:
//   * import Bootstrap JS (dropdowns rely on data-bs-toggle)
//   * import our CSS so Vite bundles it into the output
//   * construct the editor + wire callbacks
//   * mount the Examples dropdown
//   * wire the Build button + every per-action button to actions.js
//   * surface api/version in the navbar and listen for hotkeys

// ---- Bootstrap (JS + CSS) ----
import 'bootstrap/dist/css/bootstrap.min.css';
import * as bootstrap from 'bootstrap';            // exposes window.bootstrap-style helpers
// Eager-import to ensure data-bs-* handlers work on inline markup.
// We don't use the imported value directly; the side-effect mounts
// auto-init handlers for dropdowns, tooltips, etc.

// ---- Site styles (loaded after Bootstrap so we win cascade ties) ----
import './styles/site.css';
import './styles/cm6.css';

// ---- App modules ----
import { api, ApiError } from './api.js';
import { createEditor } from './editor.js';
import { mountExamples } from './examples.js';
import { runAction } from './actions.js';
import { renderBuildBanner } from './renderers.js';
import { renderError } from './error-pane.js';
import * as history from './history.js';
import { $, el, empty } from './utils/dom.js';

// Activate any Bootstrap tooltips on the page (used for example notes).
document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(
    (t) => new bootstrap.Tooltip(t),
);

// ----------------------------------------------------------------------
// Module state
// ----------------------------------------------------------------------
const ctx = {
    currentId: null,
    currentKind: null,
    currentName: null,
};

const outputEl = $('output');
const buildBtn = $('build-btn');

// ----------------------------------------------------------------------
// Editor
// ----------------------------------------------------------------------
const editor = createEditor($('editor-host'), {
    onBuild:        () => build(),
    onHistoryPrev:  () => navigateHistory('prev'),
    onHistoryNext:  () => navigateHistory('next'),
});

editor.setText('agg Dice dfreq [3] dsev [1:6]\n');
editor.focus();

// Reset the history cursor whenever the user types something new so
// arrow-up resumes from the latest entry on the next press.
editor.view.dom.addEventListener('keydown', (ev) => {
    if (!ev.ctrlKey && !ev.metaKey) history.resetCursor();
});

function navigateHistory(dir) {
    const text = dir === 'prev' ? history.prev() : history.next();
    if (text !== null) editor.setText(text);
}

// ----------------------------------------------------------------------
// Build
// ----------------------------------------------------------------------
async function build() {
    const decl = editor.getText().trim();
    if (!decl) return;

    buildBtn.disabled = true;
    buildBtn.textContent = 'Building…';
    empty(outputEl);

    // Optional log2 / bs from the Options dropdown.
    const log2 = parseInt($('opt-log2').value, 10);
    const bs   = parseFloat($('opt-bs').value);
    const opts = {};
    if (Number.isFinite(log2)) opts.log2 = log2;
    if (Number.isFinite(bs) && bs > 0) opts.bs = bs;

    try {
        const res = await api.build(decl, opts);
        ctx.currentId   = res.id;
        ctx.currentKind = res.kind;
        ctx.currentName = res.name;
        outputEl.appendChild(renderBuildBanner(res));
        outputEl.appendChild(el('div', { className: 'text-muted small' },
            'Pick an action button above to inspect this object.'));
        history.record(decl);
        updateActionEnablement();
    } catch (err) {
        ctx.currentId = ctx.currentKind = ctx.currentName = null;
        if (err instanceof ApiError) outputEl.appendChild(renderError(err));
        else outputEl.appendChild(el('div', { className: 'alert alert-danger' }, err.message));
        updateActionEnablement();
    } finally {
        buildBtn.disabled = false;
        buildBtn.innerHTML = 'Build <span class="text-white-50 small ms-1">Ctrl-Enter</span>';
    }
}

buildBtn.addEventListener('click', build);

// ----------------------------------------------------------------------
// Action buttons
// ----------------------------------------------------------------------
document.querySelectorAll('[data-action]').forEach((btn) => {
    btn.addEventListener('click', () => {
        runAction(btn.dataset.action, ctx, outputEl, btn);
    });
});

function updateActionEnablement() {
    const isPort = ctx.currentKind === 'port';
    document.querySelectorAll('[data-action="kappa"], [data-action="pricing"]')
        .forEach(b => { b.classList.toggle('disabled', !isPort); });
}

// ----------------------------------------------------------------------
// Examples dropdown
// ----------------------------------------------------------------------
mountExamples($('examples-menu'), (item) => {
    editor.setText(item.decl);
    editor.focus();
});

// ----------------------------------------------------------------------
// Version + defaults from /v1/meta
// ----------------------------------------------------------------------
api.meta().then((meta) => {
    $('version-tag').textContent = `aggregate v${meta.version}`;
    if ($('opt-log2').value === '') {
        $('opt-log2').placeholder = `default ${meta.log2_default}`;
        $('opt-log2').min = '4';
        $('opt-log2').max = String(meta.log2_cap);
    }
}).catch(() => {
    $('version-tag').textContent = '(api offline)';
});

updateActionEnablement();
