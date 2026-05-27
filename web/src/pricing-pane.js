// Inline pricing form -- renders into #output when the user picks
// "pricing at…" from the more-dropdown.
//
// The api supports two dispatch flavors:
//   * constant cost-of-capital: {p, ccoc}     -> Portfolio.price_ccoc
//   * calibrated distortion:    {p|a, name}   -> Portfolio.pricing_at
// The form exposes p, ccoc, and a free-text distortion name; submit
// behavior picks the dispatch based on whether 'distortion' is filled.

import { api, ApiError } from './api.js';
import { el, empty } from './utils/dom.js';
import { renderTable, renderInfo } from './renderers.js';
import { renderError } from './error-pane.js';

export function renderPricingForm(currentId, container) {
    empty(container);

    const pInput        = el('input', { name: 'p',          className: 'form-control', value: '0.995' });
    const ccocInput     = el('input', { name: 'ccoc',       className: 'form-control', value: '0.10' });
    const distInput     = el('input', { name: 'distortion', className: 'form-control', placeholder: '(optional)' });
    const aInput        = el('input', { name: 'a',          className: 'form-control', placeholder: '(optional)' });
    const submitBtn     = el('button', { type: 'submit', className: 'btn btn-primary' }, 'Calculate');
    const resultDiv     = el('div', { id: 'pricing-result', className: 'mt-3' });

    const form = el('form', { className: 'row g-3 align-items-end' },
        labelled('p (VaR prob.)',  pInput),
        labelled('ccoc',           ccocInput),
        labelled('a (assets)',     aInput),
        labelled('distortion',     distInput),
        el('div', { className: 'col-sm-2' }, submitBtn),
    );

    form.addEventListener('submit', async (ev) => {
        ev.preventDefault();
        submitBtn.disabled = true;
        submitBtn.textContent = '…';
        empty(resultDiv);

        const body = {};
        const p   = parseFloat(pInput.value);     if (Number.isFinite(p))    body.p   = p;
        const a   = parseFloat(aInput.value);     if (Number.isFinite(a))    body.a   = a;
        const cc  = parseFloat(ccocInput.value);  if (Number.isFinite(cc))   body.ccoc = cc;
        if (distInput.value.trim()) body.distortion = distInput.value.trim();

        try {
            const res = await api.pricing_at(currentId, body);
            resultDiv.appendChild(renderPricingResult(res));
        } catch (err) {
            if (err instanceof ApiError) resultDiv.appendChild(renderError(err));
            else resultDiv.appendChild(el('div', { className: 'alert alert-danger' }, err.message));
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Calculate';
        }
    });

    container.appendChild(form);
    container.appendChild(resultDiv);
}

function labelled(text, input) {
    input.classList.add('form-control-sm');
    return el('div', { className: 'col-sm-2' },
        el('label', { className: 'form-label small mb-1' }, text),
        input,
    );
}

function renderPricingResult(payload) {
    const wrap = document.createDocumentFragment();
    // Headline row: a, p, ccoc, distortion
    const headline = el('dl', { className: 'row mb-2 small' });
    for (const k of ['a', 'p', 'ccoc', 'distortion']) {
        if (payload[k] === null || payload[k] === undefined) continue;
        headline.appendChild(el('dt', { className: 'col-sm-2 text-muted' }, k));
        headline.appendChild(el('dd', { className: 'col-sm-10 font-monospace mb-0' }, String(payload[k])));
    }
    wrap.appendChild(headline);

    // Convert rows (list of dict) into a FrameResponse-ish (columns, rows).
    if (Array.isArray(payload.rows) && payload.rows.length) {
        const cols = Object.keys(payload.rows[0]);
        const rows = payload.rows.map(r => cols.map(c => r[c]));
        wrap.appendChild(renderTable({ columns: cols, rows }));
    } else {
        wrap.appendChild(el('div', { className: 'text-muted small' }, '(no rows)'));
    }
    return wrap;
}
