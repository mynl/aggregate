// Per-button action dispatch.
//
// Each entry in ACTIONS returns a function (currentId, currentKind)
// that fetches from the api, renders the result, and replaces the
// contents of #output. Buttons that need a Portfolio (kappa, pricing
// with allocations) check currentKind and surface a friendly notice
// rather than letting the api 400.

import { api, ApiError } from './api.js';
import { el, empty } from './utils/dom.js';
import { renderTable, renderInfo } from './renderers.js';
import { renderPlot } from './plot-pane.js';
import { renderError } from './error-pane.js';
import { renderPricingForm } from './pricing-pane.js';

const PLOT_OPTS = { kind: 'density', format: 'svg' };

export const ACTIONS = {
    info: async (id) => renderInfo(await api.info(id)),
    describe: async (id) => renderTable(await api.description(id), { caption: 'describe' }),
    stats_df: async (id) => renderTable(await api.stats_df(id), { caption: 'stats_df' }),
    density_df: async (id) => renderTable(
        await api.density_df(id, { downsample: 200 }),
        { caption: 'density_df (downsampled to 200 rows)' },
    ),
    kappa: async (id, kind) => {
        if (kind !== 'port') {
            return el('div', { className: 'alert alert-info' },
                'kappa needs a Portfolio. Build with `port …` to enable.');
        }
        return renderTable(await api.kappa(id, { downsample: 200 }),
            { caption: 'kappa (downsampled to 200 rows)' });
    },
    plot: (id) => renderPlot(api.plotUrl(id, { kind: 'density', format: 'svg' }), 'density plot'),
    cdf:  (id) => renderPlot(api.plotUrl(id, { kind: 'cdf',     format: 'svg' }), 'cdf plot'),
    qq:   (id) => renderPlot(api.plotUrl(id, { kind: 'qq',      format: 'svg' }), 'qq plot'),
};

/**
 * Run an action by name and put the result into `outputEl`.
 * Handles button enable/disable, api errors, and unknown actions.
 */
export async function runAction(name, ctx, outputEl, btnEl) {
    if (!ctx.currentId) {
        empty(outputEl);
        outputEl.appendChild(el('div', { className: 'alert alert-info' },
            'Build an object first (press Ctrl-Enter or click Build).'));
        return;
    }

    // Pricing isn't a fire-and-forget call -- it renders an
    // interactive form, so it's special-cased.
    if (name === 'pricing') {
        if (ctx.currentKind !== 'port') {
            empty(outputEl);
            outputEl.appendChild(el('div', { className: 'alert alert-info' },
                'pricing_at needs a Portfolio. Build with `port …` to enable.'));
            return;
        }
        empty(outputEl);
        renderPricingForm(ctx.currentId, outputEl);
        return;
    }

    const handler = ACTIONS[name];
    if (!handler) {
        outputEl.appendChild(el('div', { className: 'alert alert-danger' },
            `Unknown action: ${name}`));
        return;
    }

    if (btnEl) { btnEl.disabled = true; btnEl.classList.add('disabled'); }
    try {
        const node = await handler(ctx.currentId, ctx.currentKind);
        empty(outputEl);
        if (node) outputEl.appendChild(node);
    } catch (err) {
        empty(outputEl);
        if (err instanceof ApiError) outputEl.appendChild(renderError(err));
        else outputEl.appendChild(el('div', { className: 'alert alert-danger' }, err.message));
    } finally {
        if (btnEl) { btnEl.disabled = false; btnEl.classList.remove('disabled'); }
    }
}
