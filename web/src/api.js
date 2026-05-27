// Fetch wrappers for the /v1/* endpoints.
//
// One small layer over the browser's fetch:
//
// * _json() does the JSON serialize/deserialize dance and turns any
//   non-2xx response into an ApiError that carries the parsed body.
//   The body shape for parse errors is Plan B's ErrorReport, which
//   error-pane.js knows how to render.
// * api.plotUrl() returns a string (not a Promise) -- callers stick
//   it straight into <img src="…"> and let the browser fetch it.

import { API_BASE } from './config.js';

export class ApiError extends Error {
    constructor(status, body) {
        super((body && body.message) || `HTTP ${status}`);
        this.status = status;
        this.body = body;          // raw response body, JSON-parsed if possible
    }
}

function qs(params) {
    const pairs = [];
    for (const [k, v] of Object.entries(params)) {
        if (v === undefined || v === null || v === '') continue;
        pairs.push(`${encodeURIComponent(k)}=${encodeURIComponent(v)}`);
    }
    return pairs.join('&');
}

async function _json(method, path, body) {
    const r = await fetch(API_BASE + path, {
        method,
        headers: body ? { 'Content-Type': 'application/json' } : {},
        body: body ? JSON.stringify(body) : undefined,
    });
    // Try to parse as JSON whether ok or not -- 4xx/5xx still carry
    // the structured error body for the SPA to render.
    let data = null;
    const text = await r.text();
    if (text) {
        try { data = JSON.parse(text); }
        catch { data = { message: text }; }
    }
    if (!r.ok) throw new ApiError(r.status, data);
    return data;
}

export const api = {
    // Builds + cache
    build:        (decl, opts = {}) => _json('POST', '/v1/objects', { decl, ...opts }),
    list:         ()                => _json('GET',  '/v1/objects'),
    manifest:     (id)              => _json('GET',  `/v1/objects/${id}`),
    drop:         (id)              => _json('DELETE', `/v1/objects/${id}`),

    // Per-button data
    info:         (id)              => _json('GET',  `/v1/objects/${id}/info`),
    description:  (id)              => _json('GET',  `/v1/objects/${id}/description`),
    stats_df:     (id)              => _json('GET',  `/v1/objects/${id}/stats_df`),
    density_df:   (id, p = {})      => _json('GET',  `/v1/objects/${id}/density_df?${qs(p)}`),
    kappa:        (id, p = {})      => _json('GET',  `/v1/objects/${id}/kappa?${qs(p)}`),

    // Pricing
    pricing_at:   (id, body)        => _json('POST', `/v1/objects/${id}/pricing_at`, body),

    // DecL editor support
    complete:     (decl, cursor)    => _json('POST', '/v1/decl/complete', { decl, cursor }),
    lex:          (decl)            => _json('POST', '/v1/decl/lex', { decl }),

    // Metadata
    examples:     ()                => _json('GET',  '/v1/examples'),
    meta:         ()                => _json('GET',  '/v1/meta'),
    health:       ()                => _json('GET',  '/v1/health'),

    /** Plot URL: handed straight to an <img>. */
    plotUrl:      (id, p = {})      => `${API_BASE}/v1/objects/${id}/plot?${qs(p)}`,
};
