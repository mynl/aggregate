// Format Plan B's ErrorReport JSON for human reading.
//
// ErrorReport shape (from src/aggregate/parser_errors.py):
//
//   {
//     "kind":        "parse" | "lex" | "limit",
//     "message":     "Unexpected identifier 'mixedd'.",
//     "line":        1,
//     "column":      41,
//     "got":         "mixedd",
//     "expected":    ["MIXED", "NOTE", "$END"],
//     "expected_labels": ["'mixed'", "'note'", "end of input"],
//     "suggestions": ["mixed"],
//     "source":      "agg X 100 claims sev lognorm 100 cv 2 mixedd poisson 0.5",
//     "caret":       "                                        ^^^^^^",
//   }
//
// We render it as a monospaced block with the caret line under the
// source. Suggestions become a "did you mean" line in italics.

import { el } from './utils/dom.js';

export function renderError(err) {
    const body = (err && err.body) || {};
    const detail = body.detail || body || {};

    // Generic non-parse fallback (HTTP 500, network failure, etc.):
    if (!detail.line && !detail.message) {
        return el('div', { className: 'alert alert-danger', role: 'alert' },
            err?.message || 'Request failed',
        );
    }

    const parts = [];

    if (detail.line && detail.column) {
        parts.push(el('div', { className: 'small text-muted' },
            `Parse error at line ${detail.line}, column ${detail.column}.`));
    }

    if (detail.source) {
        // Render as a single <pre>, source on one line, caret beneath.
        const pre = el('pre', { className: 'agg-error-source mb-2' });
        pre.appendChild(el('code', {}, detail.source + '\n'));
        if (detail.caret) {
            pre.appendChild(el('code', { className: 'agg-error-caret' }, detail.caret));
        }
        parts.push(pre);
    }

    if (detail.message) {
        parts.push(el('div', { className: 'mb-1' }, detail.message));
    }

    if (Array.isArray(detail.suggestions) && detail.suggestions.length) {
        parts.push(el('div', { className: 'fst-italic mb-2' },
            'Did you mean: ',
            detail.suggestions.join(', '),
            '?'));
    }

    const labels = detail.expected_labels || detail.expected || [];
    if (labels.length) {
        parts.push(el('div', { className: 'small text-muted' },
            'Expected: ', labels.join(', '), '.'));
    }

    return el('div', { className: 'alert alert-warning agg-error-pane', role: 'alert' },
        ...parts,
    );
}
