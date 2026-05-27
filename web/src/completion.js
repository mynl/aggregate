// CodeMirror 6 autocomplete source backed by /v1/decl/complete.
//
// CM6 calls the source function whenever the user is typing inside
// what could be a completable token. We extract the cursor position +
// current document text, hand them to the api, and translate the
// server's {label, terminal, kind} entries into CM6 Completion shapes.
//
// On any error (network down, parse failure server-side) we fall back
// to the static keyword pool from decl-keywords.json so the editor
// stays usable offline.

import { api } from './api.js';
import { DECL_KEYWORDS, DECL_ATOMS } from './decl-mode.js';

const STATIC_POOL = [
    ...[...DECL_KEYWORDS].map(label => ({ label, type: 'keyword' })),
    ...[...DECL_ATOMS].map(label => ({ label, type: 'atom' })),
];

/**
 * CM6 CompletionSource. `context.matchBefore(/.../) ` returns
 * { from, to, text } for the identifier-shaped token ending at cursor.
 */
export async function declCompletionSource(context) {
    const word = context.matchBefore(/[\w.:~-]*/);
    // Don't pop the menu on every cursor move into empty space unless
    // the user explicitly invoked it (Ctrl-Space).
    if (!word || (word.from === word.to && !context.explicit)) return null;

    const decl = context.state.doc.toString();
    const cursor = context.pos;

    let serverOptions = null;
    try {
        const res = await api.complete(decl, cursor);
        if (res && Array.isArray(res.completions)) {
            serverOptions = res.completions.map(c => ({
                label: c.label,
                type: c.kind || 'keyword',
                // boost server-supplied entries so they outrank the
                // local fallback in mixed lists.
                boost: 5,
            }));
        }
    } catch {
        serverOptions = null;
    }

    // Always include the static pool too -- the server returns only
    // grammar-suggested keywords, but users frequently want to type a
    // distortion name that isn't currently grammar-legal.
    const merged = dedupeByLabel([...(serverOptions || []), ...STATIC_POOL]);

    return {
        from: word.from,
        options: merged,
        // Let CM6 filter; we just supply the candidate set.
        validFor: /^[\w.:~-]*$/,
    };
}

function dedupeByLabel(items) {
    const seen = new Set();
    const out = [];
    for (const it of items) {
        if (seen.has(it.label)) continue;
        seen.add(it.label);
        out.push(it);
    }
    return out;
}
