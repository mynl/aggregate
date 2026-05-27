// Tiny localStorage-backed history of successful DecL programs.
//
// Stored shape:
//   { entries: ["agg X …", "agg Y …", …], cursor: -1 }
//
// cursor === -1 means "no nav in progress" (next prev() returns the
// most recent entry). Each successful build pushes the program onto
// the front, dedup-against-latest, capped at MAX entries.
//
// Browser-local by design -- per-user history would need an account.

const KEY  = 'aggregate-web:history';
const MAX  = 20;

function load() {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return { entries: [], cursor: -1 };
        const obj = JSON.parse(raw);
        if (!Array.isArray(obj.entries)) return { entries: [], cursor: -1 };
        return obj;
    } catch { return { entries: [], cursor: -1 }; }
}

function save(state) {
    try { localStorage.setItem(KEY, JSON.stringify(state)); }
    catch { /* storage full or disabled -- silent drop */ }
}

const state = load();

/** Record a successful program. Dedup'd against the most-recent entry. */
export function record(text) {
    const trimmed = (text || '').trim();
    if (!trimmed) return;
    if (state.entries[0] === trimmed) return;
    state.entries.unshift(trimmed);
    if (state.entries.length > MAX) state.entries.length = MAX;
    state.cursor = -1;
    save(state);
}

/** Step back one entry; returns text or null if no further history. */
export function prev() {
    if (state.entries.length === 0) return null;
    if (state.cursor + 1 >= state.entries.length) return null;
    state.cursor += 1;
    save(state);
    return state.entries[state.cursor];
}

/** Step forward one entry; null when we've walked off the front. */
export function next() {
    if (state.cursor <= 0) {
        state.cursor = -1;
        save(state);
        return null;
    }
    state.cursor -= 1;
    save(state);
    return state.entries[state.cursor];
}

/** Reset the nav cursor (called after the user types a new char). */
export function resetCursor() {
    if (state.cursor !== -1) {
        state.cursor = -1;
        save(state);
    }
}
