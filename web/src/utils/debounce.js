// Standard trailing-edge debounce.
// Used by the completion source to avoid hammering /v1/decl/complete
// on every keystroke.

export function debounce(fn, ms) {
    let h = null;
    return (...args) => {
        if (h !== null) clearTimeout(h);
        h = setTimeout(() => { h = null; fn(...args); }, ms);
    };
}
