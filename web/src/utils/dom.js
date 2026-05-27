// Minimal DOM helpers. Keeps main.js readable without pulling in a
// framework. All functions accept and return real DOM nodes.

/** Quick alias for document.getElementById. */
export const $ = (id) => document.getElementById(id);

/**
 * Create a DOM element with optional attributes/classes and children.
 *
 *   el('button', { className: 'btn btn-primary', onClick: handler }, 'Build')
 *   el('div',    { className: 'output' }, child1, child2)
 *
 * Children can be strings (auto-wrapped as text nodes), other nodes,
 * or arrays of either.
 */
export function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
        if (v === undefined || v === null) continue;
        if (k === 'className') node.className = v;
        else if (k === 'dataset') Object.assign(node.dataset, v);
        else if (k.startsWith('on') && typeof v === 'function') {
            node.addEventListener(k.slice(2).toLowerCase(), v);
        } else {
            node.setAttribute(k, v);
        }
    }
    appendAll(node, children);
    return node;
}

function appendAll(parent, items) {
    for (const item of items) {
        if (item === null || item === undefined || item === false) continue;
        if (Array.isArray(item)) { appendAll(parent, item); continue; }
        if (item instanceof Node) { parent.appendChild(item); continue; }
        parent.appendChild(document.createTextNode(String(item)));
    }
}

/** Replace a container's children with the given nodes. */
export function replaceChildren(container, ...nodes) {
    container.replaceChildren(...nodes.flat().filter(Boolean));
}

/** Remove every child without churning innerHTML. */
export function empty(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
}
