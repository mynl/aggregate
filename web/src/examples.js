// Examples dropdown -- loaded once from /v1/examples.
//
// The api returns categories grouped by letter (A, B, …) each with an
// items list of {name, decl, note?}. We render the dropdown with a
// header per category and one item per example. Clicking loads the
// DecL text into the editor (no auto-build).

import { api } from './api.js';
import { el, empty } from './utils/dom.js';

let cached = null;

/** Populate the dropdown menu element. */
export async function mountExamples(menuEl, onPick) {
    let payload;
    try {
        payload = cached ?? (cached = await api.examples());
    } catch (err) {
        empty(menuEl);
        menuEl.appendChild(el('li', {}, el('span', {
            className: 'dropdown-item-text text-danger small',
        }, `failed to load examples: ${err.message}`)));
        return;
    }

    empty(menuEl);
    const categories = payload.categories || [];

    if (!categories.length) {
        menuEl.appendChild(el('li', {}, el('span', {
            className: 'dropdown-item-text text-muted small',
        }, 'no examples')));
        return;
    }

    let first = true;
    for (const cat of categories) {
        if (!first) {
            menuEl.appendChild(el('li', {}, el('hr', { className: 'dropdown-divider' })));
        }
        first = false;
        menuEl.appendChild(el('li', {}, el('h6', {
            className: 'dropdown-header',
        }, `${cat.letter}. ${cat.title}`)));

        for (const item of cat.items || []) {
            const a = el('a', {
                className: 'dropdown-item',
                href:      '#',
                title:     item.note || '',
                onClick:   (ev) => {
                    ev.preventDefault();
                    onPick?.(item);
                },
            }, item.name);
            menuEl.appendChild(el('li', {}, a));
        }
    }
}
