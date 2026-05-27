// CodeMirror 6 editor setup -- DecL language mode + extensions.
//
// CM6 is "assemble what you need from extensions" rather than "one big
// editor object with options". We bundle:
//
//   * the DecL StreamLanguage from decl-mode.js
//   * line numbers, history, default keymap (incl. autocompletion shortcut)
//   * autocomplete with our async source
//   * a small theme that picks up the site's monospace font
//   * Ctrl/Cmd-Enter -> build, Ctrl-↑/↓ -> history nav
//
// `createEditor(host, callbacks)` returns an object with helpers the
// rest of the SPA uses (getText / setText / focus / dispatch).

import { EditorState, Compartment } from '@codemirror/state';
import { EditorView, keymap, lineNumbers, highlightActiveLine,
         drawSelection, highlightSpecialChars } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import { autocompletion, completionKeymap, acceptCompletion } from '@codemirror/autocomplete';
import { bracketMatching, syntaxHighlighting, defaultHighlightStyle } from '@codemirror/language';
import { searchKeymap, highlightSelectionMatches } from '@codemirror/search';

import { declLanguage } from './decl-mode.js';
import { declCompletionSource } from './completion.js';

// Compartment lets us swap parts of the config later without rebuilding
// the whole editor (e.g. toggling a "lint" overlay when /v1/decl/lex
// returns an error). Not used yet but cheap to set up.
const languageCompartment = new Compartment();

const editorTheme = EditorView.theme({
    '&': {
        fontSize: '14px',
        // 'background' is set by site.css via .cm-editor; leave to the cascade.
    },
    '.cm-content': {
        fontFamily: '"Cascadia Mono", Menlo, Consolas, monospace',
        padding: '10px',
        minHeight: '180px',
    },
    '.cm-scroller': { overflow: 'auto', maxHeight: '50vh' },
    '.cm-gutters': {
        backgroundColor: '#f4f6fa',
        color: '#7b8794',
        borderRight: '1px solid #dde2ea',
    },
    '.cm-activeLine':       { backgroundColor: '#f7faff' },
    '.cm-activeLineGutter': { backgroundColor: '#eef3fb' },
});

/**
 * Build the editor on `host` (a DOM node).
 *
 * Callbacks:
 *   onBuild()         -- Ctrl-Enter / Cmd-Enter
 *   onHistoryPrev()   -- Ctrl-ArrowUp / Cmd-ArrowUp
 *   onHistoryNext()   -- Ctrl-ArrowDown / Cmd-ArrowDown
 */
export function createEditor(host, callbacks = {}) {
    const customKeymap = keymap.of([
        {
            key: 'Mod-Enter',
            run: () => { callbacks.onBuild?.(); return true; },
        },
        {
            key: 'Mod-ArrowUp',
            run: () => { callbacks.onHistoryPrev?.(); return true; },
        },
        {
            key: 'Mod-ArrowDown',
            run: () => { callbacks.onHistoryNext?.(); return true; },
        },
        // Tab accepts a completion if the popup is open, otherwise inserts a tab.
        { key: 'Tab', run: acceptCompletion },
    ]);

    const state = EditorState.create({
        doc: '',
        extensions: [
            lineNumbers(),
            highlightSpecialChars(),
            history(),
            drawSelection(),
            EditorState.allowMultipleSelections.of(true),
            bracketMatching(),
            highlightActiveLine(),
            highlightSelectionMatches(),
            languageCompartment.of(declLanguage),
            syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
            autocompletion({
                override: [declCompletionSource],
                activateOnTyping: true,
                closeOnBlur: true,
            }),
            customKeymap,
            keymap.of([
                ...defaultKeymap,
                ...historyKeymap,
                ...completionKeymap,
                ...searchKeymap,
                indentWithTab,
            ]),
            editorTheme,
        ],
    });

    const view = new EditorView({ state, parent: host });

    return {
        view,
        getText: () => view.state.doc.toString(),
        setText: (text) => {
            view.dispatch({
                changes: { from: 0, to: view.state.doc.length, insert: text || '' },
            });
        },
        focus: () => view.focus(),
    };
}
