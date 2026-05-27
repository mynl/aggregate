// CodeMirror 6 StreamLanguage definition for DecL.
//
// CM6 StreamLanguage is a regex-state-machine -- the right tool for a
// small DSL where we don't want to ship a real parser to the browser.
// State machine is intentionally permissive (a few false positives on
// highlighting are fine; the server-side parser is authoritative for
// errors).
//
// Token classes (mapped to CM6 standard tags by tags.js):
//   "keyword"        -- DecL keyword (agg, sev, mixed, …)
//   "atom"           -- distortion / severity name
//   "number"         -- 100, 1.5, 1e-3, 1/4
//   "string"         -- 'x', "x", note{…}
//   "comment"        -- # to end of line
//   "bracket"        -- [ ] ( )

import { StreamLanguage } from '@codemirror/language';
import keywordData from './decl-keywords.json';

// Flatten the curated keyword bundle into a single Set for O(1)
// membership checks in the tokenizer hot path.
const KEYWORDS = new Set();
const ATOMS    = new Set();
for (const [group, words] of Object.entries(keywordData)) {
    if (group.startsWith('_')) continue;
    const target = (group === 'distortions') ? ATOMS : KEYWORDS;
    for (const w of words) target.add(w);
}

// Identifier shape mirrors decl.lark's ID terminal:
// [A-Za-z][A-Za-z0-9._:~-]* (note: NO underscore).
const ID_START   = /[A-Za-z]/;
const ID_BODY    = /[A-Za-z0-9._:~\-]/;
const DIGIT      = /[0-9]/;

function readIdent(stream) {
    stream.eat(ID_START);
    while (!stream.eol() && ID_BODY.test(stream.peek())) stream.next();
    return stream.current();
}

export const declLanguage = StreamLanguage.define({
    name: 'decl',
    startState() {
        return { inBraceString: false };
    },
    token(stream, state) {
        // ----- brace string continuation (note{...} multi-line) -----
        if (state.inBraceString) {
            while (!stream.eol()) {
                const ch = stream.next();
                if (ch === '}') { state.inBraceString = false; return 'string'; }
            }
            return 'string';
        }

        // ----- whitespace -----
        if (stream.eatSpace()) return null;

        // ----- comment -----
        if (stream.peek() === '#') {
            stream.skipToEnd();
            return 'comment';
        }

        // ----- brace string (note{...}) -----
        if (stream.peek() === '{') {
            stream.next();
            state.inBraceString = true;
            while (!stream.eol()) {
                const ch = stream.next();
                if (ch === '}') { state.inBraceString = false; return 'string'; }
            }
            return 'string';
        }

        // ----- quoted strings -----
        if (stream.peek() === '"' || stream.peek() === "'") {
            const quote = stream.next();
            while (!stream.eol()) {
                const ch = stream.next();
                if (ch === '\\') { stream.next(); continue; }
                if (ch === quote) break;
            }
            return 'string';
        }

        // ----- numbers (incl. fractions like 1/4 -- only the leading int) -----
        if (DIGIT.test(stream.peek()) || (stream.peek() === '-' && DIGIT.test(stream.string[stream.pos + 1] || ''))) {
            stream.eat('-');
            while (!stream.eol() && DIGIT.test(stream.peek())) stream.next();
            // optional fractional part
            if (stream.peek() === '.' && DIGIT.test(stream.string[stream.pos + 1] || '')) {
                stream.next();
                while (!stream.eol() && DIGIT.test(stream.peek())) stream.next();
            }
            // optional exponent
            if (stream.peek() === 'e' || stream.peek() === 'E') {
                stream.next();
                if (stream.peek() === '+' || stream.peek() === '-') stream.next();
                while (!stream.eol() && DIGIT.test(stream.peek())) stream.next();
            }
            return 'number';
        }

        // ----- brackets -----
        if ('()[]'.includes(stream.peek())) {
            stream.next();
            return 'bracket';
        }

        // ----- identifier / keyword -----
        if (ID_START.test(stream.peek())) {
            const word = readIdent(stream);
            if (KEYWORDS.has(word.toLowerCase())) return 'keyword';
            if (ATOMS.has(word.toLowerCase())) return 'atom';
            return null;
        }

        // ----- fallback: consume one char so we don't loop -----
        stream.next();
        return null;
    },
    languageData: {
        commentTokens: { line: '#' },
        // Surface a hint to CM6's autocomplete that '.' is part of identifiers.
        wordChars: '._:~-',
    },
});

export { KEYWORDS as DECL_KEYWORDS, ATOMS as DECL_ATOMS };
