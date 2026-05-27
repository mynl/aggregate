// Image renderer for the plot endpoints.
//
// The api returns image bytes (SVG by default, PNG if format=png is
// requested). We hand the URL to an <img>; the browser does the
// fetching, with caching/etag behavior we don't have to manage.
//
// One small subtlety: when the SVG file is large (Tweedie plots can
// hit 50 kB+), the browser may render it before all of it is parsed.
// `loading="eager"` is the default for <img>, which is what we want.

import { el } from './utils/dom.js';

export function renderPlot(url, alt = 'aggregate plot') {
    const img = el('img', {
        src: url,
        alt,
        className: 'img-fluid border rounded bg-white p-2',
    });
    return el('div', { className: 'agg-plot-pane text-center' }, img);
}
