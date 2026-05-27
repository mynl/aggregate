// Numeric formatting picked by magnitude.
//
// The aggregate library frequently produces values that span many
// decades inside a single table (loss = 0..1e7, probability = 1e-12..1).
// A single format string can't render both nicely, so fmt() picks per
// cell.

const INT_PATTERN = /^-?\d+$/;

/**
 * Format one value for display.
 *
 *   null / undefined / NaN  -> ""
 *   strings                  -> verbatim
 *   integers (|x| < 1e9)     -> "1,234"
 *   |x| in [1e-3, 1e6)       -> 6-significant-digit fixed
 *   else                     -> "1.234e+05" scientific
 */
export function fmt(value) {
    if (value === null || value === undefined) return '';
    if (typeof value === 'string') {
        // Already-formatted strings come through stats_df sometimes.
        return INT_PATTERN.test(value) ? value : value;
    }
    if (typeof value !== 'number') return String(value);
    if (!Number.isFinite(value)) return '';
    if (value === 0) return '0';

    const abs = Math.abs(value);
    if (Number.isInteger(value) && abs < 1e9) {
        return value.toLocaleString('en-US');
    }
    if (abs >= 1e-3 && abs < 1e6) {
        // 6 sig figs is a good readable default.
        return value.toPrecision(6).replace(/\.?0+$/, '');
    }
    // Scientific. toExponential(4) -> "1.2345e+05".
    return value.toExponential(4);
}

/** True if the value should align right in a table (numeric-ish). */
export function isNumeric(value) {
    if (typeof value === 'number') return Number.isFinite(value);
    if (typeof value === 'string' && value.trim() && !Number.isNaN(Number(value))) return true;
    return false;
}
