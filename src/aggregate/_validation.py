"""Validation: the shared explanation formatter and the validation thresholds.

Extracted from ``distributions.py`` / ``_aggregate.py`` (Phase 1b, shared concerns). A leaf/near-leaf: it never imports ``_aggregate``/``_portfolio`` (takes plain data / a distribution object), which is what lets Portfolio reuse it in P4.
"""

from .constants import Validation
from .config import get_settings


# VALIDATION_NOISE: absolute dust floor below which a quantity is treated as
# exact zero / numerical noise.
VALIDATION_NOISE = get_settings().validation.noise


# ALIASING_RATIO: the ALIASING flag fires when the agg-mean relative error
# exceeds this multiple of the sev-mean relative error (FFT wrap-around).
ALIASING_RATIO = get_settings().validation.aliasing_ratio


def explain_validation(rv):
    """
    Explain the validation result rv.
    Don't over report: if you fail CV don't need to be told you fail Skew too.

    Under reinsurance the realised view has no independent theoretical and
    cannot be validated, but the SUBJECT (gross) view was validated under
    the hood (§1.3 of the aggregate refactor plan). The message reports
    that subject status alongside the ``reinsurance`` marker, so the user
    can tell whether the underlying gross object is sound.
    """
    if rv == Validation.NOT_UNREASONABLE:
        return "not unreasonable"
    if rv & Validation.NOT_UPDATED:
        return "n/a, not updated"
    # Collect failures from the SEV/AGG/ALIASING flags (suppressing higher
    # moments once a lower-order moment already failed).
    parts = []
    if rv & Validation.SEV_MEAN:
        parts.append('sev mean')
    if rv & Validation.AGG_MEAN:
        parts.append('agg mean')
    if rv & Validation.ALIASING:
        parts.append('agg mean error >> sev, possible aliasing; try larger bs')
    if not (rv & Validation.SEV_MEAN) and (rv & Validation.SEV_CV):
        parts.append('sev cv')
    if not (rv & Validation.AGG_MEAN) and (rv & Validation.AGG_CV):
        parts.append('agg cv')
    if not (rv & Validation.SEV_CV) and (rv & Validation.SEV_SKEW):
        parts.append('sev skew')
    if not (rv & Validation.AGG_CV) and (rv & Validation.AGG_SKEW):
        parts.append('agg skew')
    explanation = ', '.join(parts)
    if rv & Validation.REINSURANCE:
        if explanation:
            return f'reinsurance; subject fails {explanation}'
        return 'reinsurance; subject not unreasonable'
    return f'fails {explanation}'
