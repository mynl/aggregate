"""User-editable settings for ``aggregate``.

This module is the single source of truth for the library's tunable defaults
(grid sizing, default databases, discretization schemes, validation tolerances)
and for the "where things live" path names. Values cascade, lowest priority
first:

1. the dataclass field defaults below (always present);
2. a user TOML file at ``~/.aggregate/config.toml`` (never auto-created);
3. ``AGGREGATE_*`` environment variables (a small allow-list);
4. explicit ``build(...)`` / constructor keyword arguments (handled at the call
   sites, the highest-priority layer — not seen here).

The settings object is a lazily-built, cached singleton (:func:`get_settings`);
the file is read **once per session**, so behaviour never changes silently
mid-run. To pick up an edited file, set ``AGGREGATE_*`` before importing
``aggregate``, pass keyword arguments at the call site, or call
:func:`reload_settings`.

Design notes
------------
This is a **leaf** module: it imports only the standard library, so any
``aggregate`` module may import it without risking an import cycle. It owns the
path names (``USER_DIR_NAME`` / ``PACKAGE_DATA_DIR`` / ``TEST_SUITE_FILENAME``)
because it is the authority on where the library reads and writes. The narrow
sibling :mod:`aggregate.constants` keeps only the ``Validation`` flag enum, the
``DefectiveDistributionWarning`` class, and the structural reinsurance column
labels -- types and structural keys that are *not* user settings.

No third-party TOML reader is used: ``tomllib`` is read directly from the
standard library (Python >= 3.11).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from importlib.resources import files
import logging
import os
from pathlib import Path
import tomllib
import warnings

logger = logging.getLogger(__name__)

__all__ = [
    'BuildSettings', 'DiscretizationSettings', 'ValidationSettings',
    'MultivariateSettings', 'LabelSettings', 'Settings',
    'get_settings', 'load_settings', 'reload_settings',
    'config_path', 'user_dir', 'write_default_config', 'describe_settings',
    'USER_DIR_NAME', 'PACKAGE_DATA_DIR', 'TEST_SUITE_FILENAME',
]

# --- path names (the "where things live" authority) ------------------------
# User-local data directory, under Path.home(): the new config file, any user
# style override, and the user's own *.agg databases live here.
USER_DIR_NAME = '.aggregate'
# Subdirectory inside the installed `aggregate` package holding bundled .agg files.
PACKAGE_DATA_DIR = 'agg'
# The canonical bundled test suite filename (lives in PACKAGE_DATA_DIR).
TEST_SUITE_FILENAME = 'test_suite.agg'

# Name of the user config file (inside USER_DIR_NAME) and the shipped template.
_CONFIG_FILENAME = 'config.toml'
_TEMPLATE_RESOURCE = ('data', 'config.default.toml')


# ---------------------------------------------------------------------------
# Settings dataclasses. The field defaults here are the single source of truth
# for every tunable wired in Phase 1. Frozen so a Settings object is an
# immutable snapshot of the resolved configuration.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildSettings:
    """Defaults for :class:`~aggregate.underwriter.Underwriter` and ``build``.

    Parameters
    ----------
    log2 : int
        log2 of the number of buckets in the discrete representation
        (``2**log2`` buckets). Unifies the historical 10-vs-16 split between
        ``Underwriter.__init__`` and the module-level ``build``.
    bs : float
        Bucket size; ``0`` lets the object recommend one.
    padding : int
        FFT zero-padding (powers of two of extra space) used in ``update``.
    normalize : bool
        Whether ``update`` renormalises the discretized severity.
    databases : tuple of str
        Database name(s) loaded on construction. ``"test_suite"`` keeps the
        historical ``build`` knowledge base; ``"default"`` would load every
        bundled file, ``"all"`` bundled plus user.
    update : bool
        Whether constructed objects are auto-updated.
    """

    log2: int = 16
    bs: float = 0.0
    padding: int = 1
    normalize: bool = True
    databases: tuple[str, ...] = ('test_suite',)
    update: bool = True


@dataclass(frozen=True)
class DiscretizationSettings:
    """Defaults controlling how distributions are placed on the FFT grid.

    Parameters
    ----------
    reins_bucket : {'linear', 'nearest'}
        Scheme for rebucketing reinsurance net/ceded distributions onto the
        grid. ``'linear'`` splits off-grid mass across the two bracketing
        buckets (preserves the first moment); ``'nearest'`` rounds.
    dsev_bucket : {'linear', 'nearest'}
        Scheme for placing discrete-severity atoms (``dsev`` / ``dhistogram``
        / ``fixed``) onto the grid. Same meaning as ``reins_bucket``.
    bucket_sizing_p : float
        Percentile of the fitted distribution used by ``recommend_bucket`` to
        size ``bs`` (formerly ``RECOMMEND_P`` / the ``recommend_p`` kwarg). If
        ``> 1`` it is read as a number of nines, i.e. ``1 - 10**-p``.
    window_nines : int
        Number of nines defining the automatic 1-D aggregate output window:
        it spans roughly the ``10**-window_nines .. 1 - 10**-window_nines``
        quantiles (formerly ``WINDOW_NINES``). This is the **protected** edge
        coverage -- the tail the sign convention cares about (the right tail
        for a loss, the left for a payoff) -- sized deep to avoid clipping.
    window_nines_trim : int
        Number of nines for the **unprotected** edge of a windowed (two-sided)
        aggregate: the cheap tail the convention does not price (the left for a
        loss, the right for a payoff). Shallower than ``window_nines`` so the
        window trims dead space on that side rather than wasting grid; only
        affects a *windowed* book (mass band clears 0), never an ordinary
        0-based one. See ``dev/plan-bucket-window-2.md`` §1A (Q1/Q2).
    window_pad_skew : float
        Padding-balance skew ``Delta`` in ``[0, 0.5)``. Once a windowed band is
        placed, the power-of-2 slack is split with fraction ``f = 0.5 -/+ Delta``
        below the band (``0.5 - Delta`` for a loss -> more room on the right
        where the priced tail lives; ``0.5 + Delta`` for a payoff). ``0`` centres
        the band exactly; the legacy behaviour was ``f = 0`` (all slack above).
        See ``dev/plan-bucket-window-2.md`` §1A (Q4).
    sbj_tail_floor : float
        Numerical-depth floor for the single-big-jump (SBJ) extent floor. To
        cover the aggregate to ``p*`` the severity is probed at the deeper
        ``p** = 1 - (1 - p*)/E[N]``; for a large ``E[N]`` this pushes
        ``1 - p**`` past double precision (so ``q_X(p**)`` would be ``inf``).
        The lower tail ``1 - p**`` is floored here at the deepest level the
        severity is numerically meaningful (default ``1e-14``, about
        ``window_nines + 2`` nines). See ``dev/plan-bucket-window-2.md`` §1A-fix.
    """

    reins_bucket: str = 'linear'
    dsev_bucket: str = 'linear'
    bucket_sizing_p: float = 0.99999
    window_nines: int = 12
    window_nines_trim: int = 6
    window_pad_skew: float = 0.1
    sbj_tail_floor: float = 1e-14


@dataclass(frozen=True)
class ValidationSettings:
    """Tolerances for moment-matching validation.

    Parameters
    ----------
    eps : float
        Relative-error threshold above which a moment mismatch is flagged
        (formerly ``VALIDATION_EPS``).
    noise : float
        Absolute floor below which a quantity is treated as exact zero /
        numerical dust (formerly ``VALIDATION_NOISE``). This is a *dust floor*,
        not a coverage target -- the 12-nines coverage lives in
        ``discretization.window_nines`` (the two coincidentally both involve
        12). Sensible band 1e-12 .. 1e-14.
    """

    eps: float = 1e-4
    noise: float = 1e-12


@dataclass(frozen=True)
class MultivariateSettings:
    """Defaults for :mod:`aggregate.multivariate`.

    Parameters
    ----------
    window_nines : int
        Number of nines for the per-axis 2-D sizing window. Independent of
        :attr:`DiscretizationSettings.window_nines` because the 2-D per-axis
        window may want fewer nines for memory. (The remaining 2-D axis sizing
        knobs land here once the multivariate tuning settles them -- see
        ``dev/plan-multivariate-punchup.md``.)
    """

    window_nines: int = 12


@dataclass(frozen=True)
class LabelSettings:
    """Display labels for the two ``value_type`` sign-convention roles.

    ``value_type`` is a two-valued semantic role: *loss* convention ("more is
    worse", the default) vs *payoff* convention ("more is better"). Objects
    store the role internally as a boolean (``_is_loss_value``); these strings
    are only how the role is displayed (``info``, ``stats_df``) and spelled
    when setting ``value_type``. Changing a label renames the value everywhere
    it is shown or accepted; it never changes the underlying role of existing
    objects.

    Parameters
    ----------
    loss : str
        Label for the loss convention.
    payoff : str
        Label for the payoff convention.
    """

    loss: str = 'loss'
    payoff: str = 'payoff'


@dataclass(frozen=True)
class Settings:
    """Resolved, immutable snapshot of the library configuration.

    Access nested sections, e.g. ``settings.build.log2`` or
    ``settings.discretization.dsev_bucket``. ``sources`` maps each
    ``"section.key"`` to where its value came from (``'default'`` / ``'config'``
    / ``'env'``), backing :func:`describe_settings`.
    """

    build: BuildSettings = field(default_factory=BuildSettings)
    discretization: DiscretizationSettings = field(default_factory=DiscretizationSettings)
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    multivariate: MultivariateSettings = field(default_factory=MultivariateSettings)
    labels: LabelSettings = field(default_factory=LabelSettings)
    sources: dict = field(default_factory=dict, compare=False, repr=False)


# Registry mapping a TOML section name to its dataclass. Order is the display
# order used by describe_settings / show_settings.
_SECTIONS = {
    'build': BuildSettings,
    'discretization': DiscretizationSettings,
    'validation': ValidationSettings,
    'multivariate': MultivariateSettings,
    'labels': LabelSettings,
}


# ---------------------------------------------------------------------------
# Environment-variable allow-list (Phase 1 starter set). Each entry maps an
# AGGREGATE_* variable to (section, field, coercion). Unknown AGGREGATE_*
# variables warn loudly rather than silently no-op. AGGREGATE_CONFIG is handled
# separately in config_path.
# ---------------------------------------------------------------------------

def _env_databases(s: str) -> tuple[str, ...]:
    """Parse a comma-separated ``AGGREGATE_DATABASES`` value into a tuple."""
    return tuple(part.strip() for part in s.split(',') if part.strip())


_ENV_MAP = {
    'AGGREGATE_LOG2': ('build', 'log2', int),
    'AGGREGATE_BS': ('build', 'bs', float),
    'AGGREGATE_DATABASES': ('build', 'databases', _env_databases),
    'AGGREGATE_REINS_BUCKET': ('discretization', 'reins_bucket', str),
    'AGGREGATE_DSEV_BUCKET': ('discretization', 'dsev_bucket', str),
    'AGGREGATE_VALIDATION_EPS': ('validation', 'eps', float),
    'AGGREGATE_VALUE_TYPE_LOSS': ('labels', 'loss', str),
    'AGGREGATE_VALUE_TYPE_PAYOFF': ('labels', 'payoff', str),
}


def user_dir() -> Path:
    """Return the user-local data directory ``~/.aggregate`` (not created).

    Returns
    -------
    pathlib.Path
        ``Path.home() / USER_DIR_NAME``. This function never creates the
        directory; only :func:`write_default_config` does.
    """
    return Path.home() / USER_DIR_NAME


def config_path(env=None):
    """Resolve the config file path, honouring ``AGGREGATE_CONFIG``.

    Parameters
    ----------
    env : mapping, optional
        Environment to read (defaults to :data:`os.environ`).

    Returns
    -------
    pathlib.Path or None
        ``AGGREGATE_CONFIG=none`` (case-insensitive) returns ``None`` (ignore
        any file, for reproducible runs); a path value returns that path; unset
        returns ``user_dir() / 'config.toml'``.
    """
    if env is None:
        env = os.environ
    raw = env.get('AGGREGATE_CONFIG')
    if raw is None:
        return user_dir() / _CONFIG_FILENAME
    if raw.strip().lower() == 'none':
        return None
    return Path(raw).expanduser()


def _coerce_field(section_cls, fname, value):
    """Light coercion of a TOML value to the section field's expected type.

    Parameters
    ----------
    section_cls : type
        The section dataclass.
    fname : str
        Field name.
    value : object
        Raw value parsed from TOML.

    Returns
    -------
    object
        The coerced value (``databases`` list/str -> tuple; numeric int/float
        fields cast to match the default; everything else passed through).
    """
    if fname == 'databases':
        if isinstance(value, str):
            return (value,)
        return tuple(value)
    default = section_cls.__dataclass_fields__[fname].default
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def load_settings(*, path='__use_config_path__', env=None) -> Settings:
    """Build a :class:`Settings` snapshot by applying the cascade.

    Cascade (low -> high): dataclass defaults -> TOML file -> ``AGGREGATE_*``
    environment variables. (Explicit call-site keyword arguments are the final
    layer but live at the ``build`` / constructor call sites, not here.)

    Parameters
    ----------
    path : str, pathlib.Path, or None, optional
        The TOML file to read. The default sentinel resolves via
        :func:`config_path`; pass ``None`` to skip any file; pass an explicit
        path to read that file.
    env : mapping, optional
        Environment to read (defaults to :data:`os.environ`).

    Returns
    -------
    Settings
        The resolved settings, with a populated ``sources`` map.

    Notes
    -----
    Unknown file sections / keys and unknown ``AGGREGATE_*`` variables each
    emit a :class:`UserWarning` and are dropped, mirroring the unknown-``hints``
    behaviour -- misconfiguration is surfaced, never silently ignored.
    """
    if env is None:
        env = os.environ
    if path == '__use_config_path__':
        path = config_path(env)

    file_data = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            try:
                with open(p, 'rb') as fh:
                    file_data = tomllib.load(fh)
            except (OSError, tomllib.TOMLDecodeError) as exc:
                warnings.warn(f'Could not read config file {p}: {exc}. '
                              'Using defaults.')
                file_data = {}

    sources: dict = {}
    section_objs = {}

    for sec_name, sec_cls in _SECTIONS.items():
        field_names = {f.name for f in fields(sec_cls)}
        raw = file_data.get(sec_name, {}) or {}
        values = {}
        for key, val in raw.items():
            if key not in field_names:
                warnings.warn(
                    f"Unknown key '{key}' in [{sec_name}] of config file "
                    f"(known: {sorted(field_names)}); ignored.")
                continue
            values[key] = _coerce_field(sec_cls, key, val)
        for fname in field_names:
            sources[f'{sec_name}.{fname}'] = 'config' if fname in values else 'default'
        section_objs[sec_name] = sec_cls(**values)

    # Unknown top-level sections.
    for sec_name in file_data:
        if sec_name not in _SECTIONS:
            warnings.warn(
                f"Unknown section [{sec_name}] in config file "
                f"(known: {sorted(_SECTIONS)}); ignored.")

    # Environment overrides.
    for var, raw_val in env.items():
        if not var.startswith('AGGREGATE_') or var == 'AGGREGATE_CONFIG':
            continue
        if var not in _ENV_MAP:
            warnings.warn(
                f"Unknown environment variable '{var}' "
                f"(known: {sorted(_ENV_MAP)} and AGGREGATE_CONFIG); ignored.")
            continue
        sec_name, fname, coerce = _ENV_MAP[var]
        try:
            new_val = coerce(raw_val)
        except (TypeError, ValueError) as exc:
            warnings.warn(f"Could not parse {var}={raw_val!r}: {exc}; ignored.")
            continue
        section_objs[sec_name] = replace(section_objs[sec_name], **{fname: new_val})
        sources[f'{sec_name}.{fname}'] = 'env'

    return Settings(sources=sources, **section_objs)


# ---------------------------------------------------------------------------
# Lazy singleton + explicit reload.
# ---------------------------------------------------------------------------

_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached :class:`Settings` singleton, building it on first use.

    The config file is read once per session. To change configuration after
    import, set ``AGGREGATE_*`` before importing ``aggregate``, pass keyword
    arguments at the call site, or call :func:`reload_settings`.

    Returns
    -------
    Settings
        The shared settings snapshot.
    """
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reload_settings() -> Settings:
    """Re-read the config file and environment and refresh the module ``build``.

    Use after editing ``~/.aggregate/config.toml`` in a live session, or in
    tests. Rebuilds the cached singleton and, if the underwriter module is
    imported, refreshes the module-level ``build`` underwriter in place so that
    existing ``from aggregate import build`` references see the new defaults.

    Returns
    -------
    Settings
        The freshly resolved settings.
    """
    global _settings
    _settings = load_settings()
    try:
        from . import underwriter as _uw
        _uw._refresh_default_underwriter()
    except Exception:  # pragma: no cover - defensive; underwriter may be mid-import
        logger.exception('reload_settings: could not refresh module build')
    return _settings


def write_default_config(path=None, *, force=False) -> Path:
    """Copy the annotated, fully-commented config template into ``~/.aggregate``.

    Every line of the template is commented out, so the written file is inert
    (== pure defaults) until a line is uncommented. Refuses to overwrite an
    existing file unless ``force`` is set.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Destination. Defaults to ``user_dir() / 'config.toml'``.
    force : bool, default False
        Overwrite an existing destination file.

    Returns
    -------
    pathlib.Path
        The path written.

    Raises
    ------
    FileExistsError
        If the destination exists and ``force`` is False.
    """
    if path is None:
        path = user_dir() / _CONFIG_FILENAME
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f'{path} already exists; pass force=True to overwrite.')
    path.parent.mkdir(parents=True, exist_ok=True)
    template = files('aggregate').joinpath(*_TEMPLATE_RESOURCE)
    path.write_text(template.read_text(encoding='utf-8'), encoding='utf-8')
    logger.info('Wrote default config to %s', path)
    return path


def describe_settings(settings: Settings | None = None):
    """Return ``(key, value, source)`` rows for every setting.

    Parameters
    ----------
    settings : Settings, optional
        Defaults to :func:`get_settings`.

    Returns
    -------
    list of tuple
        ``(dotted_key, value, source)`` in section/field order, where source is
        one of ``'default'`` / ``'config'`` / ``'env'``.
    """
    if settings is None:
        settings = get_settings()
    rows = []
    for sec_name, sec_cls in _SECTIONS.items():
        section = getattr(settings, sec_name)
        for f in fields(sec_cls):
            key = f'{sec_name}.{f.name}'
            rows.append((key, getattr(section, f.name),
                         settings.sources.get(key, 'default')))
    return rows
