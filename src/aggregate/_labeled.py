"""``LabeledMixin`` -- the shared label surface for DecL-created objects.

This is the codebase's **first mixin** (see the naming note in ``CLAUDE.md``).
Five classes -- :class:`~aggregate._aggregate.Aggregate`,
:class:`~aggregate._portfolio.Portfolio`, :class:`~aggregate._pnl.PnL`,
:class:`~aggregate._severity.Severity`, and
:class:`~aggregate.spectral.Distortion` -- need an identical label surface but
share **no common base** (``Portfolio`` subclasses ``object``, ``Severity``
extends ``scipy.stats.rv_continuous``), so the surface lives here once and is
mixed in. See ``dev/plan-labels.md`` (``[DecL-Labels-Everywhere]``).

The idea: a DecL program is a **tree** of named things, and each node carries

* a **handle** -- the ID-shaped bareword ``name`` (the identity, the reference
  target, the dict key), and
* an optional **label** -- the private ``_label`` slot (the human string, the
  DecL ``as`` clause), read through the resolved :attr:`label` property, plus
* interior **labels** -- ``label_map`` for the sub-object sites (exposure,
  layer, inline severity clause, cessions) that have no Python class of their
  own and so cannot hold their own ``_label``.

Resolution is a chain that is **never blank**::

    explicit label  ->  derived default  ->  handle

The object-level label uses this chain in the :attr:`label` property; the
interior sites use it in :attr:`renamer` (the ``{handle: label}`` dict exhibits
apply as a final ``df.rename()``).

Notes
-----
Delivery is by **explicit initialization**, not cooperative ``super().__init__``:
``Severity`` sits on a heavy scipy base where MRO chaining is fragile, so every
host ``__init__`` calls :meth:`_init_labels` when it is ready. Pure presentation
-- labels land in attributes, dict keys, and rendered text, never in the FFT.
"""

from types import SimpleNamespace


class _LabelView:
    """Read-only namespace view over an object's interior ``label_map``.

    Attribute access (``a.labels.exposure``) returns the explicit interior
    label if one was declared, else ``None``; dict-valued sites (cessions)
    come back as their ``{index: label}`` dict so ``a.labels.occ_reins[0]``
    works. A missing site returns ``None`` rather than raising, so exhibit code
    can coalesce to a derived default or the handle without a guard.
    """

    __slots__ = ('_label_map',)

    def __init__(self, label_map):
        object.__setattr__(self, '_label_map', label_map)

    def __getattr__(self, key):
        # __getattr__ only fires for names not found normally, so the private
        # slot is safe. Missing interior sites resolve to None, not AttributeError.
        return self._label_map.get(key)

    def __setattr__(self, key, value):
        raise AttributeError('labels view is read-only')

    def get(self, key, default=None):
        """Dict-style access with an explicit default."""
        return self._label_map.get(key, default)

    def __repr__(self):
        return f'labels({self._label_map!r})'


class LabeledMixin:
    """Shared label surface: private ``_label`` / resolved :attr:`label` /
    ``label_map`` / :attr:`labels` / :attr:`renamer` / :attr:`use_labels`.

    Hosts supply the specifics via three small hooks, all with harmless
    defaults so a host that only wants the object-level label need override
    nothing:

    * :meth:`_label_default` -- a derived object-level label (e.g. Distortion's
      auto-pretty ``'PH(0.9)'``); default ``None``.
    * :meth:`_label_handles` -- the handles the object's exhibits key off (units
      for a Portfolio, legs for a PnL, sub-parts for an Aggregate); default
      empty, so :attr:`renamer` is the identity.
    * :meth:`_resolve_handle_label` -- how one handle maps to its display label;
      default identity (the handle prints as itself).
    """

    # ---- initialization -------------------------------------------------
    def _init_labels(self, *, label=None, label_map=None):
        """Populate label state from the spec. Called explicitly by each host
        ``__init__`` (no cooperative ``super()`` -- see the module docstring).

        Parameters
        ----------
        label : str or None
            The object-level human label (DecL ``as``); ``None`` means fall
            back to the derived default, then the handle. Stored privately as
            ``_label``; read back through the resolved :attr:`label` property.
        label_map : dict or None
            Interior labels keyed by site (``'exposure'``, ``'severity'``,
            ``'layer'``, ``'occ_reins'``, ``'agg_reins'``). ``None`` -> empty.
        """
        #: Optional explicit object-level human label (the DecL ``as`` clause),
        #: or ``None``. Presentation only; ``name`` stays the identity handle.
        #: Read through the resolved :attr:`label` property, never directly.
        self._label = label
        #: Interior labels keyed by sub-object site; see :class:`_LabelView`.
        self.label_map = dict(label_map) if label_map else {}
        #: Per-object label switch (no module global -- see ``dev/plan-labels.md``
        #: D5). ``True`` -> exhibits serve the relabeled view; ``False`` -> raw
        #: handle-keyed frame (debug / join view).
        self._use_labels = True
        #: Cached ``renamer`` dict; invalidated when ``use_labels`` flips.
        self._renamer = None

    # ---- object-level label --------------------------------------------
    @property
    def label(self):
        """The resolved object label: explicit ``_label`` -> derived default
        (:meth:`_label_default`) -> ``name`` handle. Never blank.

        Presentation only -- repr and exhibit titles prefer it; ``name`` stays
        the identity / reference handle.
        """
        return self._label or self._label_default() or self.name

    def _label_default(self):
        """Derived object-level default label, or ``None``.

        Overridden by hosts that can pretty-print themselves from structure
        (Distortion's ``'PH(0.9)'``). The default has no derived form.
        """
        return None

    @property
    def _title_name(self):
        """Exhibit-title form: ``label (handle)`` when an explicit label is set
        (human label leads, identity handle stays visible), else the handle."""
        return f'{self.label} ({self.name})' if self._label is not None \
            else self.name

    # ---- interior labels ------------------------------------------------
    @property
    def labels(self):
        """Read-only :class:`_LabelView` over :attr:`label_map` --
        ``a.labels.exposure``, ``a.labels.occ_reins[0]``. Missing sites return
        ``None``."""
        return _LabelView(self.label_map)

    # ---- switch ---------------------------------------------------------
    @property
    def use_labels(self):
        """Whether exhibits serve the relabeled view (default ``True``).

        A property, not a bare bool, so flipping it can *do more than assign* --
        today it invalidates the cached :attr:`renamer`.
        """
        return self._use_labels

    @use_labels.setter
    def use_labels(self, value):
        self._use_labels = bool(value)
        self._renamer = None  # invalidate the cached rename map

    # ---- renamer --------------------------------------------------------
    @property
    def renamer(self):
        """Cached ``{handle: display}`` map for this object's exhibit axis.

        Applied as a final ``df.rename(index=/columns=)`` at the serve step so
        relabeling happens *after* any sort on the (handle-ordered) axis and can
        never reorder an exhibit. Built from :meth:`_label_handles` /
        :meth:`_resolve_handle_label`; the identity map when a host declares no
        handles.
        """
        if self._renamer is None:
            self._renamer = {h: self._resolve_handle_label(h)
                             for h in self._label_handles()}
        return self._renamer

    def _label_handles(self):
        """The handles this object's exhibits key off. Override per host
        (Portfolio -> units, PnL -> legs, Aggregate -> sub-parts)."""
        return []

    def _resolve_handle_label(self, handle):
        """Map one exhibit handle to its display label: explicit ->
        derived default -> the handle itself. Default is the identity."""
        return handle

    def _relabel(self, df):
        """Return a **display copy** of ``df`` with :attr:`renamer` applied to
        both axes, when :attr:`use_labels` is on; otherwise ``df`` unchanged.

        The canonical (handle-keyed) frame is never mutated -- ``rename`` returns
        a copy, so joins / references / compute that key off the handle are
        untouched (dev/plan-labels.md D2). ``rename`` leaves axis labels that do
        not match a handle alone, and for an unlabeled object the renamer is the
        identity, so this is a no-op there.
        """
        if not self._use_labels:
            return df
        renamer = self.renamer
        if not renamer:
            return df
        return df.rename(index=renamer, columns=renamer)
