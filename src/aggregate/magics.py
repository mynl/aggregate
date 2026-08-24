"""
IPython magics: write DecL in a notebook cell.

``%%agg`` turns a cell of DecL into live objects. It is the notebook spelling
of :func:`aggregate.build`, and it exists because the alternative is wrapping
every program in ``a = build('''...''')``, which buries the language inside a
Python string literal and costs an editor its DecL syntax highlighting.

Load it explicitly, once per kernel::

    %load_ext aggregate.magics

Two reasons the load is explicit rather than automatic on ``import
aggregate``. First, house preference: less magic. A magic that appears without
being asked for is a name in the notebook nobody declared. Second, importing
IPython costs about a second, and :mod:`aggregate.utilities` goes to some
trouble to keep it off the ``import aggregate`` path; this module imports it at
module scope, so it must stay off that path too.

There is one magic, not two. :meth:`aggregate.Underwriter.build` is
:meth:`~aggregate.Underwriter.build_many` plus an unwrap and a count check, so
``%%agg`` always calls the plural form and unwraps when exactly one output
comes back. One statement or twenty is the same cell, with nothing to detect
and no second spelling to remember.
"""

from IPython.core.magic import Magics, cell_magic, magics_class
from IPython.core.magic_arguments import (argument, magic_arguments,
                                          parse_argstring)
from IPython.display import display

from .underwriter import build_many
from .utilities import qd

__all__ = ['AggregateMagics', 'load_ipython_extension']


@magics_class
class AggregateMagics(Magics):
    """
    The ``%%agg`` cell magic.

    Notes
    -----
    Named for IPython's own convention (``@magics_class class XMagics``)
    rather than the house ``Base<Kind>`` prefix form, which exists to sort
    sibling taxonomies together. There is one magics class and there is not
    going to be a family of them.
    """

    @magic_arguments()
    @argument('name', nargs='?', default='a',
              help='variable to bind, default a; several outputs bind a dict')
    @argument('-q', '--quiet', action='store_true',
              help='bind and report the names, skip the qd display')
    @argument('-s', '--silent', action='store_true',
              help='bind and print nothing at all')
    @argument('-p', '--plot', action='store_true',
              help='also plot, for a cell declaring a single object')
    @argument('-v', '--validation', action='store_true',
              help='qd the validation_df instead of the object summary')
    @argument('--log2', type=int, default=0,
              help='log2 bucket count, 0 (default) means auto')
    @argument('--bs', default='0',
              help='bucket size, evaluated in the notebook namespace so '
                   '1/32 works, 0 (default) means auto')
    @cell_magic
    def agg(self, line, cell):
        """
        Build a cell of DecL and bind the result into the notebook.

        Parameters
        ----------
        line : str
            The magic's own arguments, parsed by
            :func:`~IPython.core.magic_arguments.parse_argstring`.
        cell : str
            The cell body: one or more DecL statements, separated by a blank
            line or by a semicolon at the end of a line.

        Returns
        -------
        None
            The objects arrive as notebook variables, not as a return value,
            which is the point of the magic.

        Notes
        -----
        **What gets bound.** A single output binds ``name`` (``a`` unless
        given) and, when the DecL name is a legal Python identifier, that name
        too, so ``agg Dice ...`` leaves both ``a`` and ``Dice``. Several
        outputs bind ``name`` to a ``{decl_name: object}`` dict, plus each
        legal identifier individually. Names DecL allows and Python does not
        (``EV.Peel``) are reachable through the dict.

        **Three volumes.** The default builds, says what it bound, and ``qd``s
        each object. ``--quiet`` stops before the ``qd``, which is the useful
        setting when the cell declares a dozen things. ``--silent`` prints
        nothing, for a cell whose output is a later figure.

        **Plotting** with ``--plot`` is for a cell declaring one object, since
        that is the cell where the picture is the answer. Several outputs
        would put several figures under one cell with nothing to say which is
        which, so the flag reports that it declined instead. It is independent
        of the three volumes: ``--silent --plot`` draws the figure and says
        nothing, which is the notebook equivalent of a plot statement.

        **Validation** with ``--validation`` swaps each object's ``qd`` for a
        ``qd`` of its ``validation_df``, the moment vs estimate audit. An
        object without the frame (a recipe stub, an ``expr`` value) displays
        itself as usual. The three volumes apply unchanged: ``--quiet`` and
        ``--silent`` skip the display, validation frame included.
        """
        args = parse_argstring(self.agg, line)
        bs = eval(args.bs, {}, self.shell.user_ns)

        recipes = build_many(cell, log2=args.log2, bs=bs)
        if not recipes:
            if not args.silent:
                print('no output')
            return

        # A recipe that cannot stand alone, a named mixture severity, has
        # object None. Hand back the recipe itself so the spec is reachable
        # rather than dropping the entry on the floor.
        outputs = {r.name: (r.object if r.object is not None else r)
                   for r in recipes}
        bound = {args.name: (outputs[recipes[0].name] if len(recipes) == 1
                             else outputs)}
        bound.update({nm: ob for nm, ob in outputs.items() if nm.isidentifier()})
        self.shell.user_ns.update(bound)

        if args.plot:
            self._plot(recipes, outputs, quiet=args.silent)
        if args.silent:
            return
        print(f"bound {', '.join(bound)}")
        if args.quiet:
            return
        for r in recipes:
            if len(recipes) > 1:
                print(f'\n{r.kind} {r.name}')
            try:
                obj = outputs[r.name]
                if args.validation:
                    obj = getattr(obj, 'validation_df', obj)
                qd(obj)
            except Exception as e:                      # noqa: BLE001
                # qd covers the first-class classes; a Recipe stub or a bare
                # expression value falls through to the ordinary display.
                print(f'qd failed ({e}), showing the object')
                display(outputs[r.name])


    @staticmethod
    def _plot(recipes, outputs, quiet=False):
        """
        Draw the cell's object, for the ``--plot`` flag.

        Parameters
        ----------
        recipes : list of Recipe
            The cell's outputs, in declaration order.
        outputs : dict
            Name to object, as bound into the notebook.
        quiet : bool, optional
            Suppress the two declining messages (set under ``--silent``).

        Notes
        -----
        Declines on several outputs by design: see :meth:`agg`. Declines on an
        object with no ``plot``, an ``expr`` value or a recipe stub, because
        the flag asked for a picture there is no way to draw.
        """
        if len(recipes) > 1:
            if not quiet:
                print(f'{len(recipes)} outputs, --plot draws a single object only')
            return
        obj = outputs[recipes[0].name]
        plot = getattr(obj, 'plot', None)
        if plot is None:
            if not quiet:
                print(f'{recipes[0].kind} {recipes[0].name} has no plot')
            return
        plot()


def load_ipython_extension(ipython):
    """
    Register the magics. Called by ``%load_ext aggregate.magics``.

    Parameters
    ----------
    ipython : InteractiveShell
        The running shell, supplied by IPython.
    """
    ipython.register_magics(AggregateMagics)
