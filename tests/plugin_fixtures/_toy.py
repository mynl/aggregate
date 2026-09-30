"""The object the fixture plugins serve, and the chart-emitter boilerplate.

A plugin registers against its own types or against the library's first-class
classes. A toy class is used here so a fixture plugin's registrations cannot
change what ``available_charts`` or ``available_exhibits`` report for a real
``Aggregate`` or ``Portfolio``, which several other test modules snapshot.
"""

from functools import singledispatch

import pandas as pd


class Toy:
    """A stand-in object carrying one public frame.

    Parameters
    ----------
    name : str, default 'toy'
        Used in exhibit titles, like any first-class object's ``name``.
    """

    def __init__(self, name='toy'):
        self.name = name

    @property
    def toy_df(self):
        """A two-row frame, the thing a passthrough exhibit serves."""
        return pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]},
                            index=['first', 'second'])


def emitter(chart_name):
    """Build the base singledispatch generic for a plugin's chart.

    The five lines every chart emitter needs: a generic whose base raises
    ``NotImplementedError`` naming the type, which is what
    :func:`~aggregate.charts.available_charts` reads to decide the chart does
    not serve an object. The library has a private factory for this;
    a plugin writes it out, since the extension surface covers no underscore
    prefixed name.
    """
    @singledispatch
    def emit(obj, **options):
        raise NotImplementedError(
            f'chart {chart_name!r} is not implemented for {type(obj).__name__}')
    emit.__name__ = f'chart_{chart_name}'
    return emit


class OtherToy(Toy):
    """A second stand-in, so a name-reuse fixture extends rather than replaces.

    Distinct from :class:`Toy` because the case decision 3 of the plan is about
    is two *unrelated* plugins whose registrations would merge into one exhibit.
    """
