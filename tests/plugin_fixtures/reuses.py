"""A fixture plugin reusing the exhibit name ``good`` registered.

Sorts after ``good``, so loading both in one pass puts the name in the registry
before this plugin runs. ``register_simple_exhibit`` extends the existing
exhibit to :class:`OtherToy` rather than raising, which across a trust boundary
is two unrelated plugins silently merged into one document. The loader refuses
it; the extension is not unwound, and the test drops the whole key.
"""

from aggregate.exhibits import register_simple_exhibit
from aggregate.plugins import PluginLeaf

from ._toy import OtherToy
from .good import EXHIBIT_NAME


def register():
    """Extend ``good``'s exhibit to our own class and declare it as ours."""
    register_simple_exhibit(EXHIBIT_NAME, 'Other toy table', 'toy_df', [OtherToy])
    return [PluginLeaf(name=EXHIBIT_NAME, kind='exhibit', label='Other toy table')]
