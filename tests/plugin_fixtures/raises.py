"""A fixture plugin whose ``register()`` raises, to prove a failure is recorded."""

_MESSAGE = 'this plugin is deliberately broken'


def register():
    """Raise, as a half-written experiment does."""
    raise RuntimeError(_MESSAGE)
