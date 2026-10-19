"""
Print the DecL word list as JSON: ``python -m aggregate.decl_words``.

The output is one object, ``{"version": <aggregate version>, "groups":
decl_words()}``, for consumers that are not Python, such as a web build or a
script generating a syntax file.

Notes
-----
A ``__main__`` module inside a package, rather than a ``__main__`` block in a
single module file, because ``import aggregate`` already imports the word list
(the Pygments lexer reads it). Running an already imported module with ``-m``
makes ``runpy`` warn that the module was found in ``sys.modules``; running a
package's ``__main__`` does not.
"""

import json

from .. import __version__
from . import decl_words

print(json.dumps({'version': __version__, 'groups': decl_words()}, indent=2))
