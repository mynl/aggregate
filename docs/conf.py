# aggregate documentation build configuration file, created by
# sphinx-quickstart on Sat Sep  1 14:08:11 2018.
#
# This file is executed with the current directory set to its
# containing dir.
#

import re
import sys
import os

# allow RTD to find aggregate
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../src'))
import aggregate as agg
import aggregate.style

# constants preserved for any ipython blocks / extensions that import them
VALIDATION_EPS = 1e-4
RECOMMEND_P = 0.99999

# apply the aggregate house style (formerly inline knobble_fonts)
aggregate.style.use()


# silence warnings throughout
ipython_execlines = [
    'import aggregate',
    'aggregate.silence_warnings()',
    # other notebook prelude you might want, e.g.
    # 'import aggregate.style; aggregate.style.use()',
]


# ---------------------------------------------------------------------------
# Bug fix: IPython's ``.. ipython:: python`` continuation prompt, statement 10
# ---------------------------------------------------------------------------
# ``IPython.sphinxext.ipython_directive`` renders a pure-python block as a fake
# console session, then re-parses its own rendering. The two halves disagree
# about the width of the ``...:`` continuation prompt.
#
# ``process_pure_python`` writes the prompt for statement ``n`` as ``In [n]:``,
# increments its counter, and only *then* builds the continuation string from
# the counter, so the continuation lines of statement ``n`` are padded for
# ``n + 1``. ``block_parser`` reads the number back out of ``In [n]:`` and pads
# for ``n``. For a single-digit ``n`` both come to three dots and nothing is
# wrong. At ``n = 9`` the writer pads for ``10`` and emits four dots while the
# reader still expects three, so the continuation lines stop being recognized
# as input: they are filed as echoed stdout instead, and the shell is handed
# the first line of a multi-line statement on its own. That is *incomplete*
# rather than invalid, so it is buffered, never executed, and never raises.
#
# The symptom is silent and remote from its cause: the assignment vanishes, and
# the build fails hundreds of lines later with a bare ``NameError`` on a name
# whose definition is plainly there in the source. It bites the 10th statement
# of any pure-python block, and only when that statement spans lines. In this
# tree it hit the cast of examples in ``2_aggregate_overview/features.rst``,
# whose 10th statement is ``reins = build(...)`` wrapped over three lines.
#
# Fixed here rather than worked around in the prose (reordering the cast, or
# forcing the statement onto one long line) because a rule like "never let a
# multi-line statement land tenth" is invisible and unenforceable. The patch
# post-processes the writer's output, renumbering every continuation prompt to
# match the ``In [n]:`` above it. That output contains only blank lines,
# comments, pseudo-decorators and prompted input, so nothing else can match.
try:
    from IPython.sphinxext.ipython_directive import EmbeddedSphinxShell

    _RE_PROMPT_IN = re.compile(r'^In \[(\d+)\]:')
    _RE_PROMPT_CONTINUATION = re.compile(r'^   (\.+):')
    _orig_process_pure_python = EmbeddedSphinxShell.process_pure_python

    def _process_pure_python(self, content):
        lines = _orig_process_pure_python(self, content)
        out, number = [], None
        for line in lines:
            m = _RE_PROMPT_IN.match(line)
            if m:
                number = int(m.group(1))
                out.append(line)
                continue
            m = _RE_PROMPT_CONTINUATION.match(line)
            if m is not None and number is not None:
                prompt = '   %s:' % ('.' * (len(str(number)) + 2))
                out.append(prompt + line[m.end():])
                continue
            out.append(line)
        return out

    EmbeddedSphinxShell.process_pure_python = _process_pure_python
except ImportError:
    pass


# -- Project information -----------------------------------------------------
project = agg.__project__
copyright = agg.__copyright__
author = agg.__author__

# generally want True, so warning to be an error
# helpful in debugging to set equal to False
ipython_warning_is_error = True

# Lenient build mode: when AGG_DOCS_LENIENT is set in the environment
# (doc-test-uv.ps1 -Lenient does this), make the IPython sphinx directive
# permissive — exceptions in ``.. ipython::`` blocks render as inline
# tracebacks rather than aborting the build, and warnings stop being
# promoted to errors. This is the IPython-directive analogue of
# nbsphinx_allow_errors=True (which -Lenient also passes via -D).
if os.environ.get('AGG_DOCS_LENIENT'):
    ipython_warning_is_error = False
    try:
        from IPython.sphinxext.ipython_directive import IPythonDirective
        _orig_ipython_run = IPythonDirective.run
        def _lenient_ipython_run(self):
            self.options['okexcept'] = True
            self.options['okwarning'] = True
            return _orig_ipython_run(self)
        IPythonDirective.run = _lenient_ipython_run
    except ImportError:
        pass

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
release = agg.__version__
version = release[: len(release) -
                  len(release.lstrip("0123456789."))].rstrip(".")

# -- General configuration ------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    # 'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    'sphinx_toggleprompt',
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'nbsphinx',
    # 'sphinx_panels',
    'sphinxcontrib.bibtex',
    'sphinx_multitoc_numbering',
    # 'sphinx_rtd_dark_mode'
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["dollarmath", "amsmath", "deflist", "colon_fence"]

# ---------------------------------------------------------------------------
# autodoc / autosummary / napoleon
# ---------------------------------------------------------------------------
# Global autodoc defaults so the API-reference pages (``docs/3_reference/``)
# stay declarative: each page lists *what* to document, not the repeated
# ``:members:`` / ``:special-members:`` boilerplate. Per-directive options add
# to (or, for ``private-members``, switch on) these defaults -- e.g. the
# internal-architecture page turns private members on locally.
autodoc_default_options = {
    'members': True,
    'special-members': '__init__',
    'show-inheritance': True,
    'undoc-members': False,
}
# Source order (not alphabetical) keeps each class's methods in the author's
# logical grouping, which the docstrings are written to follow.
autodoc_member_order = 'bysource'
# The class docstring is the narrative; ``__init__`` is documented as a member
# (via ``special-members`` above) so its long parameter list renders once.
autoclass_content = 'class'
# Keep signatures readable: defer annotation rendering to the description.
autodoc_typehints = 'description'

# Napoleon: the project uses NumPy-style docstrings (Parameters / Returns /
# Notes). Older docstrings still use reST ``:param:`` fields, which autodoc
# renders natively -- the two coexist. Google style is off to avoid
# mis-parsing the occasional ``Note:`` line.
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = True
# Render an ``Attributes`` section as ``:ivar:`` fields inside the class
# description rather than as standalone ``.. attribute::`` directives. Without
# this, a dataclass / NamedTuple that documents its fields in an ``Attributes``
# section gets each one registered twice: once by napoleon from the docstring
# and once by autodoc from the annotated field. That is a "duplicate object
# description" warning per attribute, and it was 51 of this build's warnings
# (results, bounds, parser_errors, tail, recipe, contract_terms, and the
# massive compute leaf).
napoleon_use_ivar = True

# autosummary is used for the per-page *overview tables* only (``.. autosummary::``
# without ``:toctree:``): they link to the full ``automodule`` / ``autoclass``
# entries on the same page, so no stub pages are generated and nothing is
# documented twice.
autosummary_generate = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

exclude_patterns = [
    '_build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store',
    # Generated grammar listing. It is pulled into 4_dec_Language_Reference.rst
    # with ``.. include::``, so Sphinx must not also build it as a standalone
    # page (which made it an orphan, warned about, and shipped a stray
    # 4_agg_language_reference/ref_include.html).
    '4_agg_language_reference/ref_include.rst',
    # ``docs/flow`` holds the information-flow diagram fragments. They are
    # include targets and Quarto sources, never standalone Sphinx pages, so
    # excluding them keeps them out of the toctree-orphan warnings the same
    # way ``ref_include.rst`` is excluded. ``.. include::`` still reads them.
    'flow/*.md',
    # ``docs/AGGREGATE-MONOGRAPH`` is a symlink to the Quarto monograph repo,
    # kept here for convenience. Its pages are Quarto sources rendered by
    # Quarto, not Sphinx sources: nothing in this tree links to them, so Sphinx
    # built ~57 orphan pages, copied their figures, and emitted the bulk of the
    # build's warnings. Worse, a Quarto render running in that repo deletes and
    # recreates files mid-build, which crashed sphinx-build outright with
    # FileNotFoundError on a path it had just globbed. Excluded here; delete
    # this line to pull the monograph back into the Sphinx build.
    'AGGREGATE-MONOGRAPH/**',
    # Developer notes that live under docs/ but are not documentation pages.
    # Sphinx picked them up as sources, then warned that nothing links to them.
    'README.md',
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path

# The master toctree document.
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# GPT suggestions for the reference problem
autonumbering_enabled = True

# warnings_filters = {
#     'suppress': [
#         'ref.ref_has_no_links',
#         'ref.term_not_defined',
#         'autosectionlabel.label_from_unnamed_label',
#     ]
# }


# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# https://sphinx-toggleprompt.readthedocs.io/en/stable/#offset
toggleprompt_offset_right = 35

# bibtex options
bibtex_bibfiles = ['extract.bib', 'manual.bib']
bibtex_reference_style = 'author_year'

# user starts in light mode
default_dark_mode = False

# https://www.spinics.net/lists/linux-doc/msg77015.html
# GPT recommneded putthing these back
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': False,
    # 'display_version' was removed in sphinx_rtd_theme 3.0; the version is
    # shown via the flyout menu / html_context instead. Leaving it set only
    # earned an "unsupported theme option given" warning.
    'prev_next_buttons_location': 'both',
    'style_external_links': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'includehidden': True,
    'titles_only': False,
}

html_logo = '_static/agg_logo.png'
html_favicon = '_static/agg_favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'aggregatedoc'


# -- Options for LaTeX output ---------------------------------------------
# better unicode support
latex_engine = "xelatex"

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'a4paper',
    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '10pt',
    'extrapackages': '\\usepackage{mathrsfs}',
    # Wrap over-long code lines instead of letting them run off the page. The
    # docs contain ipython output lines over 2,000 characters; unwrapped they
    # produce badly overfull boxes and, where one lands on a page break, the
    # PDF build dies with "This can't happen (ext3)" / "(ext4)".
    'sphinxsetup': 'verbatimforcewraps=true',
    # Section numbering stops at three levels (chapter.section.subsection, e.g.
    # 2.8.1), matching ``:numbered: 3`` on the master toctree in ``index.rst``.
    # LaTeX's own counter is what numbers the PDF, and Sphinx derives it from the
    # toctree ``:maxdepth:``, not from ``:numbered:``, so the two have to be set
    # separately or the PDF numbers a level deeper than the HTML. The counter is
    # set here rather than via the ``secnumdepth`` element because the writer
    # overwrites that element whenever it computes a larger minimum; ``preamble``
    # is emitted after it in the template, so this wins.
    # LaTeX caps list nesting at 6 levels (4 for itemize) and errors with
    # "Too deeply nested". Autodoc stacks deeply: a page section, a class, a
    # method, its parameter list, and any list inside a docstring. Raise the
    # cap rather than flatten docstrings one at a time. HTML has no such limit,
    # which is why this only ever bites the PDF build.
    #
    # The third block gives the PDF one "Bibliography" heading instead of two.
    # ``docs/7_bibliography.rst`` is titled Bibliography, which becomes a
    # numbered ``\chapter``. Sphinx then wraps the entries in
    # ``sphinxthebibliography``, which ``sphinxmanual.cls`` defines as a page
    # break plus ``report.cls``'s ``thebibliography`` plus an explicit
    # ``\addcontentsline``. That inner environment opens with its own
    # *unnumbered* ``\chapter*{\bibname}``, so the heading and the contents
    # line each landed twice. HTML emits no such automatic heading, which is
    # why the duplicate was PDF-only. The environment is redefined to drop the
    # page break and the contents line, which the page's own title already
    # supplies, and ``\chapter`` is shadowed just long enough to swallow the
    # automatic heading. Shadowing beats copying ``report.cls``'s list body
    # here: it survives a change of document class, and it leaves ``\@mkboth``
    # to set the running head as before.
    'preamble': r'''
\setcounter{secnumdepth}{2}
\usepackage{enumitem}
\setlistdepth{12}
\renewlist{itemize}{itemize}{12}
\renewlist{enumerate}{enumerate}{12}
\renewlist{description}{description}{12}
\setlist[itemize]{label=\textbullet}
\setlist[enumerate]{label=\arabic*.}
\makeatletter
\renewenvironment{sphinxthebibliography}[1]
  {\let\sphinxorigchapter\chapter
   \def\chapter{\@ifstar\@gobble\sphinxorigchapter}%
   \begin{thebibliography}{#1}%
   \let\chapter\sphinxorigchapter}
  {\end{thebibliography}}
\makeatother
''',
    # 'preamble': '\\renewenvironment{DUlineblock}{}{}',
    # 'preamble': '\\renewenvironment{DUlineblock}{\\begin{comment}}{\\end{comment}}'
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'aggregate.tex', 'aggregate Documentation',
     'Stephen J. Mildenhall', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'aggregate', 'aggregate Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'aggregate', 'aggregate Documentation',
     author, 'aggregate', 'Working with aggregate (compound) probability '
     'distributions.',
     'Miscellaneous'),
]
