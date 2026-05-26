# aggregate documentation build configuration file, created by
# sphinx-quickstart on Sat Sep  1 14:08:11 2018.
#
# This file is executed with the current directory set to its
# containing dir.
#

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
    'sphinx_design',
    # 'sphinx_panels',
    'sphinxcontrib.bibtex',
    'sphinx_multitoc_numbering',
    # 'sphinx_rtd_dark_mode'
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["dollarmath", "amsmath", "deflist", "colon_fence"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

exclude_patterns = ['_build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store']

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
bibtex_bibfiles = ['extract.bib', 'books.bib']
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
    'display_version': True,
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
