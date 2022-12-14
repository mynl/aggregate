# https://pythonhosted.org/an_example_pypi_project/setuptools.html
# set HOME=c:\s\telos\python
# python setup.py sdist bdist_wininst upload.bat
# pip freeze > file to list installed packages
# pip install -r requirements.txt to install

# May 2022 to upload manually: twine upload dist/aggregate-0.9*.*
# enter user name and pword (mildenhall) (py... password! not Github)

import aggregate
from setuptools import setup
from pathlib import Path

tests_require = ['unittest', 'sly']
install_requires = [
    'cycler',
    'ipykernel',
    'jinja2',
    'matplotlib',
    'numpy',
    'pandas',
    'psutil',
    'pypandoc',
    'scipy',
    'sly',
    'titlecase',
    # docs
    'docutils',
    'jupyter-sphinx',
    'nbsphinx',
    'recommonmark',
    'setuptools',
    'sphinx',
    'sphinx-panels',
    'sphinx-rtd-dark-mode',
    'sphinxcontrib-bibtex',
    'sphinx-copybutton',
    'sphinx-toggleprompt',
    'IPython'
]


long_description = Path('Long_description.rst').read_text(encoding='utf-8')

version = aggregate.__version__

setup(name="aggregate",
      description="aggregate - working with compound probability distributions",
      long_description=long_description,
      license="""BSD""",
      version=version,
      author="Stephen J. Mildenhall",
      author_email="steve@convexrisk.com",
      maintainer="Stephen J. Mildenhall",
      maintainer_email="steve@convexrisk.com",
      packages=['aggregate'],
      package_data={'': ['*.txt', '*.rst', '*.md', 'agg/*.agg', 'examples/*.py', 'examples/*.ipynb',
                         'test/*.py']},
      tests_require=tests_require,
      install_requires=install_requires,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: BSD License',
          'Topic :: Education',
          'Topic :: Office/Business :: Financial',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Intended Audience :: Financial and Insurance Industry',
          'Intended Audience :: Education'
      ],
      project_urls={"Documentation": 'https://aggregate.readthedocs.io/en/latest/',
                    "Source Code": "https://github.com/mynl/aggregate"}
      )
