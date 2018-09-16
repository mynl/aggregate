# https://pythonhosted.org/an_example_pypi_project/setuptools.html
# set HOME=c:\s\telos\python
# python setup.py sdist bdist_wininst upload.bat


from setuptools import setup
import os

tests_require = ['unittest', 'pandas', 'matplotlib', 'sly']


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()


long_description = read('README.rst')

setup(name="aggregate_project",
      description="aggregate_project - working with compound probability distributions",
      long_description=long_description,
      license="""BSD""",
      version="0.5",
      author="Stephen J. Mildenhall",
      author_email="mildenhs@stjohns.edu",
      maintainer="Stephen J. Mildenhall",
      maintainer_email="mildenhs@stjohns.edu",
      packages=['aggregate'],
      package_data={'': ['*.txt', '*.rst', 'yaml/*.yaml', 'examples/*.py', 'examples/*.ipynb',
                         'test/*.py']},
      tests_require=tests_require,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: BSD License',
          'Topic :: Education',
          'Topic :: Office/Business :: Financial',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Intended Audience :: Financial and Insurance Industry',
          'Intended Audience :: Education'
      ],
      project_urls={"Documentation": 'http://www.mynl.com/aggregate_project/',
                    "Source Code": "https://github.com/mynl/aggregate_project"}
      )