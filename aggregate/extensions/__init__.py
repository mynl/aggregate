# coding: utf-8 -*-


# imports
from . basic import *
from . case_studies import *
from . samples import *
from . test_suite import *
from . pir_figures import *
from . figures import *
from . risk_progression import *
from . bodoff import *
from . ft import *
from . pentagon import Pentagon, mapper, make_possible_pentagons


# set up
from pathlib import Path
base_dir = Path.home() / 'aggregate'
base_dir.mkdir(exist_ok=True)

for p in ['cases', 'temp', 'generated']:
    (base_dir / p).mkdir(exist_ok=True)

del p, base_dir

# module level doc-string
__doc__ = """
Extensions contains optional, nice to have code that extends basic functionality.
For example, all CaseStudy material is included here. It is not required for core
functionality. It is not included in the default import. It is included in the
default build.
"""
