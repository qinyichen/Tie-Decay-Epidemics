"""Top-level package for Tie-Decay Epidemics."""

# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.0.1"

__author__ = "Qinyi Chen"
__email__ = "qinyichen@ucla.edu"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2020, Qinyi Chen"


from .example import Example  # noqa: F401
from .TieDecay import TieDecay_Graph, TieDecay_SIS
