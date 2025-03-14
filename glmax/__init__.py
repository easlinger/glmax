# __init__.py
# pylint: disable=unused-import

import sys
from .class_regression import Regression
from .class_moderation import Moderation
from . import utils as tl
from . import processing as pp
from . import analysis as ax
from . import simulation as sm
from . import visualization as pl
from . import class_regression, class_moderation, constants
from .constants import print_models

mod = ["ax", "pl", "pp", "tl", "sm", "Regression"]
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in mod})

DUMMY_GLOBAL = "example"

__all__ = [
    "ax", "pl", "pp", "tl", "sm", "Regression",
    "processing", "analysis", "visualization", "utils", "simulation",
    "class_regression", "constants", "DUMMY_GLOBAL"
]
