
import numpy as N
from numpy import zeros

from .constants import tol6, tol8, Ha2eV, kb_HaK

# =========================================================================== #
# Generic functions

@N.vectorize
def delta_lorentzian(x, eta):
    """The lorentzian representation of a delta function."""
    return (eta / N.pi) / (x ** 2 + eta ** 2)

