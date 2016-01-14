
import numpy as N
from numpy import zeros

from .constants import tol6, tol8, Ha2eV, kb_HaK

# =========================================================================== #
# Generic functions

@N.vectorize
def delta_lorentzian(x, eta):
    """The lorentzian representation of a delta function."""
    return (eta / N.pi) / (x ** 2 + eta ** 2)


def get_bose(natom, omega, temperatures):
  bose = N.array(zeros((3*natom, len(temperatures))))
  for imode in range(3*natom):
    if omega[imode].real > tol6:
      for tt, T in enumerate(temperatures):
        if T > tol6:
          bose[imode,tt] = 1.0 / (N.exp(omega[imode].real / (kb_HaK*T)) - 1)
  return bose

