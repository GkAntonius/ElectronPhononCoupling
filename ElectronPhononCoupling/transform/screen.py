import numpy as np
from ..core import GkkFile, DdbFile

def screen_gkk(gkkfile, ddbfile, omega, epsilon):
    """
    Screen the gkk matrix elements according to

        g[..., imode] --> g[..., imode] / epsilon(omega_ph) 

    The value of epsilon(omega_ph) will be interpolated onto the phonon
    frequencies, so the range of omega must be chosen accordingly.

    Arguments
    ---------

    gkkfile:
        A GkkFile object.

    ddbfile:
        A DdbFile object.

    omega: [nomega]
        Frequency range for epsilon, in Hartree.

    epsilon: [nomega]
        Value of epsilon(omega), in atomic units (eps0).
    """

    # Compute the g matrix elements in the mode basis
    gkkfile.get_gkk_mode(ddbfile, noscale=True)

    # Compute the phonon frequencies
    omegaph, evecs = ddbfile.compute_dynmat()
    nmode = len(omegaph)

    # Interpolate epsilon onto the phonon frequencies
    epsilon_ph = np.interp(omegaph, omega, epsilon)

    # Scale the matrix elements
    inv_epsilon_ph = epsilon_ph / np.abs(epsilon_ph) ** 2
    for imode in range(nmode):
        gkkfile.GKK_mode[...,imode] = gkkfile.GKK_mode[...,imode] * inv_epsilon_ph[imode]

    # Transform back the g matrix elements in the atom/cartesian basis
    gkkfile.get_gkk_cart(ddbfile, noscale=True)


