from __future__ import print_function

from ..core.constants import Ha2eV
from ..core.util import report_runtime
from ..core import EpcAnalyzer

__all__ = ['compute']


@report_runtime
def compute(

        # Options
        temperature = True,
        renormalization = True,
        broadening = False,
        self_energy = False,
        spectral_function = False,
        dynamical = True,
        split_active = True,
        double_grid = False,
        write = True,
        verbose = False,

        # Parameters
        nqpt = 1,
        wtq = [1.0],
        smearing_eV = 0.01,
        temp_range = [0, 600, 50],
        omega_range = [-0.1, 0.1, 0.001],

        # File names
        rootname = 'epc.out',
        eigk_fname = '',
        eigq_fnames = list(),
        ddb_fnames = list(),
        eigr2d_fnames = list(),
        eigi2d_fnames = list(),
        fan_fnames = list(),
        gkk_fnames = list(),

        **kwargs):
    """
    Compute electron-phonon coupling related quantities
    as a function of temperature or at T=0. Those include:

        - the frequency-dependent self-energy

        - the spectral function

        - the quasiparticle renormalization
          (the real part of the self-energy at the bare eigenvalues)

        - the quasiparticle broadening, or inverse lifetime
          (the imaginary part of the self-energy at the bare eigenvalues)


    Arguments (default values in parenthesis)
    =========================================


    Options
    -------

    temperature: (True)
        Compute the temperature dependence of all quantities.
        Otherwise, they are computed only at T=0.

    renormalization: (True)
        Compute the renormalization, that is,
        the real part of the self-energy at the bare eigenvalues.

    broadening: (False)
        Compute the broadening, that is, the inverse lifetime,
        which is the imaginary part of the self-energy at the bare eigenvalues.

    self_energy: (False)
        Compute the frequency-dependent self-energy,
        with or without temperature dependence.

    spectral_function: (False)
        Compute the spectral function.
        Requires frequency-dependent self-energy (self_energy=True).

    dynamical: (True)
        Use the dynamical AHC theory.
        Otherwise, the static AHC theory is used.

    split_active: (True)
        Split the active contribution from the sternheimer contribution.
        This means that Abinit was run with ieig2rf=5
        to produce both EIGR2D.nc and GKK.nc files.
        If set to False, it means that Abinit was run with ieig2rf=1 
        to produce only EIGR2D.nc files.

    write: (True)
        Write the results on the disk.

    verbose: (False)
        Print information to standard ouput as the calculation proceeds.


    Parameters
    ----------

    nqpt: (1)
        Number of q-points.

    wtq: ([1.])
        Weights of all the q-points. Should sum up to 1.

    smearing_eV: (0.01)
        Imaginary parameter (eta) used in the self-energy denominator.

    temp_range: ([0,300,50])
        Minimum, maximum and step temperature for eigenvalues dependance.

    omega_range: ([0,0,1])
        Minimum, maximum and step frequency for the self-energy.


    File names
    ----------

    rootname: ('epc.out')
        Rootname for the output files.

    ddb_fnames: ([])
        Names of _DDB files.

    eigk_fname: ('')
        Name of the _EIG.nc file at k for the eigenvalues being corrected.

    eigq_fnames: ([])
        Names of _EIG.nc files at k+q.

    eigr2d_fnames: ([])
        Names of _EIGR2D.nc files.

    gkk_fnames: ([])
        Names of _GKK.nc files.

    fan_fnames: ([])
        Names of _FAN.nc files to use instead of GKK.nc files.
        This option is maintained for backward compatibility.

    eigi2d_fnames: ([])
        Names of _EIGI2D.nc files. Only relevant when split_active=False
        and broadening=True.


    Returns
    =======

    epc: EpcAnalyzer
        Object containing the response function data
    """

    if smearing_eV is None:
        smearing_Ha = None
    else:
        smearing_Ha = smearing_eV / Ha2eV

    # Initialize epc
    epc = EpcAnalyzer(
        nqpt=nqpt, 
        wtq=wtq,
        eigk_fname=eigk_fname,
        eigq_fnames=eigq_fnames,
        ddb_fnames=ddb_fnames,
        eigr2d_fnames=eigr2d_fnames,
        eigi2d_fnames=eigi2d_fnames,
        fan_fnames=fan_fnames,
        gkk_fnames=gkk_fnames,
        temp_range=temp_range,
        omega_range=omega_range,
        smearing=smearing_Ha,
        rootname=rootname,
        verbose=verbose,
        **kwargs)


    # Call the main functions
    if self_energy:

        if temperature:
            epc.compute_td_self_energy()

            if spectral_function:
                epc.compute_td_spectral_function()

        else:
            epc.compute_zp_self_energy()

            if spectral_function:
                epc.compute_zp_spectral_function()

    if dynamical and split_active:

        if renormalization:

            if temperature:
                epc.compute_dynamical_td_renormalization()
            else:
                epc.compute_dynamical_zp_renormalization()

        if broadening:

            if temperature:
                epc.compute_dynamical_td_broadening()
            else:
                epc.compute_dynamical_zp_broadening()

    elif not dynamical and split_active:

        if renormalization:

            if temperature:
                epc.compute_static_td_renormalization()
            else:
                epc.compute_static_zp_renormalization()

        if broadening:

            if temperature:
                epc.compute_static_td_broadening()
            else:
                epc.compute_static_zp_broadening()

    elif not dynamical and not split_active:

        if renormalization:

            if temperature:
                epc.compute_static_td_renormalization_nosplit()
            else:
                epc.compute_static_zp_renormalization_nosplit()

        if broadening:

            if temperature:
                epc.compute_static_td_broadening_nosplit()
            else:
                epc.compute_static_zp_broadening_nosplit()


    # Write the files
    if write:
        epc.write_netcdf()
        epc.write_renormalization()
        if broadening:
            epc.write_broadening()

    return epc

