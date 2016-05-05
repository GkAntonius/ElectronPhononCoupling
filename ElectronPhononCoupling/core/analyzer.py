from __future__ import print_function

import sys
import os
import warnings
import copy
import multiprocessing
from functools import partial

import numpy as np
import netCDF4 as nc

from .constants import Ha2eV

from .degenerate import get_degen
from .util import create_directory, formatted_array_lines

from .tdep import (static_zpm_temp, dynamic_zpm_temp, static_zpm_lifetime, 
                   static_zpm_temp_lifetime)
from .zpm import (get_qpt_zpr_static, get_qpt_zpr_dynamical,
                  get_qpt_zpb_dynamical, get_qpt_zpb_static_control,
                  get_qpt_zp_self_energy)

from . import EigFile, Eigr2dFile, FanFile, DdbFile

from ..parallel import summer

# =========================================================================== #

class EPC_Analyzer(object):
    """
    Main class for analysing electron-phonon coupling related quantities.

    It is intented to analyse the files produced by ABINIT
    in a phonon response-function calculation, with one q-point per dataset,
    the first q-point being Gamma.

    """
    wtq = None
    total_corr = None
    total_wtq = None
    broadening = None
    temperatures = []
    omegase = []
    smearing = None

    zero_point_renormalization = None
    fan_zero_point_renormalization = None
    ddw_zero_point_renormalization = None

    temperature_dependent_renormalization = None
    fan_temperature_dependent_renormalization = None
    ddw_temperature_dependent_renormalization = None

    zero_point_broadening = None
    fan_zero_point_broadening = None
    ddw_zero_point_broadening = None

    temperature_dependent_broadening = None
    fan_temperature_dependent_broadening = None
    ddw_temperature_dependent_broadening = None

    renormalization_is_dynamical = False
    broadening_is_dynamical = False

    self_energy = None
    spectral_function = None

    def __init__(self,
                 nqpt=1,
                 wtq=None,
                 eig0_fname='',
                 eigq_fnames=list(),
                 DDB_fnames=list(),
                 EIGR2D_fnames=list(),
                 EIGI2D_fnames=list(),
                 FAN_fnames=list(),
                 ncpu=1,
                 temp_range=[0,0,1],
                 omega_range=[0,0,1],
                 smearing=0.01 / Ha2eV,
                 output='epc.out',
                    ):

        if not eig0_fname:
            raise Exception('Must provide a file for eig0_fname')

        if not EIGR2D_fnames:
            raise Exception('Must provide at least one file for EIGR2D_fnames')

        # Set basic quantities
        self.nqpt = nqpt
        self.set_temp_range(temp_range)
        self.set_omega_range(omega_range)
        self.set_smearing(smearing)
        self.set_output(output)

        # Set file names
        self.eig0_fname = eig0_fname
        self.eigq_fnames = eigq_fnames
        self.DDB_fnames = DDB_fnames
        self.EIGR2D_fnames = EIGR2D_fnames
        self.EIGI2D_fnames = EIGI2D_fnames
        self.FAN_fnames = FAN_fnames

        # Initialize the parallelization
        self.set_ncpu(ncpu)

        # Check that the first q-point is Gamma
        self.check_Gamma()


        # FIXME ============ #
        # Open the eig0 file
        self.eig0 = EigFile(eig0_fname)

        # Read the EIGR2D file at Gamma and save it in ddw
        self.EIGR2D = Eigr2dFile(EIGR2D_fnames[0])
        self.nkpt = self.EIGR2D.nkpt
        self.nband = self.EIGR2D.nband
        self.kpts = self.EIGR2D.kpt[:,:]

        self.ddw = copy.deepcopy(self.EIGR2D.EIG2D)  # (nkpt,nband,3,natom,3,natom) complex

        # Open the first FAN file for DDW
        if FAN_fnames:
            self.FAN = FanFile(FAN_fnames[0])
            self.ddw_active = copy.deepcopy(self.FAN.FAN)  # (nkpt,nband,3,natom,3,natom,nband), complex

        # FIXME ============ #

        # Find the degenerate eigenstates
        self.degen =  get_degen(self.eig0.EIG)

        # Create the Q-integration (wtq=1/nqpt):
        self.compute_wtq(wtq)

        # qpoint indicies
        self.iqpts = np.arange(self.nqpt)

        #eig0_pass = copy.deepcopy(eig0.EIG)

    def set_ncpu(self, ncpu):
        """Set the number of processors."""
        self.ncpu = ncpu
        self.pool = multiprocessing.Pool(processes=ncpu)

    def __del__(self):
        self.pool.close()
        #super(EPC_Analyzer, self).__del__()

    def set_temp_range(self, temp_range=(0, 0, 1)):
        """Set the minimum, makimum and step temperature."""
        self.temperatures = np.arange(*temp_range, dtype=float)

    def set_omega_range(self, omega_range=(0, 0, 1)):
        """Set the minimum, makimum and step frequency for the self-energy."""
        self.omegase = np.arange(*omega_range, dtype=float)
        self.nomegase = len(self.omegase)

    def set_smearing(self, smearing_Ha):
        """Set the smearing, in Hartree."""
        self.smearing = smearing_Ha
    
    def set_output(self, root):
        """Set the root for output names."""
        self.output = root
    
    def check_Gamma(self):
        """Check that the first q-point is Gamma and raise exception otherwise."""
        DDBtmp = DdbFile(self.DDB_fnames[0])
        if not np.allclose(DDBtmp.qred, [0.0,0.0,0.0]):
            raise Exception('The first Q-point is not Gamma.')

    def compute_wtq(self, wtq):
        """Compute the q-point weights."""

        if wtq is not None:
            self.wtq = np.array(wtq, dtype=np.float) / sum(wtq)
            return

        if (self.EIGR2D.wtq == 0):
            self.wtq = np.ones((self.nqpt)) * 1.0 / self.nqpt
            return

        wtq = map(lambda fname: Eigr2dFile(fname).wtq[0], self.EIGR2D_fnames)

        total_wtq = sum(wtq)

        self.wtq = np.array(wtq)

        if abs(total_wtq-1) > 0.1:
            raise Exception("The total weigth is not equal to 1.0. Check that you provide all the q-points.")


    def compute_static_td_renormalization(self):
        arguments = zip(self.iqpts, self.wtq, self.eigq_fnames, self.DDB_fnames, self.EIGR2D_fnames)
        partial_static_zpm_temp = partial(static_zpm_temp, ddw=self.ddw, temperatures=self.temperatures, degen=self.degen)
        qpt_corr = self.pool.map(partial_static_zpm_temp,arguments)
        total_corr = sum(qpt_corr)
        total_corr = np.einsum('okij->oijk', total_corr)
        self.temperature_dependent_renormalization = total_corr[0]
        self.fan_temperature_dependent_renormalization = total_corr[1]
        self.ddw_temperature_dependent_renormalization = total_corr[2]

    def compute_static_zp_renormalization(self):
        func = partial(get_qpt_zpr_static, ddw=self.ddw, degen=self.degen)
        args = zip(self.iqpts, self.wtq, self.eigq_fnames, self.DDB_fnames, self.EIGR2D_fnames)
        qpt_corr = self.pool.map(func, args)
        total_corr = sum(qpt_corr)
        self.zero_point_renormalization = total_corr[0]
        self.fan_zero_point_renormalization = total_corr[1]
        self.ddw_zero_point_renormalization = total_corr[2]

    def compute_dynamical_td_renormalization(self):
        arguments = zip(self.iqpts, self.wtq, self.eigq_fnames, self.DDB_fnames, self.EIGR2D_fnames, self.FAN_fnames)
        partial_dynamic_zpm_temp = partial(dynamic_zpm_temp, ddw=self.ddw, ddw_active=self.ddw_active,
                                           calc_type=2, temperatures=self.temperatures, smearing=self.smearing,
                                           eig0=self.eig0.EIG, degen=self.degen)
        qpt_corr = self.pool.map(partial_dynamic_zpm_temp,arguments)
        total_corr = sum(qpt_corr)
        total_corr = np.einsum('okij->oijk', total_corr)
        self.temperature_dependent_renormalization = total_corr[0]
        self.fan_temperature_dependent_renormalization = total_corr[1]
        self.ddw_temperature_dependent_renormalization = total_corr[2]
        self.renormalization_is_dynamical = True

    def compute_dynamical_zp_renormalization(self):
        arguments = zip(self.iqpts, self.wtq, self.eigq_fnames, self.DDB_fnames, self.EIGR2D_fnames, self.FAN_fnames)
        partial_dynamic_zpm = partial(get_qpt_zpr_dynamical, ddw=self.ddw, ddw_active=self.ddw_active,
                                     option='dynamical', smearing=self.smearing, eig0=self.eig0.EIG, degen=self.degen)
        qpt_corr = self.pool.map(partial_dynamic_zpm, arguments)
        total_corr = sum(qpt_corr)
        self.zero_point_renormalization = total_corr[0]
        self.fan_zero_point_renormalization = total_corr[1]
        self.ddw_zero_point_renormalization = total_corr[2]
        self.renormalization_is_dynamical = True

    def compute_static_control_td_renormalization(self):
        arguments = zip(self.iqpts, self.wtq, self.eigq_fnames, self.DDB_fnames, self.EIGR2D_fnames, self.FAN_fnames)
        partial_dynamic_zpm_temp = partial(dynamic_zpm_temp, ddw=self.ddw, ddw_active=self.ddw_active,
                                           calc_type=3, temperatures=self.temperatures, smearing=self.smearing,
                                           eig0=self.eig0.EIG, degen=self.degen)
        qpt_corr = self.pool.map(partial_dynamic_zpm_temp, arguments)
        total_corr = sum(qpt_corr)
        total_corr = np.einsum('okij->oijk', total_corr)
        self.temperature_dependent_renormalization = total_corr[0]
        self.fan_temperature_dependent_renormalization = total_corr[1]
        self.ddw_temperature_dependent_renormalization = total_corr[2]

    def compute_static_control_zp_renormalization(self):
        arguments = zip(self.iqpts, self.wtq, self.eigq_fnames, self.DDB_fnames, self.EIGR2D_fnames, self.FAN_fnames)
        partial_dynamic_zpm = partial(get_qpt_zpr_dynamical, ddw=self.ddw, ddw_active=self.ddw_active,
                                      option='static', smearing=self.smearing, eig0=self.eig0.EIG, degen=self.degen)
        qpt_corr = self.pool.map(partial_dynamic_zpm, arguments)
        total_corr = sum(qpt_corr)
        self.zero_point_renormalization = total_corr[0]
        self.fan_zero_point_renormalization = total_corr[1]
        self.ddw_zero_point_renormalization = total_corr[2]

    def compute_static_td_broadening(self):
        arguments = zip(self.iqpts, self.wtq, self.eigq_fnames, self.DDB_fnames, self.EIGI2D_fnames)
        partial_static_zpm_temp_lifetime = partial(static_zpm_temp_lifetime, ddw=self.ddw,
                                                   temperatures=self.temperatures, degen=self.degen)
        qpt_brd = self.pool.map(partial_static_zpm_temp_lifetime, arguments)
        total_brd = sum(qpt_brd)
        total_brd = np.einsum('oij->ijo', total_brd)
        self.temperature_dependent_broadening = total_brd

    def compute_static_zp_broadening(self):
        arguments = zip(self.iqpts, self.wtq, self.eigq_fnames, self.DDB_fnames, self.EIGI2D_fnames)
        partial_static_zpm_lifetime =  partial(static_zpm_lifetime, degen=self.degen)
        qpt_brd = self.pool.map(partial_static_zpm_lifetime, arguments)
        total_brd = sum(qpt_brd)
        self.zero_point_broadening = total_brd

    def compute_dynamical_td_broadening(self):
        warnings.warn("Dynamical lifetime at finite temperature is not yet implemented...proceed with static lifetime")
        arguments = zip(self.iqpts, self.wtq, self.eigq_fnames, self.DDB_fnames, self.EIGI2D_fnames)
        partial_static_zpm_temp_lifetime = partial(static_zpm_temp_lifetime, ddw=self.ddw,
                                                   temperatures=self.temperatures, degen=self.degen)
        qpt_brd = self.pool.map(partial_static_zpm_temp_lifetime, arguments)
        total_brd = sum(qpt_brd)
        total_brd = np.einsum('oij->ijo', total_brd)
        self.temperature_dependent_broadening = total_brd
        self.broadening_is_dynamical = True

    def compute_dynamical_zp_broadening(self):
        args = zip(self.iqpts, self.wtq, self.eigq_fnames, self.DDB_fnames, self.FAN_fnames)
        func = partial(get_qpt_zpb_dynamical, smearing=self.smearing, eig0=self.eig0.EIG, degen=self.degen)
        qpt_brd = self.pool.map(func, args)
        total_brd = sum(qpt_brd)
        self.zero_point_broadening = total_brd
        self.broadening_is_dynamical = True

    def compute_static_control_td_broadening(self):
        arguments = zip(self.iqpts, self.wtq, self.eigq_fnames, self.DDB_fnames, self.EIGI2D_fnames)
        partial_static_zpm_temp_lifetime = partial(static_zpm_temp_lifetime, ddw=self.ddw,
                                                   temperatures=self.temperatures, degen=self.degen)
        qpt_brd = self.pool.map(partial_static_zpm_temp_lifetime, arguments)
        total_brd = sum(qpt_brd)
        total_brd = np.einsum('oij->ijo', total_brd)
        self.temperature_dependent_broadening = total_brd

    def compute_static_control_zp_broadening(self):
        args = zip(self.iqpts, self.wtq, self.eigq_fnames, self.DDB_fnames, self.FAN_fnames)
        func = partial(get_qpt_zpb_static_control, smearing=self.smearing, eig0=self.eig0.EIG, degen=self.degen)
        qpt_brd = self.pool.map(func, args)
        total_brd = sum(qpt_brd)
        self.zero_point_broadening = total_brd
        self.broadening_is_dynamical = False

    def compute_self_energy(self):
        args = zip(self.iqpts, self.wtq, self.eigq_fnames, self.DDB_fnames, self.EIGR2D_fnames, self.FAN_fnames)
        args = [ (arg,) for arg in args ]
        func = partial(get_qpt_zp_self_energy, ddw=self.ddw, ddw_active=self.ddw_active, smearing=self.smearing,
                       eig0=self.eig0.EIG, degen=self.degen, omegase=self.omegase)

        self.self_energy = summer(func, args, self.ncpu)


    def compute_spectral_function(self):
        """
        Compute the spectral function of all quasiparticles in the semi-static approximation,
        that is, the 'upper bands' contribution to the self-energy is evaluated at the bare energy.

        The spectral function is evaluated on a frequency mesh 'omegase' that is shifted by the bare energies,
        such that, what is retured is

            A'_kn(omega) = A_kn(omega + E^0_kn)

        """
        self.spectral_function = np.zeros((self.nomegase, self.nkpt, self.nband), dtype=float)
        omega = np.einsum('ij,m->ijm', np.ones((self.nkpt, self.nband)), self.omegase)
        self.spectral_function = (1 / np.pi) * np.abs(self.self_energy.imag) / (
                                (omega - self.self_energy.real) ** 2 + self.self_energy.imag ** 2)


    def write_netcdf(self):
        """Write all data to a netCDF file."""
        fname = str(self.output) + '_EP.nc'
        create_directory(fname)

        # Write on a NC files with etsf-io name convention
        ncfile = nc.Dataset(fname, 'w')

        # Read dim from first EIGR2D file
        root = nc.Dataset(self.EIGR2D_fnames[0], 'r')

        # Determine nsppol from reading occ
        nsppol = len(root.variables['occupations'][:,0,0])
        if nsppol > 1:
          warnings.warn("nsppol > 1 has not been tested.")
        mband = len(root.dimensions['product_mband_nsppol']) / nsppol

        # Create dimension
        ncfile.createDimension('number_of_atoms',len(root.dimensions['number_of_atoms']))
        ncfile.createDimension('number_of_kpoints',len(root.dimensions['number_of_kpoints']))
        ncfile.createDimension('product_mband_nsppol',len(root.dimensions['product_mband_nsppol']))
        ncfile.createDimension('cartesian',3)
        ncfile.createDimension('cplex',2)
        ncfile.createDimension('number_of_qpoints', self.nqpt)
        ncfile.createDimension('number_of_spins',len(root.dimensions['number_of_spins']))
        ncfile.createDimension('max_number_of_states',mband)

        ncfile.createDimension('number_of_temperature',len(self.temperatures))
        ncfile.createDimension('number_of_frequencies',len(self.omegase))

        # Create variable
        data = ncfile.createVariable('reduced_coordinates_of_kpoints','d',('number_of_kpoints','cartesian'))
        data[:,:] = root.variables['reduced_coordinates_of_kpoints'][:,:]
        data = ncfile.createVariable('eigenvalues','d',('number_of_spins','number_of_kpoints','max_number_of_states'))
        data[:,:,:] = root.variables['eigenvalues'][:,:,:]
        data = ncfile.createVariable('occupations','i',('number_of_spins','number_of_kpoints','max_number_of_states'))
        data[:,:,:] = root.variables['occupations'][:,:,:]
        data = ncfile.createVariable('primitive_vectors','d',('cartesian','cartesian'))
        data[:,:] = root.variables['primitive_vectors'][:,:]

        root.close()

        data = ncfile.createVariable('renormalization_is_dynamical', 'i1')
        data[:] = self.renormalization_is_dynamical

        data = ncfile.createVariable('broadening_is_dynamical', 'i1')
        data[:] = self.broadening_is_dynamical

        data = ncfile.createVariable('temperatures','d',('number_of_temperature'))
        data[:] = self.temperatures[:]

        data = ncfile.createVariable('smearing', 'd')
        data[:] = self.smearing

        data = ncfile.createVariable('omegase','d',('number_of_frequencies'))
        data[:] = self.omegase[:]

        zpr = ncfile.createVariable('zero_point_renormalization','d',
            ('number_of_spins', 'number_of_kpoints', 'max_number_of_states'))

        fan = ncfile.createVariable('fan_zero_point_renormalization','d',
            ('number_of_spins', 'number_of_kpoints', 'max_number_of_states'))

        ddw = ncfile.createVariable('ddw_zero_point_renormalization','d',
            ('number_of_spins', 'number_of_kpoints', 'max_number_of_states'))

        if self.zero_point_renormalization is not None:
            zpr[0,:,:] = self.zero_point_renormalization[:,:].real  # FIXME number of spin
            fan[0,:,:] = self.fan_zero_point_renormalization[:,:].real  # FIXME number of spin
            ddw[0,:,:] = self.ddw_zero_point_renormalization[:,:].real  # FIXME number of spin

        data = ncfile.createVariable('temperature_dependent_renormalization','d',
            ('number_of_spins','number_of_kpoints', 'max_number_of_states','number_of_temperature'))

        if self.temperature_dependent_renormalization is not None:
            data[0,:,:,:] = self.temperature_dependent_renormalization[:,:,:].real  # FIXME number of spin

        data = ncfile.createVariable('zero_point_broadening','d',
            ('number_of_spins', 'number_of_kpoints', 'max_number_of_states'))

        if self.zero_point_broadening is not None:
            data[0,:,:] = self.zero_point_broadening[:,:].real  # FIXME number of spin

        data = ncfile.createVariable('temperature_dependent_broadening','d',
            ('number_of_spins','number_of_kpoints', 'max_number_of_states','number_of_temperature'))

        if self.temperature_dependent_broadening is not None:
            data[0,:,:,:] = self.temperature_dependent_broadening[:,:,:].real  # FIXME number of spin

        self_energy = ncfile.createVariable('self_energy','d',
            ('number_of_spins', 'number_of_kpoints', 'max_number_of_states', 'number_of_frequencies', 'cplex'))

        if self.self_energy is not None:
            self_energy[0,:,:,:,0] = self.self_energy[:,:,:].real  # FIXME number of spin
            self_energy[0,:,:,:,1] = self.self_energy[:,:,:].imag  # FIXME number of spin

        spectral_function = ncfile.createVariable('spectral_function','d',
            ('number_of_spins', 'number_of_kpoints', 'max_number_of_states', 'number_of_frequencies'))

        if self.spectral_function is not None:
            spectral_function[0,:,:,:] = self.spectral_function[:,:,:]  # FIXME number of spin

        ncfile.close()


    def write_renormalization(self):
        """Write the computed renormalization in a text file."""
        fname = str(self.output) + "_REN.txt"
        create_directory(fname)

        with open(fname, "w") as O:

            if self.zero_point_renormalization is not None:
                O.write("Total zero point renormalization (eV) for {} Q points\n".format(self.nqpt))
                for ikpt, kpt in enumerate(self.kpts):
                    O.write('Kpt: {0[0]} {0[1]} {0[2]}\n'.format(kpt))
                    for line in formatted_array_lines(self.zero_point_renormalization[ikpt,:].real*Ha2eV):
                        O.write(line)

            if self.temperature_dependent_renormalization is not None:
                O.write("Temperature dependence at Gamma (eV)\n")
                for iband in range(self.nband):
                  O.write('Band: {}\n'.format(iband))
                  for tt, T in enumerate(self.temperatures):
                    ren = self.temperature_dependent_renormalization[0,iband,tt].real * Ha2eV
                    O.write("{:>8.1f}  {:>12.8f}\n".format(T, ren))


    def write_broadening(self):
        """Write the computed broadening in a text file."""
        fname = str(self.output) + "_BRD.txt"
        create_directory(fname)

        with open(fname, "w") as O:

            if self.zero_point_broadening is not None:
                O.write("Total zero point broadening (eV) for {} Q points\n".format(self.nqpt))
                for ikpt, kpt in enumerate(self.kpts):
                    O.write('Kpt: {0[0]} {0[1]} {0[2]}\n'.format(kpt))
                    for line in formatted_array_lines(self.zero_point_broadening[ikpt,:].real*Ha2eV):
                        O.write(line)

            if self.temperature_dependent_broadening is not None:
                O.write("Temperature dependence at Gamma\n")
                for iband in range(self.nband):
                  O.write('Band: {}\n'.format(iband))
                  for tt, T in enumerate(self.temperatures):
                    brd = self.temperature_dependent_broadening[0,iband,tt].real * Ha2eV
                    O.write("{:>8.1f}  {:>12.8f}\n".format(T, brd))

