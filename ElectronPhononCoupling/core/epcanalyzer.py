from __future__ import print_function

import warnings
import copy

import numpy as np
import netCDF4 as nc

from .util import create_directory, formatted_array_lines

from .qptanalyzer import QptAnalyzer

# =========================================================================== #

class EpcAnalyzer(object):
    """
    Main class for analysing electron-phonon coupling related quantities.

    It is intented to analyse the files produced by ABINIT
    in a phonon response-function calculation, with one q-point per dataset,
    the first q-point being Gamma.

    """
    wtq = None
    broadening = None
    temperatures = []
    omegase = []
    smearing = None

    zero_point_renormalization = None
    zero_point_broadening = None
    temperature_dependent_renormalization = None
    temperature_dependent_broadening = None

    renormalization_is_dynamical = False
    broadening_is_dynamical = False

    self_energy = None
    spectral_function = None

    def __init__(self,
                 nqpt=1,
                 wtq=[1.0],
                 eig0_fname='',
                 eigq_fnames=list(),
                 DDB_fnames=list(),
                 EIGR2D_fnames=list(),
                 EIGI2D_fnames=list(),
                 FAN_fnames=list(),
                 temp_range=[0,0,1],
                 omega_range=[0,0,1],
                 smearing=0.00367,
                 output='epc.out',
                 **kwargs):

        # Check that the minimum number of files is present
        if not eig0_fname:
            raise Exception('Must provide a file for eig0_fname')

        if not EIGR2D_fnames:
            raise Exception('Must provide at least one file for EIGR2D_fnames')

        if not DDB_fnames:
            raise Exception('Must provide at least one file for DDB_fnames')

        if len(wtq) != nqpt:
            raise Exception("Must provide nqpt weights in the 'wtq' list.")

        # Set basic quantities
        self.nqpt = nqpt
        self.wtq = wtq

        # Set file names
        self.eig0_fname = eig0_fname
        self.eigq_fnames = eigq_fnames
        self.DDB_fnames = DDB_fnames
        self.EIGR2D_fnames = EIGR2D_fnames
        self.EIGI2D_fnames = EIGI2D_fnames
        self.FAN_fnames = FAN_fnames

        # Initialize a single QptAnalyzer
        self.qptanalyzer = QptAnalyzer(
            eig0_fname=self.eig0_fname,
            DDB_fname=self.DDB_fnames[0],
            EIGR2D0_fname=self.EIGR2D_fnames[0],
            FAN0_fname=self.FAN_fnames[0] if self.FAN_fnames else None,
            wtq=self.wtq[0],
            smearing=self.smearing,
            temperatures=self.temperatures,
            omegase=self.omegase,
            )

        # Read the first DDB and check that it is Gamma
        self.qptanalyzer.read_nonzero_files()

        if not self.qptanalyzer.is_gamma:
            raise Exception('The first Q-point is not Gamma.')

        # Read other files at q=0 
        self.qptanalyzer.read_zero_files()

        # Compute degeneracies
        self.qptanalyzer.eig0.get_degen()

        # Get arrays dimensions
        self.nkpt = self.qptanalyzer.eigr2d0.nkpt
        self.nband = self.qptanalyzer.eigr2d0.nband
        self.kpts = self.qptanalyzer.eigr2d0.kpt[:,:]

        # Set parameters
        self.set_temp_range(temp_range)
        self.set_omega_range(omega_range)
        self.set_smearing(smearing)
        self.set_output(output)

    def set_iqpt(self, iqpt):
        """
        Give the qptanalyzer the weight and files corresponding
        to one particular qpoint and read the files. 
        """
        self.qptanalyzer.wtq = self.wtq[iqpt]
        self.qptanalyzer.ddb.fname = self.DDB_fnames[iqpt]

        if self.EIGR2D_fnames:
            self.qptanalyzer.eigr2d.fname = self.EIGR2D_fnames[iqpt]

        if self.eigq_fnames:
            self.qptanalyzer.eigq.fname = self.eigq_fnames[iqpt]

        if self.FAN_fnames:
            self.qptanalyzer.fan.fname = self.FAN_fnames[iqpt]

        if self.EIGI2D_fnames:
            self.qptanalyzer.eigi2d.fname = self.EIGI2D_fnames[iqpt]

        self.qptanalyzer.read_nonzero_files()


    def set_temp_range(self, temp_range=(0, 0, 1)):
        """Set the minimum, makimum and step temperature."""
        self.temperatures = np.arange(*temp_range, dtype=float)
        self.qptanalyzer.temperatures = self.temperatures

    def set_omega_range(self, omega_range=(0, 0, 1)):
        """Set the minimum, makimum and step frequency for the self-energy."""
        self.omegase = np.arange(*omega_range, dtype=float)
        self.nomegase = len(self.omegase)
        self.qptanalyzer.omegase = self.omegase

    def set_smearing(self, smearing_Ha):
        """Set the smearing, in Hartree."""
        self.smearing = smearing_Ha
        self.qptanalyzer.smearing = smearing
    
    def set_output(self, root):
        """Set the root for output names."""
        self.output = root

    def sum_qpt_function(self, func_name, *args, **kwargs):
        """Call a certain function or each q-points and sum the result."""

        self.set_iqpt(0)
        q0 = getattr(self.qptanalyzer, func_name)(*args, **kwargs)
        total = copy(q0)

        for iqpt in range(1, self.nqpt):
            self.set_iqpt(iqpt)
            qpt = getattr(self.qptanalyzer, func_name)(*args, **kwargs)
            total += qpt

        return total

    def compute_static_zp_renormalization(self):
        """Compute the zero-point renormalization in a static scheme."""
        self.zero_point_renormalization = self.sum_qpt_function('get_zpr_static')
        self.renormalization_is_dynamical = False

    def compute_static_td_renormalization(self):
        """
        Compute the temperature-dependent renormalization in a static scheme.
        """
        self.temperature_dependent_renormalization = self.sum_qpt_function('get_tdr_static')
        self.renormalization_is_dynamical = False

    def compute_dynamical_td_renormalization(self):
        """
        Compute the temperature-dependent renormalization in a dynamical scheme.
        """
        self.temperature_dependent_renormalization = self.sum_qpt_function('get_tdr_dynamical')
        self.renormalization_is_dynamical = True

    def compute_dynamical_zp_renormalization(self):
        """Compute the zero-point renormalization in a dynamical scheme."""
        self.zero_point_renormalization = self.sum_qpt_function('get_zpr_dynamical')
        self.renormalization_is_dynamical = True

    def compute_static_control_td_renormalization(self):
        """
        Compute the temperature-dependent renormalization in a static scheme
        with the transitions split between active and sternheimer.
        """
        self.temperature_dependent_renormalization = self.sum_qpt_function('get_tdr_static_active')
        self.renormalization_is_dynamical = False

    def compute_static_control_zp_renormalization(self):
        """
        Compute the zero-point renormalization in a static scheme
        with the transitions split between active and sternheimer.
        """
        self.zero_point_renormalization = self.sum_qpt_function('get_zpr_static_active')
        self.renormalization_is_dynamical = False

    def compute_static_td_broadening(self):
        """
        Compute the temperature-dependent broadening in a static scheme
        from the EIGI2D files.
        """
        self.temperature_dependent_broadening = self.sum_qpt_function('get_tdb_static')
        self.broadening_is_dynamical = False

    def compute_static_zp_broadening(self):
        """
        Compute the zero-point broadening in a static scheme
        from the EIGI2D files.
        """
        self.zero_point_broadening = self.sum_qpt_function('get_zpb_static')
        self.broadening_is_dynamical = False

    def compute_dynamical_td_broadening(self):
        warnings.warn("Dynamical lifetime at finite temperature is not yet implemented...proceed with static lifetime")
        return self.compute_static_td_broadening()

    def compute_dynamical_zp_broadening(self):
        """
        Compute the zero-point broadening in a dynamical scheme.
        """
        self.zero_point_broadening = self.sum_qpt_function('get_zpb_dynamical')
        self.broadening_is_dynamical = True

    def compute_static_control_td_broadening(self):
        warnings.warn("Static lifetime at finite temperature with control over smearing is not yet implemented...proceed with static lifetime")
        return self.compute_static_td_broadening()

    def compute_static_control_zp_broadening(self):
        """
        Compute the zero-point broadening in a static scheme
        from the FAN files.
        """
        self.zero_point_broadening = self.sum_qpt_function('get_zpb_static_active')
        self.broadening_is_dynamical = False

    def compute_self_energy(self):
        """
        Compute the zp frequency-dependent self-energy from one q-point.
    
        The self-energy is evaluated on a frequency mesh 'omegase' that is shifted by the bare energies,
        such that, what is retured is
    
            Simga'_kn(omega) = Sigma_kn(omega + E^0_kn)
    
        """
        self.self_energy = self.sum_qpt_function('get_zp_self_energy')


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

        #fan = ncfile.createVariable('fan_zero_point_renormalization','d',
        #    ('number_of_spins', 'number_of_kpoints', 'max_number_of_states'))

        #ddw = ncfile.createVariable('ddw_zero_point_renormalization','d',
        #    ('number_of_spins', 'number_of_kpoints', 'max_number_of_states'))

        if self.zero_point_renormalization is not None:
            zpr[0,:,:] = self.zero_point_renormalization[:,:].real  # FIXME number of spin
            #fan[0,:,:] = self.fan_zero_point_renormalization[:,:].real  # FIXME number of spin
            #ddw[0,:,:] = self.ddw_zero_point_renormalization[:,:].real  # FIXME number of spin

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

