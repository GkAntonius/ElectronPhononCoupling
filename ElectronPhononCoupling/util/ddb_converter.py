import warnings
import numpy as np
from numpy import zeros, ones
import netCDF4 as nc


def convert_ddb_txt_to_netcdf(DDB_fname, DDB_nc_fname):
    """Read a _DDB file and convert it into a _DDB.nc file."""
    converter = DdbFileConverter()
    converter.read_txt(DDB_fname)
    converter.write_nc(DDB_nc_fname)


def convert_ddb_netcdf_to_txt(DDB_nc_fname, DDB_fname):
    """Read a _DDB file and convert it into a _DDB.nc file."""
    converter = DdbFileConverter()
    converter.read_nc(DDB_nc_fname)
    converter.write_txt(DDB_fname)


class DdbFileConverter(object):
    """Class to read a DDB file in txt format and write it in netCDF format."""
    # Note that in the netCDF file, it is written as
    # E2D[iat,icart,jat,jcart]
    # While it is read from the txt file as
    # E2D[icart,iat,jcart,jat]

    ncart = 3
    _ntypat = 1
    _natom = 1
    _nsym = 1
    _nkpt = 1
    _nband = 1

    def __init__(self):
        self.usepaw = 0
        self.natom = 1
        self.nkpt = 1
        self.nsppol = 1
        self.nsym = 1
        self.ntypat = 1
        self.occopt = 1
        self.nband = 1
        self.intxc = 0
        self.iscf = 1
        self.ixc = 1
        self.nspden = 1
        self.nspinor = 1

        self.dimekb = 1
        self.lmnmax = 1
        self.nblocks = 1

        self.dilatmx = 1.1
        self.ecut = 0.
        self.ecutsm = 0.
        self.dfpt_sciss = 0.
        self.tolwfr = 0.
        self.tphysel = 0.
        self.tsmear = 0.
        self.kptnrm = 1.

        self.rprim = np.identity(3, dtype=np.float)
        self.acell = np.ones(3, dtype=np.float)
        self.ngfft = np.zeros(3, dtype=np.int)

        self.xred = np.zeros((self.natom, 3), dtype=np.float)
        self.amu = np.zeros(self.ntypat, dtype=np.float)
        self.kpt = np.zeros((self.nkpt,3), dtype=np.float)
        self.wtk = np.ones((self.nkpt), dtype=np.float)
        self.occ = np.zeros(self.nband, dtype=np.float)
        self.spinat = np.zeros((self.natom,3), dtype=np.float)
        self.symafm = np.ones(self.nsym, dtype=np.int)
        self.symrel = np.zeros((self.nsym,3,3), dtype=np.int)
        self.tnons = np.zeros((self.nsym,3), dtype=np.float)
        self.typat = np.ones(self.natom, dtype=np.int)
        self.znucl = np.zeros(self.ntypat, dtype=np.float)
        self.zion = np.zeros(self.ntypat, dtype=np.float)

        self.pseudos = list()
        for i in range(self.ntypat):
            self.pseudos.append(list())
        self.pspso = np.zeros(self.ntypat, dtype=np.int)
        self.nekb = np.zeros(self.ntypat, dtype=np.int)

        self.qred = np.zeros(3, dtype=np.float)

        self.E2D = np.zeros((self.natom, self.ncart, self.natom, self.ncart), dtype=np.complex)
        self.BECT = np.zeros((self.ncart, self.natom, self.ncart), dtype=float)
        self.epsilon = np.zeros((self.ncart, self.ncart), dtype=complex)

    @property
    def ntypat(self):
        return self._ntypat

    @ntypat.setter
    def ntypat(self, val):
        self._ntypat = val

        # Init all arrays with that dimension
        self.pseudos = list()
        for i in range(self.ntypat):
            self.pseudos.append(list())
        self.pspso = np.zeros(self.ntypat, dtype=np.int)
        self.nekb = np.zeros(self.ntypat, dtype=np.int)
        self.znucl = np.zeros(self.ntypat, dtype=np.float)
        self.zion = np.zeros(self.ntypat, dtype=np.float)
        self.amu = np.zeros(self.ntypat, dtype=np.float)

    @property
    def natom(self):
        return self._natom

    @natom.setter
    def natom(self, val):
        self._natom = val

        # Init all arrays with that dimension
        self.xred = np.zeros((self.natom, 3), dtype=np.float)
        self.spinat = np.zeros((self.natom,3), dtype=np.float)
        self.typat = np.ones(self.natom, dtype=np.int)

        self.E2D = np.zeros((self.natom, self.ncart, self.natom, self.ncart),
                            dtype=np.complex)
        self.BECT = np.zeros((self.ncart, self.natom, self.ncart), dtype=float)

    @property
    def nsym(self):
        return self._nsym

    @nsym.setter
    def nsym(self, val):
        self._nsym = val

        # Init all arrays with that dimension
        self.symafm = np.ones(self.nsym, dtype=np.int)
        self.symrel = np.zeros((self.nsym,3,3), dtype=np.int)
        self.tnons = np.zeros((self.nsym,3), dtype=np.float)

    @property
    def nkpt(self):
        return self._nkpt

    @nkpt.setter
    def nkpt(self, val):
        self._nkpt = val

        # Init all arrays with that dimension
        self.kpt = np.zeros((self.nkpt,3), dtype=np.float)
        self.wtk = np.ones((self.nkpt), dtype=np.float)

    @property
    def nband(self):
        return self._nband

    @nband.setter
    def nband(self, val):
        self._nband = val

        # Init all arrays with that dimension
        self.occ = np.zeros(self.nband, dtype=np.float)

    def read_nc(self, fname):
        """Open the DDB.nc file and read it."""

        with nc.Dataset(fname, 'r') as root:

            self.natom = len(root.dimensions['number_of_atoms'])
            self.ncart = len(root.dimensions['number_of_cartesian_directions'])  # 3
            self.ntypat = len(root.dimensions['number_of_atom_species'])
            #self.nkpt = len(root.dimensions['number_of_kpoints'])  # Not relevant
            #self.nband = len(root.dimensions['max_number_of_states'])  # Not even there
            #self.nsppol = len(root.dimensions['number_of_spins'])  # Not relevant

            self.typat = root.variables['atom_species'][:self.natom]
            self.amu = root.variables['atomic_masses_amu'][:self.ntypat]
            self.rprim = root.variables['primitive_vectors'][:self.ncart,:self.ncart]
            self.xred = root.variables['reduced_atom_positions'][:self.natom,:self.ncart]
            self.qred = root.variables['q_point_reduced_coord'][:]


            # The d2E/dRdR' matrix
            self.E2D = np.zeros((self.natom, self.ncart, self.natom, self.ncart), dtype=np.complex)
            self.E2D.real = root.variables['second_derivative_of_energy'][:,:,:,:,0]
            self.E2D.imag = root.variables['second_derivative_of_energy'][:,:,:,:,1]
            self.E2D = np.einsum('aibj->bjai', self.E2D)  # Indicies are reversed when writing them from Fortran.

            self.BECT = root.variables['born_effective_charge_tensor'][:self.ncart,:self.natom,:self.ncart]

    def write_nc(self, fname):
        """Write a DDB.nc file."""

        with nc.Dataset(fname, 'w') as root:

            root.createDimension('number_of_atoms', self.natom)
            root.createDimension('number_of_cartesian_directions', 3)
            root.createDimension('number_of_atom_species', self.ntypat)
            root.createDimension('cplex', 2)

            data = root.createVariable('atom_species', 'i', ('number_of_atoms'))
            data[...] = self.typat[...]

            data = root.createVariable('atomic_masses_amu', 'd', ('number_of_atom_species'))
            data[...] = self.amu[...]

            data = root.createVariable('primitive_vectors', 'd', ('number_of_cartesian_directions', 'number_of_cartesian_directions'))
            data[...] = self.rprim[...]

            data = root.createVariable('reduced_atom_positions', 'd', ('number_of_atoms', 'number_of_cartesian_directions'))
            data[...] = self.xred[...]

            data = root.createVariable('q_point_reduced_coord', 'd', ('number_of_cartesian_directions'))
            data[...] = self.qred[...]

            data = root.createVariable('born_effective_charge_tensor', 'd', ('number_of_cartesian_directions',
                                                                            'number_of_atoms', 'number_of_cartesian_directions'))
            data[...] = self.BECT[...]

            # E2D dimensions must be swapped
            E2D_swap = np.zeros((self.natom, self.ncart, self.natom, self.ncart), dtype=np.complex)
            for iat in range(self.natom):
                for icart in range(self.ncart):
                    for jat in range(self.natom):
                        for jcart in range(self.ncart):
                            E2D_swap[iat, icart, jat, jcart] = self.E2D[icart, iat, jcart, jat]

            data = root.createVariable('second_derivative_of_energy', 'd', ('number_of_atoms', 'number_of_cartesian_directions',
                                                                            'number_of_atoms', 'number_of_cartesian_directions',
                                                                            'cplex'))
            E2D_swap = np.einsum('aibj->bjai', E2D_swap)  # Indicies are reversed when writing them from Fortran.
            data[...,0] = E2D_swap.real
            data[...,1] = E2D_swap.imag


    def read_txt(self, fname):
        """Open the DDB file and read it."""

        def read_int(f):
            line = f.next()
            return int(line.split()[-1])

        def read_float(f):
            line = f.next()
            return float(line.split()[-1].replace('D', 'e'))

        def read_int_array(f, shape):
            if isinstance(shape, int):
                size = shape
            else:
                size = np.prod(shape)
            values = list()
            line = f.next()
            for val in line.split()[1:]:
                values.append(int(val))
            while len(values) < size:
                line = f.next()
                for val in line.split():
                    values.append(int(val))
            arr = np.array(values, dtype=int)
            arr.reshape(shape)
            return arr

        def read_float_array(f, shape):
            if isinstance(shape, int):
                size = shape
            else:
                size = np.prod(shape)
            values = list()
            line = f.next()
            for val in line.split()[1:]:
                values.append(float(val.replace('D', 'e')))
            while len(values) < size:
                line = f.next()
                for val in line.split():
                    values.append(float(val.replace('D', 'e')))
            arr = np.array(values, dtype=float)
            arr.reshape(shape)
            return arr


        f = open(fname,'r')

        # Read header
        line = f.next()
        line = f.next()
        assert 'DERIVATIVE DATABASE' in line
        line = f.next()
        version = line.split()[-1]
        if version != '100401':
            warnings.warn(
                20*'*' + '\n' +
                'DDB version number has changed.\n'
                'expected: 100401\n' +
                'found: {}\n'.format(version) +
                'You should modify the script to adapt to this new DDB ' +
                'version. It is probably not very hard to do so.\n' +
                20*'*' + '\n')

        line = f.next()
        line = f.next()
        line = f.next()

        self.usepaw = read_int(f)
        self.natom = read_int(f)
        self.nkpt = read_int(f)
        self.nsppol = read_int(f)
        self.nsym = read_int(f)
        self.ntypat = read_int(f)
        self.occopt = read_int(f)
        self.nband = read_int(f)
        self.acell = read_float_array(f, 3)
        self.amu = read_float_array(f, self.ntypat)
        self.dilatmx = read_float(f)
        self.ecut = read_float(f)
        self.ecutsm = read_float(f)
        self.intxc = read_int(f)
        self.iscf = read_int(f)
        self.ixc = read_int(f)
        self.kpt = read_float_array(f, (self.nkpt, 3))
        self.kptnrm = read_float(f)
        self.ngfft = read_int_array(f, 3)
        self.nspden = read_int(f)
        self.nspinor = read_int(f)
        self.occ = read_float_array(f, self.nband)
        self.rprim = read_float_array(f, (3,3))
        self.dfpt_sciss = read_float(f)
        self.spinat = read_float_array(f, (self.natom, 3))
        self.symafm = read_int_array(f, self.nsym)
        self.symrel = read_int_array(f, (self.nsym, 3, 3))
        self.tnons = read_float_array(f, (self.nsym, 3))
        self.tolwfr = read_float(f)
        self.tphysel = read_float(f)
        self.tsmear = read_float(f)
        self.typat = read_int_array(f, (self.natom))
        self.wtk = read_float_array(f, self.nkpt)
        self.xred = read_float_array(f, (self.natom, 3))
        self.znucl = read_float_array(f, self.ntypat)
        self.zion = read_float_array(f, self.ntypat)

        line = f.next()

        # Read pseudopotentials
        line = f.next()
        line = f.next()
        line = f.next()
        line = f.next()
        parts = line.split()
        self.dimekb = int(parts[2])
        self.lmnmax = int(parts[4])

        for i in range(self.ntypat):
            line = f.next()
            parts = line.split()
            pspso = int(parts[4])
            nekb = int(parts[6])
            self.pspso[i] = pspso
            self.nekb[i] = nekb

            line = f.next()

            for j in range(nekb):  # I think nchannel = nekb
                line = f.next()
                parts = line.split()
                iln = int(parts[0])
                lpsang = int(parts[1])
                iproj = int(parts[2])
                channel = [iln, lpsang, iproj]
                nekb_read = 0
                for part in parts[3:]:
                    channel.append(float(part.replace('D', 'e')))
                    nekb_read += 1
                while nekb_read < nekb:
                    line = f.next()
                    for part in line.split():
                        channel.append(float(part.replace('D', 'e')))
                        nekb_read += 1

                self.pseudos[i].append(channel)

        line = f.next()

        # Read the database
        line = f.next()
        assert 'Database of total energy derivatives' in line
        
        line = f.next()
        self.nblocks = int(line.split()[-1])
        line = f.next()

        line = f.next()
        nelem = int(line.split()[-1])

        line = f.next()
        self.qred[0] = float(line.split()[1].replace('D', 'e'))
        self.qred[1] = float(line.split()[2].replace('D', 'e'))
        self.qred[2] = float(line.split()[3].replace('D', 'e'))

        for i in range(nelem):
            line = f.next()
            parts = line.split()
            icart = int(parts[0])
            ipert = int(parts[1])
            jcart = int(parts[2])
            jpert = int(parts[3])
            re = float(parts[4].replace('D', 'e'))
            im = float(parts[5].replace('D', 'e'))
            val = np.complex(re, im)

            # Dynamical matrix
            if ipert <= self.natom and jpert <= self.natom:
                self.E2D[ipert-1,icart-1,jpert-1,jcart-1] = val
            
            # Born effective charge tensor
            elif ipert <= self.natom and jpert == self.natom+2:
                self.BECT[icart-1,ipert-1,jcart-1] = re

            # Dielectric matrix
            elif ipert == self.natom+2 and jpert == self.natom+2:
                self.epsilon[icart-1,jcart-1] = val

        f.close()


    def write_txt(self, fname):
        """Write a DDB file in text format."""

        def listify(val):
            """Make a flat list out of anything"""
            if not '__iter__' in dir(val):
                return [val]
            return np.array(val).flatten().tolist()

        def format_int(key, val, n_per_line=9):
            vals = listify(val)
            S = '  {:>8}'.format(key)
            S += 5*' '
            for i, v in enumerate(vals):
                if i % n_per_line == 0 and i > 0:
                    S += '\n' + 15*' '
                S += '{:>5}'.format(v) 
            return S

        def format_float(key, val, n_per_line=3):
            vals = listify(val)
            S = '  {:>8}'.format(key)
            for i, v in enumerate(vals):
                if i % n_per_line == 0 and i > 0:
                    S += '\n' + 10*' '
                S += ' {:>21.14e}'.format(v) 
            S = S.replace('e+', 'D+')
            S = S.replace('e-', 'D-')
            return S

        # Construct header
        lines = list()

        # Header
        lines.append('')
        lines.append(' **** DERIVATIVE DATABASE ****')
        lines.append('+DDB, Version number    100401')
        lines.append('')
        lines.append('  Note : temporary (transfer) database')
        lines.append('')
        lines.append(format_int('usepaw', self.usepaw))
        lines.append(format_int('natom', self.natom))
        lines.append(format_int('nkpt', self.nkpt))
        lines.append(format_int('nsppol', self.nsppol))
        lines.append(format_int('nsym', self.nsym))
        lines.append(format_int('ntypat', self.ntypat))
        lines.append(format_int('occopt', self.occopt))
        lines.append(format_int('nband', self.nband))
        lines.append(format_float('acell', self.acell))
        lines.append(format_float('amu', self.amu))
        lines.append(format_float('dilatmx', self.dilatmx))
        lines.append(format_float('ecut', self.ecut))
        lines.append(format_float('ecutsm', self.ecutsm))
        lines.append(format_int('intxc', self.intxc))
        lines.append(format_int('iscf', self.iscf))
        lines.append(format_int('ixc', self.ixc))
        lines.append(format_float('kpt', self.kpt))
        lines.append(format_float('kptnrm', self.kptnrm))
        lines.append(format_int('ngfft', self.ngfft))
        lines.append(format_int('nspden', self.nspden))
        lines.append(format_int('nspinor', self.nspinor))
        lines.append(format_float('occ', self.occ))
        lines.append(format_float('rprim', self.rprim))
        lines.append(format_float('dfpt_sciss', self.dfpt_sciss))
        lines.append(format_float('spinat', self.spinat))
        lines.append(format_int('symafm', self.symafm, 12))
        lines.append(format_int('symrel', self.symrel))
        lines.append(format_float('tnons', self.tnons))
        lines.append(format_float('tolwfr', self.tolwfr))
        lines.append(format_float('tphysel', self.tphysel))
        lines.append(format_float('tsmear', self.tsmear))
        lines.append(format_int('typat', self.typat))
        lines.append(format_float('wtk', self.wtk))
        lines.append(format_float('xred', self.xred))
        lines.append(format_float('znucl', self.znucl))
        lines.append(format_float('zion', self.zion))

        # Information on the pseudopotentials.
        lines.append('')
        lines.append('  Description of the potentials (KB energies)')
        lines.append('  vrsio8 (for pseudopotentials)=100401')
        lines.append('  usepaw =  {}'.format(self.usepaw))
        lines.append('  dimekb = {:>2}       lmnmax= {:>2}'.format(
                        self.dimekb, self.lmnmax))

        nekb_per_line = 4
        for itypat in range(self.ntypat):
            lines.append('  Atom type= {:>4}   pspso=  {}   nekb= {:>3}'.format(
                            itypat+1, self.pspso[itypat], self.nekb[itypat]))
            lines.append('  iln lpsang iproj  ekb(:)')

            pseudo = self.pseudos[itypat]
            for channel in pseudo:
                iln = channel[0]
                lpsang = channel[1]
                iproj = channel[2]
                ekb = channel[3:]
                S = ' {:>5} {:>5} {:>5}   '.format(iln, lpsang, iproj)
                for i, E in enumerate(ekb):
                    if i % nekb_per_line == 0 and i > 0:
                        S += '\n' + 21*' '
                    S += '{:>15.7e}'.format(E) 
                lines.append(S)

        # Database
        lines.append('')
        lines.append(' **** Database of total energy derivatives ****')
        lines.append(' Number of data blocks= {:>4}'.format(self.nblocks))
        lines.append('')

        has_BECT = not np.allclose(self.BECT, 0.)
        has_epsilon = not np.allclose(self.epsilon, 0.)

        nelem = (3 * self.natom) ** 2

        if has_BECT:
            nelem += 9 * self.natom * 2

        if has_epsilon:
            nelem += 9

        lines.append(' 2nd derivatives (non-stat.)  - # elements : {:>7}'.format(
                        nelem))

        S = ' qpt {:>15.8e} {:>15.8e} {:>15.8e}   1.0'.format(*self.qred)
        S = S.replace('e+', 'D+')
        S = S.replace('e-', 'D-')
        lines.append(S)

        iperts = list(range(1,self.natom+1))
        if has_BECT or has_epsilon:
            iperts.append(self.natom+2)

        for ipert in iperts:
          for icart in range(1,4):
            for jpert in iperts:
              for jcart in range(1,4):

                if ipert <= self.natom and jpert <= self.natom:
                    val = self.E2D[ipert-1,icart-1,jpert-1,jcart-1]
                elif ipert <= self.natom and jpert == self.natom+2:
                    val = self.BECT[icart-1,ipert-1,jcart-1]
                    if not has_BECT:
                        continue
                elif ipert == self.natom+2 and jpert <= self.natom:
                    val = self.BECT[icart-1,jpert-1,jcart-1]
                    if not has_BECT:
                        continue
                elif ipert == self.natom+2 and jpert == self.natom+2:
                    val = self.epsilon[icart-1,jcart-1]
                    if not has_epsilon:
                        continue
                else:
                    raise Exception('Wrong value for ipert or jpert')

                S = 4 * ' {:>3}' + 2 * ' {:>21.14e}'
                S = S.format(icart,ipert,jcart,jpert,
                             np.real(val), np.imag(val))
                S = S.replace('e+', 'D+')
                S = S.replace('e-', 'D-')
                lines.append(S)

        # Write the file
        with open(fname,'w') as f:
            f.write('\n'.join(lines))

