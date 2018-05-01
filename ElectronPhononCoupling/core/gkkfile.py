
from __future__ import print_function

__author__ = "Gabriel Antonius"

import os

import numpy as np
from numpy import zeros, einsum
import netCDF4 as nc

from .ddbfile import DdbFile

from .mpi import MPI, comm, size, rank, mpi_watch

from . import EpcFile

from ..util.ncutil import nc_copy

__all__ = ['GkkFile']


class GkkFile(EpcFile):
    
    def read_nc(self, fname=None):
        """Open the GKK.nc file and read it."""
        fname = fname if fname else self.fname

        super(GkkFile, self).read_nc(fname)

        with nc.Dataset(fname, 'r') as root:

            self.natom = len(root.dimensions['number_of_atoms'])
            self.nkpt = len(root.dimensions['number_of_kpoints'])
            self.nband = len(root.dimensions['max_number_of_states'])
            self.nsppol = len(root.dimensions['number_of_spins'])

            # number_of_spins, number_of_kpoints, max_number_of_states
            self.occ = root.variables['occupations'][:,:,:]

            # number_of_spins, number_of_kpoints, max_number_of_states   
            self.eigenvalues = root.variables['eigenvalues'][:,:,:]

            # number_of_kpoints, 3
            self.kpt = root.variables['reduced_coordinates_of_kpoints'][:,:]
            self.qred = root.variables['current_q_point'][:]
            self.wtq = root.variables['current_q_point_weight'][:]
            self.rprimd = root.variables['primitive_vectors'][:,:]

            # nband, natom, ncart, nkpt, product_mband_nsppol*2 
            GKKtmp = root.variables['second_derivative_eigenenergies_actif'][:,:,:,:,:]
            GKKtmp2 = np.einsum('ijkno->nokji', GKKtmp)
            self.GKK = np.zeros((self.nkpt, self.nsppol*self.nband, 3, self.natom, self.nband), dtype=np.complex)
            self.GKK.real[...] = GKKtmp2[:, ::2, ...]
            self.GKK.imag[...] = GKKtmp2[:, 1::2, ...]

            # Now the second band index "n" refers to the state at "k+q,n"
            self.GKK = np.reshape(self.GKK,(self.nkpt,self.nsppol,self.nband,3,self.natom,self.nband))

            self.GKK_mode = None

    def write_nc(self, fname, overwrite=False):
        """
        Write a new GKK.nc file.
        Most arrays are directly copied from the original file,
        except for the gkk matrix elements, which are the ones stored in memory.
        This function is used when altering the gkk matrix elements.
        """

        if not overwrite and os.path.exists(fname):
            raise Exception('File exists: {}\nUse overwrite=True to overwrite'.format(fname))

        with nc.Dataset(fname, 'w') as dsout:

            with nc.Dataset(self.fname, 'r') as dsin:

                nc_copy(dsin, dsout, except_variables=('second_derivative_eigenenergies_actif',))

            data = dsout.createVariable('second_derivative_eigenenergies_actif', np.dtype('float64'),
                ('max_number_of_states', 'number_of_atoms_for_gkk',
                 'number_of_cartesian_directions_for_gkk', 'number_of_kpoints',
                 'product_mband_nsppol2')
                )

            GKKtmp3 = np.reshape(self.GKK,(self.nkpt, self.nsppol*self.nband, 3, self.natom, self.nband))
            GKKtmp2 = np.zeros((self.nkpt, 2*self.nsppol*self.nband, 3, self.natom, self.nband), dtype=np.float64)
            GKKtmp2[:, ::2, ...]  = GKKtmp3.real[...]
            GKKtmp2[:, 1::2, ...] = GKKtmp3.imag[...]
            GKKtmp = np.einsum('nokji->ijkno', GKKtmp2)

            data[...] = GKKtmp


    def get_gkk_squared(self):
        """
        Get squared values of gkk with  reordered indices.
        Returns:
            gkk2[nkpt,nband,3,natom,3,natom,nband]
        """

        # No spin polarization at the moment.

        gkk2 = zeros((self.nkpt, self.nband, 3, self.natom, 3, self.natom,
                      self.nband), dtype=np.complex)

        # nkpt,nband,3,natom,3,natom,nband
        gkk2 = einsum('ijklm,ijnom->ijklnom', self.GKK[:,0,...],
                                              self.GKK[:,0,...].conjugate())

        return gkk2

    def get_kpt_gkk_squared(self, ikpt):
        """
        Get squared values of gkk with  reordered indices for a ginven k-point.
        Returns:
            gkk2[nband,3,natom,3,natom,nband]
        """

        # No spin polarization at the moment.

        gkk2 = zeros((self.nband, 3, self.natom, 3, self.natom,
                      self.nband), dtype=np.complex)

        # nkpt,nband,3,natom,3,natom,nband
        gkk2 = einsum('jklm,jnom->jklnom', self.GKK[ikpt,0,...],
                                           self.GKK[ikpt,0,...].conjugate())

        return gkk2

    def get_gkk_mode(self, ddb, noscale=False):
        """
        Convert the gkk from the cartesian/atomic basis to the mode basis.

        The resulting gkk are aslo scaled by the root mean squared displacement
        of the mode, that is sqrt(hbar/(2 M omega)).

        Arguments
        ---------
        ddb:
            DdbFile object.
        noscale:
            Do not scale by the root mean squared displacement.

        Returns
        -------

        GKK_mode: [nkpt, nband, nband, nmode]
        """
        assert ddb.nmode == 3 * self.natom

        self.GKK_mode = np.zeros((self.nkpt, self.nband, self.nband, ddb.nmode), dtype=np.complex)

        polvec = ddb.get_reduced_displ(noscale=noscale)

        self.GKK_mode = np.einsum('kniam,oia->knmo', self.GKK[:,0,...], polvec)

        return self.GKK_mode

    def get_gkk_cart(self, ddb, noscale=False):
        """
        Convert the gkk back from the mode basis to the cartesian/atomic basis.

        Arguments
        ---------
        ddb:
            DdbFile object.
        noscale:
            Do not scale by the root mean squared displacement.

        Returns
        -------

        GKK_mode: [nkpt, nband, nband, nmode]
        """
        polvec = ddb.get_reduced_displ(noscale=noscale)

        polvec_mat =  np.zeros((ddb.nmode, ddb.nmode), dtype=np.complex)

        for imode in range(ddb.nmode):
            for icart in range(3):
                for iat in range(self.natom):
                    polvec_mat[imode,icart+3*iat] = polvec[imode,icart,iat]

        inv_polvec = np.linalg.inv(polvec_mat)

        for ikpt in range(self.nkpt):
            for iband in range(self.nband):
                for jband in range(self.nband):

                    GKK_mode = self.GKK_mode[ikpt,iband,jband,:]
                    GKK_cart = np.dot(inv_polvec, GKK_mode)

                    for icart in range(3):
                        for iat in range(self.natom):

                            self.GKK[ikpt,0,iband,icart,iat,jband] = (
                                GKK_cart[icart+3*iat])
        return self.GKK

    def get_gkk2_DW_mode(self, ddb):
        """
        Compute the squared gkk matrix elements for the Debye-Waller
        self-energy.

        The resulting gkk are aslo scaled by the mean squared displacement
        of the mode, that is hbar/(M omega).

        Note that, for a given q-point contribution,
        this function must be called with the GKK at q=Gamma, and the DDB
        and the given q-point.

        Arguments
        ---------
        ddb:
            DdbFile object.

        Returns
        -------

        GKK2_DW_mode: [nkpt, nband, nband, nmode]
        """
        assert ddb.nmode == 3 * self.natom


        self.GKK2_DW_mode = np.zeros((self.nkpt, self.nband, self.nband, ddb.nmode), dtype=np.complex)

        polvec = ddb.get_reduced_displ()

        g_r = np.einsum('kniam->knim', self.GKK[:,0,...])

        for imode in range(ddb.nmode):

            g_l1 = np.einsum('kniam,ia->knam', self.GKK[:,0,...], polvec[imode,...]) 

            g_l2 = np.einsum('knam,ia->knim', np.conj(g_l1), polvec[imode,...]) 

            g2 = np.einsum('knim,knim->knm', g_l2, g_r)

            self.GKK2_DW_mode[...,imode] = np.real(g2)

        return self.GKK2_DW_mode


    @mpi_watch
    def broadcast(self):
        """Broadcast the data from master to all workers."""
    
        comm.Barrier()

        if rank == 0:
            dim = np.array([self.natom, self.nkpt, self.nband, self.nsppol], dtype=np.int)
        else:
            dim = np.empty(4, dtype=np.int)

        comm.Bcast([dim, MPI.INT])

        if rank != 0:

            self.natom, self.nkpt, self.nband, self.nsppol = dim[:]

            self.occ = np.empty((self.nsppol, self.nkpt, self.nband), dtype=np.float)

            self.GKK = np.empty((self.nkpt,self.nsppol,self.nband,3,self.natom,self.nband),
                                dtype=np.complex)

            self.eigenvalues = np.empty((self.nsppol, self.nkpt, self.nband), dtype=np.float)

            # number_of_kpoints, 3
            self.kpt = np.empty((self.nkpt, 3), dtype=np.float)
            self.qred = np.empty(3, dtype=np.float)
            self.wtq = np.empty(1, dtype=np.float)
            self.rprimd = np.empty((3, 3), dtype=np.float)

        comm.Bcast([self.occ, MPI.DOUBLE])
        comm.Bcast([self.GKK, MPI.COMPLEX])
        comm.Bcast([self.eigenvalues, MPI.DOUBLE])
        comm.Bcast([self.kpt, MPI.DOUBLE])
        comm.Bcast([self.qred, MPI.DOUBLE])
        comm.Bcast([self.wtq, MPI.DOUBLE])
        comm.Bcast([self.rprimd, MPI.DOUBLE])
