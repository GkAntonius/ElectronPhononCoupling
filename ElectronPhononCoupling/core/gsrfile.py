from __future__ import print_function

__author__ = "Gabriel Antonius"

from copy import copy

import numpy as np
import netCDF4 as nc

from . import EpcFile

from .mpi import MPI, comm, size, rank, mpi_watch

__all__ = ['GsrFile']


class GsrFile(EpcFile):

    def __init__(self, *args, **kwargs):
        super(GsrFile, self).__init__(*args, **kwargs)
        self.degen = None

    def read_nc(self, fname=None):
        """Open the GSR.nc file and read important informations."""
        fname = fname if fname else self.fname

        #super(GsrFile, self).read_nc(fname)

        with nc.Dataset(fname, 'r') as ds:

            self.EIG = ds.variables['eigenvalues'][:,:,:] 
            self.Kptns = ds.variables['reduced_coordinates_of_kpoints'][:,:]
            self.occ = ds.variables['occupations'][:,:,:]
            self.nspin, self.nkpt, self.nband = self.EIG.shape

            self.zion = ds.variables['zionpsp'][...]
            self.species = ds.variables['atom_species'][...]
            #atom_names = ds.variables['atom_species_names'][...]

        self.atoms_zion = list()
        for iat in self.species:
            self.atoms_zion.append(self.zion[iat -1])
        self.atoms_zion = np.array(self.atoms_zion)

    @mpi_watch
    def broadcast(self):
        """Broadcast the data from master to all workers."""
        comm.Barrier()

        if rank == 0:
            nspin, nkpt, nband = self.EIG.shape
            dim = np.array([nspin, nkpt, nband], dtype=np.int)
        else:
            dim = np.empty(3, dtype=np.int)
            self.nspin, self.nkpt, self.nband = dim

        comm.Bcast([dim, MPI.INT])

        if rank != 0:
            self.EIG = np.empty(dim, dtype=np.float64)
            self.Kptns = np.empty((dim[1],3), dtype=np.float64)
            self.occ = np.empty(dim, dtype=np.float64)

        comm.Bcast([self.EIG, MPI.DOUBLE])
        comm.Bcast([self.Kptns, MPI.DOUBLE])
        comm.Bcast([self.occ, MPI.DOUBLE])

