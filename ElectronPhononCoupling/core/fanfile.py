
from __future__ import print_function

__author__ = "Gabriel Antonius"

import os

import numpy as np
from numpy import zeros
import netCDF4 as nc

from . import EpcFile

class FanFile(EpcFile):
    
    def read_nc(self, fname=None):
        """Open the EIG2D.nc file and read it."""
        fname = fname if fname else self.fname

        with nc.Dataset(fname, 'r') as root:

            self.natom = len(root.dimensions['number_of_atoms'])
            self.nkpt = len(root.dimensions['number_of_kpoints'])
            self.nband = len(root.dimensions['max_number_of_states'])
            self.nsppol = len(root.dimensions['number_of_spins'])

            # number_of_spins, number_of_kpoints, max_number_of_states
            self.occ = root.variables['occupations'][:,:,:]

            #product_mband_nsppol,number_of_atoms,  number_of_cartesian_directions, number_of_atoms, number_of_cartesian_directions,
            # number_of_kpoints, product_mband_nsppol*2
            FANtmp = root.variables['second_derivative_eigenenergies_actif'][:,:,:,:,:,:,:]
            FANtmp2 = zeros((self.nkpt,2*self.nband,3,self.natom,3,self.natom,self.nband))
            FANtmp2 = np.einsum('ijklmno->nomlkji', FANtmp)
            FANtmp3 = FANtmp2[:, ::2, ...]  # Slice the even numbers
            FANtmp4 = FANtmp2[:, 1::2, ...] # Slice the odd numbers
            self.FAN = 1j*FANtmp4
            self.FAN += FANtmp3
            del FANtmp, FANtmp2, FANtmp3, FANtmp4

            self.eigenvalues = root.variables['eigenvalues'][:,:,:] #number_of_spins, number_of_kpoints, max_number_of_states   
            self.kpt = root.variables['reduced_coordinates_of_kpoints'][:,:]
            #self.iqpt = root.variables['current_q_point'][:]
            self.qred = root.variables['current_q_point'][:]
            self.wtq = root.variables['current_q_point_weight'][:]
            self.rprimd = root.variables['primitive_vectors'][:,:]

    
