
from __future__ import print_function

__author__ = "Gabriel Antonius"

import os

import numpy as N
from numpy import zeros
import netCDF4 as nc

from . import EpcFile

class Eigr2dFile(EpcFile):

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

            # number_of_atoms, number_of_cartesian_directions, number_of_atoms, number_of_cartesian_directions,
            # number_of_kpoints, product_mband_nsppol, cplex
            EIG2Dtmp = root.variables['second_derivative_eigenenergies'][:,:,:,:,:,:,:]

            EIG2Dtmp2 = zeros((self.nkpt,2*self.nband,3,self.natom,3,self.natom,self.nband))
            EIG2Dtmp2 = N.einsum('ijklmno->mnlkjio', EIG2Dtmp)

            self.EIG2D = 1j*EIG2Dtmp2[...,1]
            self.EIG2D += EIG2Dtmp2[...,0]

            del EIG2Dtmp, EIG2Dtmp2

            self.eigenvalues = root.variables['eigenvalues'][:,:,:] #number_of_spins, number_of_kpoints, max_number_of_states   
            self.kpt = root.variables['reduced_coordinates_of_kpoints'][:,:]
            #self.iqpt = root.variables['current_q_point'][:]
            self.qred = root.variables['current_q_point'][:]
            self.wtq = root.variables['current_q_point_weight'][:]
            self.rprimd = root.variables['primitive_vectors'][:,:]

