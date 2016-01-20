
from __future__ import print_function

__author__ = "Gabriel Antonius"

import os

import numpy as N
from numpy import zeros
import netCDF4 as nc

from . import EpcFile

class EigFile(EpcFile):

    def read_nc(self, fname=None):
        """Open the Eig.nc file and read it."""
        fname = fname if fname else self.fname

        with nc.Dataset(fname, 'r') as root:

            self.EIG = root.variables['Eigenvalues'][:,:] 
            self.Kptns = root.variables['Kptns'][:,:]


