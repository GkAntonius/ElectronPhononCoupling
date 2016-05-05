from __future__ import print_function

__author__ = "Gabriel Antonius"

from copy import copy

import numpy as np
import netCDF4 as nc

from . import EpcFile

class EigFile(EpcFile):

    def __init__(self, *args, **kwargs):
        super(EigFile, self).__init__(*args, **kwargs)
        self.degen = None

    def read_nc(self, fname=None):
        """Open the Eig.nc file and read it."""
        fname = fname if fname else self.fname

        with nc.Dataset(fname, 'r') as root:

            self.EIG = root.variables['Eigenvalues'][:,:] 
            self.Kptns = root.variables['Kptns'][:,:]

    def iter_spin_band_eig(self, ikpt):
        """
        Iterator over spin index, band index, and eigenvalues at one k-point.
        Yields tuples (ispin, iband, eig) in order of increasing eig.
        """
        nspin, nkpt, nband = self.EIG.shape
        sbe = [(ispin, 0, self.EIG[ispin,ikpt,0]) for ispin in range(nspin)]
        cmp_sbe = lambda sbe1, sbe2: cmp(sbe1[2], sbe2[2])
        while sbe:
            min_sbe = sorted(sbe, cmp=cmp_sbe)[0]
            yield min_sbe
    
            i = sbe.index(min_sbe)
            ispin, iband, eig = min_sbe
    
            if iband == nband - 1:
                del sbe[i]
            else:
                sbe[i] = (ispin, iband+1, self.EIG[ispin, ikpt, iband+1])
    
    def get_degen(self):
        """
        Compute the degeneracy of the bands.
    
        Returns
        -------
        degen: 2D list (nkpt, )
            For each k-point, contains a list of groups of (s, n) tuples
            which are degenerated.
            For example, if there is only a single spin (spin unpolarized case)
            and two k-points, and at the first k-point there are two triplets,
            then the output is
                [[[(0,1), (0,2), (0,3)],  [(0,4), (0,5), (0,6)]], []]
    
        """
        nspin, nkpt, nband = self.EIG.shape
    
        degen = list()
        for ikpt in range(nkpt):
    
            kpt_degen = list()
            group = list()
            last_ispin, last_iband, last_eig = 0, 0, -float('inf')
    
            for sbe in self.iter_spin_band_eig(ikpt):
                ispin, iband, eig = sbe
    
                if np.isclose(last_eig, eig, rtol=1e-12, atol=1e-5):
                    if not group:
                        group.append((last_ispin, last_iband))
                    group.append((ispin, iband))
    
                else:
                    if group:
                        kpt_degen.append(group)
                        group = list()
    
                last_ispin, last_iband, last_eig = ispin, iband, eig
    
            degen.append(kpt_degen)

        self.degen = degen
    
        return degen

    def make_average(self, arr):
        """ 
        Average a quantity over degenerated states.
        Does not work with spin yet.
    
        Arguments
        ---------
    
        arr: numpy.ndarray(..., nkpt, nband)
            An array of any dimension, of which the two last indicies are
            the kpoint and the band.
    
        Returns
        -------
    
        arr: numpy.ndarray(..., nkpt, nband)
            The array with the values of degenerated bands averaged.
    
        """

        if not self.degen:
            self.get_degen()

        nkpt, nband = arr.shape[-2:]
    
        for ikpt in range(nkpt):
            for group in self.degen[ikpt]:
                average = copy(arr[...,ikpt,group[0][1]])
                for ispin, iband in group[1:]:
                    average += arr[...,ikpt,iband]
    
                average /= len(group)
                for ispin, iband in group:
                    arr[...,ikpt,iband] = average
    
        return arr

    def symmetrize_fan_degen(self, fan_epc):
        """
        Enforce coupling terms to be zero on the diagonal
        and in degenerate states subset.
    
        Arguments
        ---------
        fan_epc: np.ndarray, shape=(nkpt,nband,nband,nmode)
            The coupling matrix V_ij V_ji
    
        Returns
        -------
    
        fan_epc_symmetrized: np.ndarray, shape=(nkpt,nband,nband,nmode)
        """
        if not self.degen:
            self.get_degen()

        nkpt, nband, mband, nmode = fan_epc.shape
     
        offdiag = np.zeros((nkpt, nband, nband))
        offdiag[:] =  np.ones((nband, nband)) - np.identity(nband)
    
        for ikpt in range(nkpt):
            for group in self.degen[ikpt]:
                for degi in group:
                    for degj in group:
                        ieig, jeig = degi[1], degj[1]
                        offdiag[ikpt][ieig][jeig] = 0
    
        fan_epc_sym = np.einsum('ijkl,ijk->ijkl', fan_epc, offdiag)
    
        return fan_epc_sym
  




