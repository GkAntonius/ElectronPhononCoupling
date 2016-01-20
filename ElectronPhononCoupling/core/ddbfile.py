
from __future__ import print_function

__author__ = "Gabriel Antonius, Samuel Ponce"

import os
import warnings

import numpy as np
from numpy import zeros
import netCDF4 as nc

from .constants import tol6
from . import EpcFile


class DdbFile(EpcFile):
    
    def read_nc(self, fname=None):
        """Open the DDB.nc file and read it."""
        fname = fname if fname else self.fname

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

            self.BEC = root.variables['born_effective_charge_tensor'][:self.ncart,:self.natom,:self.ncart]


    # This function should not be used, but it is left here for legacy code.
    #def DDB_file_open(self, filefullpath):
    #  """Open the DDB file and read it."""
    #  if not (os.path.isfile(filefullpath)):
    #    raise Exception('The file "%s" does not exists!' %filefullpath)
    #  with open(filefullpath,'r') as DDB:
    #    Flag = 0
    #    Flag2 = False
    #    Flag3 = False
    #    ikpt = 0
    #    for line in DDB:
    #      if line.find('natom') > -1:
    #        self.natom = np.int(line.split()[1])
    #      if line.find('nkpt') > -1:
    #        self.nkpt = np.int(line.split()[1])
    #        self.kpt  = zeros((self.nkpt,3))
    #      if line.find('ntypat') > -1:
    #        self.ntypat = np.int(line.split()[1])
    #      if line.find('nband') > -1:
    #        self.nband = np.int(line.split()[1])
    #      if line.find('acell') > -1:
    #        line = line.replace('D','E')
    #        tmp = line.split()
    #        self.acell = [np.float(tmp[1]),np.float(tmp[2]),np.float(tmp[3])]
    #      if Flag2:
    #        line = line.replace('D','E')
    #        for ii in np.arange(3,self.ntypat):
    #          self.amu[ii] = np.float(line.split()[ii-3])
    #          Flag2 = False
    #      if line.find('amu') > -1:
    #        line = line.replace('D','E')
    #        self.amu = zeros((self.ntypat))
    #        if self.ntypat > 3:
    #          for ii in np.arange(3):
    #            self.amu[ii] = np.float(line.split()[ii+1])
    #            Flag2 = True 
    #        else:
    #          for ii in np.arange(self.ntypat):
    #            self.amu[ii] = np.float(line.split()[ii+1])
    #      if line.find(' kpt ') > -1:
    #        line = line.replace('D','E')
    #        tmp = line.split()
    #        self.kpt[0,0:3] = [float(tmp[1]),float(tmp[2]),float(tmp[3])]
    #        ikpt = 1
    #        continue
    #      if ikpt < self.nkpt and ikpt > 0:
    #        line = line.replace('D','E')
    #        tmp = line.split()
    #        self.kpt[ikpt,0:3] = [float(tmp[0]),float(tmp[1]),float(tmp[2])]  
    #        ikpt += 1
    #        continue
    #      if Flag == 2:
    #        line = line.replace('D','E')
    #        tmp = line.split()
    #        self.rprim[2,0:3] = [float(tmp[0]),float(tmp[1]),float(tmp[2])]
    #        Flag = 0
    #      if Flag == 1:
    #        line = line.replace('D','E')
    #        tmp = line.split()
    #        self.rprim[1,0:3] = [float(tmp[0]),float(tmp[1]),float(tmp[2])]
    #        Flag = 2
    #      if line.find('rprim') > -1:
    #        line = line.replace('D','E')
    #        tmp = line.split()
    #        self.rprim[0,0:3] = [float(tmp[1]),float(tmp[2]),float(tmp[3])]
    #        Flag = 1
    #      if Flag3:
    #        line = line.replace('D','E')
    #        for ii in np.arange(12,self.natom): 
    #          self.typat[ii] = np.float(line.split()[ii-12]) 
    #        Flag3 = False 
    #      if line.find(' typat') > -1:
    #        self.typat = zeros((self.natom))
    #        if self.natom > 12:
    #          for ii in np.arange(12):
    #            self.typat[ii] = np.float(line.split()[ii+1])
    #            Flag3 = True
    #        else:
    #          for ii in np.arange(self.natom):
    #            self.typat[ii] = np.float(line.split()[ii+1])
    #      # Read the actual d2E/dRdR matrix
    #      if Flag == 3:
    #        line = line.replace('D','E')
    #        tmp = line.split()
    #        if not tmp:
    #          break
    #        self.E2D[int(tmp[0])-1,int(tmp[1])-1,int(tmp[2])-1,int(tmp[3])-1] = \
    #          complex(float(tmp[4]),float(tmp[5]))
    #      # Read the current Q-point
    #      if line.find('qpt') > -1:
    #        line = line.replace('D','E')
    #        tmp = line.split()
    #        self.iqpt = [np.float(tmp[1]),np.float(tmp[2]),np.float(tmp[3])]
    #        Flag = 3
    #        self.E2D = zeros((3,self.natom,3,self.natom),dtype=complex)

    # TODO clean up
    def compute_dynmat(self):
        """
        Diagonalize the dynamical matrix.
    
        Returns:
          omega: the frequencies, in Ha
          eigvect: the eigenvectors, in reduced coord
          gprimd: the primitive reciprocal space vectors
        """
    
        # Retrive the amu for each atom
        amu = zeros(self.natom)
        for ii in np.arange(self.natom):
          jj = self.typat[ii]
          amu[ii] = self.amu[jj-1]
    
        # Calcul of gprimd from rprim
        gprimd = np.linalg.inv(np.matrix(self.rprim))
    
        # Transform from 2nd-order matrix (non-cartesian coordinates, 
        # masses not included, asr not included ) from self to
        # dynamical matrix, in cartesian coordinates, asr not imposed.
        E2D_cart = zeros((3,self.natom,3,self.natom),dtype=complex)
        for ii in np.arange(self.natom):
          for jj in np.arange(self.natom):
            for dir1 in np.arange(3):
              for dir2 in np.arange(3):
                for dir3 in np.arange(3):
                  for dir4 in np.arange(3):
                    E2D_cart[dir1,ii,dir2,jj] += gprimd[dir1,dir3]*self.E2D[ii,dir3,jj,dir4] \
                                                 *gprimd[dir2,dir4]
    
        # Reduce the 4 dimensional E2D_cart matrice to 2 dimensional Dynamical matrice.
        ipert1 = 0
        Dyn_mat = zeros((3*self.natom,3*self.natom),dtype=complex)
        while ipert1 < 3*self.natom:
          for ii in np.arange(self.natom):
            for dir1 in np.arange(3):
              ipert2 = 0
              while ipert2 < 3*self.natom:
                for jj in np.arange(self.natom):
                  for dir2 in np.arange(3):
                    Dyn_mat[ipert1,ipert2] = E2D_cart[dir1,ii,dir2,jj]*(5.4857990965007152E-4)/ \
                         np.sqrt(amu[ii]*amu[jj])
                    ipert2 += 1
              ipert1 += 1
    
        # Hermitianize the dynamical matrix
        dynmat = np.matrix(Dyn_mat)
        dynmat = 0.5*(dynmat + dynmat.transpose().conjugate())
    
        # Solve the eigenvalue problem with linear algebra (Diagonalize the matrix)
        [eigval,eigvect]=np.linalg.eigh(Dyn_mat)
    
        # Orthonormality relation 
        ipert = 0
        for ii in np.arange(self.natom):
          for dir1 in np.arange(3):
           eigvect[ipert] = (eigvect[ipert])*np.sqrt(5.4857990965007152E-4/amu[ii])
           ipert += 1
        kk = 0
        for jj in eigval:
          if jj < 0.0:
            warnings.warn("An eigenvalue is negative with value: {} ... but proceed with value 0.0".format(jj))
            eigval[kk] = 0.0
            kk += 1
          else:
            kk += 1
        omega = np.sqrt(eigval)  #  * 5.4857990965007152E-4)
    
        # The acoustic phonon at Gamma should NOT contribute because they should be zero.
        # Moreover with the translational invariance the ZPM will be 0 anyway for these
        # modes but the FAN and DDW will have a non physical value. We should therefore 
        # neglect these values.
        #  if np.allclose(self.iqpt,[0.0,0.0,0.0]) == True:
        #    omega[0] = 0.0
        #    omega[1] = 0.0
        #    omega[2] = 0.0

        self.omega = omega
        self.eigvect = eigvect
        self.gprimd = gprimd
    
        return omega, eigvect, gprimd
    
    # TODO clean up
    def get_reduced_displ(self):
        """
        Compute the squared reduced displacements (scaled by phonon frequencies)
        for the Fan and the DDW terms.
        """
        natom = self.natom
        omega, eigvect, gprimd = self.compute_dynmat()
        displ_FAN = zeros((3,3),dtype=complex)
        displ_DDW = zeros((3,3),dtype=complex)
        displ_red_FAN2 = zeros((3*natom,natom,natom,3,3),dtype=complex)
        displ_red_DDW2 = zeros((3*natom,natom,natom,3,3),dtype=complex)
        for imode in np.arange(3*natom): #Loop on perturbation (6 for 2 atoms)
          if omega[imode].real > tol6:
            for iatom1 in np.arange(natom):
              for iatom2 in np.arange(natom):
                for idir1 in np.arange(0,3):
                  for idir2 in np.arange(0,3):
                    displ_FAN[idir1,idir2] = eigvect[3*iatom2+idir2,imode].conj()\
                       *eigvect[3*iatom1+idir1,imode]/(2.0*omega[imode].real)
                    displ_DDW[idir1,idir2] = (eigvect[3*iatom2+idir2,imode].conj()\
                       *eigvect[3*iatom2+idir1,imode]+eigvect[3*iatom1+idir2,imode].conj()\
                       *eigvect[3*iatom1+idir1,imode])/(4.0*omega[imode].real)
                    # Now switch to reduced coordinates in 2 steps (more efficient)
                tmp_displ_FAN = zeros((3,3),dtype=complex)
                tmp_displ_DDW = zeros((3,3),dtype=complex)
                for idir1 in np.arange(3):
                  for idir2 in np.arange(3):
                    tmp_displ_FAN[:,idir1] = tmp_displ_FAN[:,idir1]+displ_FAN[:,idir2]*gprimd[idir2,idir1]
                    tmp_displ_DDW[:,idir1] = tmp_displ_DDW[:,idir1]+displ_DDW[:,idir2]*gprimd[idir2,idir1]
                displ_red_FAN = zeros((3,3),dtype=complex)
                displ_red_DDW = zeros((3,3),dtype=complex)
                for idir1 in np.arange(3):
                  for idir2 in np.arange(3):
                    displ_red_FAN[idir1,:] = displ_red_FAN[idir1,:] + tmp_displ_FAN[idir2,:]*gprimd[idir2,idir1]
                    displ_red_DDW[idir1,:] = displ_red_DDW[idir1,:] + tmp_displ_DDW[idir2,:]*gprimd[idir2,idir1]
    
                displ_red_FAN2[imode,iatom1,iatom2,:,:] = displ_red_FAN[:,:]
                displ_red_DDW2[imode,iatom1,iatom2,:,:] = displ_red_DDW[:,:]

        self.displ_red_FAN2 = displ_red_FAN2
        self.displ_red_DDW2 = displ_red_DDW2
    
        return displ_red_FAN2, displ_red_DDW2


