from __future__ import print_function
import warnings
import numpy as N
from numpy import zeros

from .constants import tol6, tol8, Ha2eV, kb_HaK



def compute_dynmat(DDB):
  """
  Diagonalize the dynamical matrix.

  Returns:
    omega: the frequencies, in Ha
    eigvect: the eigenvectors, in reduced coord
    gprimd: the primitive reciprocal space vectors
  """

  # Retrive the amu for each atom
  amu = zeros(DDB.natom)
  for ii in N.arange(DDB.natom):
    jj = DDB.typat[ii]
    amu[ii] = DDB.amu[jj-1]

  # Calcul of gprimd from rprimd
  rprimd = DDB.rprim*DDB.acell
  gprimd = N.linalg.inv(N.matrix(rprimd))

  # Transform from 2nd-order matrix (non-cartesian coordinates, 
  # masses not included, asr not included ) from DDB to
  # dynamical matrix, in cartesian coordinates, asr not imposed.
  IFC_cart = zeros((3,DDB.natom,3,DDB.natom),dtype=complex)
  for ii in N.arange(DDB.natom):
    for jj in N.arange(DDB.natom):
      for dir1 in N.arange(3):
        for dir2 in N.arange(3):
          for dir3 in N.arange(3):
            for dir4 in N.arange(3):
              IFC_cart[dir1,ii,dir2,jj] += gprimd[dir1,dir3]*DDB.IFC[dir3,ii,dir4,jj] \
            *gprimd[dir2,dir4]

  # Reduce the 4 dimensional IFC_cart matrice to 2 dimensional Dynamical matrice.
  ipert1 = 0
  Dyn_mat = zeros((3*DDB.natom,3*DDB.natom),dtype=complex)
  while ipert1 < 3*DDB.natom:
    for ii in N.arange(DDB.natom):
      for dir1 in N.arange(3):
        ipert2 = 0
        while ipert2 < 3*DDB.natom:
          for jj in N.arange(DDB.natom):
            for dir2 in N.arange(3):
              Dyn_mat[ipert1,ipert2] = IFC_cart[dir1,ii,dir2,jj]*(5.4857990965007152E-4)/ \
                   N.sqrt(amu[ii]*amu[jj])
              ipert2 += 1
        ipert1 += 1

  # Hermitianize the dynamical matrix
  dynmat = N.matrix(Dyn_mat)
  dynmat = 0.5*(dynmat + dynmat.transpose().conjugate())

  # Solve the eigenvalue problem with linear algebra (Diagonalize the matrix)
  [eigval,eigvect]=N.linalg.eigh(Dyn_mat)

  # Orthonormality relation 
  ipert = 0
  for ii in N.arange(DDB.natom):
    for dir1 in N.arange(3):
     eigvect[ipert] = (eigvect[ipert])*N.sqrt(5.4857990965007152E-4/amu[ii])
     ipert += 1
  kk = 0
  for jj in eigval:
    if jj < 0.0:
      warnings.warn("An eigenvalue is negative with value: {} ... but proceed with value 0.0".format(jj))
      eigval[kk] = 0.0
      kk += 1
    else:
      kk += 1
  omega = N.sqrt(eigval) #*5.4857990965007152E-4)

  # The acoustic phonon at Gamma should NOT contribute because they should be zero.
  # Moreover with the translational invariance the ZPM will be 0 anyway for these
  # modes but the FAN and DDW will have a non physical value. We should therefore 
  # neglect these values.
  #  if N.allclose(DDB.iqpt,[0.0,0.0,0.0]) == True:
  #    omega[0] = 0.0
  #    omega[1] = 0.0
  #    omega[2] = 0.0

  return omega,eigvect,gprimd

# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

def get_reduced_displ(natom,eigvect,omega,gprimd):
  displ_FAN =  zeros((3,3),dtype=complex)
  displ_DDW =  zeros((3,3),dtype=complex)
  displ_red_FAN2 = zeros((3*natom,natom,natom,3,3),dtype=complex)
  displ_red_DDW2 = zeros((3*natom,natom,natom,3,3),dtype=complex)
  for imode in N.arange(3*natom): #Loop on perturbation (6 for 2 atoms)
    if omega[imode].real > tol6:
      for iatom1 in N.arange(natom):
        for iatom2 in N.arange(natom):
          for idir1 in N.arange(0,3):
            for idir2 in N.arange(0,3):
              displ_FAN[idir1,idir2] = eigvect[3*iatom2+idir2,imode].conj()\
                 *eigvect[3*iatom1+idir1,imode]/(2.0*omega[imode].real)
              displ_DDW[idir1,idir2] = (eigvect[3*iatom2+idir2,imode].conj()\
                 *eigvect[3*iatom2+idir1,imode]+eigvect[3*iatom1+idir2,imode].conj()\
                 *eigvect[3*iatom1+idir1,imode])/(4.0*omega[imode].real)
              # Now switch to reduced coordinates in 2 steps (more efficient)
          tmp_displ_FAN = zeros((3,3),dtype=complex)
          tmp_displ_DDW = zeros((3,3),dtype=complex)
          for idir1 in N.arange(3):
            for idir2 in N.arange(3):
              tmp_displ_FAN[:,idir1] = tmp_displ_FAN[:,idir1]+displ_FAN[:,idir2]*gprimd[idir2,idir1]
              tmp_displ_DDW[:,idir1] = tmp_displ_DDW[:,idir1]+displ_DDW[:,idir2]*gprimd[idir2,idir1]
          displ_red_FAN = zeros((3,3),dtype=complex)
          displ_red_DDW = zeros((3,3),dtype=complex)
          for idir1 in N.arange(3):
            for idir2 in N.arange(3):
              displ_red_FAN[idir1,:] = displ_red_FAN[idir1,:] + tmp_displ_FAN[idir2,:]*gprimd[idir2,idir1]
              displ_red_DDW[idir1,:] = displ_red_DDW[idir1,:] + tmp_displ_DDW[idir2,:]*gprimd[idir2,idir1]

          displ_red_FAN2[imode,iatom1,iatom2,:,:] = displ_red_FAN[:,:]
          displ_red_DDW2[imode,iatom1,iatom2,:,:] = displ_red_DDW[:,:]

  return displ_red_FAN2,displ_red_DDW2



