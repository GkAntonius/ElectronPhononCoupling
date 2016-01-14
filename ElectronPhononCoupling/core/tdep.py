from __future__ import print_function

import sys
import warnings

import numpy as N
from numpy import zeros

from .constants import tol6, tol8, Ha2eV, kb_HaK
from .dynmat import compute_dynmat, get_reduced_displ
from .degenerate import get_degen, make_average, symmetrize_fan_degen
from .rf_mods import RFStructure 

from .mathutil import get_bose

# =========================================================================== #

#####################
# Compute temp. dep #
#####################

###############################################################################################################

# Compute the static ZPR with temperature-dependence
def static_zpm_temp(arguments,ddw,temperatures,degen):
  sys.stdout.flush()
  nbqpt,wtq,eigq_files,DDB_files,EIGR2D_files = arguments
  DDB = RFStructure(DDB_files)
  EIGR2D = RFStructure(EIGR2D_files)
  total_corr =  zeros((3,len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex)
  eigq = RFStructure(eigq_files)

  # If the q-point mesh is homogenous, retreve the weight of the q-point
  if (wtq == 0):
    wtq = EIGR2D.wtq[0]

  # Current Q-point calculated
  print("Q-point: {} with wtq = {} and reduced coord. {}".format(nbqpt, wtq, EIGR2D.iqpt))

  # Find phonon freq and eigendisplacement from _DDB
  omega,eigvect,gprimd=compute_dynmat(DDB)

  fan_corr =  zeros((len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex)
  ddw_corr = zeros((len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex)

  bose = get_bose(EIGR2D.natom,omega,temperatures)

  # Get reduced displacement (scaled with frequency)
  displ_red_FAN2,displ_red_DDW2 = get_reduced_displ(EIGR2D.natom,eigvect,omega,gprimd)
  fan_corrQ = N.einsum('ijklmn,olnkm->oij',EIGR2D.EIG2D,displ_red_FAN2)
  ddw_corrQ = N.einsum('ijklmn,olnkm->oij',ddw,displ_red_DDW2)

  fan_corr = N.einsum('ijk,il->ljk',fan_corrQ,2*bose+1.0)
  ddw_corr = N.einsum('ijk,il->ljk',ddw_corrQ,2*bose+1.0)

  eigen_corr = (fan_corr[:,:,:]- ddw_corr[:,:,:])*wtq
  total_corr[0,:,:,:] = eigen_corr[:,:,:]
  total_corr[1,:,:,:] = fan_corr[:,:,:]*wtq
  total_corr[2,:,:,:] = ddw_corr[:,:,:]*wtq

  make_average(total_corr, degen)

  return total_corr

#########################################################################################################

# Compute the dynamical ZPR with temperature dependence
def dynamic_zpm_temp(arguments,ddw,ddw_active,calc_type,temperatures,smearing,eig0,degen):

  nbqpt,wtq,eigq_files,DDB_files,EIGR2D_files,FAN_files = arguments
  FANterm = RFStructure(FAN_files)
  FAN = FANterm.FAN
  DDB = RFStructure(DDB_files)
  EIGR2D = RFStructure(EIGR2D_files)
  total_corr =  zeros((3,len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex)
  eigq = RFStructure(eigq_files)

  # If the q-point mesh is homogenous, retreve the weight of the q-point
  if (wtq == 0):
    wtq = EIGR2D.wtq[0]

  # Current Q-point calculated
  print("Q-point: {} with wtq = {} and reduced coord. {}".format(nbqpt, wtq, EIGR2D.iqpt))

  # Find phonon freq and eigendisplacement from _DDB
  omega,eigvect,gprimd=compute_dynmat(DDB)

  # Compute the displacement = eigenvectors of the DDB. 
  # Due to metric problem in reduce coordinate we have to work in cartesian
  # but then go back to reduce because our EIGR2D matrix elements are in reduced coord.
  fan_corr =  zeros((len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex)
  ddw_corr = zeros((len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex)
  fan_add = N.array(zeros((len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex))
  ddw_add = N.array(zeros((len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex))

  bose = get_bose(EIGR2D.natom,omega,temperatures)

  # Get reduced displacement (scaled with frequency)
  displ_red_FAN2,displ_red_DDW2 = get_reduced_displ(EIGR2D.natom,eigvect,omega,gprimd)

  # Einstein sum make the vector matrix multiplication ont the correct indices
  fan_corrQ = N.einsum('ijklmn,olnkm->oij',EIGR2D.EIG2D,displ_red_FAN2)
  ddw_corrQ = N.einsum('ijklmn,olnkm->oij',ddw,displ_red_DDW2)

  fan_corr = N.einsum('ijk,il->ljk',fan_corrQ,2*bose+1.0)
  ddw_corr = N.einsum('ijk,il->ljk',ddw_corrQ,2*bose+1.0)

  print("Now compute active space ...")

  # Now compute active space
  fan_addQ = N.einsum('ijklmno,plnkm->ijop',FAN,displ_red_FAN2)
  ddw_addQ = N.einsum('ijklmno,plnkm->ijop',ddw_active,displ_red_DDW2)

  if calc_type == 2: 
    occtmp = EIGR2D.occ[0,0,:]/2 # jband
    delta_E_ddw = N.einsum('ij,k->ijk',eig0[0,:,:].real,N.ones(EIGR2D.nband)) - \
              N.einsum('ij,k->ikj',eig0[0,:,:].real,N.ones(EIGR2D.nband)) - \
              N.einsum('ij,k->ijk',N.ones((EIGR2D.nkpt,EIGR2D.nband)),(2*occtmp-1))*smearing*1j

    tmp = N.einsum('ijkl,lm->mijk',ddw_addQ,2*bose+1.0) # tmp,ikpt,iband,jband
    ddw_add = N.einsum('ijkl,jkl->ijk',tmp,1.0/delta_E_ddw)
    delta_E = N.einsum('ij,k->ijk',eig0[0,:,:].real,N.ones(EIGR2D.nband)) - \
              N.einsum('ij,k->ikj',eigq.EIG[0,:,:].real,N.ones(EIGR2D.nband)) - \
              N.einsum('ij,k->ijk',N.ones((EIGR2D.nkpt,EIGR2D.nband)),(2*occtmp-1))*smearing*1j # ikpt,iband,jband
    omegatmp = omega[:].real # imode
    num1 = N.einsum('ij,k->ijk',bose,N.ones(EIGR2D.nband)) +1.0 \
          - N.einsum('ij,k->ijk',N.ones((3*EIGR2D.natom,len(temperatures))),occtmp) #imode,tmp,jband
    deno1 = N.einsum('ijk,l->ijkl',delta_E,N.ones(3*EIGR2D.natom)) \
          - N.einsum('ijk,l->ijkl',N.ones((EIGR2D.nkpt,EIGR2D.nband,EIGR2D.nband)),omegatmp) #ikpt,iband,jband,imode
    div1 = N.einsum('ijk,lmki->ijklm',num1,1.0/deno1) # (imode,tmp,jband)/(ikpt,iband,jband,imode) ==> imode,tmp,jband,ikpt,iband
    num2 = N.einsum('ij,k->ijk',bose,N.ones(EIGR2D.nband)) \
          + N.einsum('ij,k->ijk',N.ones((3*EIGR2D.natom,len(temperatures))),occtmp) #imode,tmp,jband
    deno2 = N.einsum('ijk,l->ijkl',delta_E,N.ones(3*EIGR2D.natom)) \
          + N.einsum('ijk,l->ijkl',N.ones((EIGR2D.nkpt,EIGR2D.nband,EIGR2D.nband)),omegatmp) #ikpt,iband,jband,imode
    div2 = N.einsum('ijk,lmki->ijklm',num2,1.0/deno2) # (imode,tmp,jband)/(ikpt,iband,jband,imode) ==> imode,tmp,jband,ikpt,iband
    fan_add = N.einsum('ijkl,lmkij->mij',fan_addQ,div1+div2) # ikpt,iband,jband,imode

  elif calc_type ==3:
    delta_E = N.einsum('ij,k->ijk',eig0[0,:,:].real,N.ones(EIGR2D.nband)) - \
              N.einsum('ij,k->ikj',eigq.EIG[0,:,:].real,N.ones(EIGR2D.nband))  # ikpt,iband,jband      
    delta_E_ddw = N.einsum('ij,k->ijk',eig0[0,:,:].real,N.ones(EIGR2D.nband)) - \
              N.einsum('ij,k->ikj',eig0[0,:,:].real,N.ones(EIGR2D.nband)) 
    num = N.einsum('ij,klm->ijklm',2*bose+1.0,delta_E)  # imode,tmp,ikpt,iband,jband
    deno = delta_E**2 +smearing**2 # ikpt,iband,jband
    div =  N.einsum('ijklm,klm->ijklm',num,1.0/deno)   # imode,tmp,ikpt,iband,jband 
    fan_add = N.einsum('ijkl,lmijk->mij',fan_addQ,div) #(ikpt,iband,jband,imode),(imode,tmp,ikpt,iband,jband)->tmp,ikpt,iband

    num = N.einsum('ij,klm->ijklm',2*bose+1.0,delta_E_ddw) # imode,tmp,ikpt,iband,jband
    deno = delta_E_ddw**2 +smearing**2 # ikpt,iband,jband
    div =  N.einsum('ijklm,klm->ijklm',num,1.0/deno)
    ddw_add = N.einsum('ijkl,lmijk->mij',ddw_addQ,div) #(ikpt,iband,jband,imode),(imode,tmp,ikpt,iband,jband)->tmp,ikpt,iband 

# The code above corresponds to the following loops:    
#  for ikpt in N.arange(EIGR2D.nkpt):
#    for iband in N.arange(EIGR2D.nband):
#      for jband in N.arange(EIGR2D.nband):
#        if calc_type == 2:
#          occtmp = EIGR2D.occ[0,0,jband]/2 # electronic occ should be 1
#          if occtmp > tol6:
#            delta_E = eig0[0,ikpt,iband].real-eigq.EIG[0,ikpt,jband].real - smearing*1j
#            delta_E_ddw = eig0[0,ikpt,iband].real-eig0[0,ikpt,jband].real - smearing*1j
#          else:
#            delta_E = eig0[0,ikpt,iband].real-eigq.EIG[0,ikpt,jband].real + smearing*1j
#            delta_E_ddw = eig0[0,ikpt,iband].real-eig0[0,ikpt,jband].real + smearing*1j
#          for imode in N.arange(3*EIGR2D.natom):
#            # DW is not affected by the dynamical equations
#            ddw_add[:,ikpt,iband] += ddw_addQ[ikpt,iband,jband,imode]*(2*bose[imode,:]+1.0)\
#                                    *(1.0/delta_E_ddw)
#            omegatmp = omega[imode].real
#            fan_add[:,ikpt,iband] += fan_addQ[ikpt,iband,jband,imode]*(\
#              (bose[imode,:]+1.0-occtmp)/(delta_E-omegatmp) \
#              + (bose[imode,:]+occtmp)/(delta_E+omegatmp))
#        if calc_type == 4:
#          delta_E = eig0[0,ikpt,iband].real-eigq.EIG[0,ikpt,jband].real
#          delta_E_ddw = eig0[0,ikpt,iband].real-eig0[0,ikpt,jband].real
#          for imode in N.arange(3*EIGR2D.natom):
#            fan_add[:,ikpt,iband] += fan_addQ[ikpt,iband,jband,imode]*(2*bose[imode,:]+1.0)\
#                                    *(delta_E/(delta_E**2+smearing**2))
#            ddw_add[:,ikpt,iband] += ddw_addQ[ikpt,iband,jband,imode]*(2*bose[imode,:]+1.0)\
#                                    *(delta_E_ddw/(delta_E_ddw**2+smearing**2))
#


  fan_corr += fan_add
  ddw_corr += ddw_add
  eigen_corr = (fan_corr[:,:,:] - ddw_corr[:,:,:])*wtq
  total_corr[0,:,:,:] = eigen_corr[:,:,:]
  total_corr[1,:,:,:] = fan_corr[:,:,:]*wtq
  total_corr[2,:,:,:] = ddw_corr[:,:,:]*wtq

  total_corr = make_average(total_corr, degen)

  return total_corr

#########################################################################################################
############
# LIFETIME #
############
########################################################################################################
def static_zpm_lifetime(arguments, degen):
  """Compute the static ZPR only with lifetime."""

  nbqpt,wtq,eigq_files,DDB_files,EIGI2D_files = arguments
  DDB = RFStructure(DDB_files)
  EIGI2D = RFStructure(EIGI2D_files)
  total_corr = zeros((3,EIGI2D.nkpt,EIGI2D.nband),dtype=complex)
  eigq = RFStructure(eigq_files)

  # If the q-point mesh is homogenous, retreve the weight of the q-point
  if (wtq == 0):
    wtq = EIGI2D.wtq[0]

  # Current Q-point calculated
  print("Q-point: {} with wtq = {} and reduced coord. {}".format(nbqpt, wtq, EIGI2D.iqpt))

  # Find phonon freq and eigendisplacement from _DDB
  omega,eigvect,gprimd=compute_dynmat(DDB)

  # For efficiency it is beter not to call a function
  EIG2D = EIGI2D.EIG2D
  nkpt = EIGI2D.nkpt
  nband = EIGI2D.nband
  natom = EIGI2D.natom
  
  # Compute the displacement = eigenvectors of the DDB. 
  # Due to metric problem in reduce coordinate we have to work in cartesian
  # but then go back to reduce because our EIGR2D matrix elements are in reduced coord.
  displ_FAN =  zeros((3,3),dtype=complex)
  broadening = zeros((nkpt,nband),dtype=complex)
  displ_red_FAN2, displ_red_DDW2 = get_reduced_displ(natom, eigvect, omega, gprimd)

  # Einstein sum make the vector matrix multiplication ont the correct indices
  fan_corrQ = N.einsum('ijklmn,olnkm->oij', EIG2D, displ_red_FAN2)

  for imode in N.arange(3*natom): #Loop on perturbation (6 for 2 atoms)
    broadening[:,:] += N.pi*fan_corrQ[imode,:,:]

  broadening = broadening*wtq

  if N.any(broadening[:,:].imag > 1E-12):
    warnings.warn("The real part of the broadening is non zero: {}".format(broadening))

  broadening = make_average(broadening, degen)
  #for ikpt in N.arange(nkpt):
  #  count = 0
  #  iband = 0
  #  while iband < nband:
  #    if iband < nband-2:
  #      if ((degen[ikpt,iband] == degen[ikpt,iband+1]) and (degen[ikpt,iband] == degen[ikpt,iband+2])):
  #        broadening[ikpt,iband] = (broadening[ikpt,iband]+broadening[ikpt,iband+1]+broadening[ikpt,iband+2])/3
  #        broadening[ikpt,iband+1] = broadening[ikpt,iband]
  #        broadening[ikpt,iband+2] = broadening[ikpt,iband]
  #        iband += 3
  #        continue
  #    if iband <  nband-1:
  #      if (degen[ikpt,iband] == degen[ikpt,iband+1]):
  #        broadening[ikpt,iband] = (broadening[ikpt,iband]+broadening[ikpt,iband+1])/2
  #        broadening[ikpt,iband+1]= broadening[ikpt,iband]
  #        iband +=2
  #        continue
  #    iband += 1
  
  return broadening


###############################################################################################################
# Compute the static ZPR with temperature-dependence with lifetime
def static_zpm_temp_lifetime(arguments,ddw,temperatures,degen):

  nbqpt,wtq,eigq_files,DDB_files,EIGI2D_files = arguments
  DDB = RFStructure(DDB_files)
  EIGI2D = RFStructure(EIGI2D_files)
  total_corr =  zeros((3,len(temperatures),EIGI2D.nkpt,EIGI2D.nband),dtype=complex)
  eigq = RFStructure(eigq_files)

  # If the q-point mesh is homogenous, retreve the weight of the q-point
  if (wtq == 0):
    wtq = EIGI2D.wtq[0]

  # Current Q-point calculated
  print("Q-point: {} with wtq = {} and reduced coord. {}".format(nbqpt, wtq, EIGI2D.iqpt))

  # Find phonon freq and eigendisplacement from _DDB
  omega,eigvect,gprimd=compute_dynmat(DDB)

  # For efficiency it is beter not to call a function
  EIG2D = EIGI2D.EIG2D
  nkpt = EIGI2D.nkpt
  nband = EIGI2D.nband
  natom = EIGI2D.natom

  # Compute the displacement = eigenvectors of the DDB. 
  # Due to metric problem in reduce coordinate we have to work in cartesian
  # but then go back to reduce because our EIGR2D matrix elements are in reduced coord.
  displ_FAN =  zeros((3,3),dtype=complex)
  displ_red_FAN2 = zeros((3*natom,natom,natom,3,3),dtype=complex)

  broadening =  zeros((len(temperatures),nkpt,nband),dtype=complex)
  bose = get_bose(EIGI2D.natom,omega,temperatures)

  for imode in N.arange(3*natom): #Loop on perturbation (6 for 2 atoms)
    if omega[imode].real > tol6:
      for iatom1 in N.arange(natom):
        for iatom2 in N.arange(natom):
          for idir1 in N.arange(0,3):
            for idir2 in N.arange(0,3):
              displ_FAN[idir1,idir2] = eigvect[3*iatom2+idir2,imode].conj()\
                 *eigvect[3*iatom1+idir1,imode]/(2.0*omega[imode].real)
              # Now switch to reduced coordinates in 2 steps (more efficient)
          tmp_displ_FAN = zeros((3,3),dtype=complex)
          for idir1 in N.arange(3):
            for idir2 in N.arange(3):
              tmp_displ_FAN[:,idir1] = tmp_displ_FAN[:,idir1]+displ_FAN[:,idir2]*gprimd[idir2,idir1]
          displ_red_FAN = zeros((3,3),dtype=complex)
          for idir1 in N.arange(3):
            for idir2 in N.arange(3):
              displ_red_FAN[idir1,:] = displ_red_FAN[idir1,:] + tmp_displ_FAN[idir2,:]*gprimd[idir2,idir1]
          displ_red_FAN2[imode,iatom1,iatom2,:,:] = displ_red_FAN[:,:]
  fan_corrQ = N.einsum('ijklmn,olnkm->oij',EIG2D,displ_red_FAN2)

  for imode in N.arange(3*natom): #Loop on perturbation (6 for 2 atoms)
    tt = 0
    for T in temperatures: 
      broadening[tt,:,:] += N.pi*fan_corrQ[imode,:,:]*(2*bose[imode,tt]+1.0)  # FIXME GA: Not sure about this
      tt += 1

  broadening = broadening*wtq

  broadening = make_average(broadening, degen)
  #for ikpt in N.arange(nkpt):
  #  count = 0
  #  iband = 0
  #  while iband < nband:
  #    if iband < nband-2:
  #      if ((degen[ikpt,iband] == degen[ikpt,iband+1]) and (degen[ikpt,iband] == degen[ikpt,iband+2])):
  #        broadening[:,ikpt,iband] = (broadening[:,ikpt,iband]+broadening[:,ikpt,iband+1]+broadening[:,ikpt,iband+2])/3
  #        broadening[:,ikpt,iband+1] = broadening[:,ikpt,iband]
  #        broadening[:,ikpt,iband+2] = broadening[:,ikpt,iband]
  #        iband += 3
  #        continue
  #    if iband <  nband-1:
  #      if (degen[ikpt,iband] == degen[ikpt,iband+1]):
  #        broadening[:,ikpt,iband] = (broadening[:,ikpt,iband]+broadening[:,ikpt,iband+1])/2
  #        broadening[:,ikpt,iband+1] = broadening[:,ikpt,iband]
  #        iband +=2
  #        continue
  #    iband += 1

  return broadening

