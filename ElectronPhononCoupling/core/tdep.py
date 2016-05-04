from __future__ import print_function

import sys
import warnings

import numpy as N
from numpy import zeros

from .constants import tol6, tol8, Ha2eV, kb_HaK
from .dynmat import compute_dynmat, get_reduced_displ
from .degenerate import get_degen, make_average, symmetrize_fan_degen
#from .rf_mods import RFStructure 
from . import EigFile, Eigr2dFile, FanFile, DdbFile

from .mathutil import get_bose

__author__ = "Gabriel Antonius, Samuel Ponce"

# =========================================================================== #

#####################
# Compute temp. dep #
#####################

###############################################################################################################

# Compute the static ZPR with temperature-dependence
def static_zpm_temp(arguments,ddw,temperatures,degen):
  sys.stdout.flush()
  nbqpt,wtq,eigq_files,DDB_files,EIGR2D_files = arguments  # FIXME Eyesore
  DDB = DdbFile(DDB_files)
  EIGR2D = Eigr2dFile(EIGR2D_files)
  total_corr =  zeros((3,len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex)
  eigq = EigFile(eigq_files)

  # If the q-point mesh is homogenous, retreve the weight of the q-point
  if (wtq == 0):
    wtq = EIGR2D.wtq[0]

  # Current Q-point calculated
  print("Q-point: {} with wtq = {} and reduced coord. {}".format(nbqpt, wtq, EIGR2D.qred))

  # Get reduced displacement (scaled with frequency)
  displ_red_FAN2, displ_red_DDW2 = DDB.get_reduced_displ()

  bose = get_bose(DDB.natom, DDB.omega, temperatures)

  fan_corr =  zeros((len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex)
  ddw_corr = zeros((len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex)

  fan_corrQ = N.einsum('ijklmn,olnkm->oij', EIGR2D.EIG2D, displ_red_FAN2)
  ddw_corrQ = N.einsum('ijklmn,olnkm->oij',ddw, displ_red_DDW2)

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
  FANterm = FanFile(FAN_files)
  FAN = FANterm.FAN
  DDB = DdbFile(DDB_files)
  EIGR2D = Eigr2dFile(EIGR2D_files)
  total_corr =  zeros((3,len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex)
  eigq = EigFile(eigq_files)

  # If the q-point mesh is homogenous, retreve the weight of the q-point
  if (wtq == 0):
    wtq = EIGR2D.wtq[0]

  # Current Q-point calculated
  print("Q-point: {} with wtq = {} and reduced coord. {}".format(nbqpt, wtq, EIGR2D.qred))

  # Get reduced displacement (scaled with frequency)
  displ_red_FAN2, displ_red_DDW2 = DDB.get_reduced_displ()

  bose = get_bose(DDB.natom, DDB.omega, temperatures)

  # Compute the displacement = eigenvectors of the DDB. 
  # Due to metric problem in reduce coordinate we have to work in cartesian
  # but then go back to reduce because our EIGR2D matrix elements are in reduced coord.
  fan_corr =  zeros((len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex)
  ddw_corr = zeros((len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex)
  fan_add = N.array(zeros((len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex))
  ddw_add = N.array(zeros((len(temperatures),EIGR2D.nkpt,EIGR2D.nband),dtype=complex))

  # Einstein sum make the vector matrix multiplication ont the correct indices
  fan_corrQ = N.einsum('ijklmn,olnkm->oij',EIGR2D.EIG2D,displ_red_FAN2)
  ddw_corrQ = N.einsum('ijklmn,olnkm->oij',ddw,displ_red_DDW2)

  fan_corr = N.einsum('ijk,il->ljk',fan_corrQ,2*bose+1.0)
  ddw_corr = N.einsum('ijk,il->ljk',ddw_corrQ,2*bose+1.0)

  #print("Now compute active space ...")

  # Now compute active space
  fan_addQ = N.einsum('ijklmno,plnkm->ijop',FAN,displ_red_FAN2)
  ddw_addQ = N.einsum('ijklmno,plnkm->ijop',ddw_active,displ_red_DDW2)

  if calc_type == 2: 
    if N.any(EIGR2D.occ[0,0,:] == 2.0):
        occtmp = EIGR2D.occ[0,0,:]/2 # jband
    else:
        occtmp = EIGR2D.occ[0,0,:] # jband
    delta_E_ddw = N.einsum('ij,k->ijk',eig0[0,:,:].real,N.ones(EIGR2D.nband)) - \
              N.einsum('ij,k->ikj',eig0[0,:,:].real,N.ones(EIGR2D.nband)) - \
              N.einsum('ij,k->ijk',N.ones((EIGR2D.nkpt,EIGR2D.nband)),(2*occtmp-1))*smearing*1j

    tmp = N.einsum('ijkl,lm->mijk',ddw_addQ,2*bose+1.0) # tmp,ikpt,iband,jband
    ddw_add = N.einsum('ijkl,jkl->ijk',tmp,1.0/delta_E_ddw)
    delta_E = N.einsum('ij,k->ijk',eig0[0,:,:].real,N.ones(EIGR2D.nband)) - \
              N.einsum('ij,k->ikj',eigq.EIG[0,:,:].real,N.ones(EIGR2D.nband)) - \
              N.einsum('ij,k->ijk',N.ones((EIGR2D.nkpt,EIGR2D.nband)),(2*occtmp-1))*smearing*1j # ikpt,iband,jband
    omegatmp = DDB.omega[:].real # imode

    num1 = N.einsum('ij,k->ijk',bose,N.ones(EIGR2D.nband)) +1.0 \
          - N.einsum('ij,k->ijk',N.ones((3*EIGR2D.natom,len(temperatures))),occtmp) #imode,tmp,jband
    deno1 = N.einsum('ijk,l->ijkl',delta_E,N.ones(3*EIGR2D.natom)) \
          - N.einsum('ijk,l->ijkl',N.ones((EIGR2D.nkpt,EIGR2D.nband,EIGR2D.nband)),omegatmp) #ikpt,iband,jband,imode
    #div1 = N.einsum('ijk,lmki->ijklm',num1,1.0/deno1) # (imode,tmp,jband)/(ikpt,iband,jband,imode) ==> imode,tmp,jband,ikpt,iband
    # BEGIN DEBUG
    #D = (N.real(deno1) ** 2 + N.imag(deno1) ** 2)
    #if N.any(D < 1e-10):
    #    print('Too small value of denominator: ', deno1)
    # END DEBUG
    invdeno1 = N.real(deno1) / (N.real(deno1) ** 2 + N.imag(deno1) ** 2)
    div1 = N.einsum('ijk,lmki->ijklm',num1,invdeno1) # (imode,tmp,jband)/(ikpt,iband,jband,imode) ==> imode,tmp,jband,ikpt,iband

    num2 = N.einsum('ij,k->ijk',bose,N.ones(EIGR2D.nband)) \
          + N.einsum('ij,k->ijk',N.ones((3*EIGR2D.natom,len(temperatures))),occtmp) #imode,tmp,jband
    deno2 = N.einsum('ijk,l->ijkl',delta_E,N.ones(3*EIGR2D.natom)) \
          + N.einsum('ijk,l->ijkl',N.ones((EIGR2D.nkpt,EIGR2D.nband,EIGR2D.nband)),omegatmp) #ikpt,iband,jband,imode
    #div2 = N.einsum('ijk,lmki->ijklm',num2,1.0/deno2) # (imode,tmp,jband)/(ikpt,iband,jband,imode) ==> imode,tmp,jband,ikpt,iband
    invdeno2 = N.real(deno2) / (N.real(deno2) ** 2 + N.imag(deno2) ** 2)
    div2 = N.einsum('ijk,lmki->ijklm',num2,invdeno2) # (imode,tmp,jband)/(ikpt,iband,jband,imode) ==> imode,tmp,jband,ikpt,iband

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
  DDB = DdbFile(DDB_files)
  EIGI2D = Eigr2dFile(EIGI2D_files)
  total_corr = zeros((3,EIGI2D.nkpt,EIGI2D.nband),dtype=complex)
  eigq = EigFile(eigq_files)

  # If the q-point mesh is homogenous, retreve the weight of the q-point
  if (wtq == 0):
    wtq = EIGI2D.wtq[0]

  # Current Q-point calculated
  print("Q-point: {} with wtq = {} and reduced coord. {}".format(nbqpt, wtq, EIGI2D.qred))

  # Get reduced displacement (scaled with frequency)
  displ_red_FAN2, displ_red_DDW2 = DDB.get_reduced_displ()

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
  DDB = DdbFile(DDB_files)
  EIGI2D = Eigr2dFile(EIGI2D_files)
  eigq = EigFile(eigq_files)
  broadening = zeros((len(temperatures),EIGI2D.nkpt,EIGI2D.nband),dtype=complex)

  # If the q-point mesh is homogenous, retreve the weight of the q-point
  if (wtq == 0):
    wtq = EIGI2D.wtq[0]

  # Current Q-point calculated
  print("Q-point: {} with wtq = {} and reduced coord. {}".format(nbqpt, wtq, EIGI2D.qred))

  # For efficiency it is beter not to call a function
  EIG2D = EIGI2D.EIG2D
  nkpt = EIGI2D.nkpt
  nband = EIGI2D.nband
  natom = EIGI2D.natom

  # Get reduced displacement (scaled with frequency)
  displ_red_FAN2, displ_red_DDW2 = DDB.get_reduced_displ()

  bose = get_bose(DDB.natom, DDB.omega, temperatures)

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

