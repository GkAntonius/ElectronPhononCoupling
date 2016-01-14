from __future__ import print_function

import os

import numpy as N
from numpy import zeros
import netCDF4 as nc


class RFStructure:
  natom = None
  ntypat = None
  nkpt = None
  kpt = None
  Kptns = None
  EIG = None
  nband = None
  acell = None
  occ = None
  amu = None
  rprim = N.empty((3,3))
  iqpt = None
  IFC = None
  filename = None
  filefullpath = None
  def __init__(self, filename, directory='.'):

    self.filename = filename

    self.filefullpath = os.path.join(directory, filename)

    if self.filefullpath.endswith('_DDB'):
      self.DDB_file_open(self.filefullpath)

    elif (self.filefullpath.endswith('_EIGR2D.nc') or
        self.filefullpath.endswith('_EIGI2D.nc')):
      self.EIG2Dnc_file_open(self.filefullpath)

    elif (self.filefullpath.endswith('_EIGR2D') or
        self.filefullpath.endswith('_EIGI2D')):
      self.EIG2D_file_open(self.filefullpath)

    elif self.filefullpath.endswith('_EIG.nc'):
      self.EIG_file_open(self.filefullpath)

    elif self.filefullpath.endswith('_FAN.nc'):
      self.FANnc_file_open(self.filefullpath)

    elif self.filefullpath.endswith('_FAN'):
      self.FAN_file_open(self.filefullpath)

    elif self.filefullpath.endswith('_EP.nc'):
      self.EP_file_open(self.filefullpath)

    elif self.filefullpath.endswith('_EIG'):
      raise Exception('Please provide a netCDF _EIG.nc file.')

    else:
      raise Exception('Unrecognized file extension.')

  def EP_file_open(self, filefullpath):
    """Read _EP.nc file"""
    if not (os.path.isfile(filefullpath)):
      raise Exception('The file "%s" does not exists!' %filefullpath)

    root = nc.Dataset(filefullpath, 'r')

    self.natom = len(root.dimensions['number_of_atoms'])
    self.nkpt = len(root.dimensions['number_of_kpoints'])
    self.nband = len(root.dimensions['max_number_of_states'])
    self.ntemp = len(root.dimensions['number_of_temperature'])
    self.nsppol = len(root.dimensions['number_of_spins'])
    self.nbQ = len(root.dimensions['number_of_qpoints'])
    self.temp = root.variables['temperature'][:]
    self.occ = root.variables['occupations'][:,:,:] # number_of_spins, number_of_kpoints, max_number_of_states
    self.kpt = root.variables['reduced_coordinates_of_kpoints'][:,:]
    self.eigenvalues = root.variables['eigenvalues'][:,:,:] #number_of_spins, number_of_kpoints, max_number_of_states
    self.rprimd = root.variables['primitive_vectors'][:,:]
    self.zpm = root.variables['zero_point_motion'][:,:,:,:,:] # nsppol, number_of_temperature, 
                                                   # number_of_kpoints, max_number_of_states, cplex 
    root.close()

  def EIG_file_open(self, filefullpath):
    """Read _EIG.nc file."""
    if not (os.path.isfile(filefullpath)):
      raise Exception('The file "%s" does not exists!' %filefullpath)
    root = nc.Dataset(filefullpath,'r')
    self.EIG = root.variables['Eigenvalues'][:,:] 
    self.Kptns = root.variables['Kptns'][:,:]
    NBandK = root.variables['NBandK'][:]
    self.nband =  N.int(NBandK[0,0])
    root.close()

  def FANnc_file_open(self, filefullpath):
    """Open the Fan.nc file and read it"""
    if not (os.path.isfile(filefullpath)):
      raise Exception('The file "%s" does not exists!' %filefullpath)
    root = nc.Dataset(filefullpath,'r')
    self.natom = len(root.dimensions['number_of_atoms'])
    self.nkpt = len(root.dimensions['number_of_kpoints'])
    self.nband = len(root.dimensions['max_number_of_states'])
    self.nsppol = len(root.dimensions['number_of_spins'])
    self.occ = root.variables['occupations'][:,:,:] # number_of_spins, number_of_kpoints, max_number_of_states
    FANtmp = root.variables['second_derivative_eigenenergies_actif'][:,:,:,:,:,:,:] #product_mband_nsppol,number_of_atoms, 
                                       # number_of_cartesian_directions, number_of_atoms, number_of_cartesian_directions,
                                       # number_of_kpoints, product_mband_nsppol*2
    FANtmp2 = zeros((self.nkpt,2*self.nband,3,self.natom,3,self.natom,self.nband))
    FANtmp2 = N.einsum('ijklmno->nomlkji', FANtmp)
    FANtmp3 = FANtmp2[:, ::2, ...]  # Slice the even numbers
    FANtmp4 = FANtmp2[:, 1::2, ...] # Slice the odd numbers
    self.FAN = 1j*FANtmp4
    self.FAN += FANtmp3
    self.eigenvalues = root.variables['eigenvalues'][:,:,:] #number_of_spins, number_of_kpoints, max_number_of_states   
    self.kpt = root.variables['reduced_coordinates_of_kpoints'][:,:]
    self.iqpt = root.variables['current_q_point'][:]
    self.wtq = root.variables['current_q_point_weight'][:]
    self.rprimd = root.variables['primitive_vectors'][:,:]
    root.close()

  def EIG2Dnc_file_open(self, filefullpath):
    """Open the EIG2D.nc file and read it."""
    if not (os.path.isfile(filefullpath)):
      raise Exception('The file "%s" does not exists!' %filefullpath)
    root = nc.Dataset(filefullpath,'r')
    self.natom = len(root.dimensions['number_of_atoms'])
    self.nkpt = len(root.dimensions['number_of_kpoints'])
    self.nband = len(root.dimensions['max_number_of_states'])
    self.nsppol = len(root.dimensions['number_of_spins'])
    self.occ = root.variables['occupations'][:,:,:] # number_of_spins, number_of_kpoints, max_number_of_states
    EIG2Dtmp = root.variables['second_derivative_eigenenergies'][:,:,:,:,:,:,:] #number_of_atoms, 
                                       # number_of_cartesian_directions, number_of_atoms, number_of_cartesian_directions,
                                       # number_of_kpoints, product_mband_nsppol, cplex
    EIG2Dtmp2 = zeros((self.nkpt,2*self.nband,3,self.natom,3,self.natom,self.nband))
    EIG2Dtmp2 = N.einsum('ijklmno->mnlkjio', EIG2Dtmp)
    self.EIG2D = 1j*EIG2Dtmp2[...,1]
    self.EIG2D += EIG2Dtmp2[...,0]
    self.eigenvalues = root.variables['eigenvalues'][:,:,:] #number_of_spins, number_of_kpoints, max_number_of_states   
    self.kpt = root.variables['reduced_coordinates_of_kpoints'][:,:]
    self.iqpt = root.variables['current_q_point'][:]
    self.wtq = root.variables['current_q_point_weight'][:]
    self.rprimd = root.variables['primitive_vectors'][:,:]
    root.close()

  def EIG2D_file_open(self, filefullpath):
    """Open the EIG2D file and read it."""
    if not (os.path.isfile(filefullpath)):
      raise Exception('The file "%s" does not exists!' %filefullpath)
    self.EIG2D = None
    with open(filefullpath,'r') as EIG2D:
      Flag = 0
      Flagocc = False
      ikpt = 0
      iocc = 0
      vv = 1
      for line in EIG2D:
        if line.find('natom') > -1:
          self.natom = N.int(line.split()[1])
        if line.find('nkpt') > -1:
          self.nkpt = N.int(line.split()[1])
          self.kpt = N.empty((self.nkpt,3))
        if line.find('nband') > -1:
          self.nband = N.int(line.split()[1])
      # Initialize the EIGR2D or EIGI2D matrix (nkpt,nband,3dir,natom,3dir,natom)
          self.EIG2D = N.zeros((self.nkpt,self.nband,3,self.natom,3,self.natom),dtype=complex)
      # Initialize the occupation vector
          self.occ = N.zeros((self.nband))
        if line.find('occ ') > -1:
          line = line.replace('D','E')
          self.occ[iocc] = N.int(N.float(line.split()[1]))
          if self.nband > 1:
            self.occ[iocc+1] = N.int(N.float(line.split()[2]))
          if self.nband > 2:
            self.occ[iocc+2] = N.int(N.float(line.split()[3]))
          if self.nband > 3:
            Flagocc = True
            iocc = 3
            continue # Go to the next iteration of the for loop
        if Flagocc:
          line = line.replace('D','E')
          vv +=1
          if vv < self.nband/3:
            self.occ[iocc] = N.int(N.float(line.split()[0]))
            self.occ[iocc+1] = N.int(N.float(line.split()[1]))
            self.occ[iocc+2] = N.int(N.float(line.split()[2]))
            iocc += 3
            continue # Go to the next iteration of the for loop
          elif vv == self.nband/3:
            Flagocc = False
            if self.nband%3 > 0:
              if self.nband%3 == 1:
                self.occ[iocc] = N.int(N.float(line.split()[0]))
              if self.nband%3 == 2:
                self.occ[iocc+1] = N.int(N.float(line.split()[1]))
      # Read the current Q-point
        if line.find('qpt') > -1:
          line = line.replace('D','E')
          tmp = line.split()
          self.iqpt = [N.float(tmp[1]),N.float(tmp[2]),N.float(tmp[3])]
      # Read the current K-point
        if line.find('K-point') > -1:
          line = line.replace('D','E')
          tmp = line.split()
          self.kpt[ikpt,:] = [N.float(tmp[1]),N.float(tmp[2]),N.float(tmp[3])]
          ikpt +=1
          ibd = 0
          continue # Go to the next iteration of the for loop
      # Read the current Bands 
        if line.find('Band:') > -1:
          ibd += 1
          Flag = 1
          continue
      # Read the EIG2RD or EIGI2D matrix
        if Flag == 1:
          line = line.replace('D','E')
          tmp = line.split()
          self.EIG2D[ikpt-1,ibd-1,int(tmp[0])-1,int(tmp[1])-1,int(tmp[2])-1,int(tmp[3])-1] = \
            complex(float(tmp[4]),float(tmp[5]))

  def DDB_file_open(self, filefullpath):
    """Open the DDB file and read it."""
    if not (os.path.isfile(filefullpath)):
      raise Exception('The file "%s" does not exists!' %filefullpath)
    with open(filefullpath,'r') as DDB:
      Flag = 0
      Flag2 = False
      Flag3 = False
      ikpt = 0
      for line in DDB:
        if line.find('natom') > -1:
          self.natom = N.int(line.split()[1])
        if line.find('nkpt') > -1:
          self.nkpt = N.int(line.split()[1])
          self.kpt  = zeros((self.nkpt,3))
        if line.find('ntypat') > -1:
          self.ntypat = N.int(line.split()[1])
        if line.find('nband') > -1:
          self.nband = N.int(line.split()[1])
        if line.find('acell') > -1:
          line = line.replace('D','E')
          tmp = line.split()
          self.acell = [N.float(tmp[1]),N.float(tmp[2]),N.float(tmp[3])]
        if Flag2:
          line = line.replace('D','E')
          for ii in N.arange(3,self.ntypat):
            self.amu[ii] = N.float(line.split()[ii-3])
            Flag2 = False
        if line.find('amu') > -1:
          line = line.replace('D','E')
          self.amu = zeros((self.ntypat))
          if self.ntypat > 3:
            for ii in N.arange(3):
              self.amu[ii] = N.float(line.split()[ii+1])
              Flag2 = True 
          else:
            for ii in N.arange(self.ntypat):
              self.amu[ii] = N.float(line.split()[ii+1])
        if line.find(' kpt ') > -1:
          line = line.replace('D','E')
          tmp = line.split()
          self.kpt[0,0:3] = [float(tmp[1]),float(tmp[2]),float(tmp[3])]
          ikpt = 1
          continue
        if ikpt < self.nkpt and ikpt > 0:
          line = line.replace('D','E')
          tmp = line.split()
          self.kpt[ikpt,0:3] = [float(tmp[0]),float(tmp[1]),float(tmp[2])]  
          ikpt += 1
          continue
        if Flag == 2:
          line = line.replace('D','E')
          tmp = line.split()
          self.rprim[2,0:3] = [float(tmp[0]),float(tmp[1]),float(tmp[2])]
          Flag = 0
        if Flag == 1:
          line = line.replace('D','E')
          tmp = line.split()
          self.rprim[1,0:3] = [float(tmp[0]),float(tmp[1]),float(tmp[2])]
          Flag = 2
        if line.find('rprim') > -1:
          line = line.replace('D','E')
          tmp = line.split()
          self.rprim[0,0:3] = [float(tmp[1]),float(tmp[2]),float(tmp[3])]
          Flag = 1
        if Flag3:
          line = line.replace('D','E')
          for ii in N.arange(12,self.natom): 
            self.typat[ii] = N.float(line.split()[ii-12]) 
          Flag3 = False 
        if line.find(' typat') > -1:
          self.typat = zeros((self.natom))
          if self.natom > 12:
            for ii in N.arange(12):
              self.typat[ii] = N.float(line.split()[ii+1])
              Flag3 = True
          else:
            for ii in N.arange(self.natom):
              self.typat[ii] = N.float(line.split()[ii+1])
        # Read the actual d2E/dRdR matrix
        if Flag == 3:
          line = line.replace('D','E')
          tmp = line.split()
          if not tmp:
            break
          self.IFC[int(tmp[0])-1,int(tmp[1])-1,int(tmp[2])-1,int(tmp[3])-1] = \
            complex(float(tmp[4]),float(tmp[5]))
        # Read the current Q-point
        if line.find('qpt') > -1:
          line = line.replace('D','E')
          tmp = line.split()
          self.iqpt = [N.float(tmp[1]),N.float(tmp[2]),N.float(tmp[3])]
          Flag = 3
          self.IFC = zeros((3,self.natom,3,self.natom),dtype=complex)
