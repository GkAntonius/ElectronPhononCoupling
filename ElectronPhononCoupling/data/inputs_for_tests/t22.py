from __future__ import print_function
import os
from os.path import join as pjoin

from ElectronPhononCoupling import compute_epc

# =========================================================================== #

DDB_fnames = """
../data_LiF/odat_calc_DS8_DDB.nc
../data_LiF/odat_calc_DS12_DDB.nc
../data_LiF/odat_calc_DS16_DDB.nc
""".split()

EIG_fnames = """
../data_LiF/odat_calc_DS9_EIG.nc
../data_LiF/odat_calc_DS13_EIG.nc
../data_LiF/odat_calc_DS17_EIG.nc
""".split()

EIGR2D_fnames = """
../data_LiF/odat_calc_DS10_EIGR2D.nc
../data_LiF/odat_calc_DS14_EIGR2D.nc
../data_LiF/odat_calc_DS18_EIGR2D.nc
""".split()

EIGI2D_fnames = """
../data_LiF/odat_calc_DS10_EIGI2D.nc
../data_LiF/odat_calc_DS14_EIGI2D.nc
../data_LiF/odat_calc_DS18_EIGI2D.nc
""".split()

FAN_fnames = """
../data_LiF/odat_calc_DS10_FAN.nc
../data_LiF/odat_calc_DS14_FAN.nc
../data_LiF/odat_calc_DS18_FAN.nc
""".split()

EIG0_fname = '../data_LiF/odat_calc_DS2_EIG.nc'


fnames = dict(
        eig0_fname=EIG0_fname,
        eigq_fnames=EIG_fnames,
        DDB_fnames=DDB_fnames,
        EIGR2D_fnames=EIGR2D_fnames,
        EIGI2D_fnames=EIGI2D_fnames,
        FAN_fnames=FAN_fnames,
        )

# =========================================================================== #

# This is a 2x2x2 q-point grid. The weights can be obtained from abinit.
nqpt = 3
wtq = [0.125, 0.5, 0.375]

epc = compute_epc(
        calc_type=2,
        write=True,
        output='output/t22',
        smearing_eV=0.01,
        temperature=True,
        temp_range=[0,600,300],
        lifetime=False,
        nqpt=nqpt,
        wtq=wtq,
        **fnames)


