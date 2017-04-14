"""
Filenames of the tests
"""
import os
from os.path import join as pjoin

nqpt = 8

dirname = os.path.dirname(__file__)

EIG0_fname = pjoin(dirname, 'EPC/WFK/out_data/odat_EIG.nc')

DDB_fnames = [
    pjoin(
        dirname,
        'DVSCF/qpt-{:0=4}/DVSCF/out_data/odat_DDB.nc'.format(iqpt+1)
        ) for iqpt in range(nqpt)
    ]

EIG_fnames = [
    pjoin(
        dirname,
        'EPC/qpt-{:0=4}/WFQ/out_data/odat_EIG.nc'.format(iqpt+1)
        ) for iqpt in range(nqpt)
    ]

EIGR2D_fnames = [
    pjoin(
        dirname,
        'EPC/qpt-{:0=4}/EPC/out_data/odat_EIGR2D.nc'.format(iqpt+1)
        ) for iqpt in range(nqpt)
    ]

GKK_fnames = [
    pjoin(
        dirname,
        'EPC/qpt-{:0=4}/EPC/out_data/odat_GKK.nc'.format(iqpt+1)
        ) for iqpt in range(nqpt)
    ]

fnames = dict(
    eig0_fname=EIG0_fname,
    eigq_fnames=EIG_fnames,
    DDB_fnames=DDB_fnames,
    EIGR2D_fnames=EIGR2D_fnames,
    GKK_fnames=GKK_fnames,
    )
