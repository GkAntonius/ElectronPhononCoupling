import os
from os.path import join as pjoin
from copy import copy
import unittest
import tempfile, shutil

from numpy.testing import assert_allclose
import netCDF4 as nc

from ..core.mpi import master_only
from ..interface import compute

__all__ = ['EPCTest', 'SETest']

class EPCTest(unittest.TestCase):
    """Base class for tests."""

    #def runTest(self):
    #    """Dummy method."""
    #    pass

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.local_testdir = 'tmp.Tests'
        self.local_tmpdir = os.path.join(
            self.local_testdir, os.path.split(self.tmpdir)[-1])

    @property
    def refdir(self):
        raise NotImplementedError()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def recover_tmpdir(self):
        """Debug tool to copy the execution directory locally."""
        if not os.path.exists(self.local_testdir):
            os.mkdir(self.local_testdir)
        shutil.copytree(self.tmpdir, self.local_tmpdir)

    @master_only
    def AssertClose(self, f1, f2, key, **kwargs):
        """Assert that an array in two nc files is close."""

        with nc.Dataset(f1, 'r') as ds1:
            a1 = ds1.variables[key][...]

        with nc.Dataset(f2, 'r') as ds2:
            a2 = ds2.variables[key][...]

        kwargs.setdefault('rtol', 1e-5)
        return assert_allclose(
            a1, a2, err_msg=('\n' +
                '-----------------------\n' +
                'Comparison test failed.\n' +
                'f1: {}\n'.format(f1) +
                'f2: {}\n'.format(f2) +
                'Variable compared: {}\n'.format(key) +
                '-----------------------\n'
                ),
            **kwargs)

    def check_reference_exists(self, fname):
        if not os.path.exists(fname):
            raise Exception('Reference file does not exist: {}'.format(fname))

    def run_compare_nc(self, function, key, refdir=None, nc_ref=None):
        """
        Run 'compute' by generating the arguments with 'function'
        then compare the array 'key' in nc_output.

        key:
            Key to be compared in both nc_output.
        function:
            Function generating the keyword arguments for 'compute'.
            Takes a directory as single argument.
        refdir:
            directory for reference files
        nc_ref:
            name of the netcdf file for comparison.
            Alternate argument of refdir.
        """

        out = compute(**function(self.tmpdir))

        if nc_ref is not None:
            pass
        else:
            if refdir is None:
                refdir = self.refdir
            nc_ref = out.nc_output.replace(self.tmpdir, refdir)

        self.check_reference_exists(nc_ref)
        self.AssertClose(out.nc_output, nc_ref, key)

    def generate_ref(self, function):
        """
        function:
            Function generating the keyword arguments for 'compute'.
            Takes a directory as single argument.
        """
        return compute(**function(self.refdir))


# =========================================================================== #


class SETest(EPCTest):
    """
    Base class for tests involving the electron self-energy
    """

    common = dict(
        temperature = False,
        renormalization = False,
        broadening = False,
        self_energy = False,
        spectral_function = False,
        dynamical = True,
        split_active = True,
        double_grid = False,
        write = True,
        verbose = False,

        nqpt = 1,
        wtq = [1.],
        smearing_eV = 0.01,
        temp_range = [0, 600, 300],
        omega_range = [-0.1, 0.1, 0.001],
        rootname = 'epc.out',
        )

    def get_kwargs(self, dirname, basename, **kwargs):
        """Construct the input dictionary for a test"""
        new_kwargs = copy(self.common)
        new_kwargs.update(rootname=pjoin(dirname, basename))
        new_kwargs.update(**kwargs)
        return new_kwargs

    def get_zpr_dyn(self, dirname):
        return self.get_kwargs(
            dirname,
            basename='zpr_dyn',
            renormalization=True,
            )

    def get_tdr_dyn(self, dirname):
        return self.get_kwargs(
            dirname,
            basename='tdr_dyn',
            renormalization=True,
            temperature=True,
            )

    def get_zpb_dyn(self, dirname):
        return self.get_kwargs(
            dirname,
            basename='tdb_stat',
            temperature=False,
            broadening=True,
            dynamical=True,
            )

    def get_zpr_stat(self, dirname):
        return self.get_kwargs(
            dirname,
            basename='zpr_stat',
            temperature=False,
            renormalization=True,
            dynamical=False,
            )

    def get_tdr_stat(self, dirname):
        return self.get_kwargs(
            dirname,
            basename='zpr_stat',
            temperature=True,
            renormalization=True,
            dynamical=False,
            )

    def get_zpb_stat(self, dirname):
        return self.get_kwargs(
            dirname,
            basename='zpb_stat',
            temperature=False,
            broadening=True,
            dynamical=False,
            )

    def get_tdb_stat(self, dirname):
        return self.get_kwargs(
            basename='tdb_stat',
            temperature=True,
            broadening=True,
            dynamical=False,
            )

    def get_zpr_stat_nosplit(self, dirname):
        return self.get_kwargs(
            dirname,
            basename='zpr_stat_nosplit',
            temperature=False,
            renormalization=True,
            dynamical=False,
            split_active=False,
            )

    def get_tdr_stat_nosplit(self, dirname):
        return self.get_kwargs(
            dirname,
            basename='tdr_stat_nosplit',
            temperature=True,
            renormalization=True,
            dynamical=False,
            split_active=False,
            )

    def get_zpb_stat_nosplit(self, dirname):
        return self.get_kwargs(
            dirname,
            basename='zpb_stat_nosplit',
            temperature=False,
            broadening=True,
            dynamical=False,
            split_active=False,
            )

    def get_tdb_stat_nosplit(self, dirname):
        return self.get_kwargs(
            dirname,
            basename='tdb_stat_nosplit',
            temperature=True,
            broadening=True,
            dynamical=False,
            split_active=False,
            )

    def get_zp_se(self, dirname):
        return self.get_kwargs(
            dirname,
            basename='zp_se',
            temperature=False,
            self_energy=True,
            )

    def get_td_se(self, dirname):
        return self.get_kwargs(
            dirname,
            basename='td_se',
            temperature=True,
            self_energy=True,
            )

    def get_zp_sf(self, dirname):
        return self.get_kwargs(
            dirname,
            basename='zp_sf',
            temperature=False,
            self_energy=True,
            spectral_function=True,
            )

    def get_td_sf(self, dirname):
        return self.get_kwargs(
            dirname,
            basename='td_sf',
            temperature=True,
            self_energy=True,
            spectral_function=True,
            )

