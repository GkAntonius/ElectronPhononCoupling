from __future__ import print_function
from os.path import join as pjoin
from copy import copy

from . import SETest
from ..interface import compute

from ..data.LiF_g4 import nqpt, wtq, fnames, refdir

class Test_LiF_g4(SETest):

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

        nqpt = nqpt,
        wtq = wtq,
        smearing_eV = 0.01,
        #temp_range = [0, 600, 50],
        temp_range = [0, 600, 300],
        omega_range = [-0.1, 0.1, 0.001],
        rootname = 'epc.out',
        **fnames)

    @property
    def refdir(self):
        return refdir

    # ZPR
    def test_zpr_dyn(self):
        """Dynamical zero-point renormalization"""
        self.run_compare_nc(
            function = self.get_zpr_dyn,
            key = 'zero_point_renormalization',
            )

    #def generate_zpr_dyn(self):
    #    """Generate epc data for this test."""
    #    return self.generate_test_ref(self.get_zpr_dyn)

    def test_tdr_dyn(self):
        """Dynamical temperature dependent renormalization"""
        self.run_compare_nc(
            function = self.get_tdr_dyn,
            key = 'temperature_dependent_renormalization',
            )

    #def generate_tdr_dyn(self):
    #    return self.generate_test_ref(self.get_tdr_dyn)

    # All
    def generate(self):
        """Generate epc data for all tests."""

        print('Generating test reference data in directory: {}'.format(refdir))

        for function in (
            self.get_zpr_dyn,
            self.get_tdr_dyn,
            ):
            self.generate_ref(function)


