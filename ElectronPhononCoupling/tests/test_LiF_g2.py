from os.path import join as pjoin
from copy import copy

from . import EPCTest, SETest
from ..data import LiF_g2 as test

from ..interface import compute


# FIXME

class Test_LiF_g2(SETest):

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

        nqpt=test.nqpt,
        wtq=test.wtq,
        smearing_eV=0.01,
        temp_range=[0,600,300],
        omega_range=[-.1,.1,.001],

        rootname = 'epc.out',
        **test.fnames)

    def test_zpr_stat_nosplit(self):
        """Static Zero Point Renormalization"""
        self.run_compare_nc(
            function = self.get_zpr_stat_nosplit,
            key = 'zero_point_renormalization',
            nc_ref = test.outputs['t11'],
            )

    def test_tdr_static_nosplit(self):
        """Static Temperature Dependent Renormalization"""
        self.run_compare_nc(
            function = self.get_tdr_stat_nosplit,
            key = 'temperature_dependent_renormalization',
            nc_ref = test.outputs['t12'],
            )

    def test_zpb_stat_nosplit(self):
        """Static Zero Point Broadening"""
        self.run_compare_nc(
            function = self.get_zpb_stat_nosplit,
            key = 'zero_point_broadening',
            nc_ref = test.outputs['t13'],
            )

    def test_tdb_stat_nosplit(self):
        """Static Temperature Dependent Broadening"""
        self.run_compare_nc(
            function = self.get_tdb_stat_nosplit,
            key = 'temperature_dependent_broadening',
            nc_ref = test.outputs['t14'],
            )

    def test_zpr_dyn(self):
        """Dynamical ZPR"""
        self.run_compare_nc(
            function = self.get_zpr_dyn,
            key = 'zero_point_renormalization',
            nc_ref = test.outputs['t21'],
            )

    def test_tdr_dyn(self):
        """Dynamical Tdep Ren"""
        self.run_compare_nc(
            function = self.get_tdr_dyn,
            key = 'temperature_dependent_renormalization',
            nc_ref = test.outputs['t22'],
            )

    def test_zpb_dyn(self):
        """Dynamical ZP Brd"""
        self.run_compare_nc(
            function = self.get_zpb_dyn,
            key = 'zero_point_broadening',
            nc_ref = test.outputs['t23'],
            )

    # Not implemented
    #def test_tdb_stat(self):
    #    """Static Tdep Brd"""
    #    self.run_compare_nc(
    #        function = self.get_zpb_dyn,
    #        key = 'zero_point_broadening',
    #        nc_ref = test.outputs['23'],
    #        )

    def test_zpr_stat(self):
        """Dynamical ZP Brd"""
        self.run_compare_nc(
            function = self.get_zpr_stat,
            key = 'zero_point_renormalization',
            nc_ref = test.outputs['t31'],
            )

    def test_tdr_stat(self):
        """Static Tdep Ren"""
        self.run_compare_nc(
            function = self.get_tdr_stat,
            key = 'temperature_dependent_renormalization',
            nc_ref = test.outputs['t32'],
            )

    def test_zpb_stat(self):
        """Static ZP Brd"""
        self.run_compare_nc(
            function = self.get_zpb_stat,
            key = 'zero_point_broadening',
            nc_ref = test.outputs['t33'],
            )

    # Not implemented
    #def test_tdb_stat(self):
    #    """Static Temperature Dependent Broadening"""
    #    self.run_compare_nc(
    #        function = self.get_tdb_stat,
    #        key = 'temperature_dependent_broadening',
    #        nc_ref = test.outputs['t34'],
    #        )

    def test_zp_se(self):
        """Zero Point Self-Energy"""
        self.run_compare_nc(
            function = self.get_zp_se,
            key = 'self_energy',
            nc_ref = test.outputs['t41'],
            )

    def test_zp_sf(self):
        """Zero Point Spectral Function"""
        self.run_compare_nc(
            function = self.get_zp_sf,
            key = 'spectral_function',
            nc_ref = test.outputs['t41'],
            )

    def test_td_se(self):
        """Temperature Dependent Self-Energy"""
        self.run_compare_nc(
            function = self.get_td_se,
            key = 'self_energy_temperature_dependent',
            nc_ref = test.outputs['t42'],
            )

    def test_td_sf(self):
        """Temperature Dependent Spectral Function"""
        self.run_compare_nc(
            function = self.get_td_sf,
            key = 'spectral_function_temperature_dependent',
            nc_ref = test.outputs['t42'],
            )

