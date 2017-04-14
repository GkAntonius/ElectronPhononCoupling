from os.path import join as pjoin
from copy import copy

from . import EPCTest
from ..data import LiF_g2 as test

from ..interface import compute_epc


class Test_LiF_g2(EPCTest):

    common = dict(
        write=True,
        smearing_eV=0.01,
        temp_range=[0,600,300],
        omega_range=[-.1,.1,.001],
        nqpt=test.nqpt,
        wtq=test.wtq,
        **test.fnames)

    def test_t11(self):
        """Static ZP Ren"""

        root = pjoin(self.tmpdir, 't11')
        ref = test.outputs['t11']
        out = root + '_EP.nc'

        compute_epc(
            calc_type=1,
            temperature=False,
            lifetime=False,
            output=root,
            **self.common)

        self.AssertClose(out, ref, 'zero_point_renormalization')

    def test_t12(self):
        """Static Tdep Ren"""

        root = pjoin(self.tmpdir, 't12')
        ref = test.outputs['t12']
        out = root + '_EP.nc'

        compute_epc(
            calc_type=1,
            temperature=True,
            lifetime=False,
            output=root,
            **self.common)

        self.AssertClose(out, ref, 'temperature_dependent_renormalization')

    def test_t13(self):
        """Static ZP Brd"""

        root = pjoin(self.tmpdir, 't13')
        ref = test.outputs['t13']
        out = root + '_EP.nc'

        compute_epc(
            calc_type=1,
            temperature=False,
            lifetime=True,
            output=root,
            **self.common)

        self.AssertClose(out, ref, 'zero_point_broadening')

    def test_t14(self):
        """Static Tdep Brd"""

        root = pjoin(self.tmpdir, 't14')
        ref = test.outputs['t14']
        out = root + '_EP.nc'

        compute_epc(
            calc_type=1,
            temperature=True,
            lifetime=True,
            output=root,
            **self.common)

        self.AssertClose(out, ref, 'temperature_dependent_broadening')

    def test_t21(self):
        """Dynamical ZP Ren"""

        root = pjoin(self.tmpdir, 't21')
        ref = test.outputs['t21']
        out = root + '_EP.nc'

        compute_epc(
            calc_type=2,
            temperature=False,
            lifetime=False,
            output=root,
            **self.common)

        self.AssertClose(out, ref, 'zero_point_renormalization')

    def test_t22(self):
        """Dynamical Tdep Ren"""

        root = pjoin(self.tmpdir, 't22')
        ref = test.outputs['t22']
        out = root + '_EP.nc'

        compute_epc(
            calc_type=2,
            temperature=True,
            lifetime=False,
            output=root,
            **self.common)

        self.AssertClose(out, ref, 'temperature_dependent_renormalization')

    def test_t23(self):
        """Dynamical ZP Brd"""

        root = pjoin(self.tmpdir, 't23')
        ref = test.outputs['t23']
        out = root + '_EP.nc'

        compute_epc(
            calc_type=2,
            temperature=False,
            lifetime=True,
            output=root,
            **self.common)

        self.AssertClose(out, ref, 'zero_point_broadening')

    # Not implemented
    #def test_t24(self):
    #    """Static Tdep Brd"""

    #    root = pjoin(self.tmpdir, 't24')
    #    ref = test.outputs['t24']
    #    out = root + '_EP.nc'

    #    compute_epc(
    #        calc_type=2,
    #        temperature=True,
    #        lifetime=True,
    #        output=root,
    #        **self.common)

    #    self.AssertClose(out, ref, 'temperature_dependent_broadening')

    def test_t31(self):
        """Static ZP Ren"""

        root = pjoin(self.tmpdir, 't31')
        ref = test.outputs['t31']
        out = root + '_EP.nc'

        compute_epc(
            calc_type=3,
            temperature=False,
            lifetime=False,
            output=root,
            **self.common)

        self.AssertClose(out, ref, 'zero_point_renormalization')

    def test_t32(self):
        """Static Tdep Ren"""

        root = pjoin(self.tmpdir, 't32')
        ref = test.outputs['t32']
        out = root + '_EP.nc'

        compute_epc(
            calc_type=3,
            temperature=True,
            lifetime=False,
            output=root,
            **self.common)

        self.AssertClose(out, ref, 'temperature_dependent_renormalization')

    def test_t33(self):
        """Static ZP Brd"""

        root = pjoin(self.tmpdir, 't33')
        ref = test.outputs['t33']
        out = root + '_EP.nc'

        compute_epc(
            calc_type=3,
            temperature=False,
            lifetime=True,
            output=root,
            **self.common)

        self.AssertClose(out, ref, 'zero_point_broadening')

    # Not implemented
    #def test_t34(self):
    #    """Static Tdep Brd"""

    #    root = pjoin(self.tmpdir, 't34')
    #    ref = test.outputs['t34']
    #    out = root + '_EP.nc'

    #    compute_epc(
    #        calc_type=3,
    #        temperature=True,
    #        lifetime=True,
    #        output=root,
    #        **self.common)

    #    self.AssertClose(out, ref, 'temperature_dependent_broadening')

    def test_t41(self):
        """ZP Spectral function."""

        root = pjoin(self.tmpdir, 't41')
        ref = test.outputs['t41']
        out = root + '_EP.nc'

        compute_epc(
            calc_type=4,
            temperature=False,
            lifetime=False,
            output=root,
            **self.common)

        self.AssertClose(out, ref, 'self_energy')
        self.AssertClose(out, ref, 'spectral_function')

    def test_t42(self):
        """ZP Spectral function."""

        root = pjoin(self.tmpdir, 't42')
        ref = test.outputs['t42']
        out = root + '_EP.nc'

        compute_epc(
            calc_type=4,
            temperature=True,
            lifetime=False,
            output=root,
            **self.common)

        self.AssertClose(out, ref, 'self_energy_temperature_dependent')
        self.AssertClose(out, ref, 'spectral_function_temperature_dependent')

