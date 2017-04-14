from __future__ import print_function
from os.path import join as pjoin

from . import EPCTest
from ..interface import compute_epc

from ..data.LiF_g4 import nqpt, wtq, fnames, outputdir

class Test_LiF_g4(EPCTest):

    common = dict(
        write=True,
        smearing_eV=0.01,
        temp_range=[0,600,300],
        omega_range=[-.1,.1,.001],
        nqpt=nqpt,
        wtq=wtq,
        **fnames)

    def test_zpr_dyn(self):
        """Dynamical zero-point renormalization"""

        basename = 'zpr_dyn'
        root = pjoin(self.tmpdir, basename)
        out = root + '_EP.nc'
        ref = pjoin(outputdir, basename + '_EP.nc')

        self.check_reference_exists(ref)

        compute_epc(
            calc_type=2,
            temperature=False,
            lifetime=False,
            output=root,
            **self.common)

        self.AssertClose(out, ref, 'zero_point_renormalization')

    def generate_zpr_dyn(self):
        """Generate epc data for this test."""

        basename = 'zpr_dyn'
        root = pjoin(outputdir, basename)

        compute_epc(
            calc_type=2,
            temperature=False,
            lifetime=False,
            output=root,
            **self.common)

    def test_tdr_dyn(self):
        """Dynamical temperature dependent renormalization"""

        basename = 'tdr_dyn'
        root = pjoin(self.tmpdir, basename)
        out = root + '_EP.nc'
        ref = pjoin(outputdir, basename + '_EP.nc')

        compute_epc(
            calc_type=2,
            temperature=True,
            lifetime=False,
            output=root,
            **self.common)

        self.AssertClose(out, ref, 'temperature_dependent_renormalization')

    def generate_tdr_dyn(self):
        """Dynamical temperature dependent renormalization"""

        basename = 'tdr_dyn'
        root = pjoin(outputdir, basename)

        compute_epc(
            calc_type=2,
            temperature=True,
            lifetime=False,
            output=root,
            **self.common)

    def generate(self):
        """Generate epc data for all tests."""

        print('Generating data in directory: {}'.format(outputdir))

        self.generate_zpr_dyn()
        self.generate_tdr_dyn()


