from __future__ import print_function
from os.path import join as pjoin
from copy import copy

from ..data import LiF_g4 as test
from ..interface import compute_epc

from . import EPCTest

class Test_LiF_g4(EPCTest):

    common = dict(
        write=True,
        smearing_eV=0.01,
        temp_range=[0,600,300],
        omega_range=[-.1,.1,.001],
        nqpt=test.nqpt,
        wtq=test.wtq,
        **test.fnames)

    def test_t21(self):
        """Dynamical ZP Ren"""

        basename = 't21'
        root = pjoin(self.tmpdir, basename)
        out = root + '_EP.nc'
        ref = pjoin(test.outputdir, basename + '_EP.nc')

        self.check_reference_exists(ref)

        compute_epc(
            calc_type=2,
            temperature=False,
            lifetime=False,
            output=root,
            **self.common)

        self.AssertClose(out, ref, 'zero_point_renormalization')

    def generate_t21(self, outputdir=test.outputdir):
        """Generate epc data for this test."""

        basename = 't21'
        root = pjoin(outputdir, basename)

        compute_epc(
            calc_type=2,
            temperature=False,
            lifetime=False,
            output=root,
            **self.common)

    def generate(self, outputdir=test.outputdir):
        """Generate epc data for all tests."""

        print('Generating data in directory: {}'.format(outputdir))

        self.generate_t21(outputdir)


