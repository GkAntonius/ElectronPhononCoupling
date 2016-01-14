
import unittest
import os

import ElectronPhononCoupling

inputsdir = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'data', 'inputs-for-tests'))
datadir = os.path.join(__file__, '..', '..', '..', 'data', 'diamond-rf-data')

class TestCommandLine(unittest.TestCase):

    def setUp(self):
        return

    def tearDown(self):
        os.chdir(inputsdir)
        os.system('rm -r output')

    #def make_input_file(self, ref_input, datadir):
    #    """Read a reference input file, and create a new input with appropriate files."""

    def run_pp_temperature_with_input(self, fname):
        os.chdir(inputsdir)
        os.system('pp-temperature < ' + fname)

    def test_test11(self): self.run_pp_temperature_with_input('test11.in')
    def test_test12(self): self.run_pp_temperature_with_input('test12.in')
    def test_test13(self): self.run_pp_temperature_with_input('test13.in')
    def test_test14(self): self.run_pp_temperature_with_input('test14.in')
    def test_test21(self): self.run_pp_temperature_with_input('test21.in')
    def test_test22(self): self.run_pp_temperature_with_input('test22.in')
    def test_test23(self): self.run_pp_temperature_with_input('test23.in')
    def test_test24(self): self.run_pp_temperature_with_input('test24.in')
    def test_test31(self): self.run_pp_temperature_with_input('test31.in')
    def test_test32(self): self.run_pp_temperature_with_input('test32.in')
    def test_test33(self): self.run_pp_temperature_with_input('test33.in')
    def test_test34(self): self.run_pp_temperature_with_input('test34.in')
    def test_test41(self): self.run_pp_temperature_with_input('test41.in')
