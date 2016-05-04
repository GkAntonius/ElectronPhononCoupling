
ElectronPhononCoupling
======================

Python module to analyze electron-phonon related quantities from ABINIT.


Istallation
-----------

Simply issue

    >$ python setup.py install

This should install the module somewhere in your $PYTHONPATH
and the script "ElectronPhononCoupling/scripts/pp-temperature" in your $PATH

requires numpy >= 1.8.1

Usage
-----

Interactive usage:

    >$ pp-temperature

As a python module:

    from ElectronPhononCoupling import compute_epc

See examples in ElectronPhononCoupling/data/inputs_for_tests/

