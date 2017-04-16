
ElectronPhononCoupling
======================

Python module to analyze electron-phonon related quantities from ABINIT.


Istallation
-----------

Simply issue

    >$ python setup.py install

This should install the module somewhere in your $PYTHONPATH
and the script "ElectronPhononCoupling/scripts/pp-temperature" in your $PATH

requires

    * numpy >= 1.8.1
    * mpi4py >= 2.0.0
    * netCDF4 >= 1.2.1

Usage
-----

As a python module:

    from ElectronPhononCoupling import compute

    ...

You can run a python script that calls the function 'compute_epc' 
in serial or in parallel with e.g.:

    mpirun -n 4 python my_script.py

See the examples in the Examples directory for how to use this module.

