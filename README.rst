
ElectronPhononCoupling
======================

Python module to analyze electron-phonon related quantities from ABINIT.


Istallation
-----------

From the module directory, issue

    >$ python setup.py install

requires

    * numpy >= 1.8.1
    * mpi4py >= 2.0.0
    * netCDF4 >= 1.2.1

Usage
-----

As a python module:

    from ElectronPhononCoupling import compute

    compute(
        renormalization=True,
        broadening=True,
        self_energy=True,
        spectral_function=True,
        temperature=True,


You can run such python script in parallel with, e.g.:

    mpirun -n 4 python my_script.py

Documentation
-------------
 
See the Examples and Doc directories for how to use this module.

