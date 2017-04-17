
ElectronPhononCoupling
======================

ElectronPhononCoupling (EPC) is a python module
to analyze electron-phonon related quantities computed with Abinit.


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
 
* For how to use this module, see the Examples directory.

* For the theory pertaining the electronic self-energy
    due to electron-phonon coupling, see [PRB 92, 085137 (2015)].

* For the advanced user and developer, see the Doc directory.


