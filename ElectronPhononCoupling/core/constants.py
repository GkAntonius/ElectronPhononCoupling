"""Constants"""
import scipy.constants as cst

# Tolerance criterions
tol5 = 1E-5
tol6 = 1E-6
tol8 = 1E-8
tol12 = 1E-12

# Conversion factor
#Ha2eV = 27.21138386
Ha2eV = cst.physical_constants['Hartree energy in eV'][0]

# Boltzman constant
#kb_HaK = 3.1668154267112283e-06
kb_HaK = cst.k / cst.physical_constants['Hartree energy'][0]

# Electron mass over atomical mass unit
me_amu = cst.physical_constants['atomic unit of mass'][0] / cst.physical_constants['atomic mass constant'][0]


bohr_to_ang = cst.physical_constants['Bohr radius'][0] / cst.angstrom 
