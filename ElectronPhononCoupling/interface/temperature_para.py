from __future__ import print_function

import numpy as N

from ..core import EPC_Analyzer
from ..core.constants import Ha2eV

from ..config import __version__
from ..core.util import report_runtime

# ============================================================================ #

def run_interactive():
    """Run the calculation after getting inputs interactively from the user."""
    arguments = get_user_input()
    epc = compute_epc(**arguments)



@report_runtime
def compute_epc(
        calc_type = 1,
        ncpu = 1,
        wtq = None,
        write = True,
        output='epc.out',
        temperature = False,
        lifetime = False,
        smearing_eV = 0.01,
        temp_info = [0, 0, 1],
        omega_range=[0,0,1],
        nqpt = 1,
        eig0_fname = '',
        eigq_fnames = list(),
        DDB_fnames = list(),
        EIGR2D_fnames = list(),
        EIGI2D_fnames = list(),
        FAN_fnames = list(),
        **kwargs):
    """
    Compute electron-phonon coupling related quantities, such as:
        - the zero-point renormalization
        - the temperature dependance of eigenvalues
        - the quasiparticle lifetime from the el-ph self-energy


    Keyword arguments
    -----------------

    calc_type: (1)
        Governs the type of calculation performed
            1  -  static AHC calculation
            2  -  dynamic AHC calculation
            3  -  static AHC calculation with control over active space

    ncpu: int
        Number of processors to parallelize the task

    write: (True)
        Write the results on the disk.

    output: ('epc.out')
        Rootname for the output files.

    smearing_eV: (0.01)
        Imaginary parameter for ZPR contributions and broadening.

    temperature: (False)
        Compute temperature dependance of eigenvalues corrections / broadening

    temp_info: [0,0,1]
        Minimum, maximum and step temperature for eigenvalues dependance.

    lifetime: (False)
        Compute broadening

    nqpt: (1)
        Number of q-points

    DDB_fnames: ([])
        Names of _DDB files

    eigq_fnames: ([])
        Names of _EIG.nc files at k+q

    EIGR2D_fnames: ([])
        Names of _EIGR2D.nc files

    EIGI2D_fnames: ([])
        Names of _EIGI2D.nc files

    FAN_fnames: ([])
        Names of _FAN.nc files

    eig0_fname: ('')
        Name of the _EIG.nc file for the eigenvalues being corrected


    Returns
    -------

    epc: EPC_Analyzer
        Object containing the response function data
    """

    if smearing_eV is None:
        smearing_Ha = None
    else:
        smearing_Ha = smearing_eV / Ha2eV

    # Initialize epc
    epc = EPC_Analyzer(nqpt=nqpt, 
                       wtq=wtq,
                       eig0_fname=eig0_fname,
                       eigq_fnames=eigq_fnames,
                       DDB_fnames=DDB_fnames,
                       EIGR2D_fnames=EIGR2D_fnames,
                       EIGI2D_fnames=EIGI2D_fnames,
                       FAN_fnames=FAN_fnames)

    epc.set_ncpu(ncpu)
    epc.set_temp_range(temp_info)
    epc.set_omega_range(omega_range)
    epc.set_smearing(smearing_Ha)
    epc.set_output(output)

    # Compute renormalization
    if calc_type == 1:

        if temperature:
            epc.compute_static_td_renormalization()
        else:
            epc.compute_static_zp_renormalization()

    elif calc_type == 2:

        if temperature:
            epc.compute_dynamical_td_renormalization()
        else:
            epc.compute_dynamical_zp_renormalization()

    elif calc_type == 3:

        if temperature:
            epc.compute_static_control_td_renormalization()
        else:
            epc.compute_static_control_zp_renormalization()

    elif calc_type == 4:
        epc.compute_self_energy()
        epc.compute_spectral_function()

    else:
        raise Exception('Calculation type must be 1, 2, 3 or 4')

    # Compute lifetime
    if lifetime:

        if calc_type == 1:

            if temperature:
                epc.compute_static_td_broadening()
            else:
                epc.compute_static_zp_broadening()

        elif calc_type == 2:

            if temperature:
                epc.compute_dynamical_td_broadening()
            else:
                epc.compute_dynamical_zp_broadening()

        elif calc_type == 3:

            if temperature:
                epc.compute_static_control_td_broadening()
            else:
                epc.compute_static_control_zp_broadening()


    # Write the files
    if write:
        epc.write_netcdf()
        epc.write_renormalization()
        if lifetime:
            epc.write_broadening()

    return epc



def get_user_input():
    """Get all inputs for the calculation interactively."""

    arguments = {
        'ncpu' : 1,
        'calc_type' : 1,
        'output' : '',
        'smearing_eV' : 3.6749e-4,
        'temperature' : False,
        'temp_info' : [0, 0, 1],
        'lifetime' : False,
        'nqpt' : 1,
        'DDB_fnames' : list(),
        'eigq_fnames' : list(),
        'EIGR2D_fnames' : list(),
        'EIGI2D_fnames' : list(),
        'FAN_fnames' : list(),
        'eig0_fname' : '',
        }

    # Interaction with the user
    logo = """

   ____  ____       _                                      _                   
  |  _ \|  _ \     | |_ ___ _ __ ___  _ __   ___ _ __ __ _| |_ _   _ _ __ ___  
  | |_) | |_) |____| __/ _ \ '_ ` _ \| '_ \ / _ \ '__/ _` | __| | | | '__/ _ \ 
  |  __/|  __/_____| ||  __/ | | | | | |_) |  __/ | | (_| | |_| |_| | | |  __/ 
  |_|   |_|         \__\___|_| |_| |_| .__/ \___|_|  \__,_|\__|\__,_|_|  \___| 
                                     |_|                           Version {}

    """.format(__version__)

    description = """
    This script compute the static/dynamic zero-point motion
    and the temperature dependance of eigenenergies due to electron-phonon interaction.
    The static electronic lifetime can also be computed.
    IMPORTANT: The first Q-point MUST be the Gamma point.
    """
    print(logo)
    print(description)

    def get_user(s):
        return raw_input(s.rstrip('\n') + '\n').split('#')[0]


    # Enter the number of cpu on which you want to multi-thread
    ui = get_user('Enter the number of cpu on which you want to multi-thread\n')
    ncpu = ui
    try:
      ncpu = int(ui)
    except ValueError:
      raise Exception('The value you enter is not an integer!')
    arguments.update(ncpu=ncpu)

    # Type of calculation the user want to perform
    ui = get_user("""
Please select the calculation type:
    1  Static AHC calculation.
    2  Dynamic AHC calculation.
    3  Static AHC calculation with control over active space.

Note that option 2 and 3 requires _FAN.nc files obtained
through ABINIT option 'ieig2rf 4'
""")
    calc_type = N.int(ui)
    arguments.update(calc_type=calc_type)
    
    # Define the output file name
    ui = get_user('Enter name of the output file')
    output = ui.strip()
    arguments.update(output=output)
    
    # Enter the value of the smearing parameter for dynamic AHC
    if (calc_type == 2 or calc_type == 3 ):
      ui = get_user('Enter value of the smearing parameter (in eV)')
      smearing_eV = N.float(ui)
    else:
      smearing_eV = None
    arguments.update(smearing_eV=smearing_eV)
    
    # Temperature dependence analysis?
    ui = get_user('Do you want to compute the change of eigenergies with temperature? [y/n]')
    temperature = ui.split()[0]
    if temperature.lower() == 'y':
      temperature = True
      arguments.update(temperature=temperature)
    else:
      temperature = False

    if temperature:
      ui = get_user('Introduce the starting temperature, max temperature and steps. e.g. 0 2000 100')
      temp_info = map(float, ui.split())
      arguments.update(temp_info=temp_info)
    
    # Broadening lifetime of the electron
    ui = get_user('Do you want to compute the lifetime of the electrons? [y/n]')
    tmp =ui.split()[0]
    if tmp == 'y':
      lifetime = True
    else:
      lifetime = False
    arguments.update(lifetime=lifetime)

    # Get the nb of random Q-points from user 
    ui = get_user('Enter the number of Q-points')
    try:
      nqpt = int(ui)
    except ValueError:
      raise Exception('The value you enter is not an integer!')
    arguments.update(nqpt=nqpt)
    
    # Get the path of the DDB files from user
    DDB_fnames = []
    for ii in N.arange(nqpt):
      ui = get_user('Enter the name of the %s DDB file' %ii)
      if len(ui.split()) != 1:
        raise Exception("You should provide only 1 file")
      else: # Append and TRIM the input string with STRIP
        DDB_fnames.append(ui.strip(' \t\n\r'))
    arguments.update(DDB_fnames=DDB_fnames)

    # Get the path of the eigq files from user
    eigq_fnames = []
    for ii in N.arange(nqpt):
      ui = get_user('Enter the name of the %s eigq file' %ii)
      if len(ui.split()) != 1:
        raise Exception("You should provide only 1 file")
      else:
        eigq_fnames.append(ui.strip(' \t\n\r'))
    arguments.update(eigq_fnames=eigq_fnames)
    
    # Get the path of the EIGR2D files from user
    EIGR2D_fnames = []
    for ii in N.arange(nqpt):
      ui = get_user('Enter the name of the %s EIGR2D file' %ii)
      if len(ui.split()) != 1:
        raise Exception("You should provide only 1 file")
      else:
        EIGR2D_fnames.append(ui.strip(' \t\n\r'))
    arguments.update(EIGR2D_fnames=EIGR2D_fnames)
    
    # Get the path of the EIGI2D files from user
    if lifetime:
      EIGI2D_fnames = []
      for ii in N.arange(nqpt):
        ui = get_user('Enter the name of the %s EIGI2D file' %ii)
        if len(ui.split()) != 1:
          raise Exception("You should provide only 1 file")
        else:
          EIGI2D_fnames.append(ui.strip(' \t\n\r'))
        arguments.update(EIGI2D_fnames=EIGI2D_fnames)
    
    # Get the path of the FAN files from user if dynamical calculation
    if (calc_type == 2 or calc_type == 3):
      FAN_fnames = []
      for ii in N.arange(nqpt):
        ui = get_user('Enter the name of the %s FAN file' %ii)
        if len(ui.split()) != 1:
          raise Exception("You should provide only 1 file")
        else:
          FAN_fnames.append(ui.strip(' \t\n\r'))
        arguments.update(FAN_fnames=FAN_fnames)
    
    # Take the EIG at Gamma
    ui = get_user('Enter the name of the unperturbed EIG.nc file at Gamma')
    if len(ui.split()) != 1:
      raise Exception("You sould only provide 1 file")
    else:
      eig0_fname=ui.strip(' \t\n\r')
    arguments.update(eig0_fname=eig0_fname)

    return arguments


