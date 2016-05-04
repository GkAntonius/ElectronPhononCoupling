# coding: utf-8

__all__ = [
    '__version__',
    'name',
    'description',
    'long_description',
    'license',
    '__author__',
    'author',
    'url',
    ]


__version__ = '2.5.0'

name = "ElectronPhononCoupling"

description = "Python module to analyze electron-phonon related quantities."

long_description = """"
    Compute electron-phonon coupling related quantities, such as:
        - the zero-point renormalization
        - the temperature dependance of eigenvalues
        - the quasiparticle lifetime from the el-ph self-energy
    """

license = 'GPL'

authors = {'SP': (u'Samuel Poncé', 'sponce at gmail.com'),
           'GA': ('Gabriel Antonius', 'gabriel.antonius at gmail.com'),
        }
        
author = 'The ABINIT group'

url = 'http://abinit.org'

__author__ = ''
for auth, email in authors.itervalues():
  __author__ += auth + ' <' + email + '>\n'
del auth, email


