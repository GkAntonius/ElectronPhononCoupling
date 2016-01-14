# coding: utf-8

import os
from glob import glob


from setuptools import find_packages #, setup
from distutils.core import setup

#---------------------------------------------------------------------------
# Basic project information
#---------------------------------------------------------------------------

# release.py contains version, authors, license, url, keywords, etc.
release_file = os.path.join('ElectronPhononCoupling', 'setup', 'release.py')

with open(release_file) as f:
    code = compile(f.read(), release_file, 'exec')
    exec(code)


#---------------------------------------------------------------------------
# Find scripts
#---------------------------------------------------------------------------

def find_scripts():
    """Find the scripts."""
    scripts = []
    # All python files in abipy/scripts
    pyfiles = glob(os.path.join('ElectronPhononCoupling', 'scripts', "*"))
    scripts.extend(pyfiles)
    return scripts


def get_long_desc():
    with open("README.rst") as fh:
        return fh.read()

#def find_package_data():
#    """Find data for tests and example."""
#    package_data = {
#        'ElectronPhononCoupling.data': ['*-data/*', 'inputs-for-tests/*'],
#    }
#    return package_data

def write_manifest():
    content = """\
include *.rst
recursive-include ElectronPhononCoupling *.py
recursive-include ElectronPhononCoupling/scripts *
recursive-include ElectronPhononCoupling/data *
prune ElectronPhononCoupling/data/inputs-for-tests/output
"""
    with open('MANIFEST.in', 'write') as f:
        f.write(content)


#---------------------------------------------------------------------------
# Setup
#---------------------------------------------------------------------------

write_manifest()

my_package_data = {
        'ElectronPhononCoupling.data' : ['*-data/*', 'inputs-for-tests/*.*', 'outputs-of-tests/*'],
    }

my_exclude_package_data = {
        'ElectronPhononCoupling.data' : ['inputs-for-tests/ouput/*'],
    }


setup_args = dict(
      name             = name,
      version          = __version__,
      description      = description,
      long_description = long_description,
      author           = author,
      url              = url,
      license          = license,
      packages         = find_packages(),
      scripts          = find_scripts(),
      package_data     = my_package_data,
      exclude_package_data = my_exclude_package_data,
      )

if __name__ == "__main__":
    setup(**setup_args)

