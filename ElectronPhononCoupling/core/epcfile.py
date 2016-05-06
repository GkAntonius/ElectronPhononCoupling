
from __future__ import print_function

__author__ = "Gabriel Antonius"

__all__ = ['EpcFile']


class EpcFile(object):
    """Base class for netCDF files used by EPC."""

    def __init__(self, fname=None, read=True):

        self.fname = fname

        if read and self.fname:
            self.read_nc()
    
    def read_nc(self, fname=None):
        pass
