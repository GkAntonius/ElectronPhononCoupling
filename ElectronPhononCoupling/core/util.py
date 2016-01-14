import os
from datetime import datetime

def report_runtime(f):
    """Decorator to print runtime of a function."""
    def g(*args, **kwargs):
        start = datetime.now()
        print('Start on {}'.format(str(start).split('.')[0]))
        out = f(*args, **kwargs)
        end = datetime.now()
        print('End on {}'.format(str(end).split('.')[0]))
        runtime = end - start
        print("Runtime: {} seconds ({:.1f} minutes)".format(runtime.seconds, runtime.seconds / 60.))
        return out
    g.__doc__ = f.__doc__
    return g


def create_directory(fname):
    """Create the directory containing a file name if it noes not exist."""
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.system('mkdir -p ' + dirname)


def formatted_array_lines(arr, ncol=6, form='{:>12.8f}'):
    """Iterator producing the lines to print an array in columns."""
    i, line = 0, ''
    for x in arr:
        i += 1
        line += form.format(x) + ' '
        if i == ncol:
            yield line + '\n'
            i, line = 0, ''
    if line:
        yield line + '\n'

