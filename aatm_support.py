""" Helper functions for AATM seminar topic I code base. """
import os
import sys

###############
# File helpers

def next_file(base, extension='.txt'):
    """ Returns the next valid path name based on increments. """
    i = 0
    while os.path.exists(f'{base}_{i}{extension}'):
        i += 1

    print(f'Next available file: {base}_{i}{extension}')

    return f'{base}_{i}{extension}'

def last_file(base, extension='.txt'):
    """ Returns the last valid path name based on increments. """
    i = 0
    while os.path.exists(f'{base}_{i}{extension}'):
        i += 1

    print(f'Last valid file: {base}_{i - 1 }{extension}')

    return f'{base}_{i - 1}{extension}'

####################################
# Utility functions for development

def current_ipython_memory():
    """ Returns current iPython memory requirements """
    # These are the usual ipython objects, including the ones you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    return sorted(
        [(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars],
        key=lambda x: x[1],
        reverse=True
    )
