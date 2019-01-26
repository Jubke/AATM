""" Helper functions for AATM seminar topic I code base. """
import os
import sys
import matplotlib.pyplot as plt

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
# Keras visualisation
def draw_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(1)

    ax = plt.subplot(211)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # acc_values = acc
    # val_acc_values = val_acc

    ax2 = plt.subplot(212)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplots_adjust(
        left = 0.125,
        right = 0.9,
        bottom = 0.1,
        top = 0.9,
        wspace = 0.2,
        hspace = 0.2
    )
    plt.show()

# ####################################
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
