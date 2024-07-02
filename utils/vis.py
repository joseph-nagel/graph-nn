'''Visualizations.'''

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history, figsize=(9, 3.5)):
    '''Plot training curves.'''

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    axes[0].plot(
        np.arange(history['num_epochs'] + 1),
        np.asarray(history['train_loss']),
        label='train', alpha=0.7
    )
    axes[0].plot(
        np.arange(history['num_epochs'] + 1),
        np.asarray(history['val_loss']),
        label='val.', alpha=0.7
    )
    axes[0].set(xlabel='epoch', ylabel='loss')

    axes[1].plot(
        np.arange(history['num_epochs'] + 1),
        np.asarray(history['train_acc']),
        label='train', alpha=0.7
    )
    axes[1].plot(
        np.arange(history['num_epochs'] + 1),
        np.asarray(history['val_acc']),
        label='val.', alpha=0.7
    )
    axes[1].set(xlabel='epoch', ylabel='acc.')
    axes[1].yaxis.set_label_position('right')
    axes[1].yaxis.tick_right()

    for ax in axes:
        ax.set_xlim((0, history['num_epochs']))
        ax.legend()
        ax.grid(visible=True, which='both', color='lightgray', linestyle='-')
        ax.set_axisbelow(True)

    fig.tight_layout()

    return fig, axes

