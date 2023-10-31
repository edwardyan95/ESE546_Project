import matplotlib.pyplot as plt
import numpy as np
import torch
import random
def plot_loss_and_error(train_loss, train_error, val_loss, val_error):
    """

    Function that plots:
    1. validation error over number of epochs
    2. validation loss over number of epochs
    3. training error over number of epochs
    4. training loss over number of epochs

    Return: fig, axes
    """
    ###TODO###
    fig, axes = plt.subplots(2,2)
    axes[0,0].plot(train_loss)
    axes[0,0].set(xlabel = 'epoch', ylabel = 'loss', title='training loss')
    axes[0,1].plot(train_error)
    axes[0,1].set(xlabel = 'epoch', ylabel = 'error%', title='training error')
    axes[1,0].plot(val_loss)
    axes[1,0].set(xlabel = 'epoch', ylabel = 'loss', title='validation loss')
    axes[1,1].plot(val_error)
    axes[1,1].set(xlabel = 'epoch', ylabel = 'error%', title='validation error')
    plt.tight_layout()
    return fig, axes

def set_seed(seed: int = 42) -> None:
    """
    set random seed for training
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
