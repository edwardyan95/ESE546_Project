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

def filter_exps(boc, exps, num_exps = 10, min_neuron = 0, max_neuron = 1000, behavior = False):
    
    pp_count = 0
    pp_exp_idx = []
    cell_count_idx = []
    for i in range(len(exps)):
        exp = exps[i]
        data_set = boc.get_ophys_experiment_data(exp['id'])
        cids = data_set.get_cell_specimen_ids()
        if len(cids)>=min_neuron and len(cids)<=max_neuron:
            cell_count_idx.append(i)
        try:
            _, _ = boc.get_ophys_experiment_data(exp['id']).get_pupil_size()
            pp_count+=1
            pp_exp_idx.append(i)
        except:
            continue
    # Convert lists to sets and perform union
    if behavior:
        union_list = cell_count_idx and pp_exp_idx
    else:
        union_list = cell_count_idx
    
    if len(union_list)>=num_exps:
        sampled_exp_idx = random.sample(union_list, num_exps)
    else:
        sampled_exp_idx = union_list
    return [exps[i] for i in sampled_exp_idx]
    

    
    
        
