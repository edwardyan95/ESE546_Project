import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from models import LSTMClassifier

def get_train_test_split(dataset, train_prop = 0.7, split_method = 'trial'):
    """
    split incoming dataset into random training and test dataset based on trials, subjects* or target structures*

    Parameters:
    dataset: dict with 3 keys 'model_input', 'model_labels', 'metadata'
    training prop: proportion of training dataset
    split_method: 'trials', 'subjects' or 'target structures'

    Return:
    train_dataset, test_dataset: dict with 2 keys 'model_input', 'model_labels'
    
    """
    if split_method == 'trial':
        rand_idx = np.random.permutation(len(dataset['model_input']))
        num_training_sample = int(len(dataset['model_input'])*train_prop)
        train_idx = rand_idx[:num_training_sample]
        test_idx = rand_idx[num_training_sample:]
        train_dataset = {'model_input' :[dataset['model_input'][i] for i in train_idx],
                         'model_labels':[dataset['model_labels'][i] for i in train_idx]
                        }
        test_dataset = {'model_input' :[dataset['model_input'][i] for i in test_idx],
                         'model_labels':[dataset['model_labels'][i] for i in test_idx]
                        }
        
    return train_dataset, test_dataset

def get_batch(dataset, start_idx, batch_size = 128):

    """
    get batch of data from dataset for training or testing, cast to device

    Parameters:
    dataset: dict with 'model_input' and 'model_labels'
    start_idx: starting index to slice data
    batch_size
    
    Return:
    batch_X: input data, n_sample x n_timestep x n_features
    batch_Y: labels, n_sample 
    
    """
    
    batch_idx = np.arange(start_idx, start_idx+batch_size)
    batch_X = [dataset['model_input'][i].T for i in batch_idx] # convert to num timesteps x num features 
    batch_X = torch.stack(batch_X) # first dim is n samples in batch, assume batch_first = True
    batch_Y = [dataset['model_labels'][i] for i in batch_idx]
    batch_Y = torch.stack(batch_Y)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
    return batch_X, batch_Y

def trainLSTM():
    pass

