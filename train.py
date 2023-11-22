import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
#from models import RNNClassifier
import random

def get_train_test_split(dataset, train_prop=0.7, split_method='trial', pad=False, max_features=400):
    """
    Split incoming dataset into random training and test dataset based on trials, subjects* or target structures*.
    Optionally pads each data point to a uniform size along the neuron dimension.

    Parameters:
    dataset: dict with 3 keys 'model_input', 'model_labels', 'metadata'
    train_prop: proportion of training dataset
    split_method: 'trials', 'subjects' or 'target structures'
    pad: pad along the neuron dimension with zeros, for deep set
    max_features: maximum number of features to pad to

    Return:
    train_dataset, test_dataset: dict with 2 keys 'model_input', 'model_labels'
    train_orig_num_feat, test_orig_num_feat: torch tensors of original feature counts
    """

    rand_idx = np.random.permutation(len(dataset['model_input']))
    num_training_sample = int(len(dataset['model_input']) * train_prop)
    train_idx = rand_idx[:num_training_sample]
    test_idx = rand_idx[num_training_sample:]
    if 'running_speed' in dataset and 'pupil_size' in dataset:
        behav = True
    else:
        behav = False

    def process_data(indices, pad, behavior):
        model_input = []
        orig_num_feat = []
        
        for i in indices:
            data_point = dataset['model_input'][i]
            orig_num_feat.append(data_point.shape[0])  # Original number of features
            if pad:
                # Pad the data point to have 'max_features' features
                padding_size = max_features - data_point.shape[0]
                if padding_size > 0:
                    padding = torch.zeros(padding_size, data_point.shape[1], dtype=torch.float)
                    data_point = torch.cat((data_point, padding), dim=0)
            if behavior:
                data_point = torch.cat((dataset['running_speed'][i], dataset['pupil_size'][i], data_point), dim=0)
            model_input.append(data_point)

        return model_input, torch.tensor(orig_num_feat, dtype=torch.long)

    train_dataset, train_orig_num_feat = process_data(train_idx, pad, behav)
    test_dataset, test_orig_num_feat = process_data(test_idx, pad, behav)

    train_dataset = {'model_input': train_dataset, 'model_labels': [dataset['model_labels'][i] for i in train_idx]}
    test_dataset = {'model_input': test_dataset, 'model_labels': [dataset['model_labels'][i] for i in test_idx]}

    return train_dataset, test_dataset, train_orig_num_feat, test_orig_num_feat

def get_batch(dataset, start_idx, batch_size = 128, with_replace = False):

    """
    get batch of data from dataset for training or testing, cast to device

    Parameters:
    dataset: dict with 'model_input' and 'model_labels'
    start_idx: starting index to slice data (if with_replace=True, then it doesn't matter)
    batch_size
    
    Return:
    batch_X: input data, n_sample x n_timestep x n_features
    batch_Y: labels, n_sample 
    
    """
    if not with_replace:
        batch_idx = np.arange(start_idx, start_idx+batch_size)
    else:
        batch_idx = random.sample(list(range(len(dataset['model_labels']))), batch_size)
    
    batch_X = [dataset['model_input'][i].T for i in batch_idx] # convert to num timesteps x num features 
    batch_X = torch.stack(batch_X) # first dim is n samples in batch, assume batch_first = True
    batch_Y = [dataset['model_labels'][i] for i in batch_idx]
    batch_Y = torch.stack(batch_Y)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
    return batch_X, batch_Y, batch_idx

import torch
import numpy as np
import torch.nn as nn

def trainRNN(model, criterion, optimizer, scheduler, epochs, batch_size, clip, train_dataset, test_dataset, dry_run=False):
    model.train()
    total_batch = len(train_dataset['model_labels']) // batch_size
    batch_train_loss, batch_train_error, batch_train_top5_error = [], [], []
    train_loss, train_error, train_top5_error = [], [], []
    val_loss, val_error, val_top5_error = [], [], []

    if dry_run:
        epochs = 1

    for epoch in range(epochs):
        print(f'training epoch#{epoch}')
        for batch, i in enumerate(range(0, total_batch * batch_size, batch_size)):
            X, y, _ = get_batch(train_dataset, i, batch_size, with_replace=True)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()  # Does backpropagation and calculates gradients
            _, predicted = torch.max(output, 1)

            # Calculate Top-1 error
            error = (predicted != y).sum().item() / len(y)

            # Calculate Top-5 error
            _, top5_pred = output.topk(5, 1, True, True)
            top5_error = 1 - (top5_pred == y.view(-1, 1).expand_as(top5_pred)).sum().item() / len(y)

            batch_train_loss.append(loss.item())
            batch_train_error.append(error)
            batch_train_top5_error.append(top5_error)

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()  # Updates the weights accordingly
            scheduler.step()

            if dry_run:  # end training with 1 batch
                break

            if batch % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    x_val, y_val, _ = get_batch(test_dataset, 0, batch_size, with_replace=True)
                    output_val = model(x_val)
                    loss_val = criterion(output_val, y_val)

                    _, predicted_val = torch.max(output_val, 1)
                    error_val = (predicted_val != y_val).sum().item() / len(y_val)

                    _, top5_pred_val = output_val.topk(5, 1, True, True)
                    top5_error_val = 1 - (top5_pred_val == y_val.view(-1, 1).expand_as(top5_pred_val)).sum().item() / len(y_val)

                    val_error.append(error_val)
                    val_top5_error.append(top5_error_val)
                    val_loss.append(loss_val)

                model.train()
                train_loss.append(np.mean(batch_train_loss[-1000:-1]))
                train_error.append(np.mean(batch_train_error[-1000:-1]))
                train_top5_error.append(np.mean(batch_train_top5_error[-1000:-1]))

                print(f'batch#{batch}, running train loss: {np.mean(batch_train_loss[-1000:-1]): .2f}, running train error: {np.mean(batch_train_error[-1000:-1]): .2%}, running train top-5 error: {np.mean(batch_train_top5_error[-1000:-1]): .2%}')
                print(f'batch#{batch}, val loss: {loss_val.item(): .2f}, val error: {error_val: .2%}, val top-5 error: {top5_error_val: .2%}')

    return model, train_loss, train_error, train_top5_error, val_loss, val_error, val_top5_error



def trainTransformerClassifier(model, criterion, optimizer, scheduler, epochs, batch_size, train_dataset, test_dataset, dry_run=False):
    model.train()
    total_batch = len(train_dataset['model_labels']) // batch_size
    batch_train_loss, batch_train_error, batch_train_top5_error = [], [], []
    train_loss, train_error, train_top5_error = [], [], []
    val_loss, val_error, val_top5_error = [], [], []

    if dry_run:
        epochs = 1

    for epoch in range(epochs):
        print(f'training epoch#{epoch}')
        for batch, i in enumerate(range(0, total_batch * batch_size, batch_size)):
            X, y, _ = get_batch(train_dataset, i, batch_size, with_replace=True)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()  # Does backpropagation and calculates gradients
            _, predicted = torch.max(output, 1)

            # Calculate Top-1 error
            error = (predicted != y).sum().item() / len(y)

            # Calculate Top-5 error
            _, top5_pred = output.topk(5, 1, True, True)
            top5_error = 1 - (top5_pred == y.view(-1, 1).expand_as(top5_pred)).sum().item() / len(y)

            batch_train_loss.append(loss.item())
            batch_train_error.append(error)
            batch_train_top5_error.append(top5_error)

            optimizer.step()  # Updates the weights accordingly
            scheduler.step()

            if dry_run:  # end training with 1 batch
                break

            if batch % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    x_val, y_val, _ = get_batch(test_dataset, 0, batch_size, with_replace=True)
                    output_val = model(x_val)
                    loss_val = criterion(output_val, y_val)

                    _, predicted_val = torch.max(output_val, 1)
                    error_val = (predicted_val != y_val).sum().item() / len(y_val)

                    _, top5_pred_val = output_val.topk(5, 1, True, True)
                    top5_error_val = 1 - (top5_pred_val == y_val.view(-1, 1).expand_as(top5_pred_val)).sum().item() / len(y_val)

                    val_error.append(error_val)
                    val_top5_error.append(top5_error_val)
                    val_loss.append(loss_val)

                model.train()
                train_loss.append(np.mean(batch_train_loss[-1000:-1]))
                train_error.append(np.mean(batch_train_error[-1000:-1]))
                train_top5_error.append(np.mean(batch_train_top5_error[-1000:-1]))

                print(f'batch#{batch}, running train loss: {np.mean(batch_train_loss[-1000:-1]): .2f}, running train error: {np.mean(batch_train_error[-1000:-1]): .2%}, running train top-5 error: {np.mean(batch_train_top5_error[-1000:-1]): .2%}')
                print(f'batch#{batch}, val loss: {loss_val.item(): .2f}, val error: {error_val: .2%}, val top-5 error: {top5_error_val: .2%}')

    return model, train_loss, train_error, train_top5_error, val_loss, val_error, val_top5_error
 


def trainDeepSetRNNClassifier(model, criterion, optimizer, scheduler, epochs, batch_size, clip, train_dataset, test_dataset, train_orig_num_feat, test_orig_num_feat, dry_run = False):
    model.train()
    total_batch = len(train_dataset['model_labels'])//batch_size
    batch_train_loss, batch_train_error = [], []
    train_loss, train_error = [], []
    val_loss, val_error = [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dry_run:
        epochs = 1
    for epoch in range(epochs):
        print(f'training epoch#{epoch}')
        for batch, i in enumerate(range(0, total_batch*batch_size, batch_size)):
            X, y, train_batch_idx = get_batch(train_dataset, i, batch_size, with_replace=True)
            optimizer.zero_grad()
            train_feat_counts = train_orig_num_feat[train_batch_idx]
            output= model(X, train_feat_counts.to(device))
            loss = criterion(output, y)
            loss.backward() # Does backpropagation and calculates gradients
            _, predicted = torch.max(output,1)
            error = (predicted != y).sum().item()/len(y)
            batch_train_loss.append(loss.item())
            batch_train_error.append(error)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step() # Updates the weights accordingly
            scheduler.step()
            if dry_run: # end training with 1 batch
                break
            if batch%1000 == 0:
                model.eval()
                with torch.no_grad():
                    x_val,y_val, val_batch_idx =  get_batch(test_dataset, 0, batch_size, with_replace=True)
                    val_feat_counts = test_orig_num_feat[val_batch_idx]
                    output_val = model(x_val, val_feat_counts.to(device))
                    loss_val = criterion(output_val, y_val)
                    _, predicted_val = torch.max(output_val,1)
                    error_val = (predicted_val != y_val).sum().item()/len(y_val)
                    val_error.append(error_val)
                    val_loss.append(loss_val)
                model.train()
                train_loss.append(np.mean(batch_train_loss[-1000:-1]))
                train_error.append(np.mean(batch_train_error[-1000:-1]))
                print(f'batch#{batch}, running train loss: {np.mean(batch_train_loss[-1000:-1]): .2f}, running train error: {np.mean(batch_train_error[-1000:-1]): .2%}')
                print(f'batch#{batch}, val loss: {loss_val.item(): .2f}, val error: {error_val: .2%}')
    return model, train_loss, train_error, val_error, val_loss



    

