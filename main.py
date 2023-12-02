import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt
from pathlib import Path
import pprint
import numpy as np
import pandas as pd
import random
import os
from sklearn.decomposition import PCA
from dataset import *
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from train import *
from models import *
from utils import *

if __name__ == '__main__':
    output_dir = '.'
    boc =  BrainObservatoryCache(
    manifest_file=str(Path(output_dir) / 'brain_observatory_manifest.json'))
    set_seed(2)
    cre_lines_to_use = [
        'Cux2-CreERT2',
        'Emx1-IRES-Cre',
        'Fezf2-CreER',
        'Nr5a1-Cre',
        'Ntsr1-Cre_GN220',
        'Rbp4-Cre_KL100',
        'Rorb-IRES2-Cre',
        'Scnn1a-Tg3-Cre',
        'Slc17a7-IRES2-Cre',
        'Tlx3-Cre_PL56',
        ]
    
    #TODO: First item to experiment with, whether to group labels or not
    group_labels = False
    if group_labels:
        sequence = [0, 0, 0, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 2, 7, 8, 2, 10, 11, 5, 2, 6,
                    12, 12, 13, 10, 4, 2, 13, 12, 0, 7, 14, 15, 14, 12, 13, 13, 13, 7, 11, 16, 6, 2, 2, 8, 10, 8,
                    2, 2, 16, 17, 17, 17, 17, 17, 18, 18, 18, 19, 20, 17, 21, 18, 17, 18, 20, 22, 21, 17, 21, 20,
                    17, 23, 17, 18, 21, 24, 24, 25, 25, 25, 21, 25, 17, 25, 23, 17, 17, 25, 17, 25, 26, 21, 8, 21,
                    21, 27, 21, 17, 26, 8, 25, 22, 15, 28, 28, 29, 30, 20]
    else:
        sequence = [i for i in range(118)] # no manual class combining
    mapping_dict, num_classes = get_mapping_dict(sequence)
    
    #TODO: Second item to experiment with, whether to use behavioral signals or not
    behavior = False
    
    #TODO: Third item to experiment with, how many ms before and after the image is shown should be used for model fitting
    pre = 0
    post = 0
    
    #TODO: Fourth item to experiment with, dff means full neurons vs. pca the dimension-reduced version
    #Play with different pca_comp values as well
    data_type = 'pca'
    pca_comp = 50
    
    ######## DO NOT CHANGE: GET ALL EXPERIMENTS #########
    exps = get_exps(boc, cre_lines=cre_lines_to_use, targeted_structures=['VISp'], session_types=['three_session_B'])
    #####################################################
    
    # Optional: Filter experiments if needed
    #exps = filter_exps(boc, exps, num_exps = 3, min_neuron = 100, max_neuron = 1000, behavior = False)
    
    # Optional: Try CCA Matching
    # cca_ind = True
    cca_ind = False
    
    #TODO: Fifth item to experiment with, three options:
    # 1) "single": to focus on performance on one randomly chosen experiment
    # 2) "multi": to combine multiple experiments together then perform train/test split within itself 
    # 3) "unseen": to split the 90+ experiments into a list of experiments for training vs. test
    # then once we fit a model on the experiments in the training, we evaluate on 
    # the unseen experiments in the test data. Will be used to test whether our model is generalizable
    # to unseen population in the world or not
    exp_type = 'single'
    num_exps = 10
    
    ######## DO NOT CHANGE THIS ########
    exp_chosen = 22
    ####################################
    
    #TODO: Sixth item to experiment with, train vs. test split ratio
    train_prop = 0.9
    
    #TODO: Seventh item to experiment with, pad the sequence or not
    pad_ind = False
    
    #TODO: Eighth item to experiment with, max_features
    max_features = 400

    
        
    if exp_type == 'single':
        exps = exps[exp_chosen:exp_chosen+1]
        dataset= prep_dataset(boc, exps, mapping_dict=mapping_dict, pre=pre, post=post, data_type=data_type, pca_comp=pca_comp, cca=cca_ind, behavior=behavior)
        #dataset = prep_dataset_by_static_grating(boc, exps, mapping_dict=mapping_dict, pre=pre, post=pre, behavior=False)
        train_dataset, test_dataset, train_orig_num_feat, test_orig_num_feat = get_train_test_split(dataset, train_prop = train_prop, pad = pad_ind, max_features=max_features)
    elif exp_type == 'multi':
        #exps = exps[exp_chosen:exp_chosen+num_exps]
        dataset = prep_dataset(boc, exps, mapping_dict=mapping_dict, pre=pre, post=post, data_type=data_type, pca_comp=pca_comp, cca=cca_ind, behavior=behavior)
        #dataset = prep_dataset_by_static_grating(boc, exps, mapping_dict=mapping_dict, pre=pre, post=pre, behavior=False)
        train_dataset, test_dataset, train_orig_num_feat, test_orig_num_feat = get_train_test_split(dataset, train_prop = train_prop, pad = pad_ind, max_features=max_features)
    else:
        train_exps, test_exps = split_by_exps(exps, train_prop = train_prop)
        train_dataset = prep_dataset(boc, train_exps, mapping_dict=mapping_dict, pre=pre, post=post, data_type=data_type, pca_comp=pca_comp, cca=cca_ind, behavior=behavior)
        test_dataset = prep_dataset(boc, test_exps, mapping_dict=mapping_dict, pre=pre, post=post, data_type=data_type, pca_comp=pca_comp, cca=cca_ind, behavior=behavior)
        train_idx = np.arange(len(train_dataset['model_input']))
        test_idx = np.arange(len(test_dataset['model_input']))
        train_temp, train_orig_num_feat = process_data(train_dataset, train_idx, pad=pad_ind, behavior=behavior, max_features=max_features)
        test_temp, test_orig_num_feat = process_data(test_dataset, test_idx, pad=pad_ind, behavior=behavior, max_features=max_features)
        train_dataset = {'model_input': train_temp, 'model_labels': train_dataset['model_labels']}
        test_dataset = {'model_input': test_temp, 'model_labels': test_dataset['model_labels']}
        train_dataset = sample_data(train_dataset, 0.01)
        test_dataset = sample_data(test_dataset, 0.01)
        
    if data_type == 'dff':
        input_dim = dataset['model_input'][0].shape[0]
    elif data_type == 'pca':
        input_dim = pca_comp
    else:
        input_dim = 120
    
    if behavior:
        input_dim+=2 # added features are running_speed and pupil_size
    print(f'dataset loaded...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # RNN Model 
    # TODO: Model hyper-parameters to tune:
    # 1) hidden_dim, embed_dim
    # 2) num_layers
    # 3) dropout_prob
    # 4) nn_type (GRU vs. LSTM vs. Transformer)
    # 5) label_smoothing parameter
    # 6) Learning Rate and Warm Up Schedule
    # 7) Weight Decay
    # 8) Batch Size
    # 9) Epochs
    # 10) Gradient Clips
    
    hidden_dim, embed_dim = 512, 512
    num_layers = 2
    dropout_prob = 0.5
    model_type = 'GRU'
    label_smoothing = 0.5
    initial_lr = 0.001
    warmup_epochs = 3
    weight_decay = 1e-4
    batch_size = 256
    epochs = 100
    clip = 1
    
    model = RNNClassifier(input_dim=input_dim, hidden_dim=hidden_dim, embed_dim=embed_dim, num_layers=num_layers, num_classes=num_classes, dropout_prob=dropout_prob, nn_type = model_type)
    
    # Transformer Model
    # model = TransformerClassifier(input_dim=input_dim, hidden_dim=512, nlayers=1, nhead=8, num_classes=num_classes, dropout=0.5)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    scheduler = WarmupWithScheduledDropLR(optimizer, warmup_epochs=warmup_epochs, initial_lr=initial_lr, drop_epochs=[30, 70])
   
    model, train_loss, train_error, train_top5_error, val_loss, val_error, val_top5_error = trainRNN(model, criterion, optimizer, scheduler, epochs, batch_size, clip, train_dataset, test_dataset, dry_run=False)
    # model, train_loss, train_error, train_top5_error, val_loss, val_error, val_top5_error = trainTransformerClassifier(model, criterion, optimizer, scheduler, epochs, batch_size, train_dataset, test_dataset, dry_run=False)
    
    #Deep set 
    # max_neuron = 400
    # train_dataset, test_dataset, train_orig_num_feat, test_orig_num_feat = get_train_test_split(dataset, train_prop = 0.9, pad = True, max_features=max_neuron)
    # model = DeepSetRNNClassifier(input_dim=max_neuron, hidden_dim=32, rnn_hidden_dim=256, num_classes=num_classes, num_linear_layers=3, num_rnn_layers=2, rnn_type='LSTM', dropout_prob=0.5)
    # model = model.to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # scheduler = WarmupWithScheduledDropLR(optimizer, warmup_epochs=3, initial_lr=0.001, drop_epochs=[30, 70])
    # epochs = 100
    # batch_size = 256
    # clip = 1
    # model, train_loss, train_error, val_error, val_loss = trainDeepSetRNNClassifier(model, criterion, optimizer, scheduler, epochs, batch_size, clip, train_dataset, test_dataset, train_orig_num_feat, test_orig_num_feat, dry_run = False)
    
    
    # fig, axes = plot_loss_and_error(train_loss, train_error, val_loss, val_error)
    # fig.suptitle('single exp, 2layer GRU, bsz=128, hiddim=512, dropout = 0.5, pca')
    # fig.savefig('1exp_2layerGRU_512hiddim_pca_residual.png')
    # torch.save(model.state_dict(), "RNNmodel.pth")
    # torch.save(model.state_dict(), "RNNmodel.pt")