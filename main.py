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
from train import get_train_test_split, get_batch, trainRNN, trainTransformerClassifier
from models import RNNClassifier, WarmupWithScheduledDropLR, TransformerClassifier, PositionalEncoding
from utils import plot_loss_and_error, set_seed

if __name__ == '__main__':
    output_dir = '.'
    boc =  BrainObservatoryCache(
    manifest_file=str(Path(output_dir) / 'brain_observatory_manifest.json'))
    set_seed(50)
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
    sequence = [
    0, 0, 0, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 2, 7, 8, 2, 10, 11, 5, 2, 6,
    12, 12, 13, 10, 4, 2, 13, 12, 0, 7, 14, 15, 14, 12, 13, 13, 13, 7, 11, 16, 6, 2, 2, 8, 10, 8,
    2, 2, 16, 17, 17, 17, 17, 17, 18, 18, 18, 19, 20, 17, 21, 18, 17, 18, 20, 22, 21, 17, 21, 20,
    17, 23, 17, 18, 21, 24, 24, 25, 25, 25, 21, 25, 17, 25, 23, 17, 17, 25, 17, 25, 26, 21, 8, 21,
    21, 27, 21, 17, 26, 8, 25, 22, 15, 28, 28, 29, 30, 20
    ]
    #sequence = [i for i in range(118)] # no manual class combining
    mapping_dict, num_classes = get_mapping_dict(sequence)
    exps = get_exps(boc, cre_lines=cre_lines_to_use, targeted_structures=None, session_types=['three_session_B'])
    pre = 15
    post = 7
    data_type = 'pca'
    pca_comp = 50
    dataset = prep_dataset(boc, exps[19:24], mapping_dict=mapping_dict, pre=pre, post=post, data_type=data_type, pca_comp=pca_comp, cca=False)
    if data_type == 'dff':
        input_dim = dataset['model_input'][0].shape[0] # only for raw dff as input, one exp training at a time
    else:
        input_dim = pca_comp
    print(f'dataset loaded...')
    train_dataset, test_dataset = get_train_test_split(dataset, train_prop = 0.7, split_method = 'trial')
    
    # model = RNNClassifier(input_dim=input_dim, hidden_dim=512, embed_dim=512, num_layers=2, num_classes=num_classes, dropout_prob=0.5, nn_type = 'GRU')
    model = TransformerClassifier(input_dim=input_dim, hidden_dim=512, nlayers=1, nhead=8, num_classes=num_classes, dropout=0.5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = WarmupWithScheduledDropLR(optimizer, warmup_epochs=3, initial_lr=0.001, drop_epochs=[30, 70])
    epochs = 100
    batch_size = 128
    clip = 1
    # model, train_loss, train_error, val_error, val_loss = trainRNN(model, criterion, optimizer, scheduler, epochs, batch_size, clip, train_dataset, test_dataset, dry_run=False)
    model, train_loss, train_error, val_error, val_loss = trainTransformerClassifier(model, criterion, optimizer, scheduler, epochs, batch_size, train_dataset, test_dataset, dry_run=False)
    # fig, axes = plot_loss_and_error(train_loss, train_error, val_loss, val_error)
    # fig.suptitle('single exp, 2layer GRU, bsz=128, hiddim=512, dropout = 0.5, pca')
    # fig.savefig('1exp_2layerGRU_512hiddim_pca_residual.png')
    # torch.save(model.state_dict(), "RNNmodel.pth")
    # torch.save(model.state_dict(), "RNNmodel.pt")