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
from train import get_train_test_split, get_batch, trainRNN
from models import RNNClassifier, WarmupWithScheduledDropLR
from utils import plot_loss_and_error, set_seed

if __name__ == '__main__':
    output_dir = '.'
    boc =  BrainObservatoryCache(
    manifest_file=str(Path(output_dir) / 'brain_observatory_manifest.json'))
    set_seed(0)
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
    exps = get_exps(boc, cre_lines=cre_lines_to_use, targeted_structures=None, session_types=['three_session_B'])
    pre = 15
    post = 7
    data_type = 'dff'
    dataset = prep_dataset(boc, exps[22:23], pre, post, data_type=data_type)
    if data_type == 'dff':
        input_dim = dataset['model_input'][0].shape[0] # only for raw dff as input, one exp training at a time
    else:
        input_dim = 50
    print(f'dataset loaded...')
    train_dataset, test_dataset = get_train_test_split(dataset, train_prop = 0.7, split_method = 'trial')
    model = RNNClassifier(input_dim=input_dim, hidden_dim=256, embed_dim=256, num_layers=2, num_classes=119, dropout_prob=0.8, nn_type = 'GRU')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = WarmupWithScheduledDropLR(optimizer, warmup_epochs=3, initial_lr=0.001, drop_epochs=[30, 70])
    epochs = 100
    batch_size = 128
    clip = 1
    model, train_loss, train_error, val_error, val_loss = trainRNN(model, criterion, optimizer, scheduler, epochs, batch_size, clip, train_dataset, test_dataset)
    fig, axes = plot_loss_and_error(train_loss, train_error, val_loss, val_error)
    fig.suptitle('single exp, 2layer GRU, bsz=128, hiddim=256, dropout = 0.8, dff')
    fig.savefig('1exp_2layerGRU_256hiddim_dff.png')
    torch.save(model.state_dict(), "RNNmodel.pth")
    torch.save(model.state_dict(), "RNNmodel.pt")