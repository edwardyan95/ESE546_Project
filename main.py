import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt
from pathlib import Path
import pprint
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from dataset import *
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from train import get_train_test_split, get_batch, trainLSTM
from models import LSTMClassifier, WarmupWithScheduledDropLR

if __name__ == '__main__':
    output_dir = '.'
    boc =  BrainObservatoryCache(
    manifest_file=str(Path(output_dir) / 'brain_observatory_manifest.json'))
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
    dataset = prep_dataset(boc, exps[:30], pre, post)
    print(f'dataset loaded...')
    train_dataset, test_dataset = get_train_test_split(dataset, train_prop = 0.7, split_method = 'trial')
    model = LSTMClassifier(input_dim=50, hidden_dim=256, embed_dim=256, num_layers=4, num_classes=119, dropout_prob=0.5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
   
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = WarmupWithScheduledDropLR(optimizer, warmup_epochs=5, initial_lr=0.001, drop_epochs=[10, 20])
    epochs = 30
    batch_size = 256
    clip = 2
    model, train_loss, train_error, val_error, val_loss = trainLSTM(model, criterion, optimizer, scheduler, epochs, batch_size, clip, train_dataset, test_dataset)
    torch.save(model.state_dict(), "LSTMmodel.pth")
    torch.save(model.state_dict(), "LSTMmodel.pt")