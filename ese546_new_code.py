import matplotlib.pyplot as plt
from pathlib import Path
import pprint
import numpy as np
import pandas as pd
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import torch
from sklearn.cross_decomposition import CCA
from scipy import stats

def get_exps(boc, cre_lines=None, targeted_structures=None, session_types=None):
    """
    params
    boc: BrainObservatoryCache object
    cre_lines: specific cre_lines to include, list
    targeted_structures: specific targeted_structures to include, list
    session_types: specific session_types to include, list
    assumes get all experiment if not specified
    return
    exps: list of experiment objects

    """
    if targeted_structures is None:
        targeted_structures = boc.get_all_targeted_structures()
    if cre_lines is None:
        cre_lines = boc.get_all_cre_lines()
    if session_types is None:
        session_types = ['three_session_A',
                         'three_session_B',
                         'three_session_C']
    ecs = boc.get_experiment_containers(targeted_structures=targeted_structures, cre_lines=cre_lines) # experiment containers
    ec_id = [ecs[i]['id'] for i in range(len(ecs))]
    exps = boc.get_ophys_experiments(experiment_container_ids=ec_id, session_types=session_types)
    for exp in exps:
        dataset = boc.get_ophys_experiment_data(exp['id']) # this is downloading the data

    return exps

def get_fluo(boc, exp):
    """
    params
    boc: BrainObservatoryCache object
    exp: single experiment object

    return
    dF/F traces with dim: num cells X num timesteps

    maybe need to add deconvolution?
    """
    session_data = boc.get_ophys_experiment_data(exp['id'])
    _, dFF = session_data.get_dff_traces()
    return dFF

def get_stim_df(boc, exp, stimulus_name='natural_scenes'):
    """
    params
    boc: BrainObservatoryCache object
    exp: single experiment object
    stimulus_name: stimulus type you want to extract info of

    return
    dataframe: for natural_scenes, should be three column: frame (range -1 (gray frame?) to 117), start timestep, end timestep

    """

    session_data = boc.get_ophys_experiment_data(exp['id'])
    session_stim_epoch = session_data.get_stimulus_epoch_table()
    assert stimulus_name in session_stim_epoch['stimulus'].values, 'Stimulus you want is not present in this experiment session!'
    session_stim = session_data.get_stimulus_table(stimulus_name)
    return session_stim

def pca_and_pad(data, num_comp):

    """
    Perform PCA on data and return the first 50 principal components.
    If the original data has less than 50 dimensions, pad with zeros.

    Parameters:
    - data (dFF in this case): numpy array of shape (num_dimensions, num_samples)

    Returns:
    - pca_data: numpy array of shape (50, num_samples)

    """
    num_dimensions, num_samples = data.shape

    # Perform PCA
    pca = PCA(n_components=min(num_comp, num_dimensions))
    pca_data = pca.fit_transform(data.T).T
    #reconstructed_data = pca.inverse_transform(pca_data.T).T

    # Pad with zeros if necessary
    if pca_data.shape[0] < num_comp:
        padding = np.zeros((num_comp - pca_data.shape[0], num_samples))
        pca_data = np.vstack((pca_data, padding))


    return pca_data, np.sum(pca.explained_variance_ratio_)

def plot_traces(data, x_range, input_type, figsize=(15,15)):

    """
    plot data, either single cell dff traces or pca traces, specified by input type

    Parameters:
    - data (dFF in this case): numpy array of shape (num_dimensions, num_samples)
    - x_range: range of samples to plot, list of indices
    - input_type: either "neuron" or "pca components"

    Returns:
    - ax handle
    """

    fig,ax = plt.subplots(figsize=figsize)
    numCell = data.shape[0]
    #x_range = np.arange(x_range[0],x_range[1])
    for i in range(numCell):
        data2plot = data[i,x_range]+(i)*3
        ax.plot(data2plot)

    ax.set_xlabel('timesteps')
    ax.set_yticks(ticks=np.arange(numCell)*3)
    ax.set_yticklabels([str(x+1) for x in np.arange(numCell)])
    ax.set_ylabel(input_type)
    # ax.title(input_type + ' ')
    plt.tight_layout()
    return ax

def extract_data_by_images(data, stim_df, mapping_dict=None, pre=15, post=7):
    """
    Extract segments of the data based on a pandas dataframe (natural scene stim df from get_stim_df()).
    30Hz, each image is presented for 250ms (8 frames), will also take 1s (30 frames) preceding and 0.25s (7 frames)post stim, total 1.5s data
    make sure that the data and image_df are from the same experiment!!!

    Parameters:
    - data: numpy array of shape (ndim, ntimesteps)
    - stim_df: pandas dataframe with columns 'frame', 'start', 'end'
    - dictionary for mapping images to different classes
    - pre: how many frames before image presentation should be extracted
    - post: how many frames after image presentation should be extracted
    Returns:
    - result: list of tuples with labels as first argument and segments of data as second arguent
    labels are int, data are numpy array, note that original label -1 is gray screen, convert to 118
    """
    data_segments, labels = [],[]
    desig_len = pre+8+post # desired num of timesteps to extract, 8 correspond to 8 frames, 250ms of stim presentation
    for index, row in stim_df.iterrows():
        label = row['frame']
        start_timestep = row['start']
        end_timestep = row['end']
        if start_timestep-pre < 0 or end_timestep+post > data.shape[1]:
            continue
        # Extract segment of data corresponding to the given start and end timesteps
        segment = data[:, start_timestep-pre:end_timestep + post+1]
        pad_length = max(0, desig_len - segment.shape[1])
        segment = np.pad(segment, ((0, 0), (0, pad_length)), 'constant')
        segment = segment[:,:desig_len]
        data_segments.append(segment)
        labels.append(label)
    if mapping_dict is not None:
        mapped_labels = [mapping_dict[label] for label in labels]
    else:
        mapped_labels = [118 if label == -1 else label for label in labels]

    return data_segments, mapped_labels

def cca_align(ref_data, ref_labels, target_data, target_labels):
    """
    align the pca dimensions of reference data and target data using canonical correlation analysis

    Parameters:
    - ref_data: reference and target data and labels, should be returned from extract_data_from _images()

    Returns:
    """
    num_features, num_tmstps = ref_data[0].shape # original number of components, and number of timesteps per trial
    num_trials = len(target_labels)

    ref_sort_idx = np.argsort(ref_labels) # sort trials based on image presented (labels) such that the data corresponding to same labels are concatenated together in sequence
    ref_sort_data = [ref_data[i] for i in ref_sort_idx]
    ref_sort_labels = [ref_labels[i] for i in ref_sort_idx]
    ref_concat_data = np.concatenate(ref_sort_data, axis=1)

    target_sort_idx = np.argsort(target_labels)
    target_sort_data = [target_data[i] for i in target_sort_idx]
    target_sort_labels = [target_labels[i] for i in target_sort_idx]
    target_concat_data = np.concatenate(target_sort_data, axis=1)

    cca = CCA(n_components=ref_concat_data.shape[0])
    cca.fit(ref_concat_data.T, target_concat_data.T)
    trans_data = target_concat_data.T.dot(cca.y_rotations_).dot(np.linalg.inv(cca.x_rotations_)).T # find transformed data matrix B that is most correlated to ref data matrix A
    pca_comp_corr = []
    for i in range(target_concat_data.shape[0]):
        pca_comp_corr.append(np.corrcoef(ref_concat_data[i,:], trans_data[i,:])[0,1])


    trans_data_list = [trans_data[:,i*num_tmstps:i*num_tmstps+num_tmstps] for i in range(num_trials)]
    reverse_sort_idx = np.argsort(target_sort_idx) # reverse label and data order to original input
    trans_data_list = [trans_data_list[i] for i in reverse_sort_idx]
    reverse_sort_labels = [target_sort_labels[i] for i in reverse_sort_idx]

    return trans_data_list, reverse_sort_labels, np.mean(pca_comp_corr)

def prep_dataset(boc, exps, mapping_dict=None, pre=15, post=7, data_type='pca', pca_comp = None, cca=False, behavior=False):
    """
    preparing dataset for training

    Parameters:
    - boc: BrainObservatoryCache object
    - exps: list of experiment objects, returned from get_exps()
    - pre: how many timesteps before image presentation should be extracted
    - post: how many timesteps after image presentation should be extracted
    - data_type: can be 'pca' or 'dff', if pca, forced data output to 50dim x X timesteps, if dff, data dim depend on num of neurons
    Returns:
    - out: dict containing 3 keys: model_input, model_labels, metadata, each with the same length containing all datapoints (trials)
    - metadata contains ['targeted_structures', 'experiment_container_id', 'indicator', 'cre_line', 'session_type', 'specimen_name'], indexed same as input and labels
    - each model_input is 50 x pre+8+post (PCA components x timesteps) in float32 tensor
    - each model_label corresponds to the image presented to the mouse at the trial with same index as model_input, in LongTensor
    """
    model_input, model_labels, metadata = [], [], []
    if behavior:
        running_speed_out, pupil_size_out = [], []
    meta_required = ['targeted_structures',
                     'experiment_container_id',
                     'indicator',
                     'cre_line',
                     'session_type',
                     'specimen_name']
    if cca:
        numCell = []
        for exp in exps:
            data_set = boc.get_ophys_experiment_data(exp['id'])
            cids = data_set.get_cell_specimen_ids()
            numCell.append(len(cids))
        ref_exp = exps[np.argmax(numCell)]
        dff = get_fluo(boc, exp)
        pca_dff, ref_explained_var = pca_and_pad(dff, num_comp=pca_comp)
        print(f'ref data explained variance: {ref_explained_var:.2f}')
        stim_df = get_stim_df(boc, ref_exp, stimulus_name='natural_scenes')
        ref_data, ref_labels = extract_data_by_images(pca_dff, stim_df, mapping_dict=mapping_dict, pre=pre, post=post)
    exp_count = 0
    for exp in exps:
        meta = boc.get_ophys_experiment_data(exp['id']).get_metadata()

        try:
            dff = get_fluo(boc, exp)
            dff = stats.zscore(dff, axis=1)
        except:
            print(f"dFF extraction from experiment id{meta['experiment_container_id']} failed!")
            continue
        pca_dff, explained_var = pca_and_pad(dff, num_comp=pca_comp)
        try:
            stim_df = get_stim_df(boc, exp, stimulus_name='natural_scenes')
        except:
            print(f"stim table from experiment id{meta['experiment_container_id']} failed!")
            continue
        if data_type == 'pca':
            data, labels = extract_data_by_images(pca_dff, stim_df, mapping_dict=mapping_dict, pre=pre, post=post)
            print(f'exp#{exp_count} data explained variance: {explained_var:.2f}')
        elif data_type == 'dff':
            data, labels = extract_data_by_images(dff, stim_df, mapping_dict=mapping_dict, pre=pre, post=post)

        if cca:
            data, labels, adj_corr = cca_align(ref_data, ref_labels, data, labels)
            print(f'exp#{exp_count} aligned corr: {adj_corr: .2f}')


        data = [torch.from_numpy(datum).float() for datum in data]
        labels = torch.LongTensor(labels)
        model_input.extend(data)
        model_labels.extend(labels)
        meta = {k:v for k, v in meta.items() if k in meta_required}
        meta_list = [meta.copy() for _ in range(len(labels))] # repeat metadata for each datum (natural scene trials) in the exp
        metadata.extend(meta_list)
        if behavior:
            run, _ = boc.get_ophys_experiment_data(exp['id']).get_running_speed()
            run[np.isnan(run)]=0
            run = stats.zscore(run)
            run = run.reshape(1,-1) # 1 x timepoints
            run_data, _ = extract_data_by_images(run, stim_df, mapping_dict, pre, post)
            _, pupil_size = boc.get_ophys_experiment_data(exp['id']).get_pupil_size()
            pupil_size[np.isnan(pupil_size)]=0
            pupil_size = stats.zscore(pupil_size)
            pupil_size = pupil_size.reshape(1,-1)
            pupil_size_data, _ = extract_data_by_images(pupil_size, stim_df, mapping_dict, pre, post)
            run_data = [torch.from_numpy(datum).float() for datum in run_data]
            pupil_size_data = [torch.from_numpy(datum).float() for datum in pupil_size_data]
            running_speed_out.extend(run_data)
            pupil_size_out.extend(pupil_size_data)

        exp_count+=1

    out = {'model_input':model_input,
           'model_labels':model_labels,
           'metadata':metadata}
    if behavior:
        out.update({'running_speed':running_speed_out,
                    'pupil_size':pupil_size_out})

    return out

def get_mapping_dict(seq):
    """
    to generate the dictionary where the keys corresponds to indexes of the 118 images, and the values correspond to the manually defined classes
    parameters:
    seq: list of manually defined classes for each of the 118 images. attach -1:max(seq) for gray screen

    returns:
    mapping_dict: mapping dictionary described above,
    num_classes: total number of manually defined classes
    """
    dict1 = {i-1: val for i, val in enumerate(seq, start=1)}
    dict1.update({-1:max(seq)+1})
    return dict1, len(set(dict1.values()))

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class RNNClassifier(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=100, embed_dim=100, num_layers=2, num_classes=119, dropout_prob=0.5, nn_type = 'LSTM'):
        super(RNNClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.nn_type = nn_type
        self.drop = nn.Dropout(dropout_prob)
        self.embed = nn.Linear(input_dim, embed_dim)
        if nn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        elif nn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)

        self.fc = nn.Linear(hidden_dim, num_classes)

        print('Num parameters: ', sum([p.numel() for p in self.parameters()]))

    def forward(self, x):
        x = self.drop(self.embed(x))
        #print('x shape:', x.shape)
        # Passing the input through the LSTM layers
        rnn_out, _ = self.rnn(x)
        #print('out shape:', rnn_out.shape)
        rnn_out = self.drop(rnn_out)
        rnn_out = torch.add(rnn_out, x) # residual
        # Only take the output from the final timestep
        rnn_out = rnn_out[:, -1, :]
        #print('rnn out shape:', rnn_out.shape)
        # Pass through the fully connected layers
        output = self.fc(rnn_out)
        #print('fc output:', output.shape)
        # return F.log_softmax(output)
        return output

    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters()).data
    #     hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
    #                   weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
    #     return hidden

class WarmupWithScheduledDropLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, initial_lr, drop_epochs, drop_factor=0.5, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.drop_epochs = drop_epochs
        self.drop_factor = drop_factor
        super(WarmupWithScheduledDropLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.initial_lr * alpha for base_lr in self.base_lrs]
        else:
            # Scheduled drop
            lr = self.initial_lr
            for epoch in self.drop_epochs:
                if self.last_epoch >= epoch:
                    lr *= self.drop_factor
            return [lr for base_lr in self.base_lrs]

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        seq_len: the length of the incoming sequence (default=50).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, seq_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = torch.transpose(x, 0, 1) # accommodate batch_first = True
        device = x.device  # Get the device from the input tensor
        pe = self.pe[:x.size(0), :].to(device)  # Move pe to the same device as x
        x = x + pe
        x = self.dropout(x)
        return torch.transpose(x, 0, 1)

class TransformerClassifier(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, input_dim, hidden_dim, nlayers, nhead, num_classes, dropout=0.5):
        super(TransformerClassifier, self).__init__(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, num_encoder_layers=nlayers, batch_first=True)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.embed = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_classes)
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dim))

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embed.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=False):
        if has_mask:
            device = src.device
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.embed(src) * math.sqrt(self.hidden_dim)
        src = torch.cat((self.class_token.expand(src.shape[0], 1, -1), src), dim=1)
        pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, seq_len=src.shape[1]+1) # src shape: n_batch x n_timestep (add class token) x hidden_dim
        src = pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        #print('encoder output: ', output.shape)
        output = output[:,0,:]
        output = self.decoder(output)
        #print('decoder output: ', output.shape)
        return output

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
#from models import RNNClassifier
import random

def split_by_exps(exps, train_prop = 0.9):
    """
    Split the given list of experiments into training vs. validation experiments

    Parameters:
    exps: list fo experiment obejcts
    train_prop: float, proportion of experiments to include in the train set

    Returns:
    train_exps: list of experiment objects for training
    test_exps: list of experiment objects for testing
    """
    num_train = int(train_prop * len(exps))
    random.shuffle(exps)  # Randomly shuffle experiments
    train_exps = exps[:num_train]
    test_exps = exps[num_train:]
    return train_exps, test_exps

def sample_data(dataset, sampling_ratio=1.0):
    """
    Randomly sample a specified ratio of the training dataset, ensuring each label is represented at least once.

    Parameters:
    train_dataset: dict containing the training data
    sampling_ratio: float, the proportion of the training data to sample

    Returns:
    sampled_train_dataset: dict containing the sampled training data
    """
    # Extract labels and their indices
    label_indices = {}
    for i, label in enumerate(dataset['model_labels']):
        if label.item() not in label_indices:
            label_indices[label.item()] = []
        label_indices[label.item()].append(i)

    # Ensure each label is represented at least once
    sampled_indices = []
    for label, indices in label_indices.items():
        sampled_indices.extend(random.sample(indices, 1))

    # Sample the rest of the data to meet the sampling ratio
    additional_samples_needed = int(len(dataset['model_input']) * sampling_ratio) - len(sampled_indices)
    if additional_samples_needed > 0:
        flat_indices = [i for indices in label_indices.values() for i in indices]
        additional_indices = np.random.choice(flat_indices, additional_samples_needed, replace=False)
        sampled_indices.extend(additional_indices)

    sampled_train_dataset = {
        'model_input': [dataset['model_input'][i] for i in sampled_indices],
        'model_labels': [dataset['model_labels'][i] for i in sampled_indices],
    }
    return sampled_train_dataset

def process_data(dataset, indices, pad=False, behavior=False, max_features=400):
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


def get_train_test_split(dataset, train_prop = 0.7, pad=False, max_features=400):
    """
    Split incoming dataset into random training and test dataset based on trials, subjects* or target structures*.
    Optionally pads each data point to a uniform size along the neuron dimension.

    Parameters:
    dataset: dict with 3 keys 'model_input', 'model_labels', 'metadata'
    train_prop: proportion of training dataset
    split_method: 'trials' or 'subjects'
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

    train_dataset, train_orig_num_feat = process_data(dataset, train_idx, pad, behav, max_features)
    test_dataset, test_orig_num_feat = process_data(dataset, test_idx, pad, behav, max_features)

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

def trainModel(model, criterion, optimizer, scheduler, epochs, batch_size, train_dataset, test_dataset, clip=None, dry_run=False):
    model.train()
    total_batch = len(train_dataset['model_labels']) // batch_size
    train_loss, train_error, train_top5_error = [], [], []
    val_loss, val_error, val_top5_error = [], [], []

    if dry_run:
        epochs = 1

    for epoch in range(epochs):
        batch_train_loss, batch_train_error, batch_train_top5_error = [], [], []

        for batch, i in enumerate(range(0, total_batch * batch_size, batch_size)):
            X, y, _ = get_batch(train_dataset, i, batch_size, with_replace=True)
            optimizer.zero_grad()
            output = model(X)

            loss = criterion(output, y)
            loss.backward()  # Backpropagation and calculate gradients
            if clip and model.model_type == 'RNN':
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()  # Update weights
            scheduler.step()

            _, predicted = torch.max(output, 1)

            error = (predicted != y).sum().item() / len(y)
            batch_train_loss.append(loss.item())
            batch_train_error.append(error)

            _, top5_pred = output.topk(5, 1, True, True)
            top5_error = 1 - (top5_pred == y.view(-1, 1).expand_as(top5_pred)).sum().item() / len(y)
            batch_train_top5_error.append(top5_error)

            if dry_run:
                break

        epoch_train_loss = np.mean(batch_train_loss)
        epoch_train_error = np.mean(batch_train_error)
        epoch_train_top5_error = np.mean(batch_train_top5_error)
        train_loss.append(epoch_train_loss)
        train_error.append(epoch_train_error)
        train_top5_error.append(epoch_train_top5_error)

        model.eval()  # Validation phase
        with torch.no_grad():
            batch_val_loss, batch_val_error, batch_val_top5_error = [], [], []
            for i in range(0, len(test_dataset['model_labels']), batch_size):
                X_val, y_val, _ = get_batch(test_dataset, i, batch_size, with_replace=True)
                output_val = model(X_val)
                loss_val = criterion(output_val, y_val)

                _, predicted_val = torch.max(output_val, 1)
                error_val = (predicted_val != y_val).sum().item() / len(y_val)
                batch_val_loss.append(loss_val.item())
                batch_val_error.append(error_val)

                _, top5_pred_val = output_val.topk(5, 1, True, True)
                top5_error_val = 1 - (top5_pred_val == y_val.view(-1, 1).expand_as(top5_pred_val)).sum().item() / len(y_val)
                batch_val_top5_error.append(top5_error_val)

            epoch_val_loss = np.mean(batch_val_loss)
            epoch_val_error = np.mean(batch_val_error)
            epoch_val_top5_error = np.mean(batch_val_top5_error)
            val_loss.append(epoch_val_loss)
            val_error.append(epoch_val_error)
            val_top5_error.append(epoch_val_top5_error)

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Training loss: {epoch_train_loss:.4f}, Training error: {epoch_train_error:.4f}, Training Top5 error: {epoch_train_top5_error:.4f}')
        print(f'Validation loss: {epoch_val_loss:.4f}, Validation error: {epoch_val_error:.4f}, Validation Top5 error: {epoch_val_top5_error:.4f}')

        model.train()  # Set the model back to training mode

    return model, train_loss, train_error, train_top5_error, val_loss, val_error, val_top5_error

import matplotlib.pyplot as plt
import numpy as np
import torch
import random
def plot_loss_and_error(train_loss, train_error, val_loss, val_error, figsize=(12, 12)):
    """

    Function that plots:
    1. validation error over number of epochs
    2. validation loss over number of epochs
    3. training error over number of epochs
    4. training loss over number of epochs

    Return: fig, axes
    """
    ###TODO###
    fig, axes = plt.subplots(2,2, figsize=figsize)
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
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

if __name__ == '__main__':
    output_dir = '.'
    boc =  BrainObservatoryCache(
    manifest_file=str(Path(output_dir) / 'brain_observatory_manifest.json'))
    set_seed(1)
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
    pre = 15
    post = 7

    #TODO: Fourth item to experiment with, dff means full neurons vs. pca the dimension-reduced version
    #Play with different pca_comp values as well
    data_type = 'pca'
    pca_comp = 150

    ######## DO NOT CHANGE: GET ALL EXPERIMENTS #########
    exps = get_exps(boc, cre_lines=cre_lines_to_use, targeted_structures=['VISp'], session_types=['three_session_B'])
    #####################################################

    # Optional: Filter experiments if needed
    # exps = filter_exps(boc, exps, num_exps = 1, min_neuron = pca_comp, max_neuron = 1000, behavior = True)

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
    exp_type = 'multi'
    num_exps = 10

    ######## DO NOT CHANGE THIS ########
    exp_chosen = 22
    ####################################

    #TODO: Sixth item to experiment with, train vs. test split ratio
    train_prop = 0.7

    #TODO: Seventh item to experiment with, pad the sequence or not
    pad_ind = False # No need to change

    #TODO: Eighth item to experiment with, max_features
    max_features = 400 # No need to change

    if exp_type == 'single':
        exps = exps[exp_chosen:exp_chosen+1]
        dataset = prep_dataset(boc, exps, mapping_dict=mapping_dict, pre=pre, post=post, data_type=data_type, pca_comp=pca_comp, cca=cca_ind, behavior=behavior)
        train_dataset, test_dataset, train_orig_num_feat, test_orig_num_feat = get_train_test_split(dataset, train_prop = train_prop, pad = pad_ind, max_features=max_features)
    elif exp_type == 'multi':
        exps = exps[exp_chosen:exp_chosen+num_exps]
        dataset = prep_dataset(boc, exps, mapping_dict=mapping_dict, pre=pre, post=post, data_type=data_type, pca_comp=pca_comp, cca=cca_ind, behavior=behavior)
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
        #train_dataset = sample_data(train_dataset, 0.01)
        #test_dataset = sample_data(test_dataset, 0.01)

    if data_type == 'dff':
        input_dim = train_dataset['model_input'][0].shape[0]
    else:
        input_dim = pca_comp

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

hidden_dim, embed_dim = 256, 256
num_layers = 2
dropout_prob = 0.7
model_type = 'GRU'
label_smoothing = 0.2
initial_lr = 0.001
warmup_epochs = 3
weight_decay = 1e-5
batch_size = 256
epochs = 100
clip = 1

model = RNNClassifier(input_dim=input_dim, hidden_dim=hidden_dim, embed_dim=embed_dim, num_layers=num_layers, num_classes=num_classes, dropout_prob=dropout_prob, nn_type = model_type)

    # Transformer Model
#model = TransformerClassifier(input_dim=input_dim, hidden_dim=hidden_dim, nlayers=num_layers, nhead=8, num_classes=num_classes, dropout=dropout_prob)

model = model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
scheduler = WarmupWithScheduledDropLR(optimizer, warmup_epochs=warmup_epochs, initial_lr=initial_lr, drop_epochs=[30, 70])

model, train_loss, train_error, train_top5_error, val_loss, val_error, val_top5_error = trainModel(model, criterion, optimizer, scheduler, epochs, batch_size, train_dataset, test_dataset, dry_run=False)