import matplotlib.pyplot as plt
from pathlib import Path
import pprint
import numpy as np
import pandas as pd
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import torch

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

def pca_and_pad(data):
    
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
    pca = PCA(n_components=min(50, num_dimensions))
    pca_data = pca.fit_transform(data.T).T
    reconstructed_data = pca.inverse_transform(pca_data.T).T
    
    # Pad with zeros if necessary
    if pca_data.shape[0] < 50:
        padding = np.zeros((50 - pca_data.shape[0], num_samples))
        pca_data = np.vstack((pca_data, padding))
    
    
    return pca_data, reconstructed_data

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

def extract_data_by_images(data, stim_df, pre=15, post=7):
    """
    Extract segments of the data based on a pandas dataframe (natural scene stim df from get_stim_df()).
    30Hz, each image is presented for 250ms (8 frames), will also take 1s (30 frames) preceding and 0.25s (7 frames)post stim, total 1.5s data
    make sure that the data and image_df are from the same experiment!!!
    
    Parameters:
    - data: numpy array of shape (ndim, ntimesteps)
    - stim_df: pandas dataframe with columns 'frame', 'start', 'end'
    - pre: how many frames before image presentation should be extracted
    - post: how many frames after image presentation should be extracted
    Returns:
    - result: list of tuples with labels as first argument and segments of data as second arguent
    labels are int, data are numpy array
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
    
    return data_segments, labels

def prep_dataset(boc, exps, pre, post):
    """
    preparing dataset for training

    Parameters:
    - boc: BrainObservatoryCache object
    - exps: list of experiment objects, returned from get_exps()
    - pre: how many timesteps before image presentation should be extracted
    - post: how many timesteps after image presentation should be extracted
    Returns:
    - out: dict containing 3 keys: model_input, model_labels, metadata, each with the same length containing all datapoints (trials)
    - metadata contains ['targeted_structures', 'experiment_container_id', 'indicator', 'cre_line', 'session_type', 'specimen_name'], indexed same as input and labels
    - each model_input is 50 x pre+8+post (PCA components x timesteps) in float32 tensor
    - each model_label corresponds to the image presented to the mouse at the trial with same index as model_input, in LongTensor
    """
    model_input, model_labels, metadata = [], [], []
    meta_required = ['targeted_structures', 
                     'experiment_container_id', 
                     'indicator', 
                     'cre_line', 
                     'session_type', 
                     'specimen_name']

    for exp in exps:
        meta = boc.get_ophys_experiment_data(exp['id']).get_metadata()

        try:
            dff = get_fluo(boc, exp)
        except:
            print(f"dFF extraction from experiment id{meta['experiment_container_id']} failed!")
            continue
        pca_dff, _ = pca_and_pad(dff)
        try:
            stim_df = get_stim_df(boc, exp, stimulus_name='natural_scenes')
        except:
            print(f"stim table from experiment id{meta['experiment_container_id']} failed!")
            continue
        data, labels = extract_data_by_images(pca_dff, stim_df, pre, post)
        labels  = [118 if x == -1 else x for x in labels ]
        data = [torch.from_numpy(datum).float() for datum in data]
        labels = torch.LongTensor(labels)
        model_input.extend(data)
        model_labels.extend(labels)
        meta = {k:v for k, v in meta.items() if k in meta_required}
        meta_list = [meta.copy() for _ in range(len(labels))] # repeat metadata for each datum (natural scene trials) in the exp
        metadata.extend(meta_list)
    
    out = {'model_input':model_input,
           'model_labels':model_labels,
           'metadata':metadata}
    
    return out




        


