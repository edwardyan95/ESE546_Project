import matplotlib.pyplot as plt
from pathlib import Path
import pprint
import numpy as np
import pandas as pd
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from sklearn.decomposition import PCA

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
        dataset = boc.get_ophys_experiment_data(exp['id'])
    
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

def extract_data_by_images(data, image_df, pre=30, post=7):
    """
    Extract segments of the data based on a pandas dataframe (natural scene stim df from get_stim_df()).
    30Hz, each image is presented for 250ms (8 frames), will also take 1s (30 frames) preceding and 0.25s (7 frames)post stim, total 1.5s data
    make sure that the data and image_df are from the same experiment!!!
    
    Parameters:
    - data: numpy array of shape (ndim, ntimesteps)
    - image_df: pandas dataframe with columns 'frame', 'start', 'end'
    - pre: how many frames before image presentation should be extracted
    - post: how many frames after image presentation should be extracted
    Returns:
    - result: list of tuples with labels as first argument and segments of data as second arguent
    labels are int, data are numpy array
    """
    result = []

    for index, row in image_df.iterrows():
        label = row['frame']
        start_timestep = row['start']
        end_timestep = row['end']
        if start_timestep-pre < 0 or end_timestep+post > data.shape[1]:
            continue
        # Extract segment of data corresponding to the given start and end timesteps
        segment = data[:, start_timestep-pre:end_timestep + post+1]

        result.append((label, segment))
    
    return result





