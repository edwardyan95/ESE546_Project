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
import itertools
from allensdk.brain_observatory.static_gratings import StaticGratings
import random

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
        start_timestep = row['start']-8
        end_timestep = row['end']-8
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

    arrays_per_label = 50

    # Initialize the two lists
    list1 = []
    list2 = []

    for label in range(119):
        # Extract the arrays for this label
        start_idx = label * arrays_per_label
        end_idx = start_idx + arrays_per_label
        label_arrays = target_sort_data[start_idx:end_idx]

        # Shuffle the arrays for this label
        random.shuffle(label_arrays)

        # Split into two halves
        half = len(label_arrays) // 2
        list1 += label_arrays[:half]
        list2 += label_arrays[half:]
    half1_concat_data = np.concatenate(list1, axis=1)
    half2_concat_data = np.concatenate(list2, axis=1)

    cca = CCA(n_components=ref_concat_data.shape[0])
    cca.fit(ref_concat_data.T, target_concat_data.T)
    trans_data = target_concat_data.T.dot(cca.y_rotations_).dot(np.linalg.inv(cca.x_rotations_)).T # find transformed data matrix B that is most correlated to ref data matrix A
    pca_comp_corr = []
    split_half_corr = []
    for i in range(target_concat_data.shape[0]):
        pca_comp_corr.append(np.corrcoef(ref_concat_data[i,:], trans_data[i,:])[0,1])
        split_half_corr.append(np.corrcoef(half1_concat_data[i,:], half2_concat_data[i,:])[0,1])
    

    trans_data_list = [trans_data[:,i*num_tmstps:i*num_tmstps+num_tmstps] for i in range(num_trials)]
    reverse_sort_idx = np.argsort(target_sort_idx) # reverse label and data order to original input
    trans_data_list = [trans_data_list[i] for i in reverse_sort_idx]
    reverse_sort_labels = [target_sort_labels[i] for i in reverse_sort_idx]

    return trans_data_list, reverse_sort_labels, np.mean(pca_comp_corr), np.mean(split_half_corr)

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
        dff = stats.zscore(dff, axis=1)
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
            data, labels, adj_corr, split_corr = cca_align(ref_data, ref_labels, data, labels)
            print(f'exp#{exp_count} aligned corr: {adj_corr: .2f}')
            print(f'exp#{exp_count} split half corr: {split_corr: .2f}')
        
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


#####################EXPERIMENT###########################
def index_by_combinations(df):
    # Initialize the dictionary with all possible combinations
    max_ori_sg = 5
    max_sf_sg = 4
    max_phase_sg = 3
    combinations = itertools.product(range(0, max_ori_sg + 1), range(1, max_sf_sg + 2), range(0, max_phase_sg + 1))
    indexes_dict = {combination: [] for combination in combinations}
    # print(indexes_dict)
    # Iterate through each row
    for index, row in df.iterrows():
        # Check if osi_sg is within the specified range
        if 0.5 < row['osi_sg'] <= 1.5:
            # Create a tuple of the combination
            combination = (row['ori_sg'], row['sf_sg'], row['phase_sg'])

            # Append the index to the corresponding list
            indexes_dict.get(combination, []).append(index)

    return indexes_dict

def average_rows_by_index_dict(X, index_dict):
    # Initialize an output array with the same number of columns as X and rows equal to the length of index_dict
    output_array = np.zeros((len(index_dict), X.shape[1]))

    for i, (key, indexes) in enumerate(index_dict.items()):
        if indexes:  # Check if the list of indexes is not empty
            selected_rows = X[indexes]
            output_array[i] = np.mean(selected_rows, axis=0)
        # If the list of indexes is empty, the row remains zeros as initialized

    return output_array

def prep_dataset_by_static_grating(boc, exps, mapping_dict=None, pre=15, post=7, behavior=False):
    """
    preparing dataset for training. here, output is organized by response specificity to static gratings (orix6, spatial freqx5, phasex4)
    each row is average of neural traces of neurons that belong to a specific combination of (ori, sf, phase)

    Parameters:
    - boc: BrainObservatoryCache object
    - exps: list of experiment objects, returned from get_exps()
    - pre: how many timesteps before image presentation should be extracted
    - post: how many timesteps after image presentation should be extracted
    Returns:
    - out: dict containing 3 keys: model_input, model_labels, metadata, each with the same length containing all datapoints (trials)
    - metadata contains ['targeted_structures', 'experiment_container_id', 'indicator', 'cre_line', 'session_type', 'specimen_name'], indexed same as input and labels
    - each model_input is 120 x pre+8+post (120 combinations of sg response x timesteps) in float32 tensor
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
    
    exp_count = 0
    for exp in exps:
        meta = boc.get_ophys_experiment_data(exp['id']).get_metadata()

        try:
            dff = get_fluo(boc, exp)
            dff = stats.zscore(dff, axis=1)
        except:
            print(f"dFF extraction from experiment id{meta['experiment_container_id']} failed!")
            continue
        try:
            stim_df = get_stim_df(boc, exp, stimulus_name='natural_scenes')
        except:
            print(f"stim table from experiment id{meta['experiment_container_id']} failed!")
            continue
        try:
            sg = StaticGratings(boc.get_ophys_experiment_data(exp['id']))
            sg_peak = sg.peak
        except:
            print(f"static gratings from experiment id{meta['experiment_container_id']} failed!")
            continue
        indexes_dict = index_by_combinations(sg_peak)
        output = average_rows_by_index_dict(dff, indexes_dict)
        data, labels = extract_data_by_images(output, stim_df, mapping_dict=mapping_dict, pre=pre, post=post)
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




        