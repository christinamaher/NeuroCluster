import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def subset_channels(data, ch_names):
    '''
    Extracts a subset of channels from a MNE Raw object.

    Parameters
    ----------
    data : mne.io.Raw
        MNE Raw object.

    ch_names : list of str
        List of channel names to extract.

    Returns
    -------
    np.ndarray
        4D array of shape (n_epochs, n_channels, n_times) containing the data
        from the selected channels.

    '''
    if ch_names == 'all':
        return data._data[:, :, :, :]
    else:
        ch_idx = [data.ch_names.index(ch) for ch in ch_names]
        return data._data[:, ch_idx, :, :]
    

def prepare_regressor_df(power_epochs):
    '''
    Prepare a DataFrame containing the behavioral variables.

    Parameters
    ----------
    power_epochs : mne.Epochs
        MNE Epochs object containing the power data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the behavioral variables.

    '''
    
    beh_df = []

    beh_variables = [col for col in power_epochs.metadata if col not in power_epochs.ch_names] # get behavioral variables
    beh_df = power_epochs.metadata[beh_variables] # extract behavioral variables from metadata

    # present user with list of beahvioral variables and have them decide if they want to keep them or not 
    for col in beh_df.columns:
        keep = input(f'Would you like to keep {col}? (yes or no): ')
        if keep == 'no':
            beh_df.drop(col, axis=1, inplace=True)

    # present user with list of behavioral variables and have them mark as categorical or continuous
    for col in beh_df.columns:
            data_type = input(f'Please specify data type for {col} (category or float64).')
            beh_df[col] = beh_df[col].astype(data_type)

    # present user with list of behavioral variables that are not marked as category and ask if they want to z-score them
    for col in beh_df.columns:
        if beh_df[col].dtype != 'category':
            z_score = input(f'Would you like to z-score {col}? (yes or no): ')
            if z_score == 'yes':
                beh_df[col] = (beh_df[col] - beh_df[col].mean()) / beh_df[col].std()

    return beh_df

def prepare_anat_dic(roi, file_path):
    '''
    Prepare a dictionary mapping each channel to its anatomical region.

    Args:
    roi : str
        Region of interest (e.g., 'ofc', 'hippocampus', etc.).

    file_path : str
        Path to the file containing the anatomical information. 

    Returns:
    dict
        Dictionary mapping each channel to its anatomical region.

    '''
    # check if file name is csv, if not raise error
    if file_path.split('.')[-1] != 'csv':
        raise ValueError('Anat file must be a csv file.')
    
    # read in anatomical info
    anat_df = pd.read_csv(file_path)

    # subset rows for specified ROI
    roi_info_df = anat_df[anat_df['roi'].isin(roi)]

    # get unique subj_ids for ROI
    roi_subj_ids = roi_info_df.subj_id.unique().tolist()

    # create dict with subj_id as key and elecs as values
    anat_dic = {f'{subj_id}':roi_info_df.reref_ch_names[roi_info_df.subj_id == subj_id].unique().tolist() for subj_id in roi_subj_ids}

    return anat_dic

def compare_mne_permutation_cluster_test(permute_var,tfr_data,sample_behav,split_method='median',n_permutations=1000):
    """
    Function to compare NeuroCluster's permutation cluster test to mne's permutation cluster test

    Args:
    permute_var (str): variable to permute
    tfr_data (np.array): 3D array of shape (num trials x num frequencies x num timepoints)
    sample_behav (pd.DataFrame): dataframe containing predictor variables
    split_method (str): method to split the permute_var variable (currently accepts either 'median' or 'mean')
    n_permutations (int): number of permutations to run

    Returns:
    Prints out whether significant clusters found by mne's permutation cluster test. 

    """
    if split_method == 'median':
        variable_of_interest = sample_behav[permute_var]
        median = np.median(variable_of_interest)
        low_split = tfr_data[variable_of_interest < median, :, :]    
        high_split = tfr_data[variable_of_interest >= median, :, :]
    elif split_method == 'mean':
        variable_of_interest = sample_behav[permute_var]
        mean = np.mean(variable_of_interest)
        low_split = tfr_data[variable_of_interest < mean, :, :]    
        high_split = tfr_data[variable_of_interest >= mean, :, :]
    _, clusters, cluster_pv, _ = mne.stats.permutation_cluster_test([low_split, high_split], n_permutations=n_permutations, tail=0)
    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_pv[i_c] <= 0.05:
            print(f"Cluster {i_c} p-value: {cluster_pv[i_c]}")
        else:
            print("No significant clusters found")

def create_directory(directory_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def save_plot_to_pdf(fig, directory, filename):
    """Save a plot to the specified directory with the given filename."""
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, dpi=300,bbox_inches='tight')
    plt.close(fig)  # Close the figure to avoid display and memory issues
  