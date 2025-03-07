# -*- coding: utf-8 -*-
# %% load libraries

from matplotlib import cm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.gridspec import GridSpec

os.chdir('C:\dev\workspaces\Ephys-scripts\ephys_analysis')
from Functions.PSTH import *
from Functions.neural_response import *
from Functions.behavioural import *
from Functions.time_constant import *
from Functions.user_def_analysis import *
from Functions.load import *
from Functions.backup_functions import prefered_orientation
plt.rcParams['font.family'] = 'Arial'

# Set print options to suppress scientific notation
np.set_printoptions(suppress=True, precision=2)
    
# %% Load data

# root path (local) where the data to analyse is located
analysisDir = define_directory_analysis()

# Generate a csv with all the data that is going to be anlysed in THIS script
# Should have a copy local of all the preprocessed data
csvDir = os.path.join(analysisDir, 'Inventory', 'gratings.csv')


database = pd.read_csv(
    csvDir,
    dtype={
        "Name": str,
        "Date": str,
        "Protocol": str,
        "Experiment": str,
        "Sorting": str,
    }
)

label = 'good'

# Create database for analysis
# Inicialize object to colect data from each session
stim = []
neural = []

for i in range(len(database)):

    # Path
    dataEntry = database.loc[i]
    dataDirectory = os.path.join(analysisDir, dataEntry.Name, dataEntry.Date)

    # Check if directory for figures exists
    figureDirectory = os.path.join(
        analysisDir, 'Figures', dataEntry.Name, dataEntry.Date)

    if not os.path.isdir(figureDirectory):
        os.makedirs(figureDirectory)

    # Load stimulus type (for example oddballs)
    st = load_stimuli_data(dataDirectory, dataEntry)

    # Load from the csv file WHICH experiments are concatenated for the analysis in INDEX
    # For example, all the oddballs drifting

    # TODO: IMPLEMENTED TEST. capture here the cases when there is an only experiment (no indices given)

    chopped = chop_by_experiment(st, dataEntry)

    if isinstance(dataEntry.Experiment, str):
        index = [int(e) for e in dataEntry.Experiment.split(',')]
        st = merge_experiments_by_index(chopped, index)
    else:
        st = chopped[0]

    stim.append(st)

    # Load neural data#########################################################
    
    #Set interval to load just spikes in these limits. Handle merged experiments
    #in an inefficent way. 
    extend = 20
    
    interval = np.array([st['intervals'][0][0] - extend, st['intervals'][-1][0] + extend])
    
    n = load_neural_data_local(dataDirectory, interval)

    cluster_info = n['cluster_info']
    cluster_info.set_index('cluster_id', inplace=True)
    SC_depth_limits = n['SC_depth_limits']

    # Load units
    if dataEntry.Sorting == 'curated':

        label = label.lower()

        # Keep clusters IDs within SC limits, with the label selected before (default good)
        units = cluster_info[((cluster_info.group == label) &
                             (cluster_info.depth >= SC_depth_limits[1]) &
                             (cluster_info.depth <= SC_depth_limits[0]))].index.values
        
    n['cluster_analysis'] = units

    neural.append(n)

# %% Params

#Save intermediate results
save_results = True 

# PSTH params
baseline = 0.2
stimDur = 2
prePostStim = 1
window = np.array([-prePostStim, stimDur + prePostStim])
binSize = 0.04
sigma = 0.06
alpha_val = 0.4
groups = np.unique(stim[0]['direction'])
colors = cm.rainbow(np.linspace(0, 1, len(groups)))
xticks = np.arange(-1, 3.5, 0.5)

# define windows for determine if neuron is visually responsive
no_stim_window = (-1, 0)
stim_window = (0, 1)
#stim_window_sensitization = (1.5,2)

# Behaviour params
threshold = 0.9  # for running velocity

# adaptation, fr and supression  params
window_response = np.array([0, 2])
early_interval = np.array([0, 0.5])
late_interval = np.array([1.5, 2])
window_is_suppressed = np.array([0, 1])
repetitions = 5000 # for permutation test

# Heatmap params
vmin, vmax = -12, 12
norm_interval = np.array([0, 0.15])

# Histogram and scatter plots per depth
palette = {'sSC': 'mediumturquoise', 'dSC': 'crimson'}
depth_cut = 40  #split data between superficial vs deep SC (in %)

#Tuning curves
tuning_threshold = 0.7

#Exponetial fitting 
k_fold = 20
plot_test = False
plot_models = True

# %% Behaviour and visual responsiveness
for n, st in zip(neural, stim):
    
    st['running_state'] = calculate_running_state(st['startTime'].reshape(-1),
                                                  st['endTime'].reshape(-1),
                                                  st['wheelTimestamps'].reshape(-1), 
                                                  st['wheelVelocity'].reshape(-1))

    visual = []
    clusters = n['cluster_analysis']
    
    for neuron in clusters:

        spAligned, trials = alignData(n['spike_times'][n['spike_clusters'] == neuron],
                                      st['startTime'].reshape(-1), window)
    
        res = visual_responsiveness(spAligned,
                                    trials,
                                    no_stim_window,
                                    stim_window,
                                    window_response,
                                    st['direction'].reshape(-1),
                                    baseline)
        visual.append(res)

    n['visual'] = visual
    
    
# %% Fitting params

#Params 

trials_to_include = 'strong' # 'all' or 'strong'
tuning_threshold = 0.4 # threshold for strength of trials if trials_to_include = 'strong'
plot_test = False
plot_models = False
sigma = sigma
min_trials_per_test=5
max_splits=20
save_output_files = True

# TODO: define directories function

# %% batch k-fold

i = 0

# load neural and stimulus data
for n, st in zip(neural, stim):
    clusters = n['cluster_analysis'][n['visual']]
    spike_times = n['spike_times']
    spike_clusters = n['spike_clusters']
     
    stimuli_start = st['startTime'].reshape(-1)
    stimuli_end = st['endTime'].reshape(-1)
    direction = st['direction'].reshape(-1)
    running_state = st['running_state'].reshape(-1)
    dataEntry = database.loc[i]
    
    # if i==0: 
    #     clusters = [178, 225, 329, 363, 375]
    # clusters = [324]
    
    # Initialize output matrices
    paramsAdapt = np.full((len(clusters), 3, 4), np.nan)
    paramsSens = np.full((len(clusters), 3, 4), np.nan)
    paramsMixed = np.full((len(clusters), 3, 8), np.nan)
    paramsFlat = np.full((len(clusters), 3, 2), np.nan)
    
    mseAdapt = np.full((len(clusters), 3), np.nan)
    mseSens = np.full((len(clusters), 3), np.nan)
    mseMixed = np.full((len(clusters), 3), np.nan)
    mseFlat = np.full((len(clusters), 3), np.nan)
    
    adaptIndex = np.full((len(clusters), 3, 2), np.nan)
    bestModel = np.full((len(clusters), 4), np.nan, dtype=object)  
    
    
    #iterate over neurons
    for neuron_idx, neuron_id in  enumerate(clusters):
        identifier=f'{dataEntry.Name}_{dataEntry.Date}_{neuron_id}'
        print(f'Processing neuron {identifier}')
        
        # total trials per state
        spAligned, trials = alignData( 
            spike_times[spike_clusters == neuron_id],
            stimuli_start, 
            window)
        binned, timeBins = newTracesFromPSTH_per_trial(spAligned, trials, window, direction, 
                                                    groups, baseline, binSize, sigma)
        is_suppressed = analyze_suppression(np.mean(binned, axis=0), timeBins)
        
        # check low activity per state (>70% trials with no spikes)
        is_low_activity_quiet = exclude_low_activity(trials, running_state==0)
        is_low_activity_active = exclude_low_activity(trials, running_state==1)
        if is_low_activity_quiet or is_low_activity_active:
            continue
        
        if trials_to_include == 'strong':
            # fr (single value, not the psth array)
            fr = calculate_fr_per_trial( 
                spAligned, trials, window_response, direction, baseline)
            trial_indices, _, _, _ = strong_response_trials(direction,fr,tuning_threshold,is_suppressed)
        elif trials_to_include == 'all':
            trial_indices = np.where(np.isin(direction, np.unique(direction)))[0] # from 0 to (len(direction)-1)
            
        # check if there are enough trials in each state
        if sum(running_state[trial_indices] == 1) < 10 or sum(running_state[trial_indices] == 0) < 10:
            continue
        
        # finding the average firing rate across the whole window and with baseline (-1 to 3s) to exclude neurons with low FR
        binned_w_bl, _ = newTracesFromPSTH_per_trial(spAligned, trials, window, direction[trial_indices], 
                                                    groups, baseline=0, binSize=binSize, sigma=sigma)
        if np.mean(binned_w_bl) < 1: 
            continue
        
        state_trial_indices, state_test_sets = generate_state_trial_splits(trial_indices, running_state, min_trials_per_test=min_trials_per_test, max_splits=max_splits)
        
        #TODO: write more clearly:
        # collect test errors from k-fold cross validation to find the lowest error for either pooled or separated states
        test_errors_per_state = {
            'quiet': [],
            'active': [],
            'pooled': []
        }
        state_to_index = {'quiet': 0, 'active': 1, 'pooled': 2}
        
        for state, state_trial_ind, state_test_set in zip(['quiet', 'active', 'pooled'], state_trial_indices, state_test_sets):
            
            result = perform_fitting(binned[state_trial_ind], timeBins, state, state_test_set, state_trial_ind,
                                    identifier, sigma, 
                                    is_suppressed=is_suppressed, plot_test=plot_test, plot_models=plot_models)
            # extract test errors for each fold for best model per state:
            test_errors_per_state[state].extend(result['models_details'][result['best_model']]['test_trial_errors'])

            state_idx = state_to_index[state] 
            
            # Save the results to matrices based on neuron_idx and state_idx (numeric index)
            paramsAdapt[neuron_idx, state_idx, :] = result['models_details']['Adaptation']['params']
            paramsSens[neuron_idx, state_idx, :] = result['models_details']['Sensitisation']['params']
            paramsMixed[neuron_idx, state_idx, :] = result['models_details']['Mixed']['params']
            paramsFlat[neuron_idx, state_idx, :] = result['models_details']['Flat']['params']
            
            mseAdapt[neuron_idx, state_idx] = result['models_details']['Adaptation']['rmse_k_fold']
            mseSens[neuron_idx, state_idx] = result['models_details']['Sensitisation']['rmse_k_fold']
            mseMixed[neuron_idx, state_idx] = result['models_details']['Mixed']['rmse_k_fold']
            mseFlat[neuron_idx, state_idx] = result['models_details']['Flat']['rmse_k_fold']
            
            adaptIndex[neuron_idx, state_idx, :] = result['models_details'][result['best_model']]['ai']
            bestModel[neuron_idx, state_idx] = result['best_model'] 

        state_separation = decide_state_separation(test_errors_per_state)
        bestModel[neuron_idx, 3] = state_separation
        
        if save_output_files:
            output_dir = os.path.join(analysisDir, 'Fitting_output', dataEntry.Name, dataEntry.Date, '19-02-25_strong_04')
            os.makedirs(output_dir, exist_ok=True)
            
            np.save(os.path.join(output_dir, 'gratingsPsthExpFit.clusters.npy'), clusters)    
            np.save(os.path.join(output_dir, 'gratingsPsthExpFit.paramsAdapt.npy'), paramsAdapt)
            np.save(os.path.join(output_dir, 'gratingsPsthExpFit.paramsSens.npy'), paramsSens)
            np.save(os.path.join(output_dir, 'gratingsPsthExpFit.paramsMixed.npy'), paramsMixed)
            np.save(os.path.join(output_dir,'gratingsPsthExpFit.paramsFlat.npy'), paramsFlat)
            
            np.save(os.path.join(output_dir,'gratingsPsthExpFit.mseAdapt.npy'), mseAdapt)
            np.save(os.path.join(output_dir,'gratingsPsthExpFit.mseSens.npy'), mseSens)
            np.save(os.path.join(output_dir,'gratingsPsthExpFit.mseMixed.npy'), mseMixed)
            np.save(os.path.join(output_dir,'gratingsPsthExpFit.mseFlat.npy'), mseFlat)
        
            np.save(os.path.join(output_dir,'gratingsPsthExpFit.adaptIndex.npy'), adaptIndex)
            bestModel_df = pd.DataFrame(bestModel, columns=['quiet', 'active', 'pooled', 'separate/pool'])
            bestModel_df.to_csv(os.path.join(output_dir, 'bestModel.csv'), index=False)

    i+=1

# %% open npy files and create dataframe

# output_dir = os.path.join(analysisDir, 'Fitting_output', 'FG004', '2023-06-11', '29-01-25_all')

clusters_file = np.load(os.path.join(output_dir, 'gratingsPsthExpFit.clusters.npy'))
paramsAdapt_file = np.load(os.path.join(output_dir, 'gratingsPsthExpFit.paramsAdapt.npy'))
paramsSens_file = np.load(os.path.join(output_dir, 'gratingsPsthExpFit.paramsSens.npy'))
paramsMixed_file = np.load(os.path.join(output_dir, 'gratingsPsthExpFit.paramsMixed.npy'))
paramsFlat_file = np.load(os.path.join(output_dir, 'gratingsPsthExpFit.paramsFlat.npy'))

mseAdapt_file = np.load(os.path.join(output_dir, 'gratingsPsthExpFit.mseAdapt.npy'))
mseSens_file = np.load(os.path.join(output_dir, 'gratingsPsthExpFit.mseSens.npy'))
mseMixed_file = np.load(os.path.join(output_dir, 'gratingsPsthExpFit.mseMixed.npy'))
mseFlat_file = np.load(os.path.join(output_dir, 'gratingsPsthExpFit.mseFlat.npy'))

# Load the adaptation index
adaptIndex_file = np.load(os.path.join(output_dir, 'gratingsPsthExpFit.adaptIndex.npy'))
# Load the CSV file
bestModel_file = pd.read_csv(os.path.join(output_dir, 'bestModel.csv'))
# df_exp_fit = load_exp_fit_df(output_dir)





# %% save depth for database

i = 0
for n, st in zip(neural, stim):

    clusters = n['cluster_analysis'][n['visual']]
    dataEntry = database.loc[i]
    clusters_depth = []
    for neuron in clusters:
        depth = abs(n['cluster_info']['depth'].loc[neuron] - n['SC_depth_limits'][0]
                    )/(n['SC_depth_limits'][0] - n['SC_depth_limits'][1])*100
        clusters_depth.append({
            'Neuron': neuron,
            'Depth': depth
            })
    clusters_depth_df = pd.DataFrame(clusters_depth)
    output_dir = os.path.join(analysisDir, 'Fitting_output', dataEntry.Name, dataEntry.Date, '19-02-25_strong_04')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'clusters_depth.csv')
    clusters_depth_df.to_csv(output_file, index=False)

        
    i += 1
            
# clusters_depth_FG005 = pd.read_csv(output_file)    

# %% open several datasets

database = pd.read_csv(
    csvDir,
    dtype={
        "Name": str,
        "Date": str,
        "Protocol": str,
        "Experiment": str,
        "Sorting": str,
    }
)

combined_data = []
i = 0
# Iterate through the rows of the database
for _, dataEntry in database.iterrows():
    # Define the output directory
    output_dir = os.path.join(analysisDir, 'Fitting_output', dataEntry.Name, dataEntry.Date, '19-02-25_strong_04')
    clusters_depth = pd.read_csv(os.path.join(output_dir, 'clusters_depth.csv'))    
    df_exp_fit = load_exp_fit_df(output_dir)
    
    # Add `Name` and `Date` as new columns
    df_exp_fit['Name'] = dataEntry['Name']
    df_exp_fit['Date'] = dataEntry['Date']
    df_exp_fit['Depth'] = clusters_depth['Depth']
    df_exp_fit['Dataset'] = i
    
    
    # Append the DataFrame to the list
    combined_data.append(df_exp_fit)
    i += 1

combined_df = pd.concat(combined_data, ignore_index=True)

# %% specify directories

# open database with datasets of interest
# analysisDir = define_directory_analysis()
# csvDir = os.path.join(analysisDir, 'Inventory', 'gratings.csv')

database = pd.read_csv(
    csvDir,
    dtype={
        "Name": str,
        "Date": str,
        "Protocol": str,
        "Experiment": str,
        "Sorting": str,
    }
)

# %% plot best model

i = 0

for n, st in zip(neural, stim):
    dataEntry = database.loc[i]
    output_dir = os.path.join(analysisDir, 'Fitting_output', dataEntry.Name, dataEntry.Date, '29-01-25_all')
    df_exp_fit = load_exp_fit_df(output_dir)
    # clusters = df_exp_fit['clusters']
    clusters = [621,661,662,678]
    spike_times = n['spike_times']
    spike_clusters = n['spike_clusters']

    stimuli_start = st['startTime'].reshape(-1)
    stimuli_end = st['endTime'].reshape(-1)
    direction = st['direction'].reshape(-1)
    running_state = st['running_state'].reshape(-1)
    
    for neuron in clusters:
        neuron_data = df_exp_fit[df_exp_fit['clusters'] == neuron]
        
        if neuron_data['best_model_q'].isna().all(): 
            # best model is nan if the fitting is not successful; 
            # which means that the neuron is not going to be plotted. 
            continue
        
        # Overall neuron data
        
        spAligned, trials = alignData( # total trials per state
            spike_times[spike_clusters == neuron],
            stimuli_start, 
            window)
        binned, timeBins = newTracesFromPSTH_per_trial(spAligned, trials, window, direction, 
                                                    groups, baseline, binSize, sigma)
        is_suppressed = analyze_suppression(np.mean(binned, axis=0), timeBins)
        

        if trials_to_include == 'strong':
            fr = calculate_fr_per_trial( # single value, not the psth
                spAligned, trials, window_response, direction, baseline)
            trial_indices, _, _, _ = strong_response_trials(direction,fr,tuning_threshold,is_suppressed)
        elif trials_to_include == 'all':
            trial_indices = np.where(np.isin(direction, np.unique(direction)))[0] # 0-239
            
        state_trial_indices, _ = generate_state_trial_splits(trial_indices, running_state, min_trials_per_test=min_trials_per_test, max_splits=max_splits)
        
        
        # Active state data
        
        mean_psth_a = np.mean(binned[state_trial_indices[1]], axis=0)
        if is_suppressed:
            mean_psth_a = -mean_psth_a
        
        params_a = neuron_data['params_a'].iloc[0]
        best_model_a = neuron_data['best_model_a'].iloc[0]
        if best_model_a == 'Mixed':
            AI = [f'{value:.2f}' for value in neuron_data['AI_a'].iloc[0]]
            AI_str = ", ".join(AI)  
        else:
            AI = f'{neuron_data["AI_a"].iloc[0][0]:.2f}'
            AI_str = AI
            
        start_time_for_fit_a = params_a[-1]
        
        if not np.isnan(start_time_for_fit_a):
            window_mask = (timeBins >= 0) & (timeBins <= 2 - 2 * sigma) # to avoid edge effects from convolution
            time_window = timeBins[window_mask]
            fit_mask = time_window >= start_time_for_fit_a
            time_to_fit = time_window[fit_mask] - start_time_for_fit_a # adjusted to start from 0
            
            if best_model_a == 'Adaptation': 
                fitted_firing_rate = exponential_model_adapt(time_to_fit, *params_a[:-2], params_a[-2])
            elif best_model_a == 'Sensitisation':
                fitted_firing_rate = exponential_model_sens(time_to_fit, *params_a[:-1])
            elif best_model_a == 'Mixed': 
                if start_time_for_fit_a == 0:
                    first_peak_index = np.searchsorted(timeBins, start_time_for_fit_a, side="right")
                else:
                    first_peak_index = np.searchsorted(timeBins, start_time_for_fit_a, side="right") - 1
                    
                end_time_for_fit = 2 - 2 * sigma
                end_index = np.searchsorted(timeBins, end_time_for_fit, side="right") - 1
                    
                fitted_firing_rate = mixed_response_model(timeBins, params_a[:-2], params_a[-2], first_peak_index, end_index)
            elif best_model_a == 'Flat':
                fitted_firing_rate = np.full_like(time_to_fit, fill_value=params_a[0])
        else:
            fitted_firing_rate = None
        
        fig = plt.figure(figsize=(9, 9))
        
        depth = abs(n['cluster_info']['depth'].loc[neuron] - n['SC_depth_limits'][0]
                    )/(n['SC_depth_limits'][0] - n['SC_depth_limits'][1])*100

        fig.suptitle(
            f'''Neuron: {neuron} Depth from sSC:{depth:.1f}''')
        
        # Active state plot
        
        ax1 = plt.subplot(2, 2, 1)
        plt.title('ACTIVE', loc='left', fontsize=10)
        active_trials = state_trial_indices[1]
        active_trials_mask = np.isin(trials, active_trials)
        _, trials_active_spAligned = np.unique(trials[active_trials_mask], return_inverse=True)
        newPlotPSTH(spAligned[active_trials_mask], trials_active_spAligned, window,
                    direction[active_trials], groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax2 = plt.subplot(2, 2, 3, sharex=ax1)

        plt.plot(timeBins, mean_psth_a, c='k')
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)
        plt.title(
            f'{best_model_a} AI {AI_str}', loc='right', fontsize=10)
        
        
        if fitted_firing_rate is not None:
            ax2.plot(time_to_fit+start_time_for_fit_a, fitted_firing_rate, color="orange", linewidth=2, label="Fit")
        
        if best_model_a == "Mixed":
            param_text = (f"y_start_fit={params_a[-2]:.2f}, C1={params_a[1]:.2f}, tau1={params_a[0]:.2f}\n"
                          f"A={params_a[4]:.2f}, C2={params_a[3]:.2f}, tau2={params_a[2]:.2f}\n"
                          f"t_switch={params_a[5]:.2f}")
        elif best_model_a == "Flat":
            param_text = f"Mean FR: {params_a[0]:.2f}"
        elif best_model_a == "Sensitisation":
            param_text = f"A={params_a[2]:.2f}, C={params_a[1]:.2f}, tau={params_a[0]:.2f}"
        elif best_model_a == "Adaptation":
            param_text = f"y_start_fit={params_a[-2]:.2f}, C={params_a[1]:.2f}, tau={params_a[0]:.2f}"
        else:
            param_text = "N/A"  # Default case if model_type doesn't match any known models
            
        ax2.text(
            0.05,
            0.02,
            param_text,
            transform=ax2.transAxes,
            verticalalignment="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2),
        ) 

        # Quiet state data
        
        mean_psth_q = np.mean(binned[state_trial_indices[0]], axis=0)
        if is_suppressed:
            mean_psth_q = -mean_psth_q
        
        params_q = neuron_data['params_q'].iloc[0]
        best_model_q = neuron_data['best_model_q'].iloc[0]
        if best_model_q == 'Mixed':
            AI = [f'{value:.2f}' for value in neuron_data['AI_q'].iloc[0]]
            AI_str = ", ".join(AI)  # Convert the list to a comma-separated string
        else:
            AI = f'{neuron_data["AI_q"].iloc[0][0]:.2f}'
            AI_str = AI
            
        start_time_for_fit_q = params_q[-1]
        
        
        if not np.isnan(start_time_for_fit_q):
            window_mask = (timeBins >= 0) & (timeBins <= 2 - 2 * sigma) # to avoid edge effects from convolution
            time_window = timeBins[window_mask]
            fit_mask = time_window >= start_time_for_fit_q
            time_to_fit = time_window[fit_mask] - start_time_for_fit_q # adjusted to start from 0
            
            if best_model_q == 'Adaptation': 
                fitted_firing_rate = exponential_model_adapt(time_to_fit, *params_q[:-2], params_q[-2])
            elif best_model_q == 'Sensitisation':
                fitted_firing_rate = exponential_model_sens(time_to_fit, *params_q[:-1])
            elif best_model_q == 'Mixed': 
                if start_time_for_fit_q == 0:
                    first_peak_index = np.searchsorted(timeBins, start_time_for_fit_q, side="right")
                else:
                    first_peak_index = np.searchsorted(timeBins, start_time_for_fit_q, side="right") - 1
                    
                end_time_for_fit = 2 - 2 * sigma
                end_index = np.searchsorted(timeBins, end_time_for_fit, side="right") - 1
                    
                fitted_firing_rate = mixed_response_model(timeBins, params_q[:-2], params_q[-2], first_peak_index, end_index)
            elif best_model_q == 'Flat':
                fitted_firing_rate = np.full_like(time_to_fit, fill_value=params_q[0])
        else:
            fitted_firing_rate = None


        # Quiet state plot 
        
        ax3 = plt.subplot(2, 2, 2, sharex=ax1)
        plt.title('QUIET', loc='left', fontsize=10)
        
        quiet_trials = state_trial_indices[0]
        quiet_trials_mask = np.isin(trials, quiet_trials)
        _, trials_quiet_spAligned = np.unique(trials[quiet_trials_mask], return_inverse=True)
        newPlotPSTH(spAligned[quiet_trials_mask], trials_quiet_spAligned, window,
                    direction[quiet_trials], groups, colors)
        
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax4 = plt.subplot(2, 2, 4, sharex=ax3, sharey=ax2)
        

        plt.plot(timeBins, mean_psth_q, c='k')
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)
        

        if fitted_firing_rate is not None:
            ax4.plot(time_to_fit+start_time_for_fit_q, fitted_firing_rate, color="orange", linewidth=2, label="Fit")
        
        if best_model_q == "Mixed":
            param_text = (f"y_start_fit={params_q[-2]:.2f}, C1={params_q[1]:.2f}, tau1={params_q[0]:.2f}\n"
                          f"A={params_q[4]:.2f}, C2={params_q[3]:.2f}, tau2={params_q[2]:.2f}\n"
                          f"t_switch={params_q[5]:.2f}")
        elif best_model_q == "Flat":
            param_text = f"Mean FR: {params_q[0]:.2f}"
        elif best_model_q == "Sensitisation":
            param_text = f"A={params_q[2]:.2f}, C={params_q[1]:.2f}, tau={params_q[0]:.2f}"
        elif best_model_q == "Adaptation":
            param_text = f"y_start_fit={params_q[-2]:.2f}, C={params_q[1]:.2f}, tau={params_q[0]:.2f}"
        else:
            param_text = "N/A"  # Default case if model_type doesn't match any known models
            
        ax4.text(
            0.05,
            0.02,
            param_text,
            transform=ax4.transAxes,
            verticalalignment="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2),
        ) 
        
        plt.title(
            f'{best_model_q} AI {AI_str}', loc='right', fontsize=10)

        plt.tight_layout()
        plt.show()
        
        state_separation = neuron_data['state_separation'].iloc[0]
        saving_dir = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, '29-01-25_all')
        if state_separation == 'separate':
            saving_directory = os.path.join(saving_dir, 'separate_state')
        elif state_separation == 'pooled': 
            saving_directory = os.path.join(saving_dir, 'pooled_state')
        else: 
            saving_directory = os.path.join(saving_dir, 'neither')
        
        # Create the directory if it doesn't exist
        os.makedirs(saving_directory, exist_ok=True)
        
        # Save the plot in the determined directory with a meaningful filename
        save_path = os.path.join(saving_directory, f'{neuron}_PSTH.png')
        # plt.savefig(save_path)
        # plt.close()

    i += 1
# %% polar plots 

i = 0

for n, st in zip(neural, stim):

    clusters = n['cluster_analysis'][n['visual']]
    spike_times = n['spike_times']
    spike_clusters = n['spike_clusters']

    stimuli_start = st['startTime'].reshape(-1)
    stimuli_end = st['endTime'].reshape(-1)
    direction = st['direction'].reshape(-1)
    running_state = st['running_state'].reshape(-1)

    dataEntry = database.loc[i]

    data = []
    print(f'Processing dataset {i}')

    for neuron in clusters:
        identifier=f'{dataEntry.Name}_{dataEntry.Date}_{neuron}'
        
        spAligned, trials = alignData( # total trials per state
            spike_times[spike_clusters == neuron],
            stimuli_start, 
            window)
        binned, timeBins = newTracesFromPSTH_per_trial(spAligned, trials, window, direction, 
                                                    groups, baseline, binSize, sigma)
        is_suppressed = analyze_suppression(np.mean(binned, axis=0), timeBins)
        fr = calculate_fr_per_trial( # single value, not the psth
            spAligned, trials, window_response, direction, baseline)
        trial_indices, pref_dirs, threshold_fr, max_response = strong_response_trials(direction,fr,tuning_threshold,is_suppressed)
        
            # Polar Plot##############################################################
        meanT, semT, stimT = tuning_curve(spAligned, trials, window_response,
                                              direction, groups, baseline, is_suppressed)
        dirs = stimT
        fig = plt.figure(figsize=(25, 9))

        ax3 = plt.subplot(2, 5, 3, projection='polar')
        
        stimPlot = stimT * np.pi / 180
        meanT = np.hstack((meanT, meanT[0]))
        semT = np.hstack((semT, semT[0]))
        stimPlot = np.hstack((stimPlot, stimPlot[0]))
        ax3.plot(stimPlot, np.full_like(stimPlot, threshold_fr), linestyle='--', color='r', label=f'Threshold: {threshold_fr}')
        plt.polar(stimPlot, meanT)
        plt.fill_between(stimPlot, meanT-semT, meanT+semT, alpha=alpha_val)
        plt.xticks(stimPlot)
        text_str = f'tuning threshold: {tuning_threshold}, pref dirs: {pref_dirs}, threshold (FR): {threshold_fr:.2f}, max: {max_response:.2f}'
        plt.title(f'{text_str}')


        #  Active all
        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                      stimuli_start[running_state == 1],
                                      window)
        ax1 = plt.subplot(2, 5, 1)
        plt.title('ACTIVE', loc='left', fontsize=10)
        newPlotPSTH(spAligned, trials, window,
                    direction[running_state == 1], groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax6 = plt.subplot(2, 5, 6)
        traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[running_state == 1],
                                              groups, baseline,  binSize, sigma)

        for t, s, c, l in zip(traces, sem, colors, groups):

            if np.isin(l, dirs):

                plt.plot(timeBins, t, alpha=alpha_val, c=c, label=str(l))
                plt.fill_between(timeBins, t - s, t + s, alpha=0.1, color=c)

        plt.plot(timeBins, np.nanmean(traces, axis=0), c='k')
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)


        # Quiet all

        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                      stimuli_start[running_state == 0],
                                      window)

        ax2 = plt.subplot(2, 5, 2, sharex=ax1)

        plt.title('QUIET', loc='left', fontsize=10)
        newPlotPSTH(spAligned, trials, window,
                    direction[running_state == 0], groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax7 = plt.subplot(2, 5, 7, sharey=ax6)
        traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[running_state == 0],
                                              groups, baseline, binSize, sigma)

        for t, s, c, l in zip(traces, sem, colors, groups):

            if np.isin(l, dirs):
                plt.plot(timeBins, t, alpha=alpha_val, c=c, label=str(l))
                plt.fill_between(timeBins, t - s, t + s, alpha=0.1, color=c)

        plt.plot(timeBins, np.nanmean(traces, axis=0), c='k')
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)
        
        #  Active strongest trials
        
        running_state_strong_trials = running_state[trial_indices]
        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                      stimuli_start[trial_indices][running_state_strong_trials == 1],
                                      window)
        ax4 = plt.subplot(2, 5, 4)
        plt.title('ACTIVE', loc='left', fontsize=10)
        newPlotPSTH(spAligned, trials, window,
                    direction[trial_indices][running_state_strong_trials == 1], groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax9= plt.subplot(2, 5, 9)
        traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[trial_indices][running_state_strong_trials == 1],
                                              groups, baseline,  binSize, sigma)

        for t, s, c, l in zip(traces, sem, colors, groups):

            if np.isin(l, dirs):

                plt.plot(timeBins, t, alpha=alpha_val, c=c, label=str(l))
                plt.fill_between(timeBins, t - s, t + s, alpha=0.1, color=c)

        plt.plot(timeBins, np.nanmean(traces, axis=0), c='k')
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)
        
        
        # Quiet strongest trials
        
        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                      stimuli_start[trial_indices][running_state_strong_trials == 0],
                                      window)

        ax5 = plt.subplot(2, 5, 5, sharex=ax4)

        plt.title('QUIET', loc='left', fontsize=10)
        newPlotPSTH(spAligned, trials, window,
                    direction[trial_indices][running_state_strong_trials == 0], groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax10 = plt.subplot(2, 5, 10, sharey=ax9)
        traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[trial_indices][running_state_strong_trials == 0],
                                              groups, baseline, binSize, sigma)

        for t, s, c, l in zip(traces, sem, colors, groups):

            if np.isin(l, dirs):
                plt.plot(timeBins, t, alpha=alpha_val, c=c, label=str(l))
                plt.fill_between(timeBins, t - s, t + s, alpha=0.1, color=c)

        plt.plot(timeBins, np.nanmean(traces, axis=0), c='k')
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)
        
        plt.legend(fontsize='x-small')
        fig.suptitle(
            f'''{identifier}''')
        plt.tight_layout()
        plt.close()
        
        saving_directory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, '05-02-25_polar_tun_th_04')
        os.makedirs(saving_directory, exist_ok=True)
        save_path = os.path.join(saving_directory, f'{neuron}_polar_plot.png')
        fig.savefig(save_path)

    i += 1
# %% polar plots 
            
            # Polar Plot##############################################################
        meanT, semT, stimT = tuning_curve(spAligned, trials, window_response,
                                              direction, groups, baseline, is_suppressed)
        dirs = stimT
        fig = plt.figure(figsize=(25, 9))

        ax3 = plt.subplot(2, 5, 3, projection='polar')
        
        stimPlot = stimT * np.pi / 180
        meanT = np.hstack((meanT, meanT[0]))
        semT = np.hstack((semT, semT[0]))
        stimPlot = np.hstack((stimPlot, stimPlot[0]))
        ax3.plot(stimPlot, np.full_like(stimPlot, threshold_fr), linestyle='--', color='r', label=f'Threshold: {threshold_fr}')
        plt.polar(stimPlot, meanT)
        plt.fill_between(stimPlot, meanT-semT, meanT+semT, alpha=alpha_val)
        plt.xticks(stimPlot)
        text_str = f'dirs: {pref_dirs}, threshold_fr: {threshold_fr:.2f}, max: {max_response:.2f}'
        
        
        # text_str = f'DSI: {dir_stats.statistic:.2f} (p={dir_stats.pvalue:.2f}), dirs: {pref_dirs} \nOSI: {ori_stats.statistic:.2f} (p={ori_stats.pvalue:.2f}), oris: {pref_oris}'

        plt.title(f'{text_str}')


        #  Active all
        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                      stimuli_start[running_state == 1],
                                      window)
        ax1 = plt.subplot(2, 5, 1)
        plt.title('ACTIVE', loc='left', fontsize=10)
        newPlotPSTH(spAligned, trials, window,
                    direction[running_state == 1], groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax6 = plt.subplot(2, 5, 6)
        traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[running_state == 1],
                                              groups, baseline,  binSize, sigma)

        for t, s, c, l in zip(traces, sem, colors, groups):

            if np.isin(l, dirs):

                plt.plot(timeBins, t, alpha=alpha_val, c=c, label=str(l))
                plt.fill_between(timeBins, t - s, t + s, alpha=0.1, color=c)

        plt.plot(timeBins, np.nanmean(traces, axis=0), c='k')
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)


        # Quiet all

        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                      stimuli_start[running_state == 0],
                                      window)

        ax2 = plt.subplot(2, 5, 2, sharex=ax1)

        plt.title('QUIET', loc='left', fontsize=10)
        newPlotPSTH(spAligned, trials, window,
                    direction[running_state == 0], groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax7 = plt.subplot(2, 5, 7, sharey=ax6)
        traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[running_state == 0],
                                              groups, baseline, binSize, sigma)

        for t, s, c, l in zip(traces, sem, colors, groups):

            if np.isin(l, dirs):
                plt.plot(timeBins, t, alpha=alpha_val, c=c, label=str(l))
                plt.fill_between(timeBins, t - s, t + s, alpha=0.1, color=c)

        plt.plot(timeBins, np.nanmean(traces, axis=0), c='k')
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)
        
        #  Active strongest trials
        
        running_state_strong_trials = running_state[trial_indices]
        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                      stimuli_start[trial_indices][running_state_strong_trials == 1],
                                      window)
        ax4 = plt.subplot(2, 5, 4)
        plt.title('ACTIVE', loc='left', fontsize=10)
        newPlotPSTH(spAligned, trials, window,
                    direction[trial_indices][running_state_strong_trials == 1], groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax9= plt.subplot(2, 5, 9)
        traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[trial_indices][running_state_strong_trials == 1],
                                              groups, baseline,  binSize, sigma)

        for t, s, c, l in zip(traces, sem, colors, groups):

            if np.isin(l, dirs):

                plt.plot(timeBins, t, alpha=alpha_val, c=c, label=str(l))
                plt.fill_between(timeBins, t - s, t + s, alpha=0.1, color=c)

        plt.plot(timeBins, np.nanmean(traces, axis=0), c='k')
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)
        
        
        # Quiet strongest trials
        
        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                      stimuli_start[trial_indices][running_state_strong_trials == 0],
                                      window)

        ax5 = plt.subplot(2, 5, 5, sharex=ax4)

        plt.title('QUIET', loc='left', fontsize=10)
        newPlotPSTH(spAligned, trials, window,
                    direction[trial_indices][running_state_strong_trials == 0], groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax10 = plt.subplot(2, 5, 10, sharey=ax9)
        traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[trial_indices][running_state_strong_trials == 0],
                                              groups, baseline, binSize, sigma)

        for t, s, c, l in zip(traces, sem, colors, groups):

            if np.isin(l, dirs):
                plt.plot(timeBins, t, alpha=alpha_val, c=c, label=str(l))
                plt.fill_between(timeBins, t - s, t + s, alpha=0.1, color=c)

        plt.plot(timeBins, np.nanmean(traces, axis=0), c='k')
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)
        
        
        
#         spAligned[(trials == trial) &
# (spAligned > window_response[0]) &
# (spAligned < window_response[1])]
        
        
        
        
        
        plt.legend(fontsize='x-small')
        fig.suptitle(
            f'''Neuron: {identifier}''')
        plt.tight_layout()





        saving_directory = os.path.join(analysisDir, 'Figures', '23_10', 'strong_response_trials_norm_thr_04')
        save_path = os.path.join(saving_directory, f'{identifier}_polar_plot.png')
        fig.savefig(save_path)
        # plt.show()
        plt.close()


    i+=1


# %% Plot individual raster and PSTH (F)

i = 0

for n, st in zip(neural, stim):

    clusters = n['cluster_analysis'][n['visual']]
    spike_times = n['spike_times']
    spike_clusters = n['spike_clusters']

    stimuli_start = st['startTime'].reshape(-1)
    stimuli_end = st['endTime'].reshape(-1)
    direction = st['direction'].reshape(-1)
    running_state = st['running_state'].reshape(-1)

    dataEntry = database.loc[i]

    for neuron in clusters:

        fig = plt.figure(figsize=(9, 9))

        # Active
        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                      stimuli_start[running_state == 1],
                                      window)

        ax1 = plt.subplot(2, 2, 1)
        plt.title('ACTIVE', loc='left', fontsize=10)
        newPlotPSTH(spAligned, trials, window,
                    direction[running_state == 1], groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax2 = plt.subplot(2, 2, 3, sharex=ax1)
        traces, sem, bins = newTracesFromPSTH(spAligned, trials, window, direction[running_state == 1],
                                              groups, baseline,  binSize, sigma)

        binned, bins = newTracesFromPSTH_per_trial(spAligned, trials, window, direction[running_state == 1],
                                                    groups, baseline,  binSize, sigma)

        for t, s, c, l in zip(traces, sem, colors, groups):

            plt.plot(bins, t, alpha=alpha_val, c=c, label=str(l))
            plt.fill_between(bins, t - s, t + s, alpha=0.1, color=c)

        plt.plot(bins, np.nanmean(traces, axis=0), c='k')
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)
        # plt.title(
        #     f'AI {r.mean_ai_active.loc[neuron]:.3f}', loc='right', fontsize=10)

        # Quiet
        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                      stimuli_start[running_state == 0],
                                      window)

        ax3 = plt.subplot(2, 2, 2, sharex=ax1)
        plt.title('QUIET', loc='left', fontsize=10)
        newPlotPSTH(spAligned, trials, window,
                    direction[running_state == 0], groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax4 = plt.subplot(2, 2, 4, sharex=ax3, sharey=ax2)
        traces, sem, bins = newTracesFromPSTH(spAligned, trials, window, direction[running_state == 0],
                                              groups, baseline, binSize, sigma)

        for t, s, c, l in zip(traces, sem, colors, groups):

            plt.plot(bins, t, alpha=alpha_val, c=c, label=str(l))
            plt.fill_between(bins, t - s, t + s, alpha=0.1, color=c)

        plt.plot(bins, np.nanmean(traces, axis=0), c='k')
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)
        plt.legend(fontsize='x-small')

        # plt.title(
        #     f'AI {r.mean_ai_quiet.loc[neuron]:.3f}', loc='right', fontsize=10)

        depth = abs(n['cluster_info']['depth'].loc[neuron] - n['SC_depth_limits'][0]
                    )/(n['SC_depth_limits'][0] - n['SC_depth_limits'][1])*100

        fig.suptitle(
            f'''Neuron: {neuron} Depth from sSC:{depth:.1f}''')
        #             osi:{r.osi.loc[neuron]:.2f}, pval_osi:{r.pval_osi.loc[neuron]:.2f}
        #             dsi:{r.dsi.loc[neuron]:.2f}, pval_dsi:{r.pval_dsi.loc[neuron]:.2f}
        #             pvalAI:{r.pval_ai_states.loc[neuron]:.2f}, pvalFR:{r.pval_fr_states.loc[neuron]:.2f}
        #             dAI:{r.delta_states.loc[neuron]:.2f}, BMod:{r.beh_mod.loc[neuron]:.2f}''')

        saveDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, 
                                      dataEntry.Date, '05-02-25_raster_PSTH')
        if not os.path.exists(saveDirectory):
            os.makedirs(saveDirectory)

        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(saveDirectory, f'{neuron}_PSTH'))
        plt.close()

    i += 1
 
# %% tau/AI dataframe 

nans_in_combined_df = combined_df['best_model_q'].isna().sum()

adapt_adapt = []
sens_sens = []
mixed_mixed = []
flat_flat = []

adapt_sens = []
adapt_mixed = []
adapt_flat = []

sens_adapt = []
sens_mixed = []
sens_flat = []

mixed_adapt = []
mixed_sens = []
mixed_flat = []

flat_adapt = []
flat_sens = []
flat_mixed = []


tau_data = {
    'Dataset':[],
    'Neuron': [],
    'State': [],
    'Tau': [],
    'State Separation': [],
    'Depth': [],
    'AI': [],
    'Best Model': []
}



def process_state_combinations(neuron_data):
    
    neuron = neuron_data['clusters']
    dataset = neuron_data['Dataset']
    best_model_a = neuron_data['best_model_a']
    best_model_q = neuron_data['best_model_q']
    tau_a = neuron_data['tau_a']
    tau_q = neuron_data['tau_q']
    AI_a = neuron_data['AI_a']
    AI_q = neuron_data['AI_q']
    depth = neuron_data['Depth']
    state_separation = neuron_data['state_separation']
    

    # Rule 0: Skip if Flat in both states
    if best_model_a == 'Flat' and best_model_q == 'Flat':
        flat_flat.append({'neuron': neuron, 'depth': depth})
        return
    elif best_model_a == 'Adaptation' and best_model_q == 'Adaptation':
        tau_data['Dataset'].extend([dataset] * 2)
        tau_data['Neuron'].extend([neuron] * 2)
        tau_data['State'].extend(['active', 'quiet'])
        tau_data['Tau'].extend([tau_a[0], tau_q[0]])
        tau_data['State Separation'].extend([state_separation] * 2)
        tau_data['Depth'].extend([depth] * 2)
        tau_data['AI'].extend([AI_a[0],AI_q[0]])
        tau_data['Best Model'].extend(['Adaptation', 'Adaptation'])
        adapt_adapt.append({'neuron': neuron, 'depth': depth})
    elif best_model_a == 'Sensitisation' and best_model_q == 'Sensitisation':
        tau_data['Dataset'].extend([dataset] * 2)
        tau_data['Neuron'].extend([neuron] * 2)
        tau_data['State'].extend(['active', 'quiet'])
        tau_data['Tau'].extend([tau_a[0], tau_q[0]])
        tau_data['State Separation'].extend([state_separation] * 2)
        tau_data['Depth'].extend([depth] * 2)
        tau_data['AI'].extend([AI_a[0],AI_q[0]])
        tau_data['Best Model'].extend(['Sensitisation', 'Sensitisation'])
        sens_sens.append({'neuron': neuron, 'depth': depth})
    elif best_model_a == 'Mixed' and best_model_q == 'Mixed':
        tau_data['Dataset'].extend([dataset] * 4)
        tau_data['Neuron'].extend([f'{dataset}_{neuron}'] * 4)
        tau_data['State'].extend(['active', 'quiet', 'active', 'quiet'])
        tau_data['Tau'].extend([tau_a[0], tau_q[0], tau_a[1], tau_q[1]])
        tau_data['State Separation'].extend([state_separation] * 4)
        tau_data['Depth'].extend([depth] * 4)
        tau_data['AI'].extend([AI_a[0],AI_q[0], AI_a[1], AI_q[1]])
        tau_data['Best Model'].extend(['Mixed', 'Mixed', 'Mixed', 'Mixed'])
        mixed_mixed.append({'neuron': neuron, 'depth': depth})
    elif best_model_a == 'Adaptation' and best_model_q == 'Sensitisation':
        tau_data['Dataset'].extend([dataset] * 2)
        tau_data['Neuron'].extend([neuron] * 2)
        tau_data['State'].extend(['active', 'quiet'])
        tau_data['Tau'].extend([tau_a[0], tau_q[0]])
        tau_data['State Separation'].extend([state_separation] * 2)
        tau_data['Depth'].extend([depth] * 2)
        tau_data['AI'].extend([AI_a[0],AI_q[0]])
        tau_data['Best Model'].extend(['Adaptation', 'Sensitisation'])
        adapt_sens.append({'neuron': neuron, 'depth': depth})
    elif best_model_a == 'Adaptation' and best_model_q == 'Mixed':
        tau_data['Dataset'].extend([dataset] * 2)
        tau_data['Neuron'].extend([neuron] * 2)
        tau_data['State'].extend(['active', 'quiet'])
        tau_data['Tau'].extend([tau_a[0], tau_q[0]])
        tau_data['State Separation'].extend([state_separation] * 2)
        tau_data['Depth'].extend([depth] * 2)
        tau_data['AI'].extend([AI_a[0],AI_q[0]])
        tau_data['Best Model'].extend(['Adaptation', 'Mixed'])
        adapt_mixed.append({'neuron': neuron, 'depth': depth})
    elif best_model_a == 'Adaptation' and best_model_q == 'Flat':
        tau_data['Dataset'].extend([dataset] * 2)
        tau_data['Neuron'].extend([neuron] * 2)
        tau_data['State'].extend(['active', 'quiet'])
        tau_data['Tau'].extend([tau_a[0], tau_q[0]])
        tau_data['State Separation'].extend([state_separation] * 2)
        tau_data['Depth'].extend([depth] * 2)
        tau_data['AI'].extend([AI_a[0],AI_q[0]])
        tau_data['Best Model'].extend(['Adaptation', 'Flat'])
        adapt_flat.append({'neuron': neuron, 'depth': depth})
    elif best_model_a == 'Sensitisation' and best_model_q == 'Adaptation':
        tau_data['Dataset'].extend([dataset] * 2)
        tau_data['Neuron'].extend([neuron] * 2)
        tau_data['State'].extend(['active', 'quiet'])
        tau_data['Tau'].extend([tau_a[0], tau_q[0]])
        tau_data['State Separation'].extend([state_separation] * 2)
        tau_data['Depth'].extend([depth] * 2)
        tau_data['AI'].extend([AI_a[0],AI_q[0]])
        tau_data['Best Model'].extend(['Sensitisation', 'Adaptation'])
        sens_adapt.append({'neuron': neuron, 'depth': depth})
    elif best_model_a == 'Sensitisation' and best_model_q == 'Mixed':
        tau_data['Dataset'].extend([dataset] * 2)
        tau_data['Neuron'].extend([neuron] * 2)
        tau_data['State'].extend(['active', 'quiet'])
        tau_data['Tau'].extend([tau_a[0], tau_q[1]])
        tau_data['State Separation'].extend([state_separation] * 2)
        tau_data['Depth'].extend([depth] * 2)
        tau_data['AI'].extend([AI_a[0], AI_q[1]])
        tau_data['Best Model'].extend(['Sensitisation', 'Mixed'])
        sens_mixed.append({'neuron': neuron, 'depth': depth})
    elif best_model_a == 'Sensitisation' and best_model_q == 'Flat':
        tau_data['Dataset'].extend([dataset] * 2)
        tau_data['Neuron'].extend([neuron] * 2)
        tau_data['State'].extend(['active', 'quiet'])
        tau_data['Tau'].extend([tau_a[0], tau_q[0]])
        tau_data['State Separation'].extend([state_separation] * 2)
        tau_data['Depth'].extend([depth] * 2)
        tau_data['AI'].extend([AI_a[0],AI_q[0]])
        tau_data['Best Model'].extend(['Sensitisation', 'Flat'])
        sens_flat.append({'neuron': neuron, 'depth': depth})
    elif best_model_a == 'Mixed' and best_model_q == 'Adaptation':
        tau_data['Dataset'].extend([dataset] * 2)
        tau_data['Neuron'].extend([neuron] * 2)
        tau_data['State'].extend(['active', 'quiet'])
        tau_data['Tau'].extend([tau_a[0], tau_q[0]])
        tau_data['State Separation'].extend([state_separation] * 2)
        tau_data['Depth'].extend([depth] * 2)
        tau_data['AI'].extend([AI_a[0],AI_q[0]])
        tau_data['Best Model'].extend(['Mixed', 'Adaptation'])
        mixed_adapt.append({'neuron': neuron, 'depth': depth})
    elif best_model_a == 'Mixed' and best_model_q == 'Sensitisation':
        tau_data['Dataset'].extend([dataset] * 2)
        tau_data['Neuron'].extend([neuron] * 2)
        tau_data['State'].extend(['active', 'quiet'])
        tau_data['Tau'].extend([tau_a[1], tau_q[0]])
        tau_data['State Separation'].extend([state_separation] * 2)
        tau_data['Depth'].extend([depth] * 2)
        tau_data['AI'].extend([AI_a[1],AI_q[0]])
        tau_data['Best Model'].extend(['Mixed', 'Sensitisation'])
        mixed_sens.append({'neuron': neuron, 'depth': depth})
    elif best_model_a == 'Mixed' and best_model_q == 'Flat':
        tau_data['Dataset'].extend([dataset] * 4)
        tau_data['Neuron'].extend([f'{dataset}_{neuron}'] * 4)
        tau_data['State'].extend(['active', 'quiet', 'active', 'quiet'])
        tau_data['Tau'].extend([tau_a[0], tau_q[0], tau_a[1], tau_q[0]])
        tau_data['State Separation'].extend([state_separation] * 4)
        tau_data['Depth'].extend([depth] * 4)
        tau_data['AI'].extend([AI_a[0],AI_q[0], AI_a[1],AI_q[0]])
        tau_data['Best Model'].extend(['Mixed', 'Flat', 'Mixed', 'Flat'])
        mixed_flat.append({'neuron': neuron, 'depth': depth})
    elif best_model_a == 'Flat' and best_model_q == 'Adaptation':
        tau_data['Dataset'].extend([dataset] * 2)
        tau_data['Neuron'].extend([neuron] * 2)
        tau_data['State'].extend(['active', 'quiet'])
        tau_data['Tau'].extend([tau_a[0], tau_q[0]])
        tau_data['State Separation'].extend([state_separation] * 2)
        tau_data['Depth'].extend([depth] * 2)
        tau_data['AI'].extend([AI_a[0],AI_q[0]])
        tau_data['Best Model'].extend(['Flat', 'Adaptation'])
        flat_adapt.append({'neuron': neuron, 'depth': depth})
    elif best_model_a == 'Flat' and best_model_q == 'Sensitisation':
        tau_data['Dataset'].extend([dataset] * 2)
        tau_data['Neuron'].extend([neuron] * 2)
        tau_data['State'].extend(['active', 'quiet'])
        tau_data['Tau'].extend([tau_a[0], tau_q[0]])
        tau_data['State Separation'].extend([state_separation] * 2)
        tau_data['Depth'].extend([depth] * 2)
        tau_data['AI'].extend([AI_a[0],AI_q[0]])
        tau_data['Best Model'].extend(['Flat', 'Sensitisation'])
        flat_sens.append({'neuron': neuron, 'depth': depth})
    elif best_model_a == 'Flat' and best_model_q == 'Mixed':
        tau_data['Dataset'].extend([dataset] * 4)
        tau_data['Neuron'].extend([f'{dataset}_{neuron}'] * 4)
        tau_data['State'].extend(['active', 'quiet', 'active', 'quiet'])
        tau_data['Tau'].extend([tau_a[0], tau_q[0], tau_a[0], tau_q[1]])
        tau_data['State Separation'].extend([state_separation] * 4)
        tau_data['Depth'].extend([depth] * 4)
        tau_data['AI'].extend([AI_a[0],AI_q[0], AI_a[0], AI_q[1]])
        tau_data['Best Model'].extend(['Flat', 'Mixed', 'Flat', 'Mixed'])
        flat_mixed.append({'neuron': neuron, 'depth': depth})



for _, neuron_row in combined_df.iterrows():
    
    process_state_combinations(neuron_row)
    
# Convert tau_data to DataFrame
tau_df = pd.DataFrame(tau_data)

# %% hist distribution models in states

depth_cut = 40

def categorize_depth(depth, depth_cut=40):
    return 'sSC' if depth < depth_cut else 'dSC'

# Categorize depths and count for each state combination
categories = {
    'adapt_adapt': adapt_adapt,
    'sens_sens': sens_sens,
    'mixed_mixed': mixed_mixed,
    'flat_flat': flat_flat,
    'adapt_sens': adapt_sens,
    'adapt_mixed': adapt_mixed,
    'adapt_flat': adapt_flat,
    'sens_adapt': sens_adapt,
    'sens_mixed': sens_mixed,
    'sens_flat': sens_flat,
    'mixed_adapt': mixed_adapt,
    'mixed_sens': mixed_sens,
    'mixed_flat': mixed_flat,
    'flat_adapt': flat_adapt,
    'flat_sens': flat_sens,
    'flat_mixed': flat_mixed,
}

# Categorize depth and count for each state combination
sSC_counts = {}
dSC_counts = {}

for category, neurons in categories.items():
    sSC_counts[category] = 0
    dSC_counts[category] = 0
    for neuron_info in neurons:
        depth_category = categorize_depth(neuron_info['depth'], depth_cut)
        
        if depth_category == 'sSC':
            sSC_counts[category] += 1
        else:
            dSC_counts[category] += 1



# Plot counts of neurons in each state combination along with total counts
categories_list = list(categories.keys())
sSC_values = [sSC_counts[category] for category in categories_list]
dSC_values = [dSC_counts[category] for category in categories_list]

# Extend the categories and values to include total counts
total_sSC = sum(sSC_counts.values())
total_dSC = sum(dSC_counts.values())
categories_list.extend(['Total sSC', 'Total dSC'])
sSC_values.extend([total_sSC, 0])
dSC_values.extend([0, total_dSC])

# Plot
x = range(len(categories_list))

plt.figure(figsize=(12, 6))
bars_sSC = plt.bar(x, sSC_values, width=0.4, label='sSC', align='center', color='skyblue')
bars_dSC = plt.bar(x, dSC_values, width=0.4, bottom=sSC_values, label='dSC', align='center', color='lightcoral')

# Add value labels on top of bars
for i, bar in enumerate(bars_sSC):
    if sSC_values[i] > 0:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(sSC_values[i]), ha='center', va='bottom', fontsize=8)

for i, bar in enumerate(bars_dSC):
    if dSC_values[i] > 0:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sSC_values[i] + 1, str(dSC_values[i]), ha='center', va='bottom', fontsize=8)

plt.xticks(x, categories_list, rotation=90)
plt.xlabel('State-Model Combinations (Active-Quiet)', fontsize = '12', fontweight='bold')
plt.ylabel('Counts', fontsize = '12', fontweight='bold')
plt.title('Counts of sSC and dSC Neurons for Each State-Model Combination', fontsize = '14', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# %% tau scatter plot (separate vs pooled)

# Extract active and quiet tau values
active_taus = tau_df[tau_df['State'] == 'active']['Tau'].reset_index(drop=True)
quiet_taus = tau_df[tau_df['State'] == 'quiet']['Tau'].reset_index(drop=True)
n = len(active_taus) - (active_taus.isna().sum()+quiet_taus.isna().sum()+len(mixed_mixed)+len(mixed_flat)+len(flat_mixed))
state_separation = tau_df[tau_df['State'] == 'active']['State Separation'].reset_index(drop=True)
#n_sep = len(state_separation)-

scatter_df = pd.DataFrame({'Active Tau': active_taus, 'Quiet Tau': quiet_taus, 'State Separation': state_separation})
color_map = {'pooled': 'grey', 'separate': 'blue'}

# mean for both pooled and separate
mean_active_tau = active_taus.median()
mean_quiet_tau = quiet_taus.median()

# mean for separate only
separate_state_df = tau_df[tau_df['State Separation'] == 'separate']
active_taus_separate = separate_state_df[separate_state_df['State'] == 'active']['Tau'].reset_index(drop=True)
quiet_taus_separate = separate_state_df[separate_state_df['State'] == 'quiet']['Tau'].reset_index(drop=True)
mean_active_tau = active_taus_separate.mean()
mean_quiet_tau = quiet_taus_separate.mean()

# Scatter plot
plt.figure(figsize=(6, 6))
sns.scatterplot(x='Active Tau', y='Quiet Tau', hue='State Separation', palette=color_map, data=scatter_df)

# Set the title and labels
plt.title(f'Tau Values: Active vs Quiet State (n = {n})', fontsize=12, fontweight='bold')
plt.xlabel('Active State Tau', fontsize=10, fontweight='bold')
plt.ylabel('Quiet State Tau', fontsize=10, fontweight='bold')

# Add a diagonal line for reference
min_val = min(active_taus.min(), quiet_taus.min())
max_val = max(active_taus.max(), quiet_taus.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

# Display mean tau values
plt.scatter(mean_active_tau, mean_quiet_tau, color='red', s=100, label='Mean Tau (for separate)', marker='x')
text_offset = 0.25  # You can adjust this value as needed
plt.text(mean_active_tau, -text_offset, f'{mean_active_tau:.2f}', 
         horizontalalignment='center', verticalalignment='top', fontsize=10, color='red')
plt.text(-text_offset, mean_quiet_tau, f'{mean_quiet_tau:.2f}', 
         horizontalalignment='right', verticalalignment='center', fontsize=10, color='red')

# Remove spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend(title='States')

plt.show()

# %% tau scatter (dSC vs sSC)

# Extract depth values
depth_values = tau_df[tau_df['State'] == 'active']['Depth'].reset_index(drop=True)  

# Categorize depths into 'sSC' and 'dSC'
depth_cut = 40
depth_category = depth_values.apply(lambda x: 'sSC' if x < depth_cut else 'dSC')

# Create a DataFrame with both active and quiet tau values and depth category
scatter_df = pd.DataFrame({
    'Active Tau': active_taus,
    'Quiet Tau': quiet_taus,
    'Depth Category': depth_category
})

# Scatter plot using depth category as hue
plt.figure(figsize=(6, 6))
color_map = {'sSC': 'skyblue', 'dSC': 'lightcoral'}
sns.scatterplot(x='Active Tau', y='Quiet Tau', hue='Depth Category', palette=color_map, data=scatter_df)

# Set the title and labels
plt.title(f'Tau Values: Active vs Quiet State (n = {n})', fontsize=12, fontweight='bold')
plt.xlabel('Active State Tau', fontsize=10, fontweight='bold')
plt.ylabel('Quiet State Tau', fontsize=10, fontweight='bold')

# Add a diagonal line for reference
plt.plot([min_val, max_val], [min_val, max_val], 'r--')


# Display mean values on the plot
plt.scatter(mean_active_tau, mean_quiet_tau, color='red', s=100, label='Mean Tau', marker='x')
plt.text(mean_active_tau, -text_offset, f'{mean_active_tau:.2f}', 
          horizontalalignment='center', verticalalignment='center', fontsize=10, color='red')
plt.text(-text_offset, mean_quiet_tau, f'{mean_quiet_tau:.2f}', 
          horizontalalignment='right', verticalalignment='center', fontsize=10, color='red')

# Remove spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend(title='Depth')

plt.tight_layout()
plt.show()

# %% tau scatter (dataset highlighted)

# Extract depth values
dataset_no = tau_df[tau_df['State'] == 'active']['Dataset'].reset_index(drop=True)  

# Create a DataFrame with both active and quiet tau values and depth category
scatter_df = pd.DataFrame({
    'Active Tau': active_taus,
    'Quiet Tau': quiet_taus,
    'Dataset': dataset_no
})

scatter_df['Dataset'].astype(str)

# Scatter plot using depth category as hue
plt.figure(figsize=(6, 6))
sns.scatterplot(x='Active Tau', y='Quiet Tau', hue='Dataset', palette='tab10', data=scatter_df)

# Set the title and labels
plt.title(f'Tau Values: Active vs Quiet State (n = {n})', fontsize=12, fontweight='bold')
plt.xlabel('Active State Tau', fontsize=10, fontweight='bold')
plt.ylabel('Quiet State Tau', fontsize=10, fontweight='bold')

# Add a diagonal line for reference
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

# Display mean values on the plot
plt.scatter(mean_active_tau, mean_quiet_tau, color='red', s=100, label='Mean Tau', marker='x')
plt.text(mean_active_tau, -text_offset, f'{mean_active_tau:.2f}', 
          horizontalalignment='center', verticalalignment='center', fontsize=10, color='red')
plt.text(-text_offset, mean_quiet_tau, f'{mean_quiet_tau:.2f}', 
          horizontalalignment='right', verticalalignment='center', fontsize=10, color='red')

# Remove spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend(title='Dataset')

plt.tight_layout()
plt.show()

# %% hist for flat in quiet state

active_tau_df = tau_df[tau_df['State'] == 'active']

# Define your desired bin size
bin_size = 0.05  # Adjust the bin size as needed

# Calculate the range of the data
tau_min = active_tau_df['Tau'].min()
tau_max = active_tau_df['Tau'].max()

# Calculate the number of bins based on the range and the bin size
bins = np.arange(tau_min, tau_max + bin_size, bin_size)

# Plot the histogram
plt.figure()
plt.hist(active_tau_df['Tau'], bins=bins, color='blue', edgecolor='black')
plt.title('Tau Values in Active State for Neurons with Flat in Quiet State')
plt.xlabel('Tau')
plt.ylabel('Frequency')
plt.show()

# %%hist for flat in active state

quiet_tau_df = tau_df[tau_df['State'] == 'quiet']

# Define your desired bin size
bin_size = 0.05  # Adjust the bin size as needed

# Calculate the range of the data
tau_min = quiet_tau_df['Tau'].min()
tau_max = quiet_tau_df['Tau'].max()

# Calculate the number of bins based on the range and the bin size
bins = np.arange(tau_min, tau_max + bin_size, bin_size)

# Plot the histogram
plt.hist(quiet_tau_df['Tau'], bins=bins, color='blue', edgecolor='black')
plt.title('Tau Values in Quiet State for Neurons with Flat in Active State')
plt.xlabel('Tau')
plt.ylabel('Frequency')
plt.show()



# %% AI scatter (separate vs pooled) 

# Remove rows with inf or -inf in the AI column
tau_df_AI = tau_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['AI'])
n = len(tau_df_AI)/2-len(mixed_mixed)

# Extract active and quiet AI values
active_ai = tau_df_AI[tau_df_AI['State'] == 'active']['AI'].reset_index(drop=True)
quiet_ai = tau_df_AI[tau_df_AI['State'] == 'quiet']['AI'].reset_index(drop=True)
model_preferences = tau_df_AI[tau_df_AI['State'] == 'active']['State Separation'].reset_index(drop=True)

# Clip AI values to the range [-4, 4]
active_ai_clipped = np.clip(active_ai, -1, 3)
quiet_ai_clipped = np.clip(quiet_ai, -1, 3)
# Create a DataFrame with both active and quiet AI values
ai_scatter_df = pd.DataFrame({'Active AI': active_ai_clipped, 'Quiet AI': quiet_ai_clipped, 'State Separation': model_preferences})

# Set color palette for State Separation
color_map = {'pooled': 'grey', 'separate': 'blue'}

# Scatter plot for AI values
plt.figure(figsize=(6, 6))
sns.scatterplot(x='Active AI', y='Quiet AI', hue='State Separation', palette=color_map, data=ai_scatter_df)

# Set the title and labels
plt.title(f'AI Values: Active vs Quiet State n={n}', fontsize=12, fontweight='bold')
plt.xlabel('Active State AI', fontsize=10, fontweight='bold')
plt.ylabel('Quiet State AI', fontsize=10, fontweight='bold')

# Add a diagonal line for reference
min_val_ai = min(active_ai_clipped.min(), quiet_ai_clipped.min())
max_val_ai = max(active_ai_clipped.max(), quiet_ai_clipped.max())
plt.plot([min_val_ai, max_val_ai], [min_val_ai, max_val_ai], 'r--')

# Calculate and add mean AI values for 'separate' State Separation
# separate_df = ai_scatter_df[ai_scatter_df['State Separation'] == 'separate']
# mean_active_ai_separate = separate_df['Active AI'].mean()
# mean_quiet_ai_separate = separate_df['Quiet AI'].mean()
# plt.scatter(mean_active_ai_separate, mean_quiet_ai_separate, color='red', s=100, label='Mean AI (for separate)', marker='x')

# # Display mean values on the plot
# text_offset = 1.25
# plt.text(mean_active_ai_separate, -text_offset, f'{mean_active_ai_separate:.2f}', 
#          horizontalalignment='center', verticalalignment='top', fontsize=10, color='red')
# plt.text(-text_offset, mean_quiet_ai_separate, f'{mean_quiet_ai_separate:.2f}', 
#          horizontalalignment='right', verticalalignment='center', fontsize=10, color='red')

# Remove spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend(title='States') #, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
# %% AI scatter (dSC vs sSC) 

# Remove rows with inf or -inf in the AI column
tau_df_AI = tau_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['AI'])


# Extract depth values}
depth_values = tau_df_AI[tau_df_AI['State'] == 'active']['Depth'].reset_index(drop=True)  

# Categorize depths into 'sSC' and 'dSC'
depth_cut = 40
depth_category = depth_values.apply(lambda x: 'sSC' if x < depth_cut else 'dSC')

# Extract active and quiet AI values
active_ai = tau_df_AI[(tau_df_AI['State'] == 'active')& (tau_df_AI['State Separation'] == 'separate')]['AI'].reset_index(drop=True)
quiet_ai = tau_df_AI[(tau_df_AI['State'] == 'quiet')& (tau_df_AI['State Separation'] == 'separate')]['AI'].reset_index(drop=True)
n = len(active_ai)
# Clip AI values to the range [-4, 4]
active_ai_clipped = np.clip(active_ai, -1, 3)
quiet_ai_clipped = np.clip(quiet_ai, -1, 3)
# Create a DataFrame with both active and quiet AI values
ai_scatter_df = pd.DataFrame({'Active AI': active_ai_clipped, 'Quiet AI': quiet_ai_clipped, 'Depth Category': depth_category})

# Set color palette for State Separation
color_map = {'sSC': 'skyblue', 'dSC': 'lightcoral'}
# Scatter plot for AI values
plt.figure(figsize=(6, 6))
sns.scatterplot(x='Active AI', y='Quiet AI', hue='Depth Category', palette=color_map, data=ai_scatter_df)

# Set the title and labels
plt.title(f'AI Values: Active vs Quiet State n={n} (separate states only)', fontsize=12, fontweight='bold')
plt.xlabel('Active State AI', fontsize=10, fontweight='bold')
plt.ylabel('Quiet State AI', fontsize=10, fontweight='bold')

# Add a diagonal line for reference
min_val_ai = min(active_ai_clipped.min(), quiet_ai_clipped.min())
max_val_ai = max(active_ai_clipped.max(), quiet_ai_clipped.max())
plt.plot([min_val_ai, max_val_ai], [min_val_ai, max_val_ai], 'r--')

# Display mean values on the plot
# plt.scatter(mean_active_ai_separate, mean_quiet_ai_separate, color='red', s=100, label='Mean AI (for separate)', marker='x')
# text_offset = 1.25
# plt.text(mean_active_ai_separate, -text_offset, f'{mean_active_ai_separate:.2f}', 
#          horizontalalignment='center', verticalalignment='top', fontsize=10, color='red')
# plt.text(-text_offset, mean_quiet_ai_separate, f'{mean_quiet_ai_separate:.2f}', 
#          horizontalalignment='right', verticalalignment='center', fontsize=10, color='red')

# Remove spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend(title='States') #, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


# %% AI values histogram 

# Clip values to be within the -4 to 4 range
active_ai_clipped = np.clip(active_ai, -1, 2)
quiet_ai_clipped = np.clip(quiet_ai, -1, 2)

# Define common bins
common_bins = np.histogram_bin_edges(np.concatenate([active_ai_clipped, quiet_ai_clipped]), bins=100)

# Plot the histogram
plt.figure(figsize=(8, 5))
sns.histplot(active_ai_clipped, color='blue', label='Active AI', kde=False, bins=common_bins, alpha=0.6)
sns.histplot(quiet_ai_clipped, color='orange', label='Quiet AI', kde=False, bins=common_bins, alpha=0.6)

# Set the title and labels
plt.title('Distribution of AI Values: Active vs Quiet State', fontsize=12, fontweight='bold')
plt.xlabel('AI Value', fontsize=10, fontweight='bold')
plt.ylabel('Frequency', fontsize=10, fontweight='bold')

# Improve plot aesthetics
plt.legend(title='State')
plt.tight_layout()
plt.show()

# %% AI scatter adaptation only
# Remove rows with inf or -inf in the AI column
tau_df_AI = tau_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['AI'])
# Extract depth values
depth_values = tau_df_AI[tau_df_AI['State'] == 'active']['Depth'].reset_index(drop=True)  

# Categorize depths into 'sSC' and 'dSC'
depth_cut = 40
depth_category = depth_values.apply(lambda x: 'sSC' if x < depth_cut else 'dSC')

# Extract active and quiet AI values
active_ai = tau_df_AI[(tau_df_AI['State'] == 'active') & (tau_df_AI['Best Model'] == 'Adaptation') & (tau_df_AI['State Separation'] == 'separate')]['AI'].reset_index(drop=True)
quiet_ai = tau_df_AI[(tau_df_AI['State'] == 'quiet') & (tau_df_AI['Best Model'] == 'Adaptation')& (tau_df_AI['State Separation'] == 'separate')]['AI'].reset_index(drop=True)
# Clip AI values to the range [-4, 4]
active_ai_clipped = np.clip(active_ai, -1, 3)
quiet_ai_clipped = np.clip(quiet_ai, -1, 3)
n = len(active_ai)
# Create a DataFrame with both active and quiet AI values
ai_scatter_df = pd.DataFrame({'Active AI': active_ai_clipped, 'Quiet AI': quiet_ai_clipped, 'Depth Category': depth_category})

# Set color palette for State Separation
color_map = {'sSC': 'skyblue', 'dSC': 'lightcoral'}
# Scatter plot for AI values
plt.figure(figsize=(6, 6))
sns.scatterplot(x='Active AI', y='Quiet AI', hue='Depth Category', palette=color_map, data=ai_scatter_df)

# Set the title and labels
plt.title(f'AI Values: Active vs Quiet State (Adaptation Model only) n={n}', fontsize=12, fontweight='bold')
plt.xlabel('Active State AI', fontsize=10, fontweight='bold')
plt.ylabel('Quiet State AI', fontsize=10, fontweight='bold')

# Add a diagonal line for reference
min_val_ai = min(active_ai_clipped.min(), quiet_ai_clipped.min())
max_val_ai = max(active_ai_clipped.max(), quiet_ai_clipped.max())
plt.plot([min_val_ai, max_val_ai], [min_val_ai, max_val_ai], 'r--')

# Display mean values on the plot
mean_active_ai = ai_scatter_df['Active AI'].median()
mean_quiet_ai = ai_scatter_df['Quiet AI'].median()


plt.scatter(mean_active_ai, mean_quiet_ai, color='red', s=100, label='Median AI (for separate)', marker='x')
text_offset = 0.25
plt.text(mean_active_ai, -text_offset, f'{mean_active_ai:.2f}', 
         horizontalalignment='center', verticalalignment='top', fontsize=10, color='red')
plt.text(-text_offset, mean_quiet_ai, f'{mean_quiet_ai:.2f}', 
         horizontalalignment='right', verticalalignment='center', fontsize=10, color='red')

# Remove spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend(title='States') #, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
#%% batch raster and psth

i = 0

for n,st in zip(neural, stim):    

    clusters = n['cluster_analysis'][n['visual']]
    spike_times = n['spike_times']
    spike_clusters = n['spike_clusters']
    
    stimuli_start = st['startTime'].reshape(-1)
    stimuli_end = st['endTime'].reshape(-1)
    direction = st['direction'].reshape(-1)
    running_state = st['running_state'].reshape(-1)
   
    for neuron in clusters:
        
        fig = plt.figure(figsize=(9, 9))

        # Active
        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                            stimuli_start[running_state == 1], 
                                            window)
        
        ax1 = plt.subplot(2,2,1)
        plt.title('ACTIVE', loc='left', fontsize = 10)
        newPlotPSTH(spAligned, trials, window, direction[running_state == 1], groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x= np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax2 = plt.subplot(2,2,3, sharex = ax1)
        traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[running_state == 1],
                                              groups, baseline,  binSize, sigma)
        # traces, sem, timeBins = newTracesFromPSTH_trialSmoooth(spAligned, trials, window, direction[running_state == 1],
        #                                       groups, baseline,  binSize, sigma)
        mean_trace_active = np.mean(traces, axis=0)
        
        ax2.plot(timeBins, mean_trace_active, color='black', label='Mean Active', linewidth=2)
        ax2.legend()

        
        

        for t, s, c, l in zip(traces, sem, colors, groups):

            plt.plot(timeBins, t, alpha = alpha_val, c = c, label = str(l) )
            plt.fill_between(timeBins, t - s, t + s, alpha= 0.1, color= c)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x= np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

            
        # Quiet
        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                            stimuli_start[running_state == 0], 
                                            window)
    
        ax3 = plt.subplot(2,2,2, sharex = ax1)
        plt.title('QUIET', loc='left', fontsize = 10)
        newPlotPSTH(spAligned, trials, window, direction[running_state == 0], groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x= np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax4 = plt.subplot(2,2,4, sharex = ax3, sharey = ax2)
        traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[running_state == 0],
                                              groups, baseline, binSize, sigma)
        mean_trace_quiet = np.mean(traces, axis=0)
        
        ax4.plot(timeBins, mean_trace_quiet, color='black', label='Mean Quiet', linewidth=2)
        ax4.legend()


        for t, s, c, l in zip(traces, sem, colors, groups):

            plt.plot(timeBins, t, alpha = alpha_val, c = c, label = str(l) )
            plt.fill_between(timeBins, t - s, t + s, alpha= 0.1, color= c)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x= np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        
        depth = abs(n['cluster_info']['depth'].loc[neuron] - n['SC_depth_limits'][0]
                    )/(n['SC_depth_limits'][0] - n['SC_depth_limits'][1])*100
        
        fig.suptitle(
            f'Neuron: {neuron} Depth from sSC:{depth:.1f}')
        
        dataEntry = database.loc[i]
               
        saveDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, 'raster_PSTH_12_11_24')
        # saveDirectory = os.path.join(analysisDir, 'Figures', 'FG005', 'raster_PSTH_trial_smooth')
        if not os.path.isdir(saveDirectory):
            os.makedirs(saveDirectory)
        filename = os.path.join(saveDirectory, f'{neuron}_PSTH.png')
        plt.savefig(filename)
        plt.close()
        
    i += 1






# %% mean vs train fit fails

# TODO: ensure that it makes sense to use test_trial_errors as the max number of train fits performed 
df = df_results
df['test_trial_errors_len'] = df['test_trial_errors'].apply(len)
df = df.drop(['test_trial_errors','Parameters', 'AI', 'fitted_firing_rates', 'rmse_k_fold', 'time_to_fit', 'start_time_for_fit', 'firing_rate', 'fr_sem', 'y_start_fit', 'SC_depth'], axis=1)

# Create dataframes for adaptation vs mixed model fail fits
df_a = df[df['train_fit_fail_a'] != 0]
df_m = df[df['train_fit_fail_m'] != 0]
# Calculate the proportion of train_fit_fail_a and train_fit_fail_m relative to the length of test_trial_errors
df_a['proportion_train_fit_fail_a'] = df_a['train_fit_fail_a'] / df_a['test_trial_errors_len']
df_m['proportion_train_fit_fail_m'] = df_m['train_fit_fail_m'] / df_m['test_trial_errors_len']

# Distribution for `train_fit_fail_a` based on `mean_fit_fail_a`
proportion_fit_fail_a_true = df_a[df_a['mean_fit_fail_a'] == True]['proportion_train_fit_fail_a']
proportion_fit_fail_a_false = df_a[df_a['mean_fit_fail_a'] == False]['proportion_train_fit_fail_a']

# Distribution for `train_fit_fail_m` based on `mean_fit_fail_m`
proportion_fit_fail_m_true = df_m[df_m['mean_fit_fail_m'] == True]['proportion_train_fit_fail_m']
proportion_fit_fail_m_false = df_m[df_m['mean_fit_fail_m'] == False]['proportion_train_fit_fail_m']

rows_fit_fail_a_true = df_a[(df_a['mean_fit_fail_a'] == True)]
rows_fit_fail_a_false = df_a[(df_a['mean_fit_fail_a'] == False)] 

rows_fit_fail_m_true = df_m[(df_m['mean_fit_fail_m'] == True)]
rows_fit_fail_m_false = df_m[(df_m['mean_fit_fail_m'] == False)]

# Plot for `proportion_train_fit_fail_a`
plt.figure()
plt.hist(proportion_fit_fail_a_true, bins=20, alpha=0.5, label='mean_fit_fail_a True')
plt.hist(proportion_fit_fail_a_false, bins=20, alpha=0.5, label='mean_fit_fail_a False')
plt.legend()
plt.title('Proportion of train fit fails in adaptation')
plt.xlabel('Proportion of train_fit_fail_a')
plt.ylabel('Frequency')
plt.show()

# Plot for `proportion_train_fit_fail_m`
plt.figure()
plt.hist(proportion_fit_fail_m_true, bins=20, alpha=0.5, label='mean_fit_fail_m True')
plt.hist(proportion_fit_fail_m_false, bins=20, alpha=0.5, label='mean_fit_fail_m False')
plt.legend()
plt.title('Proportion of train fit fails in mixed model')
plt.xlabel('Proportion of train_fit_fail_m')
plt.ylabel('Frequency')
plt.show()

plt.figure()
rows_fit_fail_a_false_05 = df_a[df_a['proportion_train_fit_fail_a'] == 0.5]
plt.hist(rows_fit_fail_a_false_05['test_trial_errors_len'], bins=20, alpha=0.7, color='blue')
plt.xlabel('Number of folds')
plt.ylabel('Frequency')
plt.title('Total number of folds for proportion of fit failures of 0.5')
plt.show()

plt.figure()
rows_fit_fail_m_false_05 = df_m[df_m['proportion_train_fit_fail_m'] == 0.5]
plt.hist(rows_fit_fail_m_false_05['test_trial_errors_len'], bins=20, alpha=0.7, color='blue')
plt.xlabel('Number of folds')
plt.ylabel('Frequency')
plt.title('Total number of folds for proportion of fit failures of 0.5')
plt.show()

plt.figure()
rows_fit_fail_m_false_1 = df_m[(df_m['proportion_train_fit_fail_m'] == 1) & (df_m['mean_fit_fail_m'] == False)]
plt.hist(rows_fit_fail_m_false_1['test_trial_errors_len'], bins=20, alpha=0.7, color='blue')
plt.xlabel('Number of folds')
plt.ylabel('Frequency')
plt.title('Total number of folds for proportion of fit failures of 1 with the mean fit success')
plt.show()

if ds_neuron: 
    plt.title(f'DS neuron: {text_str} pref: {pref_dir}')      
elif os_neuron: 
    plt.title(f'OS neuron: {text_str} pref: {pref_ori}')      
else: 
    plt.title(f'{text_str}')

spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                              stimuli_start[running_state == 1],
                              window)
ax4 = plt.subplot(2, 3, 1)
plt.title('ACTIVE', loc='left', fontsize=10)
newPlotPSTH(spAligned, trials, window,
            direction[running_state == 1], groups, colors)
plt.axvline(x=0, c='k', ls='--')
plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
plt.xticks(xticks)

ax2 = plt.subplot(2, 3, 4)
traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[running_state == 1],
                                      groups, baseline,  binSize, sigma)

for t, s, c, l in zip(traces, sem, colors, groups):

    if np.isin(l, dirs):

        plt.plot(timeBins, t, alpha=alpha_val, c=c, label=str(l))
        plt.fill_between(timeBins, t - s, t + s, alpha=0.1, color=c)

plt.plot(timeBins, np.nanmean(traces, axis=0), c='k')
plt.axvline(x=0, c='k', ls='--')
plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
plt.xticks(xticks)


# Quiet

spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                              stimuli_start[running_state == 0],
                              window)

ax5 = plt.subplot(2, 3, 2, sharex=ax4)

plt.title('QUIET', loc='left', fontsize=10)
newPlotPSTH(spAligned, trials, window,
            direction[running_state == 0], groups, colors)
plt.axvline(x=0, c='k', ls='--')
plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
plt.xticks(xticks)

ax3 = plt.subplot(2, 3, 5, sharey=ax2)
traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[running_state == 0],
                                      groups, baseline, binSize, sigma)

for t, s, c, l in zip(traces, sem, colors, groups):

    if np.isin(l, dirs):
        plt.plot(timeBins, t, alpha=alpha_val, c=c, label=str(l))
        plt.fill_between(timeBins, t - s, t + s, alpha=0.1, color=c)

plt.plot(timeBins, np.nanmean(traces, axis=0), c='k')
plt.axvline(x=0, c='k', ls='--')
plt.axvline(x=np.mean(stimuli_end - stimuli_start), c='k', ls='--')
plt.xticks(xticks)
plt.legend(fontsize='x-small')
identifier=f'{dataEntry.Name}_{dataEntry.Date}_{neuron}'

fig.suptitle(
    f'''Neuron: {identifier}''')
plt.tight_layout()

saveDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, 'polar_plots', '06_08_24', 'test_dsi_osi')

if not os.path.exists(saveDirectory):
    os.makedirs(saveDirectory)
save_path = os.path.join(saveDirectory, f'{neuron}_polar_plot.png')
fig.savefig(save_path)
# plt.show()
plt.close()



    




#%% Individual raster and PSTH (clean plot)
def newPlotPSTH(spAligned, trials, window, stimuli_id, groups, color_map, marker=2.5):
    """
    Plot raster of grouped events (spikes) and add other (rare) events (e.g.,
    behavioural events like wheel movements) with dots colored black.

    Parameters
    ----------
    spAligned : np.array
        Spike times aligned to event of interest.
    trials : np.array
        Trial ID of spike.
    window : np.array
        Time interval of aligned trials.
    stimuli_id : np.array
        Total stimulus id considered.
    groups : np.array
        Stimulus type in session.
    color_map : ignored in this version, all markers are black.
    marker : float, optional
        Marker size for the raster. The default is 2.5.

    Returns
    -------
    None.
    """
    # Loop over each group of trials (trial = each stimulus presentation)
    k = 0
    for group in groups:
        trials_type_selection = np.where(stimuli_id == group)[0]
        
        for trial in trials_type_selection:
            spikes = spAligned[trials == trial]
            plt.plot(spikes, np.ones(np.shape(spikes)) * k, '.', c='black', markersize=marker)
            k += 1
# adapt = 728a, sens = 699q/0818:404a, flat = 699q/704q/0818: , mixed = 890a

colors = cm.rainbow(np.linspace(0, 1, len(groups)))

for n,st in zip(neural, stim):    

    clusters = n['cluster_analysis'][n['visual']]
    spike_times = n['spike_times']
    spike_clusters = n['spike_clusters']
    
    stimuli_start = st['startTime'].reshape(-1)
    stimuli_end = st['endTime'].reshape(-1)
    direction = st['direction'].reshape(-1)
    running_state = st['running_state'].reshape(-1)
    


plt.rcParams.update({
    'font.size': 10,        # Adjust the size
    'font.weight': 'bold',  # Make it bold
    'font.family': 'sans-serif'  # Change font family
})
# Assuming alignData, newPlotPSTH, and newTracesFromPSTH functions are predefined and imported.

neuron = 891

fig = plt.figure(figsize=(3, 6))
gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 3, 1])  # Define a GridSpec of 4 rows, but we will use only 3 slots

# Active
# Assuming spike_times, spike_clusters, and other necessary variables are defined in your provided code.
spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                    stimuli_start[running_state == 0], 
                                    window)

ax1 = fig.add_subplot(gs[0, 0])  
plt.title('Adaptation', loc='left', fontsize=12, fontweight='bold', color='red')
newPlotPSTH(spAligned, trials, window, direction[running_state == 0], groups, None)
# plt.axvline(x=0, color='black', linestyle='--')
# plt.axvline(x=np.mean(stimuli_end - stimuli_start), color='black', linestyle='--')
ax1.set_ylabel('Trials', fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_linewidth(2)

ax1.set_xticklabels([])  # Remove x-axis tick labels
ax1.set_yticklabels([])  # Remove y-axis tick labels
ax1.set_xticks([])
ax1.set_yticks([])

ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Using the third row
ax2.set_ylabel('Firing rate (spikes/s)', fontweight='bold')
traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[running_state == 1],
                                      groups, baseline, binSize, sigma)
mean_trace_active = np.mean(traces, axis=0)
ax2.plot(timeBins, mean_trace_active, color='black', linewidth=2)
# plt.axvline(x=0, color='black', linestyle='--')
# plt.axvline(x=np.mean(stimuli_end - stimuli_start), color='black', linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_linewidth(2)
ax2.set_xticklabels([])  # Remove x-axis tick labels
ax2.set_yticklabels([])  # Remove y-axis tick labels
ax2.set_xticks([])
ax2.set_yticks([])


ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)  # Using the third row
# Define step stimulus
time_stim = np.array([-1, 0, 1, 2, 3])  # Time points for the step
stimulus = np.array([0, 1, 1, 0, 0])  # Step goes up at t=0 and down at t=2
ax3.step(time_stim, stimulus, where='post', color='green', linewidth=2)
ax3.set_xlim(-1, 3)
ax3.set_ylim(-1, 2)
ax3.set_ylabel('Stimulus', fontweight='bold')
ax3.set_xlabel('Time (s)', fontweight='bold')
# Modify spines for ax3
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_linewidth(2)
ax3.spines['bottom'].set_linewidth(2)

# ax3.set_xlim(-1, 3)
# # ax3.set_ylim(0, 17)
# ax3.add_patch(FancyArrowPatch((0, 4), (0, 4),
#                              arrowstyle='-|>', color='black',
#                              mutation_scale=20))

ax3.set_xticklabels([])  # Remove x-axis tick labels
ax3.set_yticklabels([])  # Remove y-axis tick labels
ax3.set_xticks([])
ax3.set_yticks([])

plt.tight_layout()
plt.show()

saveDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, 'symp')
if not os.path.isdir(saveDirectory):
    os.makedirs(saveDirectory)
filename = os.path.join(saveDirectory, f'{neuron}_PSTH_sens.png')
plt.savefig(filename, dpi=150)

# %% individual mean fit (clean plot) - to revise

stimDur = 2
prePostStim = 1
no_stim_window = (-0.5,0)
stim_window = (0, 0.5)
baseline = 0.2
window = np.array([-prePostStim, stimDur + prePostStim])
sigma = 0.06
alpha_val = 0.3

groups = np.unique(stim[0]['direction'])
colors = cm.rainbow(np.linspace(0, 1, len(groups)))
# xticks = np.arange(-0.5, 2.5, 0.5)
threshold = 0.9 #for running velocity

interval_fr = np.array([0, 2])
binSize=0.005


for n,st in zip(neural, stim):    

    clusters = n['cluster_analysis'][n['visual']]
    spike_times = n['spike_times']
    spike_clusters = n['spike_clusters']
    
    stimuli_start = st['startTime'].reshape(-1)
    stimuli_end = st['endTime'].reshape(-1)
    direction = st['direction'].reshape(-1)
    running_state = st['running_state'].reshape(-1)
   
    i = 3
    
neuron = 300
state_mean_psths = {}
results = {}

for state in ['active', 'quiet']:
    if state == 'quiet':
        state_condition = running_state == 0
    else:
        state_condition = running_state == 1

    spAligned_active, trials_active = alignData(
        spike_times[spike_clusters == neuron],
        stimuli_start[state_condition], 
        window
    )
    
    
    binned, timeBins = newTracesFromPSTH_per_trial(spAligned, trials, window, direction[state_condition], groups, baseline,
                                binSize, sigma)
    
    spAligned, trials = alignData(spike_times[spike_clusters == neuron], stimuli_start[state_cond], window)
    # else:
        # spAligned, trials = spAligned_pooled, trials_pooled  # Use pooled data

    binned, timeBins = newTracesFromPSTH_per_trial(spAligned, trials, window, direction[state_cond], groups, baseline, binSize, sigma)

    
    state_mean_psths[state] = np.mean(binned, axis=0)
    k = 20
    results[state] = perform_k_fold(binned, timeBins, state, k, identifier = f'{neuron}', is_suppressed = False, plot_test=False, plot_models = False)
max_firing_rate = max(
    np.max(state_mean_psths['active']),
    np.max(state_mean_psths['quiet'])
)
min_firing_rate = min(
    np.min(state_mean_psths['active']),
    np.min(state_mean_psths['quiet'])
)

# Adding a 10% margin to the maximum firing rate
y_max = max_firing_rate * 1.1
y_min = min_firing_rate * 1.1 #if np.min(state_mean_psths['active']) < 0 or np.min(state_mean_psths['quiet']) < 0 else 0


fig, axs = plt.subplots(2, 1, figsize=(6, 12))  # Two subplots for active and quiet states
# fig.suptitle(f'Neuron {neuron}')


states = ['active', 'quiet']

# Iterate over each state to plot
for i, state in enumerate(states):
    state_data = results[state]
    mean_psth = state_mean_psths[state]
    is_suppressed,_,_ = analyze_suppression(mean_psth, timeBins)
    if is_suppressed:
        mean_psth = -1 * mean_psth  # Invert the data if suppressed
    # Extracting the best model's fitted response and times
    best_model = state_data['best_model']
    best_model_details = state_data['model_details']
    fitted_firing_rates = best_model_details['fitted_firing_rates']
    time_to_fit_adjusted = best_model_details['time_to_fit_adjusted']
    start_time_for_fit = best_model_details['start_time_for_fit']
    ai_value = best_model_details.get('ai', 'N/A')  # Safely get AI value, default to 'N/A' if not present
    
    if best_model in ['Adaptation', 'Sensitisation']:
        tau = best_model_details['params'][0]
        tau_str = f"tau: {tau:.2f}"
    elif best_model == 'Mixed':
        tau1 = best_model_details['params'][1]
        tau2 = best_model_details['params'][3]
        tau_str = f"tau: {tau1:.2f}, {tau2:.2f}"
    else: 
        tau_str = ''




    # Plotting the actual neural response
    axs[i].plot(timeBins, mean_psth, label='Actual Response', color='blue', linewidth=3)

    # Plotting the fitted model response
    # adjusted_time = [t + start_time_for_fit for t in time_to_fit_adjusted]
    axs[i].plot(time_to_fit_adjusted, fitted_firing_rates, label='Exponential Fit', color='orange', linewidth=3)
    if is_suppressed: 
        axs_supp = axs[i].twinx()
        axs_supp.plot(timeBins, -1*mean_psth, color='grey', alpha=0.5, label='Suppressed response')
        axs_supp.set_ylabel('Original Firing Rate')
        axs_supp.legend(loc='upper right')
    # axs[i].set_title(f"{state.capitalize()} State: {state_data['best_model']} (AI: {ai_value})", loc='left', fontsize=10, fontweight='bold')
    axs[i].set_ylabel('Firing Rate (spikes/s)', fontsize=12, fontweight='bold')
    axs[i].tick_params(axis='y', labelsize=12)  # Setting y-axis tick label font size
    axs[i].axvline(x=0, color='black', linestyle='--', linewidth=1)  # Dashed line at 0 seconds
    axs[i].axvline(x=2, color='black', linestyle='--', linewidth=1)  # Dashed line at 2 seconds

    
    # Customize specific axes differently
    if state == 'active':
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_linewidth(2)
        axs[i].set_xticklabels([])  # Remove x-axis tick labels
        # axs[i].set_yticklabels([])  # Remove y-axis tick labels
        axs[i].set_xticks([])
        # axs[i].set_yticks([])
        # axs[i].legend()

    elif state == 'quiet':
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['left'].set_linewidth(2)
        axs[i].spines['bottom'].set_linewidth(2)
        axs[i].set_xlabel('Time (s)')
    axs[i].set_ylim(y_min, y_max)
    # Set a blank title to reserve space
    axs[i].set_title(" ", loc='left')
       
    axs[i].text(
        0.01, 1.02,
        f"{state_data['best_model']}",
        transform=axs[i].transAxes,
        fontsize=14,
        fontweight='bold',
        color='red',
        verticalalignment='bottom'
    )
    axs[i].text(
        0.25, 1.02,
        f" AI: {ai_value} {tau_str}",
        transform=axs[i].transAxes,
        fontsize=14,
        fontweight='bold',
        verticalalignment='bottom'
    )
# fig.text(0.015, 0.75, 'Quiet', va='center', ha='center', rotation='vertical', fontsize=12, fontweight='bold', bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
# fig.text(0.015, 0.25, 'Active', va='center', ha='center', rotation='vertical', fontsize=12, fontweight='bold', bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round',pad=0.5))
plt.subplots_adjust(left=0.2)  # Increase the left margin

    # figureDirectory = r'/Users/rg483/Analysis/Figures'
    # exponential_neuronal_folder = os.path.join(figureDirectory, 'exponential_neuronal_data_club_best')
    # if not os.path.isdir(exponential_neuronal_folder):
    #     os.makedirs(exponential_neuronal_folder)
    # filename = os.path.join(exponential_neuronal_folder, f'{neuron}_{state}.png')
    # plt.savefig(filename)
    # plt.close()

plt.tight_layout()
plt.show()

