# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:06:54 2025

@author: Raikhan
"""
# %% load libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

os.chdir('C:\dev\workspaces\Ephys-scripts\ephys_analysis')
from Functions.user_def_analysis import *
from Functions.load import *
from Functions.PSTH import *
from Functions.neural_response import *
from Functions.shift_warping import *
from Functions.behavioural import *
from Functions.time_constant import *


from scipy.ndimage import gaussian_filter1d
from affinewarp import ShiftWarping, SpikeData
from affinewarp.crossval import heldout_transform

# %% load gratings temporal data

### same as another loading function except that there is manual sorting in this one and the new one is curated sorting and label is 'mua' here instead of 'good'

#root path (local) where the data to analyse is located
analysisDir =  define_directory_analysis()

#Generate a csv with all the data that is going to be anlysed in THIS script
#Should have a copy local of all the preprocessed data
csvDir = os.path.join(analysisDir, 'Inventory', 'gratings_temporal.csv')


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

label = 'mua'

# Create database for analysis
#Inicialize object to colect data from each session
stim = []
neural = []

for i in range(len(database)):
    
    
    #Path
    dataEntry = database.loc[i]
    dataDirectory = os.path.join(analysisDir, dataEntry.Name, dataEntry.Date)
    
    #Check if directory for figures exists
    figureDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date)
    
    if not os.path.isdir(figureDirectory):
        os.makedirs(figureDirectory)
    
    
    #Load stimulus type (for example oddballs)
    s = load_stimuli_data(dataDirectory, dataEntry)
    
    #Load from the csv file WHICH experiments are concatenated for the analysis in INDEX
    #For example, all the oddballs drifting
    
    #TODO: IMPLEMENTED TEST. capture here the cases when there is an only experiment (no indices given)
        
    chopped = chop_by_experiment(s, dataEntry)

    if isinstance(dataEntry.Experiment, str):
        index = [int(e) for e in dataEntry.Experiment.split(',')]
        s = merge_experiments_by_index(chopped, index)
    else:
        s = chopped[0]
        
    stim.append(s)
    
    # Load neural data#########################################################
    
    #Set interval to load just spikes in these limits. Handle merged experiments
    #in an inefficent way. 
    extend = 20
    
    interval = np.array([s['intervals'][0][0] - extend, s['intervals'][-1][0] + extend])
    
    n = load_neural_data_local(dataDirectory, interval)
    
    cluster_info = n['cluster_info']
    cluster_info.set_index('cluster_id', inplace=True)
    SC_depth_limits = n['SC_depth_limits']
    
    #Find somatic units
    if dataEntry.Sorting == 'manual':
        
        label = label.lower()     
        
        # Keep clusters IDs within SC limits, with the label selected before (default good)
        soma = cluster_info[((cluster_info.group == label) &
                                              (cluster_info.depth >= SC_depth_limits[1]) &
                                              (cluster_info.depth <= SC_depth_limits[0]))].index.values
    
    if dataEntry.Sorting == 'bombcell':
        label = label.upper()     
    
        soma = cluster_info[((cluster_info.bc_unitType == label) &
                                              (cluster_info.depth >= SC_depth_limits[1]) &
                                              (cluster_info.depth <= SC_depth_limits[0]))].index.values
    
        #Find non somatic units
        nonSoma = find_units_bomcell(dataDirectory, 'NON-SOMA')
        nonSoma = cluster_info.loc[nonSoma][((cluster_info.depth >= SC_depth_limits[1]) &
                                      (cluster_info.depth <= SC_depth_limits[0]))].index.values
    
    n['cluster_analysis'] = soma
    #n['cluster_analysis'] = np.vstack((soma,nonSoma))
    
    neural.append(n)

# Params

#PSTH params
baseline = 0.2
stimDur = 2
prePostStim = 1
window = np.array([-prePostStim, stimDur + prePostStim])
binSize=0.005
sigma = 0.06
alpha_val = 0.4
groups = np.unique(stim[0]['direction'])
colors = cm.rainbow(np.linspace(0, 1, len(groups)))
xticks = np.arange(-1, 4, 1)

#define windows for determine if neuron is visually responsive
no_stim_window = (-1,0)
stim_window = (0,1)
#stim_window_sensitization = (1.5,2)

#Behaviour params
threshold = 0.9 #for running velocity

#adaptation, fr and supression  params
window_response = np.array([0, 2])

# Define behavior

data = []

for st in stim:

    running_state = np.array([])    

    active = 0
    quiet = 0
    not_considered = 0
    
    stimuli_start = st['startTime'].reshape(-1)
    stimuli_end = st['endTime'].reshape(-1)
    ard_time_adjust = st['wheelTimestamps'].reshape(-1)
    velocity = st['wheelVelocity'].reshape(-1)
    
    for start, end in zip(stimuli_start, stimuli_end):
        
        interval_velocity = velocity[np.where((ard_time_adjust >= start) &
                                              (ard_time_adjust <= end))[0]]
        if sum(interval_velocity < 1) > int(len(interval_velocity) * threshold):
            state = 0
            quiet += 1
        elif sum(interval_velocity >= 1) > int(len(interval_velocity) * threshold):
            state = 1
            active += 1 
        else:
            state = np.nan
            not_considered += 1 
        
        running_state = np.hstack((running_state,state))    
        
    st['running_state'] = running_state

# Compute visual responsiveness


for n,st in zip(neural,stim):    

    clusters = n['cluster_analysis']
    spike_times = n['spike_times']
    spike_clusters = n['spike_clusters']
    stimuli_start = st['startTime'].reshape(-1)
    direction = st['direction'].reshape(-1)
    
    visual = []
    
    for neuron in clusters:
        
        #Calculate vr using info from one state (because of differences in baseline)
        #Use state with higher response
        
        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                            stimuli_start,window)
        #Calculate vr in early response
        r_before, r_after = visual_responsiveness(spAligned,
                                          trials,
                                          no_stim_window,
                                          stim_window,
                                          direction,
                                          baseline)

        res = permutation_test((r_before, r_after), vr_statistic, permutation_type = 'samples',
                                vectorized=True, n_resamples=5000, alternative='two-sided')
                
        if res.pvalue < 0.05:
        
            #Calculate vr in late response (for sensitizing neurons)
            # r_before, r_after = visual_responsiveness(spAligned,
            #                                   trials,
            #                                   no_stim_window,
            #                                   stim_window_sensitization,
            #                                   direction,
            #                                   baseline)

            # res = permutation_test((r_before, r_after), vr_statistic, permutation_type = 'samples',
            #                         vectorized=True, n_resamples=5000, alternative='two-sided')
            # if res.pvalue <= 0.05:
                
            #     visual.append(False)
            # else:
            #     #Test if the firing rate is low
            fr = np.mean(calculate_fr_per_trial(spAligned, trials, window_response, 
                                        direction, baseline))
            if abs(fr) < 0.15:
                visual.append(False)
            else:
                visual.append(True)
            
        else:
            visual.append(False)
   
    #line for selecting and checking the non-visual            
    #visual = [not val for val in visual]     
    n['visual'] = visual     


# %% Load gratings data

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

# Params


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

# Histogram and scatter plots per depth
palette = {'sSC': 'mediumturquoise', 'dSC': 'crimson'}
depth_cut = 40  #split data between superficial vs deep SC (in %)

#Tuning curves
tuning_threshold = 0.7


# Define behavior and calculate visual responsiveness

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
#%% batch raster and psth 

i = 0

for n,st in zip(neural, stim):    

    clusters = n['cluster_analysis'][n['visual']]
    spike_times = n['spike_times']
    spike_clusters = n['spike_clusters']
    
    mask = (st['temporalF'] == 2).reshape(-1)
    stimuli_start = st['startTime'][mask].reshape(-1)
    stimuli_end = st['endTime'][mask].reshape(-1)
    direction = st['direction'][mask].reshape(-1)
    running_state = st['running_state'][mask].reshape(-1)
   
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
               
        saveDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, 'rasterPSTH')
        if not os.path.isdir(saveDirectory):
            os.makedirs(saveDirectory)
        filename = os.path.join(saveDirectory, f'{neuron}_PSTH.png')
        plt.savefig(filename)
        plt.close()
        
    i += 1
    
# %% parameters for trial alignment 

tmin = 0.4
tmax = 2

NBINS = 160        # Number of time bins per trial (time window btwn tmin and tmax)
SMOOTH_REG = 10   # Strength of roughness penalty
WARP_REG = 0.0      # Strength of penalty on warp magnitude
L2_REG = 0.0        # Strength of L2 penalty on template magnitude
MAXLAG = 0.5       # Maximum amount of shift allowed.

trials_to_include = 'strong'

# %% removing low activity neurons and finding strong response directions

filtered_clusters = {}

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
            trial_indices, pref_dirs, _, _ = strong_response_trials(direction,fr,tuning_threshold,is_suppressed)
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
        
        filtered_clusters[neuron_id] = pref_dirs

# %% package data per direction (all neurons - all trials - one direction)


desired_directions = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]

packaged_spike_data_all_dirs = {}

for dir_id in desired_directions:
    
    spAligned_list = [] # will be list of arrays
    trials_list = [] # will be list of arrays
    neuron_ids_list = []
    
    i = 0
    
    for n, st in zip(neural, stim):
        clusters = list(filtered_clusters.keys())
        spike_times = n['spike_times']
        spike_clusters = n['spike_clusters']
         
        stimuli_start = st['startTime'].reshape(-1)
        stimuli_end = st['endTime'].reshape(-1)
        direction = st['direction'].reshape(-1)
        running_state = st['running_state'].reshape(-1)
        dataEntry = database.loc[i]
        
        # clusters = [741, 743, 745, 746, 753, 771, 773, 778]
        
        for neuron_index, neuron_id in enumerate(clusters):
            identifier=f'{dataEntry.Name}_{dataEntry.Date}_{neuron_id}'
            print(f'Packaging neuron {identifier} - direction {dir_id}')
            # Mask to filter trials for the desired direction
            direction_mask = (direction == dir_id)
    
            spAligned, trials = alignData( # total trials per state
                spike_times[spike_clusters == neuron_id],
                stimuli_start[direction_mask], 
                window)
            
            # Append data to lists
            spAligned_list.append(spAligned)
            trials_list.append(trials)
            neuron_ids_list.extend([neuron_index] * len(spAligned))
            
        
    spAligned_flat = np.concatenate(spAligned_list, axis=0)
    trials_flat = np.concatenate(trials_list, axis=0)
        
    spAligned_array = np.array(spAligned_flat, dtype=float)  
    trials_array = np.array(trials_flat, dtype=int)         
    neuron_ids_array = np.array(neuron_ids_list, dtype=int)
    clusters_packaged = clusters
    clusters_idx = list(range(len(clusters)))
    
    packaged_spike_data = SpikeData(
        trials_array,
        spAligned_array,
        neuron_ids_array,
        tmin=tmin,
        tmax=tmax,
    )
    
    packaged_spike_data_all_dirs[dir_id] = packaged_spike_data
    
    

# %% fit the model   

# Specify model.
shift_model = ShiftWarping(
    maxlag=MAXLAG,
    smoothness_reg_scale=SMOOTH_REG,
    warp_reg_scale=WARP_REG,
    l2_reg_scale=L2_REG,
)

validated_alignments = {}

for dir_id, packaged_spike_data in packaged_spike_data_all_dirs.items():
    # Fit and apply warping to held out neurons.
    validated_alignment = heldout_transform(
        shift_model, packaged_spike_data.bin_spikes(NBINS), packaged_spike_data, iterations=100)
    
    validated_alignments[dir_id] = validated_alignment


    # template = shift_model.template
    # plt.figure()
    # plt.plot(template)
    
    # binned_spikes = our_data.bin_spikes(NBINS)
    # plt.figure()
    # plt.plot(binned_spikes[0,:,0]) 
    
    # Fit model to full dataset (used to align sniffs).
    # shift_model.fit(our_data.bin_spikes(NBINS))

# %% plot columns (raw, sorted, aligned) per direction

example_neurons = [0,1,2,3,4,5,6,7,8,9] 
# example_neurons = [10,11,12,13,14,15,16,17,18,19]#191
# example_neurons = [20,21,22,23,24,25,26,27,28,29] #215
# example_neurons = [30,31,32,33,34,35,36,37,38,39] 
# example_neurons = [40,41,42,43,44,45,46] 

for key in packaged_spike_data_all_dirs:
    packaged_spike_data = packaged_spike_data_all_dirs[key]
    validated_alignment = validated_alignments[key]
    # Create figure.
    fig, axes = plt.subplots(10, 3, figsize=(9.5, 8))
    fig.suptitle(f'Neurons: {example_neurons} Direction: {key}')

    # First column, raw data.
    _plot_column(
        axes[:, 0],
        packaged_spike_data,
        example_neurons
    )
    
    # Second column, re-sorted trials by warping function.
    _plot_column(
        axes[:, 1],
        packaged_spike_data.reorder_trials(shift_model.argsort_warps()),
        example_neurons
    )
    
    # Third column, shifted alignment.
    _plot_column(
        axes[:, 2],
        validated_alignment,
        example_neurons
    )
    
    # Final formatting.
    for ax in axes.ravel():
        ax.set_xlim(-0.5, 3)
    # for ax in axes[-1]:
    #     ax.set_xlabel("time (ms)")
    
    axes[0, 0].set_title("raw data")
    axes[0, 1].set_title("sorted by warp")
    axes[0, 2].set_title("aligned by model")
    
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=.3)

# %% extract shifted spikes
        
saved_neurons = clusters_idx
shifted_spikes_all_dirs = {}
original_spikes_all_dirs = {}

for (dir_id, validated_alignment), (_, packaged_spike_data) in zip(validated_alignments.items(), packaged_spike_data_all_dirs.items()):
    shifted_spikes = save_shifted_spikes(validated_alignment, saved_neurons)
    original_spikes = save_shifted_spikes(packaged_spike_data, saved_neurons)
    
    shifted_spikes_all_dirs[dir_id] = shifted_spikes
    original_spikes_all_dirs[dir_id] = original_spikes

# %% plot raster and psth (before and after shift) 

#TODO: needs to have a plotting for all dirs

smoothed_psths_original = {}

for neuron_id, data in original_spikes.items():
    spiketimes = data['spiketimes']
    trials = data['trials']
    
    # Define the analysis window and parameters
    stimDur = 2
    prePostStim = 1
    window = np.array([-prePostStim, stimDur + prePostStim])
    bin_size = 0.05  # Bin size for PSTH
    sigma = 0.06  # Smoothing parameter
    baseline = 0.2  # Baseline duration for correction

    # Compute PSTH for the neuron
    bins, smoothed_psth = compute_psth_with_baseline_correction(
        spiketimes, trials, window, bin_size=bin_size, sigma=sigma, baseline=baseline
    )

    smoothed_psths_original[neuron_id] = smoothed_psth
    # Save path for the figure
    # save_path = rf"Q:\Analysis\Figures\{dataEntry.Name}/{dataEntry.Date}/27-01-25/trial_shifts\before_shift" 
    save_path = None
    # Plot combined raster and PSTH
    plot_raster_and_psth(
        neuron_id,
        spiketimes,
        trials,
        bins,
        smoothed_psth,
        window,
        save_path=save_path
    )
    
# Initialize a dictionary to store smoothed PSTH for all neurons
smoothed_psths_shifted = {}
    
for neuron_id, data in shifted_spikes.items():
    spiketimes = data['spiketimes']
    trials = data['trials']
    
    # Define the analysis window and parameters
    stimDur = 2
    prePostStim = 1
    window = np.array([-prePostStim, stimDur + prePostStim])
    bin_size = 0.05  # Bin size for PSTH
    sigma = 0.06  # Smoothing parameter
    baseline = 0.2  # Baseline duration for correction

    # Compute PSTH for the neuron
    bins, smoothed_psth = compute_psth_with_baseline_correction(
        spiketimes, trials, window, bin_size=bin_size, sigma=sigma, baseline=baseline
    )
    
    # Inside your loop, after computing the smoothed PSTH:
    smoothed_psths_shifted[neuron_id] = smoothed_psth

    # Save path for the figure
    # save_path = rf"Q:\Analysis\Figures\{dataEntry.Name}/{dataEntry.Date}/27-01-25/trial_shifts\after_shift"
    save_path = None

    # Plot combined raster and PSTH
    plot_raster_and_psth(
        neuron_id,
        spiketimes,
        trials,
        bins,
        smoothed_psth,
        window,
        save_path=save_path
    )


# %% restructure data based on neurons 

spShifted_all_dirs_all_neurons = {}
for neuron_id in saved_neurons: 
    dataEntry = database.loc[i]
    identifier=f'{dataEntry.Name}_{dataEntry.Date}_{neuron_id}'
    print(f'Processing neuron {identifier}')
    
    spShifted_all_dirs = {}
    for dir_idx, shifted_spikes_dir in shifted_spikes_all_dirs.items():
        all_neurons = shifted_spikes_dir
        one_neuron = all_neurons[neuron_id]
        spShifted_all_dirs[dir_idx] = {
            'spiketimes': one_neuron['spiketimes'], # all directions one neuron
            'trials': one_neuron['trials']
            }
    
    spShifted_all_dirs_all_neurons[neuron_id] = spShifted_all_dirs
 
# %% use strong directions 

spShifted_strong_dirs_all_neurons = {}

for i, (key, value) in enumerate(filtered_clusters.items()):
    one_neuron_data = spShifted_all_dirs_all_neurons[i]
    spShifted_strong_dirs = {k: v for k, v in one_neuron_data.items() if k in value}
    
    spShifted_strong_dirs_all_neurons[i] = spShifted_strong_dirs
    
# %% oscillations params 

stimulus_freq = 2 #Hz
f1_f0_threshold = 0.4
tuning_threshold = 0.4

# %% plot fft boolean
# plot_fft = True
plot_fft = False   
# %% find oscillating neurons using strong directions

oscillating_neurons_strong = []
f1_f0_values_strong = {}  # Dictionary to store F1/F0 per neuron
traces_oscillating_neurons = {}

# if specific neurons:
# neurons_of_interest = {356} 
# neurons_filter = [index for index, cluster in enumerate(clusters_packaged) if cluster in neurons_of_interest]

# if all neurons:
neurons_of_interest = clusters
neurons_filter = clusters_idx # if all neurons 


spShifted_neurons_of_int = {key: spShifted_strong_dirs_all_neurons[key] for key in neurons_filter if key in spShifted_strong_dirs_all_neurons}

for cluster_id, (neuron_id, neuron_data) in zip(neurons_of_interest, spShifted_neurons_of_int.items()):
    identifier=f'{dataEntry.Name}_{dataEntry.Date}_{neuron_id}'
    print(f'Processing neuron {identifier}')
    traces_dict = {}
    for dir_id, neuron_data_dir in neuron_data.items():
        spShifted_dir = neuron_data_dir['spiketimes']
        trialsShifted_dir = neuron_data_dir['trials']
        direction_mask = direction == dir_id
    
        traces, sem, timeBins = newTracesFromPSTH(spShifted_dir, trialsShifted_dir, window, direction[direction_mask],
                                              groups, baseline=0,  binSize=0.0005, sigma=sigma)
        dir_idx = np.where(groups == dir_id)[0][0]
        traces_dict[dir_id] = traces[dir_idx] 
        # plt.plot(traces[dir_idx])
        
    f1_f0_values_strong[identifier] = {}
    # Compute F1/F0 per direction
    for dir_id, trace in traces_dict.items():
        if not np.any(np.isnan(trace)):  # Ensure the trace does not contain NaNs
            # trace = - trace
            _, f1_f0 = compute_f1_f0(identifier, trace, stimulus_freq, timeBins, threshold=f1_f0_threshold, plot=plot_fft)
            dir_key = f'Direction_{dir_id}'
            f1_f0_values_strong[identifier][dir_key] = f1_f0  # Store per direction
        
    # Decide if the neuron is oscillatory
    mean_f1_f0 = np.median(list(f1_f0_values_strong[identifier].values()))  # Average across directions
    if mean_f1_f0 > f1_f0_threshold:
        oscillating_neurons_strong.append(cluster_id)
        traces_oscillating_neurons[cluster_id] = traces_dict
        
    
#     i+=1    

# %% Filter design


# Calculate sampling frequency from time bins
time_diffs = np.diff(timeBins)  # Calculate time differences between consecutive points
fs = 1 / np.mean(time_diffs)  # Sampling frequency is the reciprocal of the mean time difference

# Filter specifications
cutoff = [1, 3]  # Desired cutoff frequencies in Hz
bandpass = cutoff[1]-cutoff[0]
ripple_db = 60  # Desired stopband attenuation in dB - 30 -> attentuated to 1/30th of original amplitude
width = 2/fs   # 
width = 0.002
# width = 0.1*bandpass # Transition width - 10% of passband width ('cutoff') - numtaps become too low 
numtaps, beta = kaiserord(ripple_db, width) # essentially the length of the filter 
numtaps = 5500
print(f"Calculated number of taps: {numtaps}")
group_delay = numtaps / (2 * fs) # how much the delay would be in seconds
print(f'group delay: {group_delay}')

# Generate filter coefficients using the Kaiser window
fir_coeff = firwin(numtaps, cutoff, window=('kaiser', beta), pass_zero=False, fs=fs)

# Calculate frequency response of the filter
w, h = freqz(fir_coeff, worN=8000, fs=fs)

# Plot the frequency response
plt.figure()
plt.plot(w, abs(h), label=f'Kaiser window numtaps={numtaps}')
plt.title('Frequency Response with Kaiser Window')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.xlim(0, 10)
plt.grid(True)
plt.legend(loc='best')
plt.show()

# %% Filtering dir traces

# if specific neurons:
neurons_of_interest = {368}
#if all neurons:
# neurons_of_interest = oscillating_neurons_strong


traces_oscillating_neurons_of_int = {k: v for k, v in traces_oscillating_neurons.items() if k in neurons_of_interest}

filtered_traces_all = {}
for cluster_id, traces in traces_oscillating_neurons_of_int.items():
    filtered_traces = {}
    for dir_id, trace in traces.items():
        filtered_response = filter_F1(cluster_id, timeBins, trace, sigma, fir_coeff, pad_length = 6000, plot = True)
        filtered_traces[dir_id] = filtered_response
    filtered_traces_all[cluster_id] = filtered_traces
    
        
 


# %% Data for filtering old
# Define the analysis window and parameters
stimDur = 2
prePostStim = 0.5
window = np.array([-prePostStim, stimDur + prePostStim])
bin_size = 0.0005  # Bin size for PSTH
sigma = 0.06  # Smoothing parameter
baseline = 0.2  # Baseline duration for correction
    
smoothed_psths_original = {}

for neuron_id, data in original_spikes.items():
    spiketimes = data['spiketimes']
    trials = data['trials']
    

    # Compute PSTH for the neuron
    bins, smoothed_psth = compute_psth_with_baseline_correction(
        spiketimes, trials, window, bin_size=bin_size, sigma=sigma, baseline=baseline
    )

    smoothed_psths_original[neuron_id] = smoothed_psth
    
# Initialize a dictionary to store smoothed PSTH for all neurons
smoothed_psths_shifted = {}
    
for neuron_id, data in shifted_spikes_all_dirs.items():
    spiketimes = data['spiketimes']
    trials = data['trials']
    
    # Compute PSTH for the neuron
    bins, smoothed_psth = compute_psth_with_baseline_correction(
        spiketimes, trials, window, bin_size=bin_size, sigma=sigma, baseline=baseline
    )
    
    # Inside your loop, after computing the smoothed PSTH:
    smoothed_psths_shifted[neuron_id] = smoothed_psth
    
    

timeBins = bins
neuron_id = 8
psth_trace = smoothed_psths_shifted[neuron_id]
plt.figure()
plt.plot(psth_trace)



# %% find time shifts per trial

binsize_shift_model = (tmax-tmin)/NBINS
shifts = shift_model.shifts
shifts_s = shifts*binsize_shift_model


# checking the difference between original and shifted spikes to confirm
# it seems that they are not always exactly as the binsize suggests - difference of 0.01 here and there and in different neurons
# but consistent per trial per in a given neuron
original_spikes_1 = original_spikes[1]['spiketimes']
shifted_spikes_1 = shifted_spikes[1]['spiketimes']
trials_1 = original_spikes[1]['trials']

diff_spikes_1 = original_spikes_1 - shifted_spikes_1



# %% subtract F1 from original PSTH

pad_value = 639
padded_F1 = np.pad(filtered_data, pad_width=pad_value, mode='constant', constant_values=0)
F1_trial = padded_F1/20

original_psth_nonshift = smoothed_psths_original[neuron_id]

binsize_signal = timeBins[-1]-timeBins[-2]

F1_trial_shifted = np.zeros((len(shifts_s), len(F1_trial)))  # Placeholder for all shifted versions

for i, shift in enumerate(shifts_s):
    shift_bins = int(shift / binsize_signal)  # Convert shift in seconds to bins
    # shift_bins = - shift_bins
    if shift_bins >= 0:
        F1_trial_shifted[i, shift_bins:] = F1_trial[:len(F1_trial) - shift_bins]
    else:
        F1_trial_shifted[i, :shift_bins] = F1_trial[-shift_bins:]

F1_subtracted_signal = original_psth_nonshift.copy()

# Subtract each trial's shifted signal from the psth_trace
for i in range(len(shifts_s)):
    F1_subtracted_signal -= F1_trial_shifted[i]



plt.figure()
plt.plot(timeBins, original_psth_nonshift, label = 'Original')
plt.plot(timeBins, padded_F1, label = 'F1')
plt.plot(timeBins, F1_trial, label = 'F1 trial no shift')
plt.plot(timeBins, F1_trial_shifted[2], label = 'F1 trial shift 0.14 s')
plt.plot(timeBins, F1_subtracted_signal, label = 'F1 subtracted')
plt.legend()


plt.figure()
plt.plot(timeBins, F1_trial, label = 'F1 trial no shift', linewidth=5, color='black')
for idx, trial in enumerate(F1_trial_shifted, start=1):
    plt.plot(timeBins, trial, label=f'Trial {idx}')
plt.legend()

# %% sliced plot

pad_value = 850
# Define slicing indices
slice_start = pad_value
slice_end = -pad_value if pad_value > 0 else None  # Avoid issues if pad_value is 0

# Slice the relevant arrays
F1_trial_sliced = F1_trial[slice_start:slice_end]
F1_trial_shifted_sliced = F1_trial_shifted[:, slice_start:slice_end]
F1_subtracted_signal_sliced = F1_subtracted_signal[slice_start:slice_end]
timeBins_sliced = timeBins[slice_start:slice_end]
filtered_data_sliced = filtered_data[211:-211]
# Plot sliced versions
plt.figure()
plt.plot(timeBins_sliced, original_psth_nonshift[slice_start:slice_end], label='Original', linewidth=2)
plt.plot(timeBins_sliced, filtered_data_sliced, label = 'F1', linewidth=2)
plt.plot(timeBins_sliced, F1_trial_sliced, label='F1 trial no shift', linewidth=2)
plt.plot(timeBins_sliced, F1_trial_shifted_sliced[2], label='F1 trial shift 0.14 s', linewidth=2)
plt.plot(timeBins_sliced, F1_subtracted_signal_sliced, label='F1 subtracted', linewidth=2)
plt.title('F1 subtraction')
plt.legend()

plt.figure()
plt.plot(timeBins_sliced, F1_subtracted_signal_sliced, label='F1 subtracted', linewidth=2)
plt.title('F1 subtracted response')
# %% F1 subtracted directly

F1_subtracted_wo_trial = original_psth_nonshift - padded_F1

F1_subtracted_wo_trial=F1_subtracted_wo_trial[211:-211]
plt.figure()
# plt.plot(timeBins_sliced, original_psth_nonshift[slice_start:slice_end], label = 'Original')
# plt.plot(timeBins, padded_F1, label = 'F1')
plt.plot(timeBins_sliced, F1_subtracted_wo_trial, label = 'F1 subtracted directly')
# plt.plot(timeBins, F1_subtracted_signal, label = 'F1 subtracted each trial')

plt.legend()

# %% find oscillating neurons using all directions

oscillating_neurons = []
f1_f0_values = {}  # Dictionary to store F1/F0 per neuron

# if specific neurons:
# neurons_of_interest = {244} 
# neurons_filter = [index for index, cluster in enumerate(clusters_packaged) if cluster in neurons_of_interest]

# if all neurons:
neurons_of_interest = clusters
neurons_filter = clusters_idx # if all neurons 
spShifted_neurons_of_int = {key: spShifted_all_dirs_all_neurons[key] for key in neurons_filter if key in spShifted_all_dirs_all_neurons}

for cluster_id, (neuron_id, neuron_data) in zip(neurons_of_interest, spShifted_neurons_of_int.items()):
    identifier=f'{dataEntry.Name}_{dataEntry.Date}_{neuron_id}'
    print(f'Processing neuron {identifier}')
    traces_dict = {}
    i=0
    for dir_idx, neuron_data_dir in neuron_data.items():
        spShifted_dir = neuron_data_dir['spiketimes']
        trialsShifted_dir = neuron_data_dir['trials']
        direction_mask = direction == dir_idx
    
        traces, sem, timeBins = newTracesFromPSTH(spShifted_dir, trialsShifted_dir, window, direction[direction_mask],
                                              groups, baseline=0,  binSize=binSize, sigma=sigma)
        traces_dict[dir_idx] = traces[i]
        # plt.plot(traces[0])
        i += 1
        
    f1_f0_values[identifier] = {}
    # Compute F1/F0 per direction
    for dir_idx, trace in traces_dict.items():
        if not np.any(np.isnan(trace)):  # Ensure the trace does not contain NaNs
            _, f1_f0 = compute_f1_f0(identifier, trace, stimulus_freq, timeBins, threshold=f1_f0_threshold, plot=plot_fft)
            dir_key = f'Direction_{dir_idx}'
            f1_f0_values[identifier][dir_key] = f1_f0  # Store per direction
        
    # Decide if the neuron is oscillatory
    mean_f1_f0 = np.median(list(f1_f0_values[identifier].values()))  # Average across directions
    if mean_f1_f0 > f1_f0_threshold:
        oscillating_neurons.append(cluster_id)
        
    


# %% find oscillating neurons old

oscillating_neurons = []
f1_f0_values = {}  # Dictionary to store F1/F0 per neuron
low_fr_neurons = []

i=0
 
for n, st in zip(neural, stim):
    clusters = n['cluster_analysis'][n['visual']]
    # clusters = [719, 741, 743, 753, 771, 773, 778, 1064, 1107]
    clusters = [263]
    spike_times = n['spike_times']
    spike_clusters = n['spike_clusters']
     
    mask = (st['temporalF'] == stimulus_freq).reshape(-1)
    stimuli_start = st['startTime'][mask].reshape(-1)
    stimuli_end = st['endTime'][mask].reshape(-1)
    direction = st['direction'][mask].reshape(-1)
    running_state = st['running_state'][mask].reshape(-1)
    dataEntry = database.loc[i]
    
    
    for neuron_index, neuron_id in enumerate(clusters):
        identifier=f'{dataEntry.Name}_{dataEntry.Date}_{neuron_id}'
        print(f'Processing neuron {identifier}')
        # Specify the direction of interest

        spAligned, trials = alignData( # total trials per state
            spike_times[spike_clusters == neuron_id],
            stimuli_start, 
            window)
        binned, timeBins = newTracesFromPSTH_per_trial(spAligned, trials, window, direction, 
                                                    groups, baseline, binSize, sigma)
        is_suppressed = analyze_suppression(np.mean(binned, axis=0), timeBins)
        
        # check low activity (>70% trials with no spikes)
        is_low_activity = exclude_low_activity(trials, (running_state == 0) | (running_state == 1))
        if is_low_activity:
            continue
        
        fr = calculate_fr_per_trial( 
            spAligned, trials, window_response, direction, baseline)
        trial_indices, pref_dirs, _, _ = strong_response_trials(direction,fr,tuning_threshold,is_suppressed)
        
        traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[trial_indices],
                                              groups, baseline,  binSize, sigma)
        if is_suppressed:
            traces = -traces
        
        #TODO: mean_psth should be within the stimulus window
        mean_psth = np.nanmean(traces, axis=0)
        if np.mean(mean_psth) < 1:
            low_fr_neurons.append(identifier)
            continue
        
        
        f1_f0_values[identifier] = {}

        # Compute F1/F0 per direction
        for dir_idx, trace in enumerate(traces):
            if not np.any(np.isnan(trace)):  # Ensure the trace does not contain NaNs
                _, f1_f0 = compute_f1_f0(identifier, trace, stimulus_freq, timeBins, threshold=f1_f0_threshold, plot=plot_fft)
                dir_key = f'Direction_{dir_idx}'
                f1_f0_values[identifier][dir_key] = f1_f0  # Store per direction
            
        # Decide if the neuron is oscillatory
        mean_f1_f0 = np.median(list(f1_f0_values[identifier].values()))  # Average across directions
        if mean_f1_f0 > f1_f0_threshold:
            oscillating_neurons.append(identifier)
    
    i+=1
            


    