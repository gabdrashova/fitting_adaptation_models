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
from Functions.shift_warping import _plot_column
from Functions.behavioural import *
from Functions.time_constant import *
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
import sys
sys.path.insert(0, r'C:\dev\workspaces/affinewarp')
from affinewarp.shiftwarp import ShiftWarping
from affinewarp.shiftwarp import ShiftWarping
from affinewarp.spikedata import SpikeData
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
    
# %% plot saving directory and other params

analysisDir = os.path.join(analysisDir, '09-04-2025')    
trials_to_include = 'strong'
# %% batch raster and psth 

i = 0

for n,st in zip(neural, stim):    

    clusters = n['cluster_analysis'][n['visual']]
    # clusters = [366, 367, 368]
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
    filtered_clusters[i] = {}
    
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
        
        filtered_clusters[i][neuron_id] = pref_dirs
    i+=1

# %% package data per direction (all neurons - all trials - one direction)


desired_directions = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]

packaged_spike_data_all_dirs = {}
clusters_idx = {}
i = 0
for n, st in zip(neural, stim):
    packaged_spike_data_all_dirs[i] = {}
    clusters_idx[i] = {}

    for dir_id in desired_directions:
        
        spAligned_list = [] # will be list of arrays
        trials_list = [] # will be list of arrays
        neuron_ids_list = []
        
        for n, st in zip(neural, stim):
            clusters = list(filtered_clusters[i].keys())
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
        clusters_idx[i] = list(range(len(clusters)))
        
        packaged_spike_data = SpikeData(
            trials_array,
            spAligned_array,
            neuron_ids_array,
            tmin=tmin,
            tmax=tmax,
        )
        
        packaged_spike_data_all_dirs[i][dir_id] = packaged_spike_data
        
    i+=1
    
    

# %% fit the model, find the time shifts  

tmin = 0.4
tmax = 2

NBINS = 160        # Number of time bins per trial (time window btwn tmin and tmax)
SMOOTH_REG = 10   # Strength of roughness penalty
WARP_REG = 0.0      # Strength of penalty on warp magnitude
L2_REG = 0.0        # Strength of L2 penalty on template magnitude
MAXLAG = 0.3       # Maximum amount of shift allowed.



# Specify model.
shift_model = ShiftWarping(
    maxlag=MAXLAG,
    smoothness_reg_scale=SMOOTH_REG,
    warp_reg_scale=WARP_REG,
    l2_reg_scale=L2_REG,
)
binsize_shift_model = (tmax-tmin)/NBINS

validated_alignments = {}
time_shifts = {}

for i, packaged_spike_data_all_dirs_one_dataset in packaged_spike_data_all_dirs.items():
    validated_alignments[i] = {}
    time_shifts[i] = {}
    for dir_id, packaged_spike_data in packaged_spike_data_all_dirs_one_dataset.items():
        # Fit and apply warping to held out neurons.
        validated_alignment = heldout_transform(
            shift_model, packaged_spike_data.bin_spikes(NBINS), packaged_spike_data, iterations=100)
        
        validated_alignments[i][dir_id] = validated_alignment
        time_shifts[i][dir_id] = shift_model.shifts*binsize_shift_model
    i+=1
        
        
        # shifts = shift_model.shifts
        # shifts_s = shifts*binsize_shift_model
    
    
        # template = shift_model.template
        # plt.figure()
        # plt.plot(template)
        
        # binned_spikes = our_data.bin_spikes(NBINS)
        # plt.figure()
        # plt.plot(binned_spikes[0,:,0]) 
    


# %% plot columns (raw, sorted, aligned) per direction

example_neurons = [0,1,2,3,4,5,6,7,8,9] 
# example_neurons = [10,11,12,13,14,15,16,17,18,19]#191
# example_neurons = [20,21,22,23,24,25,26,27,28,29] #215
# example_neurons = [30,31,32,33,34,35,36,37,38,39] 
# example_neurons = [40,41,42,43,44,45,46] 
for (i, packaged_spike_data_all_dirs_one_dataset), validated_alignments_one_dataset in zip(packaged_spike_data_all_dirs.items(), validated_alignments.values()):
    for key in packaged_spike_data_all_dirs_one_dataset:
        packaged_spike_data = packaged_spike_data_all_dirs_one_dataset[key]
        validated_alignment = validated_alignments_one_dataset[key]
        # Create figure.
        fig, axes = plt.subplots(10, 3, figsize=(9.5, 8))
        fig.suptitle(f'Dataset {i} Neurons: {example_neurons} Direction: {key}')
    
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

# %% extract shifted spikes and original spikes
        
saved_neurons = clusters_idx
shifted_spikes_all_dirs = {}
original_spikes_all_dirs = {}

i=0
for (validated_alignments_one_dataset, packaged_spike_data_all_dirs_one_dataset) in zip(validated_alignments.values(), packaged_spike_data_all_dirs.values()):
    shifted_spikes_all_dirs[i] = {}
    original_spikes_all_dirs[i] = {}
    for (dir_id, validated_alignment), (_, packaged_spike_data) in zip(validated_alignments_one_dataset.items(), packaged_spike_data_all_dirs_one_dataset.items()):
        shifted_spikes = save_shifted_spikes(validated_alignment, saved_neurons[i])
        original_spikes = save_shifted_spikes(packaged_spike_data, saved_neurons[i])
        
        shifted_spikes_all_dirs[i][dir_id] = shifted_spikes
        original_spikes_all_dirs[i][dir_id] = original_spikes
    i+=1

# %% restructure data based on neurons 

spShifted_all_dirs_all_neurons = {}
spOrig_all_dirs_all_neurons = {}

for i in shifted_spikes_all_dirs.keys():
    spShifted_all_dirs_all_neurons[i] = {}
    spOrig_all_dirs_all_neurons[i] = {}
    
    for neuron_id in saved_neurons[i]: 
        dataEntry = database.loc[i]
        identifier=f'{dataEntry.Name}_{dataEntry.Date}_{neuron_id}'
        print(f'Processing neuron {identifier}')

        
        spShifted_all_dirs = {}
        for dir_idx, shifted_spikes_dir in shifted_spikes_all_dirs[i].items():
            all_neurons = shifted_spikes_dir
            one_neuron = all_neurons[neuron_id]
            spShifted_all_dirs[dir_idx] = {
                'spiketimes': one_neuron['spiketimes'], # all directions one neuron
                'trials': one_neuron['trials']
                }
        
        spShifted_all_dirs_all_neurons[i][neuron_id] = spShifted_all_dirs
        
        spOrig_all_dirs = {}
        for dir_idx, original_spikes_dir in original_spikes_all_dirs[i].items():
            all_neurons = original_spikes_dir
            one_neuron = all_neurons[neuron_id]
            spOrig_all_dirs[dir_idx] = {
                'spiketimes': one_neuron['spiketimes'], # all directions one neuron
                'trials': one_neuron['trials']
                }
        
        spOrig_all_dirs_all_neurons[i][neuron_id] = spOrig_all_dirs
        
    

    
 
# %% use strong directions for shifted and original spikes and time shifts

spShifted_strong_dirs_all_neurons = {}
spOrig_strong_dirs_all_neurons = {}
for i in filtered_clusters.keys():
    
    spShifted_strong_dirs_all_neurons[i] = {}
    spOrig_strong_dirs_all_neurons[i] = {}
    for n, (key, value) in enumerate(filtered_clusters[i].items()):
        one_neuron_data = spShifted_all_dirs_all_neurons[i][n]
        spShifted_strong_dirs = {k: v for k, v in one_neuron_data.items() if k in value}
        spShifted_strong_dirs_all_neurons[i][key] = spShifted_strong_dirs
        
        one_neuron_data = spOrig_all_dirs_all_neurons[i][n]
        spOrig_strong_dirs = {k: v for k, v in one_neuron_data.items() if k in value}
        spOrig_strong_dirs_all_neurons[i][key] = spOrig_strong_dirs
    
        
# %% separating spikes and traces by state


# Define neurons of interest for each dataset individually
neurons_of_interest_by_dataset = {
    0: filtered_clusters[0].keys(),
    1: filtered_clusters[1].keys(),
    2: filtered_clusters[2].keys(),
    3: filtered_clusters[3].keys()
}

# Initialize nested dictionaries
spShifted_neurons_of_int = {}
spOrig_neurons_of_int = {}

for ds, neurons in neurons_of_interest_by_dataset.items():
    spShifted_neurons_of_int[ds] = {}
    spOrig_neurons_of_int[ds] = {}

    for neuron_id in neurons:
        if neuron_id in spShifted_strong_dirs_all_neurons[ds]:
            spShifted_neurons_of_int[ds][neuron_id] = spShifted_strong_dirs_all_neurons[ds][neuron_id]
        if neuron_id in spOrig_strong_dirs_all_neurons[ds]:
            spOrig_neurons_of_int[ds][neuron_id] = spOrig_strong_dirs_all_neurons[ds][neuron_id]


shifted_data_by_state_dict = {}
orig_data_by_state_dict = {}
for dataset in spShifted_neurons_of_int.keys():
    spShifted_neurons_of_int_one_ds = spShifted_neurons_of_int[dataset]
    spOrig_neurons_of_int_one_ds = spOrig_neurons_of_int[dataset]
    shifted_data_by_state_dict[dataset] = {}
    orig_data_by_state_dict[dataset] = {}
    for (neuron_idx, shifted_neuron_data), (_, original_neuron_data) in zip(spShifted_neurons_of_int_one_ds.items(), spOrig_neurons_of_int_one_ds.items()):
        dataEntry = database.loc[dataset]
        identifier = f'{dataEntry.Name}_{dataEntry.Date}_{neuron_idx}'
        print(f'Processing neuron {identifier}')
    
        shifted_traces_dict_active = {}
        shifted_traces_dict_quiet = {}
    
        # Collect all spikes in a concatenated array per state
        shifted_spikes_all_active = []
        shifted_spikes_all_quiet = []
        orig_spikes_all_active = []
        orig_spikes_all_quiet = []
    
        # Trial indices per state (continuous count across directions)
        shifted_trials_all_active = []
        shifted_trials_all_quiet = []
        orig_trials_all_active = []
        orig_trials_all_quiet = []
    
        # Direction labels per trial per state
        shifted_dirs_all_active = []
        shifted_dirs_all_quiet = []
        orig_dirs_all_active = []
        orig_dirs_all_quiet = []
    
        orig_traces_dict_active = {}
        orig_traces_dict_quiet = {}
    
        shifted_data_by_state_dict[dataset][neuron_idx] = {
            'active': {},
            'quiet': {}
        }
        orig_data_by_state_dict[dataset][neuron_idx] = {
            'active': {},
            'quiet': {}
        }
    
        trial_counter_active = 0
        trial_counter_quiet = 0
    
        for (dir_id, shifted_neuron_data_dir), (_, original_neuron_data_dir) in zip(shifted_neuron_data.items(), original_neuron_data.items()):
            spShifted_dir = shifted_neuron_data_dir['spiketimes']
            spOrig_dir = original_neuron_data_dir['spiketimes']
            trials_dir = shifted_neuron_data_dir['trials']
            dir_idx = np.where(groups == dir_id)[0][0]
            direction = stim[dataset]['direction'].reshape(-1)
            running_state = stim[dataset]['running_state'].reshape(-1)
            dir_indices = np.where(direction == dir_id)[0]
    
            # ACTIVE
            relative_active_indices = np.nonzero(running_state[dir_indices] == 1)[0]
            active_spike_mask = np.isin(trials_dir, relative_active_indices)
            spShifted_dir_active = spShifted_dir[active_spike_mask]
            spOrig_dir_active = spOrig_dir[active_spike_mask]
            trials_dir_a = trials_dir[active_spike_mask]
            unique_active_trials = np.sort(np.unique(trials_dir_a))
            trial_mapping = {old: new + trial_counter_active for new, old in enumerate(unique_active_trials)}
            trials_dir_active = np.array([trial_mapping[t] for t in trials_dir_a])
            trial_counter_active += len(unique_active_trials)
            
            trial_mapping_for_traces = {old: new for new, old in enumerate(unique_active_trials)}
            trials_dir_active_for_traces = np.array([trial_mapping_for_traces[t] for t in trials_dir_a])
    
            shifted_spikes_all_active.append(spShifted_dir_active)
            shifted_trials_all_active.append(trials_dir_active)
            shifted_dirs_all_active.append(np.full(len(unique_active_trials), dir_id))
    
            orig_spikes_all_active.append(spOrig_dir_active)
            orig_trials_all_active.append(trials_dir_active)
            orig_dirs_all_active.append(np.full(len(unique_active_trials), dir_id))
            shifted_traces_active, sem, timeBins = newTracesFromPSTH(
                spShifted_dir_active, trials_dir_active_for_traces, window,
                direction[(running_state == 1) & (direction == dir_id)],
                groups, baseline=0, binSize=0.0005, sigma=sigma)
    
            orig_traces_active, _, _ = newTracesFromPSTH(
                spOrig_dir_active, trials_dir_active_for_traces, window,
                direction[(running_state == 1) & (direction == dir_id)],
                groups, baseline=0, binSize=0.0005, sigma=sigma)
    
            shifted_traces_dict_active[dir_id] = shifted_traces_active[dir_idx]
            orig_traces_dict_active[dir_id] = orig_traces_active[dir_idx]
    
            # QUIET
            relative_quiet_indices = np.nonzero(running_state[dir_indices] == 0)[0]
            quiet_spike_mask = np.isin(trials_dir, relative_quiet_indices)
            spShifted_dir_quiet = spShifted_dir[quiet_spike_mask]
            spOrig_dir_quiet = spOrig_dir[quiet_spike_mask]
            trials_dir_q = trials_dir[quiet_spike_mask]
            unique_quiet_trials = np.sort(np.unique(trials_dir_q))
            trial_mapping = {old: new + trial_counter_quiet for new, old in enumerate(unique_quiet_trials)}
            trials_dir_quiet = np.array([trial_mapping[t] for t in trials_dir_q])
            trial_counter_quiet += len(unique_quiet_trials)
            
            trial_mapping_for_traces = {old: new for new, old in enumerate(unique_quiet_trials)}
            trials_dir_quiet_for_traces = np.array([trial_mapping_for_traces[t] for t in trials_dir_q])
    
            shifted_spikes_all_quiet.append(spShifted_dir_quiet)
            shifted_trials_all_quiet.append(trials_dir_quiet)
            shifted_dirs_all_quiet.append(np.full(len(unique_quiet_trials), dir_id))
    
            orig_spikes_all_quiet.append(spOrig_dir_quiet)
            orig_trials_all_quiet.append(trials_dir_quiet)
            orig_dirs_all_quiet.append(np.full(len(unique_quiet_trials), dir_id))
    
            shifted_traces_quiet, sem, timeBins = newTracesFromPSTH(
                spShifted_dir_quiet, trials_dir_quiet_for_traces, window,
                direction[(running_state == 0) & (direction == dir_id)],
                groups, baseline=0, binSize=0.0005, sigma=sigma)
    
            orig_traces_quiet, _, _ = newTracesFromPSTH(
                spOrig_dir_quiet, trials_dir_quiet_for_traces, window,
                direction[(running_state == 0) & (direction == dir_id)],
                groups, baseline=0, binSize=0.0005, sigma=sigma)
    
            shifted_traces_dict_quiet[dir_id] = shifted_traces_quiet[dir_idx]
            orig_traces_dict_quiet[dir_id] = orig_traces_quiet[dir_idx]
    
        # Save data to final dicts
        shifted_data_by_state_dict[dataset][neuron_idx]['active']['traces'] = shifted_traces_dict_active
        shifted_data_by_state_dict[dataset][neuron_idx]['quiet']['traces'] = shifted_traces_dict_quiet
        orig_data_by_state_dict[dataset][neuron_idx]['active']['traces'] = orig_traces_dict_active
        orig_data_by_state_dict[dataset][neuron_idx]['quiet']['traces'] = orig_traces_dict_quiet
        shifted_data_by_state_dict[dataset][neuron_idx]['active']['spikes'] = np.concatenate(shifted_spikes_all_active)
        shifted_data_by_state_dict[dataset][neuron_idx]['quiet']['spikes'] = np.concatenate(shifted_spikes_all_quiet)
        shifted_data_by_state_dict[dataset][neuron_idx]['active']['trials'] = np.concatenate(shifted_trials_all_active)
        shifted_data_by_state_dict[dataset][neuron_idx]['quiet']['trials'] = np.concatenate(shifted_trials_all_quiet)
        shifted_data_by_state_dict[dataset][neuron_idx]['active']['dirs'] = np.concatenate(shifted_dirs_all_active)
        shifted_data_by_state_dict[dataset][neuron_idx]['quiet']['dirs'] = np.concatenate(shifted_dirs_all_quiet)
    
        
        orig_data_by_state_dict[dataset][neuron_idx]['active']['spikes'] = np.concatenate(orig_spikes_all_active)
        orig_data_by_state_dict[dataset][neuron_idx]['quiet']['spikes'] = np.concatenate(orig_spikes_all_quiet)
        orig_data_by_state_dict[dataset][neuron_idx]['active']['trials'] = np.concatenate(orig_trials_all_active)
        orig_data_by_state_dict[dataset][neuron_idx]['quiet']['trials'] = np.concatenate(orig_trials_all_quiet)
        orig_data_by_state_dict[dataset][neuron_idx]['active']['dirs'] = np.concatenate(orig_dirs_all_active)
        orig_data_by_state_dict[dataset][neuron_idx]['quiet']['dirs'] = np.concatenate(orig_dirs_all_quiet)

# %% plot shifted rasterPSTH
        
for dataset in shifted_data_by_state_dict.keys():
    shifted_data_by_state_dict_one_ds = shifted_data_by_state_dict[dataset]
    for neuron_id, neuron_data in shifted_data_by_state_dict_one_ds.items():
        
        spShifted_active = neuron_data['active']['spikes']
        spShifted_trials_active = neuron_data['active']['trials']
        spShifted_traces_active = neuron_data['active']['traces']
        spShifted_stimuli_id_active = neuron_data['active']['dirs']
        
        fig = plt.figure(figsize=(9, 9))
        
        ax1 = plt.subplot(2,2,1)
        plt.title('ACTIVE', loc='left', fontsize = 10)
        newPlotPSTH(spShifted_active, spShifted_trials_active, window, spShifted_stimuli_id_active, groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x= 2, c='k', ls='--')
        plt.xticks(xticks)
    
        ax2 = plt.subplot(2,2,3, sharex = ax1)
        traces_array = np.stack(list(spShifted_traces_active.values()))
        mean_trace_active = np.mean(traces_array, axis=0)
        ax2.plot(timeBins, mean_trace_active, color='black', label='Mean Active', linewidth=2)
        ax2.legend()
    
        
        for dir_id, trace in spShifted_traces_active.items():
            if dir_id in groups:
                idx = np.where(groups == dir_id)[0][0]
                color = colors[idx]
                label = str(groups[idx])
                plt.plot(timeBins, trace, alpha=alpha_val, c=color, label=label)
    
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x= 2, c='k', ls='--')
        plt.xticks(xticks)
    
            
        # Quiet
    
        spShifted_quiet = neuron_data['quiet']['spikes']
        spShifted_trials_quiet = neuron_data['quiet']['trials']
        spShifted_traces_quiet = neuron_data['quiet']['traces']
        spShifted_stimuli_id_quiet = neuron_data['quiet']['dirs']
        
        ax3 = plt.subplot(2,2,2, sharex = ax1)
        plt.title('QUIET', loc='left', fontsize = 10)
        newPlotPSTH(spShifted_quiet, spShifted_trials_quiet, window, spShifted_stimuli_id_quiet, groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=2, c='k', ls='--')
        plt.xticks(xticks)
    
        ax4 = plt.subplot(2,2,4, sharex = ax3, sharey = ax2)
        traces_array = np.stack(list(spShifted_traces_quiet.values()))
        mean_trace_quiet = np.mean(traces_array, axis=0)
        ax4.plot(timeBins, mean_trace_quiet, color='black', label='Mean Quiet', linewidth=2)
        ax4.legend()
    
        for dir_id, trace in spShifted_traces_quiet.items():
            if dir_id in groups:
                idx = np.where(groups == dir_id)[0][0]
                color = colors[idx]
                label = str(groups[idx])
                plt.plot(timeBins, trace, alpha=alpha_val, c=color, label=label)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=2, c='k', ls='--')
        plt.xticks(xticks)
    
        
        # depth = abs(n['cluster_info']['depth'].loc[neuron] - n['SC_depth_limits'][0]
        #             )/(n['SC_depth_limits'][0] - n['SC_depth_limits'][1])*100
        
        fig.suptitle(
            f'Neuron: {neuron_id} shifted spikes')
        
        dataEntry = database.loc[dataset]
        saveDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, 'rasterPSTH_shiftSpikes_strong_dirs_03lag')
        if not os.path.isdir(saveDirectory):
            os.makedirs(saveDirectory)
        filename = os.path.join(saveDirectory, f'{neuron_id}_PSTH.png')
        plt.savefig(filename)
        plt.close()
        
# %% plot original rasterPSTH (strong directions)
        
for dataset in orig_data_by_state_dict.keys():
    orig_data_by_state_dict_one_ds = orig_data_by_state_dict[dataset]
    for neuron_id, neuron_data in orig_data_by_state_dict_one_ds.items():
        
        spShifted_active = neuron_data['active']['spikes']
        spShifted_trials_active = neuron_data['active']['trials']
        spShifted_traces_active = neuron_data['active']['traces']
        spShifted_stimuli_id_active = neuron_data['active']['dirs']
        
        fig = plt.figure(figsize=(9, 9))

        ax1 = plt.subplot(2,2,1)
        plt.title('ACTIVE', loc='left', fontsize = 10)
        newPlotPSTH(spShifted_active, spShifted_trials_active, window, spShifted_stimuli_id_active, groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x= 2, c='k', ls='--')
        plt.xticks(xticks)
    
        ax2 = plt.subplot(2,2,3, sharex = ax1)
        traces_array = np.stack(list(spShifted_traces_active.values()))
        mean_trace_active = np.mean(traces_array, axis=0)
        ax2.plot(timeBins, mean_trace_active, color='black', label='Mean Active', linewidth=2)
        ax2.legend()
    
        
        for dir_id, trace in spShifted_traces_active.items():
            if dir_id in groups:
                idx = np.where(groups == dir_id)[0][0]
                color = colors[idx]
                label = str(groups[idx])
                plt.plot(timeBins, trace, alpha=alpha_val, c=color, label=label)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x= 2, c='k', ls='--')
        plt.xticks(xticks)
    
            
        # Quiet
        spShifted_quiet = neuron_data['quiet']['spikes']
        spShifted_trials_quiet = neuron_data['quiet']['trials']
        spShifted_traces_quiet = neuron_data['quiet']['traces']
        spShifted_stimuli_id_quiet = neuron_data['quiet']['dirs']
        
        ax3 = plt.subplot(2,2,2, sharex = ax1)
        plt.title('QUIET', loc='left', fontsize = 10)
        newPlotPSTH(spShifted_quiet, spShifted_trials_quiet, window, spShifted_stimuli_id_quiet, groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=2, c='k', ls='--')
        plt.xticks(xticks)
    
        ax4 = plt.subplot(2,2,4, sharex = ax3, sharey = ax2)

        traces_array = np.stack(list(spShifted_traces_quiet.values()))
        mean_trace_quiet = np.mean(traces_array, axis=0)
        ax4.plot(timeBins, mean_trace_quiet, color='black', label='Mean Quiet', linewidth=2)
        ax4.legend()

        for dir_id, trace in spShifted_traces_quiet.items():
            if dir_id in groups:
                idx = np.where(groups == dir_id)[0][0]
                color = colors[idx]
                label = str(groups[idx])
                plt.plot(timeBins, trace, alpha=alpha_val, c=color, label=label)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x=2, c='k', ls='--')
        plt.xticks(xticks)
    
        
        # depth = abs(n['cluster_info']['depth'].loc[neuron] - n['SC_depth_limits'][0]
        #             )/(n['SC_depth_limits'][0] - n['SC_depth_limits'][1])*100
        
        fig.suptitle(
            f'Neuron: {neuron_id} original spikes')
        
        dataEntry = database.loc[dataset]
        saveDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, 'rasterPSTH_origSpikes_strong_dirs')
        if not os.path.isdir(saveDirectory):
            os.makedirs(saveDirectory)
        filename = os.path.join(saveDirectory, f'{neuron_id}_PSTH.png')
        plt.savefig(filename)
        plt.close()
        
# %% oscillations params 

stimulus_freq = 2 #Hz
f1_f0_threshold = 0.4
tuning_threshold = 0.4

  
# %% find oscillating neurons using strong directions
plot_fft = True
# plot_fft = False 

f1_f0_values_strong = {}  # Dictionary to store F1/F0 per neuron
oscillating_neurons_strong_dict = {}
# Define neurons of interest for each dataset individually
neurons_of_interest_by_dataset = {
    0: filtered_clusters[0].keys(),
    1: filtered_clusters[1].keys(),
    2: filtered_clusters[2].keys(),
    3: filtered_clusters[3].keys()
}

# Initialize nested dictionaries
spShifted_neurons_of_int = {}
spOrig_neurons_of_int = {}

for ds, neurons in neurons_of_interest_by_dataset.items():
    spShifted_neurons_of_int[ds] = {}
    spOrig_neurons_of_int[ds] = {}

    for neuron_id in neurons:
        if neuron_id in spShifted_strong_dirs_all_neurons[ds]:
            spShifted_neurons_of_int[ds][neuron_id] = spShifted_strong_dirs_all_neurons[ds][neuron_id]
        if neuron_id in spOrig_strong_dirs_all_neurons[ds]:
            spOrig_neurons_of_int[ds][neuron_id] = spOrig_strong_dirs_all_neurons[ds][neuron_id]
            
            
for dataset in spShifted_neurons_of_int.keys():
    spShifted_neurons_of_int_one_ds = spShifted_neurons_of_int[dataset]
    spOrig_neurons_of_int_one_ds = spOrig_neurons_of_int[dataset]
    f1_f0_values_strong[dataset] = {}
    oscillating_neurons_strong_dict[dataset] = {}
    oscillating_neurons_strong = []
    for (neuron_id, shifted_neuron_data), (_, original_neuron_data) in zip(spShifted_neurons_of_int_one_ds.items(), spOrig_neurons_of_int_one_ds.items()):
        dataEntry = database.loc[dataset]
        identifier = f'{dataEntry.Name}_{dataEntry.Date}_{neuron_id}'
        print(f'Processing neuron {identifier}')            
        shifted_traces_dict = {}
        
        for (dir_id, shifted_neuron_data_dir), (_, original_neuron_data_dir) in zip(shifted_neuron_data.items(), original_neuron_data.items()):
            spShifted_dir = shifted_neuron_data_dir['spiketimes']
            trials_dir = shifted_neuron_data_dir['trials']
            spOrig_dir = original_neuron_data_dir['spiketimes']
            direction = stim[dataset]['direction'].reshape(-1)
            direction_mask = direction == dir_id
        
            shifted_traces, sem, timeBins = newTracesFromPSTH(spShifted_dir, trials_dir, window, direction[direction_mask],
                                                  groups, baseline=0,  binSize=0.0005, sigma=sigma)
            dir_idx = np.where(groups == dir_id)[0][0]
            shifted_traces_dict[dir_id] = shifted_traces[dir_idx] 
            # plt.plot(shifted_traces[dir_idx])
            
        f1_f0_values_strong[dataset][neuron_id] = {}
        # Compute F1/F0 per direction
        for dir_id, trace in shifted_traces_dict.items():
            if not np.any(np.isnan(trace)):  # Ensure the trace does not contain NaNs
                dataEntry = database.loc[dataset]
                saveDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, 'F1-F0', str(neuron_id))
                _, f1_f0 = compute_f1_f0(identifier, dir_id, trace, stimulus_freq, timeBins, threshold=f1_f0_threshold, saveDirectory = saveDirectory, plot=plot_fft)
                f1_f0_values_strong[dataset][neuron_id][dir_id] = f1_f0  # Store per direction
            
        # Decide if the neuron is oscillatory
        mean_f1_f0 = np.mean(list(f1_f0_values_strong[dataset][neuron_id].values()))  # Average across directions
        if mean_f1_f0 > f1_f0_threshold:
            oscillating_neurons_strong.append(neuron_id)
    oscillating_neurons_strong_dict[dataset] = oscillating_neurons_strong
 
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

# # Plot the frequency response
# plt.figure()
# plt.plot(w, abs(h), label=f'Kaiser window numtaps={numtaps}')
# plt.title('Frequency Response with Kaiser Window')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Gain')
# plt.xlim(0, 10)
# plt.grid(True)
# plt.legend(loc='best')
# plt.show()  

# %% filtering dir traces by state new 
plot_filt_trace = True
# Define neurons of interest for each dataset individually
neurons_of_interest_by_dataset = {
    0: oscillating_neurons_strong_dict[0],
    1: oscillating_neurons_strong_dict[1], 
    2: oscillating_neurons_strong_dict[2], 
    3: oscillating_neurons_strong_dict[3], 
    
}

# Initialize nested dictionaries
spShifted_neurons_of_int = {}

for ds, neurons in neurons_of_interest_by_dataset.items():
    spShifted_neurons_of_int[ds] = {}

    for neuron_id in neurons:
        if neuron_id in shifted_data_by_state_dict[ds]:
            spShifted_neurons_of_int[ds][neuron_id] = shifted_data_by_state_dict[ds][neuron_id]
            
            
            

filtered_traces_by_state_dict = {}
for dataset in spShifted_neurons_of_int.keys():
    shifted_data_by_state_dict_one_ds = spShifted_neurons_of_int[dataset]
    filtered_traces_by_state_dict[dataset] = {}
    for neuron_id, shifted_neuron_data in shifted_data_by_state_dict_one_ds.items():
        dataEntry = database.loc[dataset]
        saveDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, 'Filtered', str(neuron_id))
        identifier = f'{dataEntry.Name}_{dataEntry.Date}_{neuron_id}'
        print(f'Processing neuron {identifier}') 
        filtered_traces_active = {}
        filtered_traces_quiet = {}
        
        filtered_traces_by_state_dict[dataset][neuron_id] = {}
        shifted_neuron_active_traces = shifted_neuron_data['active']['traces']
        shifted_neuron_quiet_traces = shifted_neuron_data['quiet']['traces']
        
        for dir_id, shifted_neuron_trace_dir in shifted_neuron_active_traces.items():
            
            filtered_trace_active = filter_F1(neuron_id, dir_id, timeBins, shifted_neuron_trace_dir, sigma, fir_coeff, state=1,pad_length = 6000, saveDirectory = saveDirectory, plot = plot_filt_trace)
            filtered_traces_active[dir_id] = filtered_trace_active
    
            
        for dir_id, shifted_neuron_trace_dir in shifted_neuron_quiet_traces.items():
            
            filtered_trace_quiet = filter_F1(neuron_id, dir_id, timeBins, shifted_neuron_trace_dir, sigma, fir_coeff, state=0, pad_length = 6000, saveDirectory = saveDirectory, plot = plot_filt_trace)
            filtered_traces_quiet[dir_id] = filtered_trace_quiet
            
    
        filtered_traces_by_state_dict[dataset][neuron_id]['active'] = filtered_traces_active
        filtered_traces_by_state_dict[dataset][neuron_id]['quiet'] = filtered_traces_quiet



# %% separate time shifts by state 

time_shifts_by_state = {}
for dataset in time_shifts.keys():
    time_shifts_by_state[dataset] = {'active': {}, 'quiet': {}}
    time_shifts_one_ds = time_shifts[dataset]
    for dir_id, time_shifts_dir in time_shifts_one_ds.items():
        dir_indices = np.where(stim[dataset]['direction'].reshape(-1) == dir_id)[0]
        relative_active_indices = np.nonzero(stim[dataset]['running_state'].reshape(-1)[dir_indices] == 1)[0]
        active_shifts = time_shifts_dir[relative_active_indices]
        time_shifts_by_state[dataset]['active'][dir_id] = active_shifts
        
        relative_quiet_indices = np.nonzero(stim[dataset]['running_state'].reshape(-1)[dir_indices] == 0)[0]
        quiet_shifts = time_shifts_dir[relative_quiet_indices]
        time_shifts_by_state[dataset]['quiet'][dir_id] = quiet_shifts
        
 
# %% subtract F1 
plot_F1_subtr = True
# Define neurons of interest for each dataset individually
neurons_of_interest_by_dataset = {
    0: oscillating_neurons_strong_dict[0],
    1: oscillating_neurons_strong_dict[1],
    2: oscillating_neurons_strong_dict[2], 
    3: oscillating_neurons_strong_dict[3], 
    
}

# Initialize nested dictionaries
spOrig_neurons_of_int = {}

for ds, neurons in neurons_of_interest_by_dataset.items():
    spOrig_neurons_of_int[ds] = {}

    for neuron_id in neurons:
        if neuron_id in orig_data_by_state_dict[ds]:
            spOrig_neurons_of_int[ds][neuron_id] = orig_data_by_state_dict[ds][neuron_id]
            
            

binsize_signal = timeBins[-1]-timeBins[-2] # 
F1_subtracted_all_dirs_all_neurons = {}        

for dataset in filtered_traces_by_state_dict.keys(): 
    filtered_traces_by_state_dict_one_ds = filtered_traces_by_state_dict[dataset]
    orig_data_by_state_dict_one_ds = spOrig_neurons_of_int[dataset]
    F1_subtracted_all_dirs_all_neurons[dataset] = {}
    for (neuron_id, filt_traces_one_neuron), (_, original_data_one_neuron) in zip(filtered_traces_by_state_dict_one_ds.items(), orig_data_by_state_dict_one_ds.items()):
        filt_traces_one_neuron_active = filt_traces_one_neuron['active']
        filt_traces_one_neuron_quiet = filt_traces_one_neuron['quiet']
        
        original_trace_one_neuron_active = original_data_one_neuron['active']['traces']
        original_trace_one_neuron_quiet = original_data_one_neuron['quiet']['traces']
        F1_subtracted_all_dirs_all_neurons[dataset][neuron_id] = {}
        F1_subtracted_one_neuron_active = {}
        dataEntry = database.loc[dataset]
        saveDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, 'F1_subtr', str(neuron_id))
    
        for (dir_id, filt_trace_dir), (_, original_trace_dir) in zip(filt_traces_one_neuron_active.items(), original_trace_one_neuron_active.items()):
            shifts_s = time_shifts_by_state[dataset]['active'][dir_id]
            if len(shifts_s) > 0: 
                F1_trial_shifted, F1_subtracted_signal = subtract_F1(neuron_id, dir_id, timeBins, filt_trace_dir, original_trace_dir, shifts_s, state=1, binsize_signal=binsize_signal, sigma = sigma, saveDirectory=saveDirectory, plot=plot_F1_subtr)
                F1_subtracted_one_neuron_active[dir_id] = F1_subtracted_signal
            else:
                continue
        
        F1_subtracted_all_dirs_all_neurons[dataset][neuron_id]['active'] = F1_subtracted_one_neuron_active
        
        F1_subtracted_one_neuron_quiet = {}
        for (dir_id, filt_trace_dir), (_, original_trace_dir) in zip(filt_traces_one_neuron_quiet.items(), original_trace_one_neuron_quiet.items()):
            shifts_s = time_shifts_by_state[dataset]['quiet'][dir_id]
            F1_trial_shifted, F1_subtracted_signal = subtract_F1(neuron_id, dir_id, timeBins, filt_trace_dir, original_trace_dir, shifts_s, state=0, binsize_signal=binsize_signal, sigma = sigma, saveDirectory=saveDirectory, plot=plot_F1_subtr)
            F1_subtracted_one_neuron_quiet[dir_id] = F1_subtracted_signal
        
        F1_subtracted_all_dirs_all_neurons[dataset][neuron_id]['quiet'] = F1_subtracted_one_neuron_quiet

# %% Hilbert
plot_hilbert = True
amplitude_envelope_all_neurons_all_dirs = {}
for dataset in filtered_traces_by_state_dict.keys(): 
    filtered_traces_by_state_dict_one_ds = filtered_traces_by_state_dict[dataset]
    amplitude_envelope_all_neurons_all_dirs[dataset] = {}
    for neuron_id, filtered_traces in filtered_traces_by_state_dict_one_ds.items():
        active_filtered_traces = filtered_traces['active']
        quiet_filtered_traces = filtered_traces['quiet']
        
        amplitude_envelope_all_neurons_all_dirs[dataset][neuron_id] = {}
        amplitude_envelope_one_neuron_all_dirs_active = {}
        amplitude_envelope_one_neuron_all_dirs_quiet = {}
        dataEntry = database.loc[dataset]
        saveDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, 'F1_hilbert', str(neuron_id))
        for dir_id, filtered_trace in active_filtered_traces.items():
            amplitude_envelope_filtered = hilbert_transform(neuron_id, dir_id, filtered_trace, timeBins, state = 1, saveDirectory=saveDirectory, plot=plot_hilbert)
            amplitude_envelope_one_neuron_all_dirs_active[dir_id] = amplitude_envelope_filtered
        for dir_id, filtered_trace in quiet_filtered_traces.items():
            amplitude_envelope_filtered = hilbert_transform(neuron_id, dir_id, filtered_trace, timeBins, state = 0, saveDirectory=saveDirectory, plot=plot_hilbert)
            amplitude_envelope_one_neuron_all_dirs_quiet[dir_id] = amplitude_envelope_filtered
            
        amplitude_envelope_all_neurons_all_dirs[dataset][neuron_id]['active'] = amplitude_envelope_one_neuron_all_dirs_active
        amplitude_envelope_all_neurons_all_dirs[dataset][neuron_id]['quiet'] = amplitude_envelope_one_neuron_all_dirs_quiet
# %% average F1 subtracted across directions
plot_F1_subtr_ave = True
average_F1_subtracted = {}

for dataset in F1_subtracted_all_dirs_all_neurons.keys():
    F1_subtracted_all_dirs_all_neurons_one_ds = F1_subtracted_all_dirs_all_neurons[dataset]
    orig_data_by_state_dict_one_ds = orig_data_by_state_dict[dataset]
    average_F1_subtracted[dataset] = {}
    for neuron_id, F1_subtracted in F1_subtracted_all_dirs_all_neurons_one_ds.items(): 
        
        average_F1_subtracted[dataset][neuron_id] = {}
        F1_subtracted_active = F1_subtracted['active']
        F1_subtracted_quiet = F1_subtracted['quiet']
        
        active_arrays = list(F1_subtracted_active.values())
        average_F1_subtracted_active = np.mean(np.stack(active_arrays), axis=0)
        
        quiet_arrays = list(F1_subtracted_quiet.values())
        average_F1_subtracted_quiet = np.mean(np.stack(quiet_arrays), axis=0)
        
        average_F1_subtracted[dataset][neuron_id]['active'] = average_F1_subtracted_active
        average_F1_subtracted[dataset][neuron_id]['quiet'] = average_F1_subtracted_quiet
        
        orig_traces_one_neuron_active = orig_data_by_state_dict_one_ds[neuron_id]['active']['traces']
        orig_traces_one_neuron_active_array = np.array(list(orig_traces_one_neuron_active.values()))
        
        
        orig_traces_one_neuron_quiet = orig_data_by_state_dict_one_ds[neuron_id]['quiet']['traces']
        orig_traces_one_neuron_quiet_array = np.array(list(orig_traces_one_neuron_quiet.values()))
        
        if plot_F1_subtr_ave: 
            dataEntry = database.loc[dataset]
            identifier=f'{dataEntry.Name}_{dataEntry.Date}_{neuron_id}'
            
            expansion_amount = 3 * sigma
            start_time = 0 - expansion_amount
            end_time = 2 + expansion_amount
            start_idx = np.searchsorted(timeBins, start_time)
            end_idx = np.searchsorted(timeBins, end_time)
            timeBins_stim_window = timeBins[start_idx:end_idx]
            plt.figure()
            plt.plot(timeBins_stim_window, average_F1_subtracted_active, linewidth = 2, label = 'F1 subtracted', color='red')
            for array in active_arrays:
                plt.plot(timeBins_stim_window, array, alpha=0.5)  # Plot individual active arrays with some transparency
            plt.plot(timeBins, np.mean(orig_traces_one_neuron_active_array, axis=0), linewidth = 2, label = 'Original', color='black')
            plt.legend()
            plt.title(f'Active F1 subtracted - {identifier}')
            plt.xlabel('Time')
            plt.ylabel('FR')
            saveDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, 'F1_subtr_ave')
            if not os.path.isdir(saveDirectory):
                os.makedirs(saveDirectory)
            filename = os.path.join(saveDirectory, f'{neuron_id}_PSTH_active.png')
            plt.savefig(filename)
            plt.close()
            
            plt.figure()
            plt.plot(timeBins_stim_window, average_F1_subtracted_quiet, linewidth = 2, label = 'F1 subtracted', color='red')
            for array in quiet_arrays:
                plt.plot(timeBins_stim_window, array, alpha=0.5)  # Plot individual quiet arrays with some transparency
            plt.plot(timeBins, np.mean(orig_traces_one_neuron_quiet_array, axis=0), linewidth = 2, label = 'Original', color='black')
            plt.legend()
            plt.title(f'Quiet F1 subtracted - {identifier}')
            plt.xlabel('Time')
            plt.ylabel('FR')
            filename = os.path.join(saveDirectory, f'{neuron_id}_PSTH_quiet.png')
            plt.savefig(filename)
            plt.close()
            
# %% average amplitude envelopes across directions

plot_hilbert_ave = True
average_amplitude_envelopes = {}
for dataset in amplitude_envelope_all_neurons_all_dirs.keys():
    amplitude_envelope_all_neurons_all_dirs_one_ds = amplitude_envelope_all_neurons_all_dirs[dataset]
    average_amplitude_envelopes[dataset] = {}
    for neuron_id, amplitude_envelopes in amplitude_envelope_all_neurons_all_dirs_one_ds.items():
        
        average_amplitude_envelopes[dataset][neuron_id] = {}
        amplitude_envelopes_active = amplitude_envelopes['active']
        amplitude_envelopes_quiet = amplitude_envelopes['quiet']
        
        active_arrays = list(amplitude_envelopes_active.values())
        average_amplitude_envelope_active = np.mean(np.stack(active_arrays), axis=0)
        
        quiet_arrays = list(amplitude_envelopes_quiet.values())
        average_amplitude_envelope_quiet = np.mean(np.stack(quiet_arrays), axis=0)
        
        average_amplitude_envelopes[dataset][neuron_id]['active'] = average_amplitude_envelope_active
        average_amplitude_envelopes[dataset][neuron_id]['quiet'] = average_amplitude_envelope_quiet
        if plot_hilbert_ave: 
            dataEntry = database.loc[dataset]
            identifier=f'{dataEntry.Name}_{dataEntry.Date}_{neuron_id}'
            
            expansion_amount = 3 * sigma
            start_time = 0 - expansion_amount
            end_time = 2 + expansion_amount
            start_idx = np.searchsorted(timeBins, start_time)
            end_idx = np.searchsorted(timeBins, end_time)
            timeBins_stim_window = timeBins[start_idx:end_idx]
            
            # Plot for Active Amplitude Envelope
            plt.figure()
            plt.plot(timeBins_stim_window, average_amplitude_envelope_active, linewidth = 2, label=f'Average')
            for array in active_arrays:
                plt.plot(timeBins_stim_window, array, alpha=0.5)  # Plot individual active arrays with some transparency
            plt.legend()
            plt.title(f'Active Amplitude Envelope - {identifier}')
            plt.xlabel('Time')
            plt.ylabel('FR')
            saveDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, 'F1_hilbert_ave')
            if not os.path.isdir(saveDirectory):
                os.makedirs(saveDirectory)
            filename = os.path.join(saveDirectory, f'{neuron_id}_PSTH_active.png')
            plt.savefig(filename)
            plt.close()
            
            # Plot for Quiet Amplitude Envelope
            plt.figure()
            plt.plot(timeBins_stim_window, average_amplitude_envelope_quiet, linewidth = 2, label=f'Average')
            for array in quiet_arrays:
                plt.plot(timeBins_stim_window, array, alpha=0.5)  # Plot individual quiet arrays with some transparency
            plt.legend()
            plt.title(f'Quiet Amplitude Envelope - {identifier}')
            plt.xlabel('Time')
            plt.ylabel('FR')
            filename = os.path.join(saveDirectory, f'{neuron_id}_PSTH_quiet.png')
            plt.savefig(filename)
            plt.close()
            
    

    

