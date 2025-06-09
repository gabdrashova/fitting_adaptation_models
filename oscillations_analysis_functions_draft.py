# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:22:33 2025

@author: Experimenter
"""
# %% load libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
from scipy.fftpack import fft
from Functions.neural_response import *
from scipy.signal import firwin, lfilter, filtfilt, hilbert, kaiserord, freqz

# %% functions
def _plot_column(axc, spks, example_neurons):
    """
    Plots column of subplots.
    
    Parameters
    ----------
    axc : array, holding list of axes in column.
    spks : SpikeData object
    sniffs : array, holding sniff time on each trial.
    """
    
    

    # Plot raster plot for each neuron.
    raster_kws = dict(s=4, c='k', lw=0)
    for n, ax in zip(example_neurons, axc):
        ax.scatter(
            spks.spiketimes[spks.neurons == n],
            spks.trials[spks.neurons == n],
            **raster_kws,
        )
        # ax.set_ylim(-1, len(trials))
        ax.axis('off')

def save_shifted_spikes(spks, example_neurons):
    
    shifted_spikes = {}

    for n in example_neurons:
        spiketimes = spks.spiketimes[spks.neurons == n]
        trials = spks.trials[spks.neurons == n]
        
        # Store spike times and trials in a sub-dictionary
        shifted_spikes[n] = {
            'spiketimes': spiketimes,
            'trials': trials
        }
        
    return shifted_spikes

# Function to calculate the range of spike times for each neuron and trial
def calculate_shifted_spike_ranges(shifted_spikes):
    trial_ranges = {}

    for neuron_id, data in shifted_spikes.items():
        spiketimes = data['spiketimes']
        trials = data['trials']
        
        # Create a dictionary to store ranges per trial
        neuron_trial_ranges = {}
        
        for trial in set(trials):
            trial_spiketimes = spiketimes[trials == trial]
            min_time = trial_spiketimes.min()
            max_time = trial_spiketimes.max()
            
            neuron_trial_ranges[trial] = (min_time, max_time)
        
        # Store the ranges for this neuron
        trial_ranges[neuron_id] = neuron_trial_ranges
    
    return trial_ranges

def compute_psth_with_baseline_correction(spiketimes, trials, window, bin_size=0.05, sigma=0.02, baseline=0):
    """
    Compute PSTH with fixed window, Gaussian smoothing, baseline correction, and bin centering.

    Parameters:
    - spiketimes: np.array
        Array of spike times aligned to an event.
    - trials: np.array
        Array of trial IDs corresponding to the spike times.
    - window: tuple
        Time window for PSTH calculation (start, end).
    - bin_size: float
        Size of bins for the PSTH (in seconds).
    - sigma: float
        Standard deviation of the Gaussian smoothing kernel (in seconds).
    - baseline: float
        Duration for baseline calculation (seconds before zero). If 0, no baseline correction.

    Returns:
    - bins: np.array
        Bin centers.
    - smoothed_spike_counts: np.array
        Smoothed PSTH values (spikes/s).
    """
    # Generate fixed bin edges and centers
    if window[0]*window[-1] < 0:
    # Calculate edges and bins centers        
        edges = np.concatenate((-np.arange(0, -window[0], bin_size)[::-1], 
                                np.arange(bin_size, window[1], bin_size)))
    else:
        edges = np.arange(window[0], window[-1], bin_size)
    bins = edges[:-1] + 0.5 * bin_size

    # Initialize an array to accumulate histograms for all trials
    binned_spike_counts = []

    for trial in np.unique(trials):  # Iterate through trials
        spiketimes_in_trial = spiketimes[trials == trial]
        trial_spike_counts, _ = np.histogram(spiketimes_in_trial, bins=edges)
        trial_spike_counts = trial_spike_counts / bin_size  # Convert to spikes/second
        binned_spike_counts.append(trial_spike_counts)

    binned_spike_counts = np.array(binned_spike_counts)

    # Compute mean firing rate across trials
    mean_spike_counts = np.mean(binned_spike_counts, axis=0)

    # Baseline correction
    if baseline > 0:
        baseline_bins = (edges >= -baseline) & (edges < 0)
        baseline_mean = np.mean(mean_spike_counts[baseline_bins[:-1]])  # Match indexing to bins
        mean_spike_counts -= baseline_mean

    # Gaussian smoothing
    sigma_bins = sigma / bin_size
    smoothed_spike_counts = gaussian_filter1d(mean_spike_counts, sigma_bins)

    return bins, smoothed_spike_counts

# def plot_raster_and_psth(neuron_id, spiketimes, trials, bins, smoothed_spike_counts, axes=None, window=None, save_path=None):
#     """
#     Plot a combined raster plot and PSTH for a neuron and save the figure.

#     Parameters:
#     - neuron_id: int
#         The ID of the neuron being plotted.
#     - spiketimes: np.array
#         Array of spike times aligned to an event.
#     - trials: np.array
#         Array of trial IDs corresponding to the spike times.
#     - bins: np.array
#         Bin centers for the PSTH.
#     - smoothed_spike_counts: np.array
#         Smoothed firing rates for the PSTH.
#     - window: tuple
#         Time window for the plots (start, end).
#     - save_path: str or None
#         Path to save the figure. If None, the figure is not saved.
#     """
#     plt.figure(figsize=(8, 8))

#     # Raster plot (top subplot)
#     plt.subplot(2, 1, 1)
#     for trial in np.unique(trials):  # Iterate through trials
#         spiketimes_in_trial = spiketimes[trials == trial]
#         plt.scatter(
#             spiketimes_in_trial,
#             [trial] * len(spiketimes_in_trial),  # Keep the trial index as the y-axis
#             s=5,
#             color='black'
#         )
#     plt.title(f"Raster Plot for Neuron {neuron_id}")
#     plt.ylabel("Trials")
#     if window:
#         plt.xlim(window)

#     # PSTH (bottom subplot)
#     plt.subplot(2, 1, 2)
#     plt.plot(bins, smoothed_spike_counts, label="Smoothed PSTH", color="blue")
#     plt.title(f"PSTH for Neuron {neuron_id}")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Firing Rate (Spikes/s)")
#     if window:
#         plt.xlim(window)
#     plt.legend()

#     # Adjust layout and save
#     plt.tight_layout()
#     if save_path:
#         # Create the directory if it doesn't exist
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         # Save the file with neuron_id appended to the filename
#         filename = os.path.join(save_path, f"neuron_{neuron_id}.png")
#         plt.savefig(filename)
#         plt.close()
        

def plot_raster_and_psth(neuron_id, spiketimes, trials, bins, smoothed_spike_counts, axes=None, window=None, save_path=None):
    """
    Plot a combined raster plot and PSTH for a neuron. Supports integration with external figures.

    Parameters:
    - neuron_id: int
        The ID of the neuron being plotted.
    - spiketimes: np.array
        Array of spike times aligned to an event.
    - trials: np.array
        Array of trial IDs corresponding to the spike times.
    - bins: np.array
        Bin centers for the PSTH.
    - smoothed_spike_counts: np.array
        Smoothed firing rates for the PSTH.
    - axes: list of Matplotlib Axes or None
        If provided, the function will plot on the given axes instead of creating a new figure.
    - window: tuple
        Time window for the plots (start, end).
    - save_path: str or None
        Path to save the figure. If None, the figure is not saved.
    """
    # If axes are not provided, create a standalone figure
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # Raster plot (top subplot)
    axes[0].scatter(spiketimes, trials, s=5, color='black')
    axes[0].set_title(f"Raster Plot for Neuron {neuron_id}")
    axes[0].set_ylabel("Trials")
    if window:
        axes[0].set_xlim(window)

    # PSTH (bottom subplot)
    axes[1].plot(bins, smoothed_spike_counts, label="Smoothed PSTH", color="blue")
    axes[1].set_title(f"PSTH for Neuron {neuron_id}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Firing Rate (Spikes/s)")
    if window:
        axes[1].set_xlim(window)
    axes[1].legend()

    # Adjust layout and save if needed
    if axes is None:
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"neuron_{neuron_id}.png"))
            plt.close()
        else:
            plt.show()



def compute_f1_f0(neuron_id, dir_id, psth, stimulus_freq, timebins, threshold=0.5, saveDirectory=None, plot=True):
    """
    Computes F1/F0 ratio from a PSTH and determines if the response is oscillatory.
    Sampling rate is inferred from `timebins`.

    Parameters:
    - psth (1D array): Peri-Stimulus Time Histogram (neural response).
    - stimulus_freq (float): Stimulus frequency (Hz).
    - timebins (1D array): Time bin edges (seconds) for the PSTH, used to infer sampling rate.
    - threshold (float, optional): Threshold for determining oscillation (default=0.5).
    - plot (bool, optional): If True, plots the FFT power spectrum.

    Returns:
    - f1_f0 (float): Ratio of F1 (modulation strength) to F0 (mean response).
    - is_oscillating (bool): True if F1/F0 exceeds threshold, otherwise False.
    """
    
    is_suppressed = analyze_suppression(psth, timebins)
    if is_suppressed:
        psth=-psth
    # Compute bin size and sampling rate
    binsize = np.diff(timebins)
    fs = 1 / np.mean(binsize)  # Sampling rate (Hz)
    
    # Window psth
    valid_idx = np.where((timebins[:-1] >= 0) & (timebins[:-1] < 2))[0]
    # Slice psth to keep only data in the time range [0, 2] sec
    psth_window = psth[valid_idx]
    # Compute F0 (Mean Firing Rate)
    f0 = np.abs(np.mean(psth_window))
    # Compute FFT and get the power spectrum
    n = len(psth_window)
    freqs = np.fft.fftfreq(n, d=1/fs)  # Frequency bins
    fft_values = fft(psth_window)
    power_spectrum = np.abs(fft_values[:n//2]) / n  # Start with correct normalization
    power_spectrum[1:] *= 2  # Double only non-DC components


    # Find index of the stimulus frequency in the FFT result
    f1_idx = np.argmin(np.abs(freqs - stimulus_freq))
    f1 = power_spectrum[f1_idx]  

    # Compute F1/F0 ratio
    f1_f0 = f1 / f0 if f0 > 0 else 0  # Avoid division by zero

    # Determine if the response is oscillatory
    is_oscillating = f1_f0 > threshold

    # Plot only FFT results
    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].plot(timebins, psth)
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Firing Rate (spikes/s)')
        ax[0].set_title(f'PSTH - Neuron {neuron_id} - Dir {dir_id}')

        ax[1].plot(freqs[:n//2], power_spectrum)
        ax[1].axvline(stimulus_freq, color='red', linestyle='dashed', label=f'Stimulus Frequency ({stimulus_freq} Hz)')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Power')
        ax[1].set_title(f'FFT Power Spectrum - Neuron {neuron_id} - F1/F0 = {f1_f0:.2f}, F0 = {f0:.2f}')
        ax[1].set_xlim(0, stimulus_freq * 2)

        plt.tight_layout()  # Adjust layout for clarity

        if saveDirectory is not None:
            if not os.path.isdir(saveDirectory):
                os.makedirs(saveDirectory)
            filename = os.path.join(saveDirectory, f'{neuron_id}_{dir_id}_PSTH.png')
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    return is_oscillating, f1_f0


def filter_F1(cluster_id, dir_id, timeBins, psth_trace, sigma, fir_coeff, state, pad_length = 6000, saveDirectory = None, plot = False):
    # padding from 0s
    expansion_amount = 3 * sigma
    
    start_time = 0 - expansion_amount
    end_time = 2 + expansion_amount
    
    start_idx = np.searchsorted(timeBins, start_time)
    end_idx = np.searchsorted(timeBins, end_time)
    stim_window_psth = psth_trace[start_idx:end_idx]
    padded_data = np.pad(stim_window_psth, pad_length, mode='edge')# Pad length is half the number of taps
    timeBins_stim_window = timeBins[start_idx:end_idx]
    time_step = np.mean(np.diff(timeBins))
    timeBins_padded = np.arange(
        timeBins[start_idx] - pad_length * time_step,
        timeBins[end_idx - 1] + pad_length * time_step + time_step,
        time_step
    )[:len(padded_data)]
    
    filtered_padded_data = filtfilt(fir_coeff, 1.0, padded_data)
    filtered_data = filtered_padded_data[pad_length:-pad_length]
    
    if plot:
        fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        
        if state == 0: 
            state = 'quiet'
        if state ==1: 
            state = 'active'
    
        axes[0].plot(timeBins, psth_trace)
        axes[0].set_title(f'Neuron {cluster_id}: Shifted mean PSTH - {state} state - Dir {dir_id}', fontsize=14, fontweight='bold')
    
        axes[1].plot(timeBins_padded, padded_data)
        axes[1].set_title('Shifted mean PSTH window padded', fontsize=14, fontweight='bold')
    
        axes[2].plot(timeBins_padded, filtered_padded_data)
        axes[2].set_title('Filtered Signal Padded', fontsize=14, fontweight='bold')
    
        axes[3].plot(timeBins_stim_window, filtered_data)
        axes[3].set_title('Filtered Signal', fontsize=14, fontweight='bold')
    
        plt.tight_layout()
        if not os.path.isdir(saveDirectory):
            os.makedirs(saveDirectory)
        filename = os.path.join(saveDirectory, f'{cluster_id}_{dir_id}_{state}_PSTH.png')
        plt.savefig(filename)
        plt.close()
        
    return filtered_data

def subtract_F1(neuron_id, dir_id, timeBins, filtered_data, original_psth_nonshift, shifts_s, state, binsize_signal, sigma, saveDirectory = None, plot=False):
    
    expansion_amount = 3 * sigma
    start_time = 0 - expansion_amount
    end_time = 2 + expansion_amount
    
    start_idx = np.searchsorted(timeBins, start_time)
    end_idx = np.searchsorted(timeBins, end_time)
    stim_window_ori_psth = original_psth_nonshift[start_idx:end_idx]
    timeBins_stim_window = timeBins[start_idx:end_idx]
    
    F1_trial = filtered_data/len(shifts_s) # F1 signal per trial
    F1_trial_shifted = np.zeros((len(shifts_s), len(F1_trial)))  # Placeholder for all shifted versions
    
    # for i, shift in enumerate(shifts_s):
    #     shift_bins = int(shift / binsize_signal)  # Convert shift in seconds to bins
    #     if shift_bins >= 0:
    #         F1_trial_shifted[i, shift_bins:] = F1_trial[:len(F1_trial) - shift_bins]
    #     else:
    #         F1_trial_shifted[i, :shift_bins] = F1_trial[-shift_bins:]
    
    for i, shift in enumerate(shifts_s):
        shift_bins = int(shift / binsize_signal)
        if shift_bins > 0:
            # Shift right (delay): fill from shift_bins onward
            F1_trial_shifted[i, shift_bins:] = F1_trial[:len(F1_trial) - shift_bins]
        elif shift_bins < 0:
            # Shift left (advance): fill up to shift_bins from the end
            F1_trial_shifted[i, :shift_bins] = F1_trial[-shift_bins:]
        else:
            # No shift
            F1_trial_shifted[i] = F1_trial

    F1_subtracted_signal = stim_window_ori_psth.copy()

    # Subtract each trial's shifted signal from the psth_trace
    for i in range(len(shifts_s)):
        F1_subtracted_signal -= F1_trial_shifted[i]
    
    if plot: 
        
        if state == 0: 
            state = 'quiet'
        if state ==1: 
            state = 'active'
            
        plt.figure()
        plt.plot(timeBins_stim_window, stim_window_ori_psth, label = 'Original')
        plt.plot(timeBins_stim_window, filtered_data, label = 'F1 (shifted spikes)')
        plt.plot(timeBins_stim_window, F1_trial, label = 'F1 trial no shift')
        if F1_trial_shifted[0] is not None and len(F1_trial_shifted[0]) > 0:
            plt.plot(timeBins_stim_window, F1_trial_shifted[0], label = 'F1 trial shift example')
        plt.plot(timeBins_stim_window, F1_subtracted_signal, label = 'F1 subtracted')
        plt.title(f'Neuron {neuron_id} - {state} state - dir {dir_id}')
        plt.legend()
        
        if not os.path.isdir(saveDirectory):
            os.makedirs(saveDirectory)
        filename = os.path.join(saveDirectory, f'{neuron_id}_{dir_id}_{state}_PSTH.png')
        plt.savefig(filename)
        plt.close()
    
    
        # plt.figure()
        # plt.plot(timeBins_stim_window, F1_trial, label = 'F1 trial no shift', linewidth=5, color='black')
        # for idx, trial in enumerate(F1_trial_shifted, start=1):
        #     plt.plot(timeBins_stim_window, trial, label=f'Trial {idx}')
        # plt.legend()
        
    return F1_trial_shifted, F1_subtracted_signal

def hilbert_transform(neuron, dir_id, filtered_psth, timeBins, state, saveDirectory=None, plot=False): 
    analytic_signal_filtered = hilbert(filtered_psth)
    signal_imag_filtered = np.imag(analytic_signal_filtered)
    amplitude_envelope_filtered = np.abs(analytic_signal_filtered)
    instantaneous_angle_filtered = np.angle(analytic_signal_filtered)

    important_phases = [0, np.pi/2, np.pi, -np.pi/2, -np.pi]
    
    if plot: 
        
        if state == 0: 
            state = 'quiet'
        if state ==1: 
            state = 'active'
            
            
        expansion_amount = 3 * sigma
        
        start_time = 0 - expansion_amount
        end_time = 2 + expansion_amount
        
        start_idx = np.searchsorted(timeBins, start_time)
        end_idx = np.searchsorted(timeBins, end_time)
        timeBins_stim_window = timeBins[start_idx:end_idx]
        # Plot the original and filtered signals along with their amplitude envelopes
        plt.figure(figsize=(15, 10))
    
        plt.plot(timeBins_stim_window, filtered_psth, label='Filtered (F1) Signal')
        plt.plot(timeBins_stim_window, signal_imag_filtered, color='red', label='Imaginary part')
        plt.plot(timeBins_stim_window, amplitude_envelope_filtered, label='Amplitude Envelope')
        plt.plot(timeBins_stim_window, instantaneous_angle_filtered, color='green', label='Angle')
    
        for phase in important_phases:
            plt.axhline(y=phase, color='gray', linestyle='--')
        plt.title(f'Neuron {neuron_id} - {state} state - dir {dir_id}')
        plt.legend()
        plt.tight_layout()
        if not os.path.isdir(saveDirectory):
            os.makedirs(saveDirectory)
        filename = os.path.join(saveDirectory, f'{neuron_id}_{dir_id}_{state}_PSTH.png')
        plt.savefig(filename)
        plt.close()
        
    return amplitude_envelope_filtered