# -*- coding: utf-8 -*-
# %% load libraries
"""
Created on Mon Mar 25 10:06:10 2024

@author: Experimenter
"""
# Population analysis
import os
import warnings
# import re
# import json
# import csv
# import io
# import struct
# from copy import deepcopy
# from collections import Counter

import numpy as np
from numpy.typing import NDArray
import pandas as pd
# from DataProcessingFunctions import DetectWheelMove_old
# import scipy
# import scipy as sp
# from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.optimize import minimize, Bounds
# import scipy.signal
# import scipy.stats as stats
# from scipy.stats import pearsonr, spearmanr, shapiro, ttest_ind
# from scipy.ndimage import gaussian_filter
# from scipy.stats import permutation_test
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error as mse
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
# from statsmodels.formula.api import ols
# import statsmodels.api as sm
import matplotlib.pyplot as plt
# from matplotlib import cm
# import seaborn as sns

# TODO: Import only functions that are used.
# import ephys
from Functions.PSTH import *
from Functions.neural_response import *


def adaptation_index(y_start_fit, y_end_fit, invert=False):
    """
    Calculate the adaptation index between two values, y_start_fit and y_end_fit.

    The adaptation index is computed as (y_start_fit - y_end_fit) / abs(y_start_fit + y_end_fit), and the result can be inverted 
    if `sens` is set to True. If certain conditions are met, such as opposite signs between y_start_fit and y_end_fit, 
    and the magnitude of y_start_fit is less than the negative of y_end_fit, the function returns infinity.

    Parameters:
    ----------
    y_start_fit : float
        The first value used in the calculation of the adaptation index.
    y_end_fit : float
        The second value used in the calculation of the adaptation index.
    sens : bool, optional
        A flag to indicate whether the result should be inverted.

    Returns:
    -------
    ai : float
        The adaptation index between y_start_fit and y_end_fit. Returns infinity if specific conditions are met.
    """
    # Check for conditions to return infinity
    if (y_start_fit < 0 and y_end_fit > 0 and y_start_fit < -y_end_fit) or (y_start_fit > 0 and y_end_fit < 0 and y_start_fit < -y_end_fit):
        return float("inf")

    ai = (y_start_fit - y_end_fit) / abs(y_start_fit + y_end_fit)

    # Invert the result if `invert` is True; this is for mixed sensitising phase of the mixed model which is flipped
    if invert:
        ai = -ai

    return ai


def exclude_low_activity(trials_spAligned, running_state):
    """
    Determine whether a neuron should be excluded based on low firing rate activity, by calculating the percentage of trials 
    with zero spikes and excluding the neuron if 70% or more trials have no spikes.

    Parameters:
    ----------
    trials : np.ndarray
        An array that shows which trial each spike belongs to. Hence, it will only show the number of a trial if a spike occurred in said trial
    
    running_state should be explicitly stated - either 0 or 1 or both with | operator:  (running_state == 0) | (running_state == 1)
    Returns:
    -------
    exclude : bool
        True if the neuron should be excluded (i.e., 70% or more trials have zero spikes), False otherwise.
    """
    state_trials = np.where(running_state)[0]
    state_trials_mask = np.isin(trials_spAligned, state_trials)
    trials_spAligned_filt = trials_spAligned[state_trials_mask]
    num_trials_with_spikes = len(set(trials_spAligned_filt))
    max_trials = len(state_trials)
    
    zero_spike_trials = max_trials - num_trials_with_spikes
    zero_spike_percentage = (zero_spike_trials / max_trials) * 100
    
    return zero_spike_percentage >= 70


def dir_selectivity_check(
    direction,
    fr,
    groups,
    tuning_threshold,
    is_suppressed,
):
    """
    Check for direction and orientation selectivity of a neuron based on firing rate (fr) and stimulus directions.

    This function performs a permutation test to determine direction selectivity index (DSI) and orientation selectivity index (OSI).
    If the neuron shows significant selectivity (p-value < 0.05) for direction or orientation, the function calculates the preferred 
    direction or orientation and adjusts stimulus tuning accordingly.

    Parameters:
    ----------
    direction : np.ndarray
        Array of stimulus directions used during the experiment.
    fr : np.ndarray
        Firing rate per trial.
    meanT : np.ndarray
        Array of mean firing rates for each stimulus condition.
    semT : np.ndarray
        Array of standard error of the mean firing rates for each stimulus condition.
    stimT : np.ndarray
        Array of stimulus directions tested.
    filtered_ind : np.ndarray
        Indices of all trials.
    groups : np.ndarray
        Array representing grouping of stimulus directions or orientations.
    tuning_threshold : float
        Threshold for including nearby directions as preferred if their firing rates are close to the preferred direction's rate.
    is_suppressed : bool
        Boolean flag indicating whether the neuron shows suppression.

    Returns:
    -------
    filtered_ind : np.ndarray
        Updated trial indices after filtering based on selectivity. If neuron is not selective, all trials are included.
    ds_neuron : bool
        True if the neuron is direction-selective.
    os_neuron : bool
        True if the neuron is orientation-selective.
    """

    # Perform permutation test for direction selectivity
    dir_stats = dsi_stat(direction, fr, repetitions=5000, is_suppressed=is_suppressed)
    dir_ind = []
    pref_dirs = []
    ds_neuron = False
    
    
    fr_ = fr.copy()
    if is_suppressed:
        fr_ = -fr_
    stimuli_tuning = np.unique(direction)
    mean_tuning = np.array([np.mean(fr_[direction == angle]) for angle in stimuli_tuning])
    
    if dir_stats.pvalue < 0.05: 
        ds_neuron = True
        prefD = pref_dir(direction, fr, is_suppressed=is_suppressed)
        if np.isin(360, stimuli_tuning):
            stimuli_tuning[np.where(stimuli_tuning == 360)[0]] = 0  
        closer_dir = groups[np.argmin(np.abs(stimuli_tuning - prefD))] # direction closest to the preferred

        pref_dirs = [closer_dir]
        if np.isin(0, stimuli_tuning):  # Hardcoded conversion back to degrees
            stimuli_tuning[np.where(stimuli_tuning == 0)[0]] = 360
        preferred_index = np.where(stimuli_tuning == closer_dir)[0][0]
        
        adjacent_indices = [
            (preferred_index - 1) % len(stimuli_tuning), # % len(stimuli_tuning) ensures that if the index goes outside the bounds of the array,
            (preferred_index - 2) % len(stimuli_tuning), # it wraps around to the other end of the array
            (preferred_index + 1) % len(stimuli_tuning),
            (preferred_index + 2) % len(stimuli_tuning),
        ]

        # Add adjacent directions if their firing rates are close to the preferred direction's
        for ai in adjacent_indices:
            if mean_tuning[ai] > mean_tuning[stimuli_tuning == closer_dir]: # if the adjacent is more active than pref
                pref_dirs.append(stimuli_tuning[ai])
            elif mean_tuning[ai] > (mean_tuning[stimuli_tuning == closer_dir] * tuning_threshold):
                pref_dirs.append(stimuli_tuning[ai])

        dir_ind = np.where(np.isin(direction, pref_dirs))[0]
            
    return ds_neuron, dir_ind, dir_stats, pref_dirs 

def ori_selectivity_check(
    direction,
    fr,
    groups,
    tuning_threshold,
    is_suppressed,
):           
            
            
    ori_stats = osi_stat(direction, fr, repetitions=5000, is_suppressed=is_suppressed)  
    # osi_manual, p_value, osi_null = osi_stat_manual(direction, fr, repetitions=5000, is_suppressed=is_suppressed)
    ori_ind = []  
    pref_oris = []
    os_neuron = False
    stimuli_tuning = np.unique(direction)
    fr_ = fr.copy()
    if is_suppressed:
        fr_ = -fr_
    mean_tuning = np.array([np.mean(fr_[direction == angle]) for angle in stimuli_tuning])
    if ori_stats.pvalue < 0.05:
        os_neuron = True
        prefO = pref_ori(direction, fr, is_suppressed=is_suppressed)
        if np.isin(360, stimuli_tuning):
            stimuli_tuning[np.where(stimuli_tuning == 360)[0]] = 0
        closer_ori = groups[np.argmin(np.abs(stimuli_tuning - prefO))] 
        if closer_ori == 360: # TODO: is this correct? check again
            closer_ori = [closer_ori, closer_ori - 180]
        else:
            closer_ori = [closer_ori, closer_ori + 180]
        pref_oris = closer_ori
        if np.isin(0, stimuli_tuning):  # Hardcoded conversion back to degrees
            stimuli_tuning[np.where(stimuli_tuning == 0)[0]] = 360
        preferred_index = np.where(stimuli_tuning == closer_ori[0])[0][0] # based on the first pref ori  
        adjacent_indices = [
            (preferred_index - 1) % len(stimuli_tuning), # % len(stimuli_tuning) ensures that if the index goes outside the bounds of the array,
            (preferred_index + 1) % len(stimuli_tuning),
        ]
        for ai in adjacent_indices:
            if mean_tuning[ai] > mean_tuning[stimuli_tuning == closer_ori[0]]: # if the adjacent is more active than pref
                pref_oris.append(stimuli_tuning[ai])
                pref_oris.append(stimuli_tuning[ai]+180)
            elif mean_tuning[ai] > (mean_tuning[stimuli_tuning == closer_ori[0]] * tuning_threshold):
                pref_oris.append(stimuli_tuning[ai])
                pref_oris.append(stimuli_tuning[ai]+180)
        ori_ind = np.where(np.isin(direction, pref_oris))[0]
        
    return os_neuron, ori_ind, ori_stats, pref_oris


def strong_response_trials(
    direction : NDArray,
    fr : NDArray,
    tuning_threshold : float,
    is_suppressed : bool,
) -> tuple[NDArray, NDArray, float, float]:
    """
    Identifies trials with strong neuronal responses based on tuning threshold.

    Parameters
    ----------
    direction : np.array
        1D array of shape (n_trials,) representing the stimulus direction
        presented in each trial.
    fr : np.array
        1D array of shape (n_trials,) containing the mean firing rate
        in each trial.
    tuning_threshold : float
        A scalar between 0 and 1 that sets the relative threshold for
        selecting strong responses. Determines the proportion of the
        maximum normalized firing rate used to identify significant tuning.
    is_suppressed : bool
        Flag indicating whether the neuron's responses are suppressed.
        If `True`, firing rates are inverted.

    Returns
    -------
    dir_ind : np.array
        1D array of indices corresponding to trials where the stimulus
        direction is among the preferred directions exceeding the threshold.
    pref_dirs : np.array
        1D array of unique stimulus directions that have an average
        firing rate above the specified tuning threshold.
    threshold_fr : float
        The firing rate threshold used to determine strong responses.
    max_response_fr : float
        The highest average firing rate observed across all stimulus directions.
    """

    if is_suppressed:
        fr = -fr

    # Assuming fr_ is the firing rate array and direction contains the stimulus directions
    stimuli_tuning = np.unique(direction)

    # Calculate the mean tuning for each unique stimulus direction
    mean_tuning = np.array([np.mean(fr[direction == angle]) for angle in stimuli_tuning])
    mean_tuning = np.where(mean_tuning < 0, 0, mean_tuning)
    mean_tuning_norm = mean_tuning/np.nansum(mean_tuning, keepdims=True)
    # Find the index of the maximum response in mean_tuning
    max_idx = np.argmax(mean_tuning_norm)

    # Find the indices where the response is greater than the threshold
    threshold = tuning_threshold * mean_tuning_norm[max_idx]
    indices_above_threshold = np.where(mean_tuning_norm > threshold)[0]
    threshold_fr = threshold * np.nansum(mean_tuning)
    # Get the corresponding stimulus directions from stimuli_tuning
    pref_dirs = stimuli_tuning[indices_above_threshold]
    # Find the trial indices in direction that match any of the preferred directions
    dir_ind = np.where(np.isin(direction, pref_dirs))[0]

    # Output results
    max_response_fr = mean_tuning[max_idx]

    return dir_ind, pref_dirs, threshold_fr, max_response_fr


# %%  mixed exponential


def mixed_response_model(timeBins, params, y_start_fit, first_peak_index, end_index):
    """
    Models a mixed neural response that includes an adaptation phase followed by a sensitisation phase.

    This function describes a mixed model response with an initial adaptation phase (exponential decay)
    and a later sensitisation phase (flipped exponential decay). The two phases are concatenated to 
    produce the final pooled response, switching at a specified time point (t_switch).

    Parameters:
    ----------
    timeBins : np.ndarray
        Array of time points corresponding to the neural response.
    params : list or tuple
        Model parameters in the following order:
        [tau1 (float): time constant for adaptation phase,
         C1 (float): asymptotic value for adaptation phase,
         tau2 (float): time constant for sensitisation phase,
         C2 (float): asymptotic value for sensitisation phase,
         A (float): amplitude scaling factor for sensitisation phase,
         t_switch (float): time of switch from adaptation to sensitisation phase].
    y_start_fit : float
        Initial firing rate at the start of the adaptation phase.
    first_peak_index : int
        Index of the timeBins array where the fitting begins.
    end_index : int
        Index of the timeBins array where the fitting ends.

    Returns:
    -------
    pooled_response : np.ndarray
        The pooled neural response modeled by the adaptation and sensitisation phases.
    """

    tau1, C1, tau2, C2, A, t_switch = params
    switch_index = np.searchsorted(timeBins, t_switch, side="right") - 1

    t_a = timeBins[first_peak_index : switch_index + 1] - timeBins[first_peak_index]
    t_s = timeBins[switch_index:end_index] - timeBins[switch_index]

    adaptation_response = (y_start_fit - C1) * np.exp(-t_a / tau1) + C1
    sensitisation_response = A * np.exp(-t_s / tau2) + C2
    sensitisation_response = np.flip(sensitisation_response)
    pooled_response = np.concatenate((adaptation_response, sensitisation_response))

    return pooled_response

def fit_mixed_exp_best_t_guess(
    psth_to_fit,
    timeBins,
    params_adapt,
    params_sens,
    y_start_fit_adapt,
    start_time_for_fit_adapt,
    sigma
):
    """

    This function iterates over a range of possible switch times between adaptation and sensitization
    phases to find the best t_switch for a mixed model based on root mean squared error (RMSE) 
    between the actual PSTH and the model predictions. The t_switch is then used as a guess in the fitting process.

    Parameters:
    ----------
    psth_to_fit : np.ndarray
        PSTH data that represents the firing rate to be fit by the model.
    timeBins : np.ndarray
        Array of time points corresponding to the original PSTH data (not psth_to_fit).
    params_adapt : tuple or list
        Parameters from the adaptation model (tau1, C1) to be used as initial guesses, or None if not provided.
    params_sens : tuple or list
        Parameters from the sensitization model (tau2, A) to be used as initial guesses, or None if not provided.
    y_start_fit_adapt : float
        Initial starting value of the firing rate for fitting the adaptation phase.
    start_time_for_fit_adapt : float
        Start time for fitting the adaptation phase.
    sigma : float
        Margin to account for edge effects during convolution, used to limit the upper bound of the time window.

    Returns:
    -------
    t_switch : float or None
        The best switch time between adaptation and sensitization phases that minimizes RMSE, or
        None if optimization fails.
    """


    #TODO: add comment about why you are doing this 
    if start_time_for_fit_adapt == 0:
        first_peak_index = np.searchsorted(timeBins, start_time_for_fit_adapt, side="right")
    else:
        first_peak_index = np.searchsorted(timeBins, start_time_for_fit_adapt, side="right") - 1
    
    end_time_for_fit = 2 - 2 * sigma
    end_index = np.searchsorted(timeBins, end_time_for_fit, side="right") - 1

    switch_times = np.arange(0.2, 1.1, 0.1)
    
    
    
    
    # t_switch = 0.5
    # switch_index = np.searchsorted(timeBins, t_switch, side="right") - 1

    # t_a = timeBins[first_peak_index : switch_index + 1] - timeBins[first_peak_index]
    # t_s = timeBins[switch_index:end_index] - timeBins[switch_index]
    
    # t_a_s = np.concatenate((t_a, t_s))
    
    # the error occurs when switch_index is before first_peak_index
    
    
    
    
    best_result = None
    min_rmse = np.inf

    if params_adapt is not None:
        C1_guess = params_adapt[1]
        tau1_guess = params_adapt[0]
    else:
        C1_guess = np.min(psth_to_fit)  
        tau1_guess = 1  

    if params_sens is not None:
        tau2_guess = params_sens[0]
        A_guess = params_sens[2]
    else:
        tau2_guess = 1  
        A_guess = np.max(psth_to_fit) 

    for t_switch in switch_times:
        switch_index = np.searchsorted(timeBins, t_switch, side="right") - 1
        if switch_index < first_peak_index:
            continue
        constraints = [
            {
                "type": "eq",
                "fun": continuity_constraint,
                "args": (timeBins, y_start_fit_adapt, first_peak_index, end_index),
            },
            {
                "type": "ineq",
                "fun": lambda params: y_start_fit_adapt - 1.2 * params[1],
            },  # (y_start_fit - C1) > 0.2 * C1
            {
                "type": "ineq",
                "fun": lambda params: (y_start_fit_adapt - params[1]) / params[1]
                - params[0]
                if params[0] > 1
                else y_start_fit_adapt - params[1],
            },  # tau1 >= (y_start_fit - C1)/C1 if C1 > 1 else tau1 >= y_start_fit
            {
                "type": "ineq",
                "fun": lambda params: params[4] - 0.2 * params[3],
            },  # A > 0.2 * C2
            {
                "type": "ineq",
                "fun": lambda params: (1.5 * params[4] / params[3]) - params[2]
                if params[2] > 1
                else params[5] - params[3],
            },  # tau2 <= 1.5 * A / C2
            {
                "type": "ineq",
                "fun": lambda params: (params[4] / (y_start_fit_adapt - min(psth_to_fit)))
                - 0.2,
            },  # A / range of response > 0.2
        ]

        initial_guess = [
            tau1_guess, # tau1
            C1_guess, # C1
            tau2_guess, # tau2
            np.min(psth_to_fit), # C2
            A_guess, # A
            t_switch, # t_switch
        ]

        bounds = [
            (0.01, 20),  # tau1
            (np.min(psth_to_fit), y_start_fit_adapt),  # C1
            (0.3, 2),  # tau2
            (np.min(psth_to_fit), 2 * np.max(psth_to_fit)),  # C2
            (0.05 * np.max(psth_to_fit), 3 * np.max(psth_to_fit)),  # A
            (max(0.2, start_time_for_fit_adapt), 1.2),  # t_switch
        ]

        try:
            with warnings.catch_warnings():
                #TODO: check if this message is not covered by the simple filter
                warnings.filterwarnings(
                    "ignore",
                    message="delta_grad == 0.0. Check if the approximated function is linear.",
                )
                warnings.simplefilter("ignore", RuntimeWarning)
                
                result = minimize(
                    objective_function,
                    initial_guess,
                    args=(timeBins, psth_to_fit, y_start_fit_adapt, first_peak_index, end_index),
                    constraints=constraints,
                    method="SLSQP",
                    bounds=bounds,
                )

                # If the optimization succeeds, calculate the RMSE and update the best result
                if result.success:
                    fitted_firing_rates = mixed_response_model(
                        timeBins, result.x, y_start_fit_adapt, first_peak_index, end_index
                    )
                    rmse = np.sqrt(np.mean((psth_to_fit - fitted_firing_rates) ** 2))
                    if rmse < min_rmse:
                        min_rmse = rmse
                        best_result = {
                            "params": result.x,
                            "t_switch": t_switch,
                        }
        except Exception:
            pass

    if best_result:
        return best_result["t_switch"]
    else:
        # print("Optimization failed for all switch times.")
        return None


def fit_mixed_exp(
    psth_to_fit,
    timeBins,
    params_adapt,
    params_sens,
    y_start_fit_adapt,
    start_time_for_fit_adapt,
    t_switch,
    sigma
):
    """
    Fits a mixed exponential model (adaptation and sensitization phases) to PSTH data using constrained optimization.

    The function attempts to fit a mixed model consisting of adaptation (exponential decay) and 
    sensitization (flipped exponential decay) phases by minimizing the error between the model and the 
    peristimulus time histogram (PSTH). 

    Parameters:
    ----------
    psth_to_fit : np.ndarray
        The peristimulus time histogram (PSTH) data to fit the model to.
    timeBins : np.ndarray
        Array of time points corresponding to the original PSTH data (not psth_to_fit).
    params_adapt : tuple or list
        Parameters from the adaptation model (tau1, C1) to be used as initial guesses, or None if not provided.
    params_sens : tuple or list
        Parameters from the sensitization model (tau2, A) to be used as initial guesses, or None if not provided.
    y_start_fit_adapt : float
        Initial starting value of the firing rate for fitting the adaptation phase.
    start_time_for_fit_adapt : float
        Start time for fitting the adaptation phase.
    t_switch : float
        Switch time between adaptation and sensitization phases.
    sigma : float
        Margin to account for edge effects during convolution, used to limit the upper bound of the time window.

    Returns:
    -------
    result : dict or None
        A dictionary containing the optimized parameters, fitted firing rates, adaptation indices, 
        and switch time. Returns None if optimization fails.
    """
    #TODO: add comment about why you are doing this 
    if start_time_for_fit_adapt == 0:
        first_peak_index = np.searchsorted(timeBins, start_time_for_fit_adapt, side="right")
    else:
        first_peak_index = np.searchsorted(timeBins, start_time_for_fit_adapt, side="right") - 1

    end_time_for_fit = 2 - 2 * sigma
    end_index = np.searchsorted(timeBins, end_time_for_fit, side="right") - 1    

    constraints = [
        {
            "type": "eq",
            "fun": continuity_constraint,
            "args": (timeBins, y_start_fit_adapt, first_peak_index, end_index),
        },
        {
            "type": "ineq",
            "fun": lambda params: y_start_fit_adapt - 1.2 * params[1],
        },  # (y_start_fit - C1) > 0.2 * C1
        {
            "type": "ineq",
            "fun": lambda params: (y_start_fit_adapt - params[1]) / params[1] - params[0]
            if params[0] > 1
            else y_start_fit_adapt - params[1],
        },  # tau1 >= (y_start_fit - C1) / C1 if C1 > 1 else tau1 >= y_start_fit
        {
            "type": "ineq",
            "fun": lambda params: params[4] - 0.2 * params[3],
        },  # A > 0.2 * C2
        {
            "type": "ineq",
            "fun": lambda params: (1.5 * params[4] / params[3]) - params[2]
            if params[2] > 1
            else params[5] - params[3],
        },  # tau2 <= 1.5 * A / C2
        {
            "type": "ineq",
            "fun": lambda params: (params[4] / (y_start_fit_adapt - np.min(psth_to_fit))) - 0.2,
        },  # A / range of response > 0.2
    ]

    if params_adapt is not None:
        C1_guess = params_adapt[1]
        tau1_guess = params_adapt[0]
    else:
        C1_guess = np.min(psth_to_fit)  # Default guess if params_adapt is None
        tau1_guess = 1  # Default guess for tau1

    if params_sens is not None:
        tau2_guess = params_sens[0]
        A_guess = params_sens[2]
    else:
        tau2_guess = 1  # Default guess for tau2
        A_guess = np.max(psth_to_fit)  # Default guess for A

    initial_guess = [
        tau1_guess, # tau1
        C1_guess, # C1
        tau2_guess, # tau2
        np.min(psth_to_fit), # C2
        A_guess, # A
        t_switch, # t_switch
    ]

    bounds = [
        (0.01, 20),  # tau1
        (np.min(psth_to_fit), y_start_fit_adapt),  # C1
        (0.3, 2),  # tau2
        (np.min(psth_to_fit), 2 * np.max(psth_to_fit)),  # C2
        (0.05 * np.max(psth_to_fit), 3 * np.max(psth_to_fit)),  # A
        (0.2, 1.2),  # t_switch
    ]

    try:
        with warnings.catch_warnings():
            #TODO: check if this message is not covered by the simple filter
            warnings.filterwarnings(
                "ignore",
                message="delta_grad == 0.0. Check if the approximated function is linear. If the function is linear better results can be obtained by defining the Hessian as zero instead of using quasi-Newton approximations.",
            )
            warnings.simplefilter("ignore", RuntimeWarning)

            # Perform constrained optimization using SLSQP method
            result = minimize(
                objective_function,
                initial_guess,
                args=(timeBins, psth_to_fit, y_start_fit_adapt, first_peak_index, end_index),
                constraints=constraints,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": 1000},
            )

            if result.success:
                optimized_params = result.x
                fitted_firing_rates = mixed_response_model(
                    timeBins,
                    optimized_params,
                    y_start_fit_adapt,
                    first_peak_index,
                    end_index,
                )

                t_switch = optimized_params[5]
                C1 = optimized_params[1]
                tau1 = optimized_params[0]
                y_end_fit_adapt = (y_start_fit_adapt - C1) * np.exp(-2 / tau1) + C1

                C2 = optimized_params[3]
                tau2 = optimized_params[2]
                A = optimized_params[4]
                y_end_fit_sens = A * np.exp(-2 / tau2) + C2
                y_start_fit_sens = A + C2


                ai_1 = adaptation_index(
                    y_start_fit_adapt, y_end_fit_adapt, invert=False
                )
                ai_2 = adaptation_index(
                    y_start_fit_sens, y_end_fit_sens, invert=True
                )

                return optimized_params, fitted_firing_rates, ai_1, ai_2

    except Exception:
        # print("Optimization failed in mixed")  
        return None, None, None, None


def objective_function(
    params, timeBins, psth_to_fit, y_start_fit_adapt, first_peak_index, end_index
):
    """
    Objective function to minimize during model fitting.

    This function computes the sum of squared residuals between the actual PSTH data and the 
    model's predicted firing rate based on the current set of parameters. This function is 
    used in the optimization process to find the best fit of the mixed response model.

    Parameters:
    ----------
    params : np.ndarray
        Array of model parameters to be optimized, including tau1, C1, tau2, C2, A, and t_switch.
    timeBins : np.ndarray
        Array of time points corresponding to the original PSTH data.
    psth_to_fit : np.ndarray
        The actual PSTH data that the model is trying to fit.
    y_start_fit_adapt : float
        The initial firing rate at the start of the adaptation phase.
    first_peak_index : int
        The index in timeBins where the adaptation phase begins.
    end_index : int
        The index in timeBins where the fitting ends.

    Returns:
    -------
    float
        The sum of squared residuals between the model predictions and the actual PSTH data.
    """

    fitted_data = mixed_response_model(
        timeBins,
        params,
        y_start_fit_adapt,
        first_peak_index,
        end_index,
    )
    residuals = fitted_data - psth_to_fit
    return np.sum(residuals**2)


def continuity_constraint(params, timeBins, y_start_fit, first_peak_index, end_index):
    """
    Continuity constraint to ensure that the model's adaptation and sensitization phases match at the switch time.

    This function calculates the difference between the firing rate at the switch time for both the adaptation 
    and sensitization phases which should be equal to 0 because this is an equality constraint. 

    Parameters:
    ----------
    params : np.ndarray
        Array of model parameters, including tau1, C1, tau2, C2, A, and t_switch.
    timeBins : np.ndarray
        Array of time points corresponding to the original PSTH data.
    y_start_fit : float
        Initial firing rate at the start of the adaptation phase.
    first_peak_index : int
        The index in timeBins where the adaptation phase begins.
    end_index : int
        The index in timeBins where the fitting ends.

    Returns:
    -------
    float
        The difference between the firing rates of the adaptation and sensitization phases at the switch point.
        A return value of 0 indicates continuity at the switch point.
    """

    tau1, C1, tau2, C2, A, t_switch = params

    switch_index = np.searchsorted(timeBins, t_switch, side="right") - 1

    adaptation_at_switch = (y_start_fit - C1) * np.exp(
        -(timeBins[switch_index] - timeBins[first_peak_index]) / tau1
    ) + C1

    sensitization_at_switch = (
        A * np.exp(-(timeBins[end_index] - timeBins[switch_index]) / tau2) + C2
    )
    # Flip the sensitization response to match the rising phase behavior
    sensitization_at_switch = np.flip(sensitization_at_switch)

    return adaptation_at_switch - sensitization_at_switch

# %% single exponentials

def data_window_psth(psth_data, timeBins, sigma):
    """
    Extract and prepare a time window of PSTH data for fitting analysis.

    This function selects a time window (from 0 to approximately 2 seconds)
    from the PSTH data, avoiding edge effects by applying a margin defined by
    `sigma`. The function then identifies the starting point (time and value)
    for fitting based on the maximum firing rate in the first 0.5 seconds or
    on peaks in the data. If a peak is found before 0.5 seconds, it is used as
    the fitting start point; otherwise, the maximum value within the first 0.5
    seconds is used. If the first peak is smaller than the second, start time
    of the first peak is used and the starting value (y_start_fit) is taken as
    the average between the first and the second peak values.

    Parameters:
    ----------
    psth_data : np.ndarray
        Array of PSTH firing rate data for the neuron.
    timeBins : np.ndarray
        Array of time timeBins corresponding to the `psth_data`, representing when each data point occurred.
    sigma : float
        Margin to account for edge effects during convolution, used to limit the upper bound of the time window.

    Returns:
    -------
    y_start_fit : float
        Firing rate at the determined start point for fitting, based on peaks or the maximum in the first 0.5 seconds.
    start_time_for_fit : float
        The time point where fitting starts, based on identified peaks or maximum firing rates.
    psth_to_fit : np.ndarray
        The portion of the PSTH data selected for fitting, starting from `start_time_for_fit`.
    time_to_fit : np.ndarray
        Time timeBins corresponding to `psth_to_fit`, adjusted to start from 0 (relative to `start_time_for_fit`).
    """

    window_mask = (timeBins >= 0) & (timeBins <= 2 - 2 * sigma) # to avoid edge effects from convolution
    time_window = timeBins[window_mask]
    psth_window = psth_data[window_mask]

    # Find the maximum value in the first quarter of the time window
    time_index_05s = np.searchsorted(time_window, 0.5)  # < 0.5s
    index_of_y_start_fit = np.argmax(psth_window[:time_index_05s])

    # Find indices of peaks in the PSTH data
    peaks, _ = find_peaks(psth_window, height=np.max(psth_window) * 0.1)

    # Determine the start time for fitting based on the first peak or max point before 0.5s
    if len(peaks) == 0 or time_window[peaks[0]] > 0.5:
        # if no peaks found or peak after 0.5s, use the maximum in the first 0.5 s
        start_time_for_fit = time_window[index_of_y_start_fit]
        y_start_fit = psth_window[index_of_y_start_fit]
    elif time_window[peaks[0]] <= 0.5:
        # if first peak occurs within the first 0.5s, use that as start time
        start_time_for_fit = time_window[peaks[0]]
        y_start_fit = psth_window[peaks[0]]
        if len(peaks) > 1: # check if second peak is higher than the first
            first_peak_value = psth_window[peaks[0]]
            second_peak_value = psth_window[peaks[1]]
            if time_window[peaks[1]] <= 0.5 and second_peak_value > first_peak_value:
                y_start_fit = (first_peak_value + second_peak_value) / 2

    fit_mask = time_window >= start_time_for_fit
    time_to_fit = time_window[fit_mask] - start_time_for_fit # adjusted to start from 0
    psth_to_fit = psth_window[fit_mask]

    return y_start_fit, start_time_for_fit, psth_to_fit, time_to_fit


def exponential_model_sens(t, tau, C, A):
    """
    Exponential decay model to describe sensitising response.

    The model represents the firing rate as an exponential decay with a time constant `tau`, an asymptotic 
    constant value `C`, and an amplitude `A`. The negative sign in front of 
    `A` causes the response to increase over time instead of decrease, starting from a lower value and 
    approaching the constant value `C` asymptotically.

    Parameters:
    ----------
    t : float or np.ndarray
        Time points at which the firing rate is calculated.
    tau : float
        Time constant of the exponential decay.
    C : float
        Asymptotic constant value the firing rate approaches.
    A : float
        Amplitude of the exponential decay.

    Returns:
    -------
    float or np.ndarray
        Firing rate at the given time point(s).
    """
    return -A * np.exp(-t / tau) + C


def to_fit_sens_check(psth_to_fit_mean):
    """
    Checks whether the sensitization model should be fitted based on firing rates.

    This function compares the mean firing rates of the first and second halves of the PSTH (Peri-Stimulus 
    Time Histogram). If the mean firing rate in the second half is greater than the first half, this suggests 
    a sensitisation effect (increased firing rate over time), and the function returns True, indicating 
    that the sensitization model should be fitted.

    Parameters:
    ----------
    psth_to_fit_mean : np.ndarray
        A 1D array representing the mean PSTH across trials, where each element corresponds to the firing 
        rate at a specific time bin.

    Returns:
    -------
    bool
        True if the sensitization model should be fitted (i.e., if the mean firing rate in the second half 
        is greater than the first half), otherwise False.
    """
    half_index = len(psth_to_fit_mean) // 2
    first_half = psth_to_fit_mean[:half_index]
    second_half = psth_to_fit_mean[half_index:]

    mean_first_half = np.mean(first_half)
    mean_second_half = np.mean(second_half)
    to_fit_sens = mean_second_half > mean_first_half
    
    return to_fit_sens


def fit_sens_model(psth_to_fit, time_to_fit):
    """
    Fit a sensitisation model to the given peristimulus time histogram (PSTH) data using a negative exponential decay function.

    This function uses a constrained optimization approach to fit the model to the PSTH data. The model is 
    described by three parameters: tau (time constant), C (constant), and A (amplitude). The function also 
    calculates the adaptation index (AI).

    Parameters:
    ----------
    psth_to_fit : np.ndarray
        PSTH data that represents the firing rates to be fit by the model.
    time_to_fit : np.ndarray
        Time points corresponding to the PSTH data.

    Returns:
    -------
    popt : np.ndarray or None
        The optimized parameters for the model [tau, C, A], or None if the fitting fails.
    fitted_firing_rate : np.ndarray or None
        Firing rate calculated from the model using the optimized parameters, or None if the fitting fails.
    ai : float or None
        Adaptation index calculated based on the fitted parameters, or None if the fitting fails.
    """
    
    def objective_sens(params, time_to_fit, psth_to_fit):
        """
        Objective function to minimize, calculating the sum of squared errors between the model and the given data.
        The goal is to minimize this error to achieve the best fit of the model to the data.
        
        Returns:
        -------
        float
            The sum of squared errors between the model predictions and the PSTH data.
        """
        tau, C, A = params
        return np.sum((exponential_model_sens(time_to_fit, tau, C, A) - psth_to_fit) ** 2)

     
    A_C_guess = np.max(psth_to_fit)
    initial_guess = [1, A_C_guess, A_C_guess]

    bounds = Bounds(
        [0.5, max(1, min(psth_to_fit)), 0.05 * A_C_guess],  # Lower bounds [tau, C, A]
        [2, 2 * A_C_guess, 3 * A_C_guess],  # Upper bounds [tau, C, A]
    )

    constraints = [
        {"type": "ineq", "fun": lambda params: params[2] - 0.2 * params[1]},  # A > 0.2 * C rewritten as A - 0.2*C
        {"type": "ineq", "fun": lambda params: 2.5 * params[2] / params[1] - params[0]}  # tau <= 2.5 * A / C rewritten as 2.5*A/C - tau
    ]

    with warnings.catch_warnings():
        # Suppress runtime warnings related to optimization
        warnings.simplefilter("ignore", RuntimeWarning)

        result = minimize(
            objective_sens,
            initial_guess,
            args=(time_to_fit, psth_to_fit), # arguments of the objective function
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-2},
        )

    if result.success:
        popt = result.x  # Optimized parameters [tau, C, A]
        C, A = popt[1], popt[2]  # Extract optimized C and A
        y_start_fit_sens = -A + C
        y_end_fit = exponential_model_sens(2, *popt)
        fitted_firing_rate = exponential_model_sens(time_to_fit, *popt)


        ai = adaptation_index(y_start_fit_sens, y_end_fit, invert=False)
        
        return popt, fitted_firing_rate, ai
    else:
        return None, None, None


def exponential_model_adapt(t, tau, C, y_start_fit):
    """
    Exponential decay model describing the adaptation response.

    The model represents the firing rate as an exponential decay from an initial starting point `y_start_fit` to an
    asymptotic constant value `C` over time `t` with time constant `tau`.
    
    Parameters:
    ----------
    t : float or np.ndarray
        Time point(s) at which the firing rates are calculated.
    tau : float
        Time constant of the exponential decay, controlling how quickly the firing rate approaches the
        asymptote.
    C : float
        The constant value that the firing rate approaches over time.
    y_start_fit : float
        The initial firing rate at time t = 0.  This is provided as a fixed input and is not optimized during fitting.

    Returns:
    -------
    float or np.ndarray
        Firing rate(s) at the given time point(s).
    """
    return (y_start_fit - C) * np.exp(-t / tau) + C


def fit_adapt_model(y_start_fit, psth_to_fit, time_to_fit):
    """
    Fits an adaptation model to a given peristimulus time histogram (PSTH) using exponential decay function.

    This function uses a constrained optimization approach to fit the model to the PSTH data. The model is 
    described by two parameters: tau (time constant) and C (constant). The function also 
    calculates the adaptation index (AI).

    Parameters:
    ----------
    y_start_fit : float
        Initial starting value of the firing rate for the fitting process.
    psth_to_fit : np.ndarray
        PSTH data that represents the firing rate to be fit by the model.
    time_to_fit : np.ndarray
        Time points corresponding to the PSTH data.

    Returns:
    -------
    popt : np.ndarray or None
        The optimized parameters for the model [tau, C], or None if the fitting fails.
    fitted_firing_rate : np.ndarray or None
        Firing rate calculated from the model using the optimized parameters, or None if the fitting fails.
    ai : float or None
        Adaptation index calculated based on the fitted parameters, or None if the fitting fails
    """
    
    def objective_adapt(params, time_to_fit, psth_to_fit, y_start_fit):
        
        """
        Objective function to minimize, calculating the sum of squared errors between the model and the given data.
        The goal is to minimize this error to achieve the best fit of the model to the data.
        
        Returns:
        -------
        float
            The sum of squared errors between the model predictions and the PSTH data.
        """
        
        tau, C = params
        return np.sum(
            (exponential_model_adapt(time_to_fit, tau, C, y_start_fit) - psth_to_fit) ** 2
        )

    initial_guess = [1, np.min(psth_to_fit)] # Initial guesses for tau and C

    bounds = Bounds(
        [0.01, np.min(psth_to_fit)],  # Lower bounds [tau, C]
        [2, y_start_fit],  # Upper bounds [tau, C]
    )
    constraints = [
        {"type": "ineq", "fun": lambda params: y_start_fit - 1.2 * params[1]},  # (y_start_fit - C) > 0.2 * C re-written as (y_start_fit - 1.2*C) 
        {"type": "ineq", "fun": lambda params: (y_start_fit - params[1]) / params[1] - params[0] # tau <= (y_start_fit - C) / C re-written as (y_start_fit - C) / C - tau
            if params[1] > 1
            else y_start_fit - params[0], # tau <= y_start_fit if C < 1
        },  
    ]

    with warnings.catch_warnings():
        # Suppress runtime warnings related to optimization 
        warnings.simplefilter("ignore", RuntimeWarning)

        result = minimize(
            objective_adapt,
            initial_guess,
            args=(time_to_fit, psth_to_fit, y_start_fit), # arguments of the objective function
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-2},
        )

    # Check if the fitting was successful
    if result.success:
        popt = result.x  # Optimized parameters [tau, C]
        y_end_fit = exponential_model_adapt(2, *popt, y_start_fit) # Calculate the firing rate at time t = 2 (for adaptation index calculation)
        fitted_firing_rates = exponential_model_adapt(time_to_fit, *popt, y_start_fit)


        ai = adaptation_index(y_start_fit, y_end_fit, invert=False)
        
        return popt, fitted_firing_rates, ai
    else:
        return None, None, None


def fit_flat(psth_to_fit):
    """
    Fit a non-adapting (flat) model to the provided peristimulus time histogram (PSTH) data.

    This model assumes that the firing rate is constant over time, so it calculates the mean firing rate
    of the input PSTH data and returns an array where each value is the mean, representing the 
    non-adapting response.

    Parameters:
    ----------
    psth_to_fit : np.ndarray
        PSTH data that represents the firing rates to be fit by the model.

    Returns:
    -------
    mean_psth_array : np.ndarray
        An array with the same shape as `psth_to_fit`, where each value is the mean firing rate.
    mean_psth : float
        The mean firing rate of the input PSTH data.
    """
    mean_psth = np.mean(psth_to_fit)

    # Create an array filled with the mean PSTH value
    mean_psth_array = np.full_like(psth_to_fit, fill_value=mean_psth)

    return mean_psth_array, mean_psth


# %% k-fold functions


def generate_state_trial_splits(trial_indices, running_state, min_trials_per_test=5, max_splits=20):
    """
    Segments trials based on their running state and generates test sets for k-fold cross-validation.
    Utilizes `KFold` from scikit-learn to generate the test splits with shuffling and a fixed random state for reproducibility.
    Pooled test sets combine the test splits from both active and quiet states.

    Parameters
    ----------
    trial_indices : np.array
        1D array of trial indices to be analysed (e.g., all trials or those with strong responses).
    running_state : np.array
        1D array with the same length as the total number of trials, indicating the state of each trial:
            - `1` for active
            - `0` for quiet
            - `np.nan` for undefined state
    min_trials_per_test : int, optional
        Minimum number of trials required in each test set (default is 5).
    max_splits : int, optional
        Maximum number of k-fold splits to generate for each state (default is 20).

    Returns
    -------
    state_trial_indices : list of lists
        A list containing three sublists, each corresponding to a state:
            - First sublist: Indices of quiet trials.
            - Second sublist: Indices of active trials.
            - Third sublist: Indices of pooled trials (active and quiet).
    state_test_sets : list of lists
        A list containing three sublists, each containing arrays of test sets:
            - First sublist: Test sets for quiet trials.
            - Second sublist: Test sets for active trials.
            - Third sublist: Test sets for pooled trials.
    """

    # Filter trials based on state
    active_trial_ind = [ind for ind in trial_indices if running_state[ind] == 1]
    quiet_trial_ind = [ind for ind in trial_indices if running_state[ind] == 0]

    # Initialize dictionary for k-fold splits
    kf_test_sets = {}

    # Create KFold splits for each state
    for state, state_ind in zip(["active", "quiet"], [active_trial_ind, quiet_trial_ind]):
        k = min((round(len(state_ind) / min_trials_per_test)), max_splits)
        # TODO: Set random state for reproducibility in scripts rather than hardcoding here
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        kf_test_sets[state] = [np.array(state_ind)[test_set] for _, test_set in kf.split(state_ind)]

    # Pooled test indices and trial indices across states
    pooled_test_sets = kf_test_sets["active"] + kf_test_sets["quiet"]
    pooled_trial_ind = active_trial_ind + quiet_trial_ind

    # Output structures for each state
    state_trial_indices = [quiet_trial_ind, active_trial_ind, pooled_trial_ind]
    state_test_sets = [kf_test_sets["quiet"], kf_test_sets["active"], pooled_test_sets]

    return state_trial_indices, state_test_sets


def perform_fitting(
    trial_psths,
    timeBins,
    state,
    test_sets,
    trial_indices,
    identifier,
    sigma,
    is_suppressed=False,
    plot_test=False,
    plot_models=False,
):
    
    """
    The function fits four models to the mean PSTH: Adaptation, Sensitisation, Mixed, and Flat.
    Cross-validation is performed using the provided `test_sets`, and RMSE is calculated for each model on the test sets.
    The best model is selected based on the lowest average RMSE across all cross-validation folds.
    Optional plotting functions can visualize the model fits and cross-validation results.
    If model fitting fails for a particular model, its RMSE is set to infinity (`np.inf`), effectively excluding it from being selected as the best model.
    
    Parameters
    ----------
    trial_psths : np.ndarray
        2D array of shape (n_trials, n_time_bins) containing the peristimulus time histograms (PSTHs) for each trial
        corresponding to the state being analysed.
    timeBins : np.ndarray
        1D array representing the time bins corresponding to the PSTH data.
    state : str
        The state being analyzed ('quiet', 'active', 'pooled').
    test_sets : list of np.ndarray
        List containing arrays of test trial indices for each cross-validation split.
    trial_indices : list or np.ndarray
        List or array of trial indices corresponding to the state being analyzed.
    identifier : str
        Unique identifier for the neuron, formatted as '{Name}_{Date}_{neuron_id}'.
    sigma : float
        Standard deviation used for Gaussian smoothing of the PSTH.
    is_suppressed : bool, optional
        If `True`, inverts the PSTH to account for suppressed neuronal responses.
    plot_test : bool, optional
        If `True`, generates plots showing each model fitted to the test sets alongside the train sets (default is `False`).
    plot_models : bool, optional
        If `True`, plots the fits of each model to the mean PSTH (default is `False`).
    
    Returns
    -------
    dict
        A dictionary containing the best fitting model and detailed information about each model including:
            - Fitted parameters
            - Adaptation indices
            - Average RMSE 
            - List of RMSE values for each cross-validation fold

    """
    
    mean_psth = np.mean(trial_psths, axis=0)
    if is_suppressed:
        mean_psth = -1 * mean_psth
    
    # find data window to fit
    y_start_fit_mean_psth, start_time_for_fit_mean_psth, psth_to_fit_mean, time_to_fit_mean_psth = data_window_psth(mean_psth, timeBins, sigma)
    
    # adaptation model
    params_adapt_mean_psth, fitted_firing_rate_adapt_mean_psth, ai_adapt = None, None, None
    try:
        params_adapt_mean_psth, fitted_firing_rate_adapt_mean_psth, ai_adapt = fit_adapt_model(
            y_start_fit_mean_psth,
            psth_to_fit_mean,
            time_to_fit_mean_psth)
    except Exception:
        pass
    
    # sensitisation model
    # If the second half is greater than the first, fit the sensitisation model
    to_fit_sens = to_fit_sens_check(psth_to_fit_mean)
    params_sens_mean_psth, fitted_firing_rate_sens_mean_psth, ai_sens = None, None, None
    if to_fit_sens:
        try:
            params_sens_mean_psth, fitted_firing_rate_sens_mean_psth, ai_sens = fit_sens_model(
                psth_to_fit_mean,
                time_to_fit_mean_psth
            )
        except Exception:
            pass
    
    # mixed model
    t_switch = fit_mixed_exp_best_t_guess(
        psth_to_fit_mean,
        timeBins,
        params_adapt_mean_psth,
        params_sens_mean_psth,
        y_start_fit_mean_psth,
        start_time_for_fit_mean_psth,
        sigma
    )
    
    if t_switch is None:
        t_switch = 0.5
    params_mixed_model_mean_psth, fitted_firing_rate_mixed_mean_psth, ai_mixed_1, ai_mixed_2 = None, None, None, None
    try:
        params_mixed_model_mean_psth, fitted_firing_rate_mixed_mean_psth, ai_mixed_1, ai_mixed_2 = fit_mixed_exp(
            psth_to_fit_mean,
            timeBins,
            params_adapt_mean_psth,
            params_sens_mean_psth,
            y_start_fit_mean_psth,
            start_time_for_fit_mean_psth,
            t_switch,
            sigma,
        )
    except Exception:
        pass
    
    # flat model
    flat_line_mean_psth, mean_fr_mean_psth = fit_flat(psth_to_fit_mean)
    
    
    # cross validation
    
    fold_number = 1
    
    test_errors_adapt = []
    test_errors_sens = []
    test_errors_mixed_model = []
    test_errors_flat = []
    
    for test_indices in test_sets:
        
        if is_suppressed:
            trial_psths = -trial_psths
        
        # get test and train trials
        original_positions = [np.where(trial_indices == i)[0][0] for i in test_indices] 
        mask_test = np.isin(np.arange(trial_psths.shape[0]), original_positions)
        mask_train = ~mask_test
        train_psths = trial_psths[mask_train]
        test_psths = trial_psths[mask_test]
        
        # fit to mean train data
        train_psths_mean = np.mean(train_psths, axis=0)
        (
        psth_to_fit_train, start_time_for_fit_train, 
        fitted_firing_rate_adapt_train, fitted_firing_rate_sens_train, 
        fitted_firing_rate_mixed_train, flat_line_train 
        )= fit_train_set(
                train_psths_mean, timeBins, 
                sigma, 
                ai_adapt, 
                ai_sens, 
                ai_mixed_1, 
                t_switch)
        
            
        # get the RMSE from comparing the train fit with the mean psth of the test set 
        (
            rmse_adapt,
            rmse_sens,
            rmse_mixed,
            rmse_flat
        ) = get_test_errors(
            psth_to_fit_train,
            timeBins,
            start_time_for_fit_train,
            test_psths, 
            fitted_firing_rate_adapt_train,
            fitted_firing_rate_sens_train,
            fitted_firing_rate_mixed_train,
            flat_line_train,
            state,
            identifier,
            sigma,
            fold_number,
            plot_test=plot_test
        )

        test_errors_adapt.append(rmse_adapt)
        test_errors_sens.append(rmse_sens)
        test_errors_mixed_model.append(rmse_mixed)
        test_errors_flat.append(rmse_flat)

        fold_number += 1
    
    # get average test error to find best model
    # checking ai because it will be None if the fitting to the mean psth was unsuccessful which will deem
    # the average error calculation for that model and it would be set to infinity to ensure it's not selected.
    if ai_adapt is not None:    
        avg_rmse_adapt = np.nanmean(test_errors_adapt)
    else: 
        avg_rmse_adapt = np.inf
    if to_fit_sens and ai_sens is not None:
        avg_rmse_sens = np.nanmean(test_errors_sens)
    else:
        avg_rmse_sens = np.inf
    if ai_mixed_1 is not None:
        avg_rmse_mixed = np.nanmean(test_errors_mixed_model)
    else: 
        avg_rmse_mixed = np.inf
    avg_rmse_flat = np.mean(test_errors_flat)

    best_model = select_best_model(avg_rmse_adapt,avg_rmse_sens,avg_rmse_mixed,avg_rmse_flat)
    
    # package output
    
    models_details = {

        "Adaptation": {
            "params": np.concatenate((params_adapt_mean_psth, [y_start_fit_mean_psth, start_time_for_fit_mean_psth])) if params_adapt_mean_psth is not None else np.full(4, np.nan),
            "ai": np.array([ai_adapt, np.nan]) if ai_adapt is not None else np.array([np.nan, np.nan]),
            "rmse_k_fold": avg_rmse_adapt,
            "test_trial_errors": test_errors_adapt
        },
        "Sensitisation": {
            "params": np.concatenate((params_sens_mean_psth, [start_time_for_fit_mean_psth])) if params_sens_mean_psth is not None else np.full(4, np.nan),
            "ai": np.array([ai_sens, np.nan]) if ai_sens is not None else np.array([np.nan, np.nan]),
            "rmse_k_fold": avg_rmse_sens,
            "test_trial_errors": test_errors_sens
        },
        "Mixed": {
            "params": np.concatenate((params_mixed_model_mean_psth, [y_start_fit_mean_psth, start_time_for_fit_mean_psth])) if params_mixed_model_mean_psth is not None else np.full(8, np.nan),
            "ai": np.array([ai_mixed_1, ai_mixed_2]) if ai_mixed_1 is not None and ai_mixed_2 is not None else np.array([np.nan, np.nan]),
            "rmse_k_fold": avg_rmse_mixed,
            "test_trial_errors": test_errors_mixed_model
        },
        "Flat": {
            "params": np.concatenate(([mean_fr_mean_psth], [start_time_for_fit_mean_psth])) if mean_fr_mean_psth is not None else np.full(2, np.nan),
            "ai": np.array([0, np.nan], dtype=float),
            "rmse_k_fold": avg_rmse_flat,
            "test_trial_errors": test_errors_flat
        },
    }
        
    if plot_models: 
        # Collect relevant details for plotting
        mean_psth_details = {
            "time_to_fit": time_to_fit_mean_psth,
            "mean_psth": mean_psth,
            "y_start_fit": y_start_fit_mean_psth,
            "start_time": start_time_for_fit_mean_psth,
            "timeBins": timeBins
        }
        models_fits = {
            "Adaptation": fitted_firing_rate_adapt_mean_psth,
            "Sensitisation": fitted_firing_rate_sens_mean_psth,
            "Mixed": fitted_firing_rate_mixed_mean_psth,
            "Flat": flat_line_mean_psth
        }
        plot_all_models(
            mean_psth_details,
            models_fits,
            models_details,
            identifier,
            best_model,
            state,
            is_suppressed=is_suppressed
        )


    return {
        "best_model": best_model,
        "models_details": models_details
    }



def fit_train_set(
        train_psths_mean, timeBins, 
        sigma, 
        ai_adapt, 
        ai_sens, 
        ai_mixed_1, 
        t_switch):
    
    """
    Fits models to the mean training PSTH within a specified data window.
    
    The function first identifies the relevant data window for fitting using `data_window_psth`.
    It conditionally fits the Adaptation, Sensitisation, and Mixed models based on 
    the presence of adaptation index (AI) for each model (`ai_adapt`, `ai_sens`, `ai_mixed_1`).
    If the AI is `None`, the corresponding models are skipped.
    The Flat model is always fitted to the PSTH data.
    
    Parameters
    ----------
    train_psths_mean : np.ndarray
        1D array representing the mean peristimulus time histogram (PSTH) across training trials.
    timeBins : np.ndarray
        1D array of time bins corresponding to the PSTH data.
    sigma : float
        Standard deviation for Gaussian smoothing applied to the PSTH.
    ai_adapt : Any
        AI used as indicator of the success of fitting the adaptation model to the mean response. 
    ai_sens : Any
        As above but for sensitisation model. 
    ai_mixed_1 : Any
        As above but for mixed model. 
    t_switch : float
        The switch time parameter used in fitting the mixed exponential model.
    
    Returns
    -------
    psth_to_fit : np.ndarray
        The PSTH data of the training set that was fitted.
    start_time_for_fit : float
        The start time of the data window used for fitting.
    fitted_firing_rate_adapt : np.ndarray or None
        1D array of fitted firing rates from the Adaptation model. `None` if the model was not fitted.
    fitted_firing_rate_sens : np.ndarray or None
        1D array of fitted firing rates from the Sensitisation model. `None` if the model was not fitted.
    fitted_firing_rate_mixed : np.ndarray or None
        1D array of fitted firing rates from the Mixed model. `None` if the model was not fitted.
    flat_line : np.ndarray
        1D array of fitted firing rates from the Flat model.

    
    
    """
    # find data window to fit
    y_start_fit, start_time_for_fit, psth_to_fit, time_to_fit = data_window_psth(
        train_psths_mean, timeBins, sigma)
    
    # fit adaptation model
    params_adapt, fitted_firing_rate_adapt = None, None
    if ai_adapt is not None: 
        try:
            params_adapt, fitted_firing_rate_adapt, _ = fit_adapt_model(
                y_start_fit, psth_to_fit, time_to_fit)
        except Exception:
            pass
    
    # fit sensitisation model
    params_sens, fitted_firing_rate_sens = None, None
    if ai_sens is not None: 
        try:
            params_sens, fitted_firing_rate_sens, _= fit_sens_model(
                psth_to_fit,time_to_fit)
        except Exception:
            pass
    # fit mixed model
    _, fitted_firing_rate_mixed = None, None
    if ai_mixed_1 is not None: 
        try:
            _, fitted_firing_rate_mixed, _, _ = fit_mixed_exp(
                psth_to_fit,
                timeBins,
                params_adapt,
                params_sens,
                y_start_fit,
                start_time_for_fit,
                t_switch,
                sigma,
            )
        except Exception:
            pass
    # fit flat model
    flat_line, _ = fit_flat(psth_to_fit)
    
    return psth_to_fit, start_time_for_fit, fitted_firing_rate_adapt, fitted_firing_rate_sens, fitted_firing_rate_mixed, flat_line
        


def get_test_errors(
    psth_to_fit,
    timeBins,
    start_time_for_fit_adapt,
    test_psths,
    fitted_firing_rate_adapt,
    fitted_firing_rate_sens,
    fitted_firing_rate_mixed,
    flat_line,
    state,
    identifier,
    sigma,
    fold_number,
    plot_test=False
):
    
    """
    Calculates the Root Mean Square Error (RMSE) for response models on test PSTH data.
    
    RMSE provides a quantitative measure of how well each model fits the test PSTH data. Lower RMSE values indicate better model performance.
    Fallback Mechanism: If a specific model (Adaptation, Sensitisation, or Mixed) was not successfully fitted (`fitted_firing_rate_*` is `None`), 
    its RMSE is set to 1.1 times the RMSE of the Flat model. This penalizes models that failed to fit, ensuring they are not selected as the best model based on error metrics.
    When `plot_test` is enabled, the function visualizes the train PSTH, test PSTH, and the fitted models.
    
    Parameters
    ----------
    psth_to_fit : np.ndarray
        The PSTH data of the training set that was fitted.
    timeBins : np.ndarray
        1D array of time bins corresponding to the full PSTH data.
    start_time_for_fit_adapt : float
        The start time of the data window used for fitting the Adaptation and Mixed models.
    test_psths : np.ndarray
        2D array of shape (n_test_trials, n_time_bins) containing PSTH data for each test trial.
    fitted_firing_rate_adapt : np.ndarray or None
        1D array of fitted firing rates from the Adaptation model. `None` if the model was not fitted.
    fitted_firing_rate_sens : np.ndarray or None
        1D array of fitted firing rates from the Sensitisation model. `None` if the model was not fitted.
    fitted_firing_rate_mixed : np.ndarray or None
        1D array of fitted firing rates from the Mixed model. `None` if the model was not fitted.
    flat_line : np.ndarray
        1D array of fitted firing rates from the Flat model.
    state : str
        The state being analyzed (quiet/active/pooled).
    identifier : str
        Unique identifier for the neuron, formatted as '{Name}_{Date}_{neuron_id}'.
    sigma : float
        Standard deviation used for Gaussian smoothing in PSTH.
    fold_number : int
        The current fold number in cross-validation.
    plot_test : bool, optional
        If `True`, generates a plot showing the train and test PSTHs along with model fits (default is `False`).
    
    Returns
    -------
    rmse_adapt : float
        Root Mean Square Error (RMSE) for the Adaptation model. 
        Calculated as the square root of the mean squared difference between 
        the test PSTH mean and the fitted Adaptation model firing rates (from the train set). 
        If the Adaptation model was not fitted, it is set to 1.1 times the RMSE of the Flat model.
        
    rmse_sens : float
        Same as above but for Sensitisation model
        
    rmse_mixed : float
        Same as above but for Mixed model
        
    rmse_flat : float
        RMSE for the Flat model. 
        Calculated as the square root of the mean squared difference between 
        the test PSTH mean and the fitted Flat model firing rates.
    
    
    """
    
    fit_indices = (timeBins >= start_time_for_fit_adapt) & (timeBins <= 2 - 2 * sigma)
    test_psths_mean = np.mean(test_psths, axis=0)
    test_psths_mean = test_psths_mean[fit_indices]
        
    # 1. find error for flat model
    rmse_flat = np.sqrt(np.mean((test_psths_mean - flat_line) ** 2))
    # 2. find error for adaptation model
    if fitted_firing_rate_adapt is not None:
        rmse_adapt = np.sqrt(np.mean((test_psths_mean - fitted_firing_rate_adapt) ** 2))
    else:
        rmse_adapt = rmse_flat * 1.1
    # 3. find error for sensitisation model
    if fitted_firing_rate_sens is not None:
        rmse_sens = np.sqrt(np.mean((test_psths_mean - fitted_firing_rate_sens) ** 2))
    else:
        rmse_sens = rmse_flat * 1.1
    # 4. find error for mixed model
    if fitted_firing_rate_mixed is not None:
        rmse_mixed = np.sqrt(np.mean((test_psths_mean - fitted_firing_rate_mixed) ** 2))
    else:
        rmse_mixed = rmse_flat * 1.1

    if plot_test:
        plt.figure(figsize=(7, 5))
    
        # Plot the train and test sets, and the fits
        test_time_window = timeBins[fit_indices]
        plt.plot(test_time_window, psth_to_fit, label="Train set", linewidth=2)
        plt.plot(test_time_window, test_psths_mean, label="Test set", linestyle="dashed", linewidth=2)
        
        # Plot the different fitted models if they exist
        if fitted_firing_rate_adapt is not None:
            plt.plot(test_time_window, fitted_firing_rate_adapt, label="Adaptation", linewidth=2)
        if fitted_firing_rate_sens is not None:
            plt.plot(test_time_window, fitted_firing_rate_sens, label="Sensitisation", linewidth=2)
        if fitted_firing_rate_mixed is not None:
            plt.plot(test_time_window, fitted_firing_rate_mixed, label="Mixed", linewidth=2)
        plt.plot(test_time_window, flat_line, label="Flat", linewidth=2)
    
        plt.title(f"Train and Test set - {identifier} - {state} - fold {fold_number}", fontsize=14)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Firing Rate (FR)", fontsize=12)
        plt.legend(loc="best", fontsize="small")
    
        plt.tight_layout()
        plt.show()
    
    return rmse_adapt, rmse_sens, rmse_mixed, rmse_flat



def select_best_model(
    avg_rmse_adapt,
    avg_rmse_sens,
    avg_rmse_mixed,
    avg_rmse_flat
):
    """
    Selects the best model based on the lowest average RMSE.
    
    Parameters
    ----------
    avg_rmse_adapt : float
        The average Root Mean Square Error (RMSE) for the Adaptation model across cross-validation folds.
    avg_rmse_sens : float
        The average RMSE for the Sensitisation model across cross-validation folds.
    avg_rmse_mixed : float
        The average RMSE for the Mixed model across cross-validation folds.
    avg_rmse_flat : float
        The average RMSE for the Flat model across cross-validation folds.
    
    Returns
    -------
    best_model : str
        The name of the model with the lowest average RMSE. 
        Possible values are "Adaptation", "Sensitisation", "Mixed", or "Flat".
    
    """

    models = {
        "Adaptation": {"RMSE": avg_rmse_adapt},
        "Sensitisation": {"RMSE": avg_rmse_sens},
        "Mixed": {"RMSE": avg_rmse_mixed},
        "Flat": {"RMSE": avg_rmse_flat},
    }
    
    best_model = min(models, key=lambda x: models[x]["RMSE"])
    
    return best_model

def decide_state_separation(test_errors_per_state):
    
    """
    This function assesses whether pooling trial states (active and quiet) results in better model performance 
    compared to treating them separately.
    
    Parameters
    ----------
    test_errors_per_state : dict
        A dictionary containing lists of test errors for each state across cross-validation folds:
            - `'quiet'`: List of floats representing errors for the quiet state.
            - `'active'`: List of floats representing errors for the active state.
            - `'pooled'`: List of floats representing errors for the pooled state.
    
    Returns
    -------
    str
        A string indicating the recommended state separation strategy:
            - `"pooled"`: If the average error for the pooled state is lower than the combined average errors of the active and quiet states.
            - `"separate"`: If the average error for the pooled state is higher than the combined average errors of the active and quiet states.
    
    """
    pooled_errors = np.array(test_errors_per_state['pooled'], dtype=float)
    active_errors = np.array(test_errors_per_state['active'], dtype=float)
    quiet_errors = np.array(test_errors_per_state['quiet'], dtype=float)

    mean_pooled_error = pooled_errors.mean()
    mean_active_quiet_error = np.concatenate([active_errors, quiet_errors]).mean()

    if mean_pooled_error < mean_active_quiet_error:
        return "pooled"
    elif mean_pooled_error > mean_active_quiet_error:
        return "separate"


        
def plot_all_models(mean_psth_details,
                    models_fits,
                    models_details,
                    identifier,
                    best_model,
                    state,
                    is_suppressed = False):
    # TODO: sharey instad of limits and check

    params_adapt = models_details["Adaptation"]["params"]
    params_sens = models_details["Sensitisation"]["params"]
    params_mixed_model = models_details["Mixed"]["params"]
    mean_fr = models_details["Flat"]["params"][0]
    
    mean_psth = mean_psth_details['mean_psth']
    timeBins = mean_psth_details["timeBins"]
    y_start_fit_mean_psth = mean_psth_details['y_start_fit']
    time_to_fit_mean_psth = mean_psth_details['time_to_fit']
    start_time_for_fit_mean_psth = mean_psth_details['start_time']

    fitted_firing_rate_adapt = models_fits["Adaptation"]
    fitted_firing_rate_sens = models_fits["Sensitisation"]
    fitted_firing_rate_mixed = models_fits["Mixed"]
    flat_line = models_fits["Flat"]
    
    avg_error_adapt = models_details["Adaptation"]["rmse_k_fold"]
    avg_error_sens = models_details["Sensitisation"]["rmse_k_fold"]
    avg_error_mixed_model = models_details["Mixed"]["rmse_k_fold"]
    avg_error_flat_model = models_details["Flat"]["rmse_k_fold"]


    fig, axs = plt.subplots(1, 4, figsize=(20, 4), sharey=False)
    fig.suptitle(f"Neuron {identifier} - {state} state (Best Model: {best_model})", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    axs_adapt = axs[0]
    axs_sens = axs[1]
    axs_mixed = axs[2]
    axs_flat = axs[3]

    xlabel = "Time(s)"
    ylabel = "Firing rate (sp/s)"
    min_psth = min(mean_psth)
    max_psth = max(mean_psth)
    range_psth = max_psth - min_psth
    new_min = min_psth - 0.1 * range_psth
    new_max = max_psth + 0.1 * range_psth
    titles = [
        f"Adaptation Model\n k-fold RMSE: {avg_error_adapt:.2f}",
        f"Sensitisation Model\n k-fold RMSE: {avg_error_sens:.2f}",
        f"Mixed Model\n k-fold RMSE: {avg_error_mixed_model:.2f}",
        f"Flat Model\n k-fold RMSE: {avg_error_flat_model:.2f}",
    ]

    param_texts = [
        # Adaptation parameters with exception handling
        (
            f"y_start_fit={y_start_fit_mean_psth:.2f}, C={params_adapt[1]:.2f}, tau={params_adapt[0]:.2f}"
            if params_adapt is not None
            else "Adaptation fitting failed"
        ),
    
        # sensitisation parameters with exception handling
        (
            f"A={params_sens[2]:.2f}, C={params_sens[1]:.2f}, tau={params_sens[0]:.2f}"
            if params_sens is not None
            else "Sensitisation was not fit or failed"
        ),
    
        # Mixed model parameters with exception handling
        # (
        #     f"y_start_fit={safe_format(y_start_fit_mean_psth)}, C1={safe_format(safe_param(params_mixed_model, 1))}, tau1={safe_format(safe_param(params_mixed_model, 0))}\n"
        #     + f"A={safe_format(safe_param(params_mixed_model, 4))}, C2={safe_format(safe_param(params_mixed_model, 3))}, tau2={safe_format(safe_param(params_mixed_model, 2))}\n"
        #     + f"t_switch = {safe_format(safe_param(params_mixed_model, 5))}"
        #     if params_mixed_model is not None
        #     else "Mixed model fitting failed"
        # ),
        
        (
            f"y_start_fit={y_start_fit_mean_psth:.2f}, C1={params_mixed_model[1]:.2f}, tau1={params_mixed_model[0]:.2f}\n"
            + f"A={params_mixed_model[4]:.2f}, C2={params_mixed_model[3]:.2f}, tau2={params_mixed_model[2]:.2f}\n"
            + f"t_switch = {params_mixed_model[5]:.2f}"
            if params_mixed_model is not None
            else "Mixed model fitting failed"
        ),
        
        
        # Flat
        (
            f"Mean FR: {mean_fr:.2f}"
        )
    ]

    for ax, title, params_text in zip(axs, titles, param_texts):
        ax.set_ylim([new_min, new_max])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.text(
            0.05,
            0.02,
            params_text,
            transform=ax.transAxes,
            verticalalignment="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    axs_adapt.plot(timeBins, mean_psth)
    if fitted_firing_rate_adapt is not None:
        axs_adapt.plot(
            (time_to_fit_mean_psth + start_time_for_fit_mean_psth),
            fitted_firing_rate_adapt,
        )
    axs_sens.plot(timeBins, mean_psth)
    if fitted_firing_rate_sens is not None:
        axs_sens.plot(
            (time_to_fit_mean_psth + start_time_for_fit_mean_psth),
            fitted_firing_rate_sens,
        )
    axs_mixed.plot(timeBins, mean_psth)
    if fitted_firing_rate_mixed is not None:
        axs_mixed.plot(
            (time_to_fit_mean_psth + start_time_for_fit_mean_psth),
            fitted_firing_rate_mixed,
        )
    axs_flat.plot(timeBins, mean_psth)
    axs_flat.plot(
        (time_to_fit_mean_psth + start_time_for_fit_mean_psth), flat_line
    )
    if is_suppressed:
        axs_supp = axs_adapt.twinx()
        axs_supp.plot(
            timeBins,
            -1 * mean_psth,
            color="grey",
            alpha=0.5,
            label="Suppressed response",
        )
        axs_supp.set_ylabel("Original Firing Rate")
        axs_supp.legend(loc="upper right")
    plt.show()
            


# %% load output files 

def load_exp_fit_df(output_dir):
    # Load files
    prefix = 'gratingsPsthExpFit'
    
    clusters = np.load(os.path.join(output_dir, f'{prefix}.clusters.npy'))
    paramsAdapt = np.load(os.path.join(output_dir, f'{prefix}.paramsAdapt.npy'))
    paramsSens = np.load(os.path.join(output_dir, f'{prefix}.paramsSens.npy'))
    paramsMixed = np.load(os.path.join(output_dir, f'{prefix}.paramsMixed.npy'))
    paramsFlat = np.load(os.path.join(output_dir, f'{prefix}.paramsFlat.npy'))
    
    adaptIndex = np.load(os.path.join(output_dir, f'{prefix}.adaptIndex.npy'))
    bestModel_df = pd.read_csv(os.path.join(output_dir, 'bestModel.csv'))

    # Prepare the DataFrame
    data = []
    for i, cluster in enumerate(clusters):
        state_separation = bestModel_df.iloc[i, 3]  # Retrieve the state separation for the cluster
        
        AI_q = adaptIndex[i, 0]
        AI_a = adaptIndex[i, 1]
        
        best_model_q = bestModel_df.iloc[i, 0]  # Best model for quiet state
        best_model_a = bestModel_df.iloc[i, 1]  # Best model for active state
        
        # Extract parameters for quiet state
        
        if best_model_q == 'Adaptation':
            params_q = paramsAdapt[i, 0]
            tau_q = [params_q[0], np.nan]
        elif best_model_q == 'Sensitisation':
            params_q = paramsSens[i, 0]
            tau_q = [params_q[0], np.nan]
        elif best_model_q == 'Mixed':
            params_q = paramsMixed[i, 0]
            tau_q = [params_q[0], params_q[2]]
        elif best_model_q == 'Flat':
            params_q = paramsFlat[i, 0]
            tau_q = [np.nan, np.nan]
        else: # when best_model_a is np.nan
            params_q = [np.nan, np.nan]
            tau_q = [np.nan, np.nan]
        
        # Extract parameters for active state
        
        if best_model_a == 'Adaptation':
            params_a = paramsAdapt[i, 1]
            tau_a = [params_a[0], np.nan]
        elif best_model_a == 'Sensitisation':
            params_a = paramsSens[i, 1]
            tau_a = [params_a[0], np.nan]
        elif best_model_a == 'Mixed':
            params_a = paramsMixed[i, 1]
            tau_a = [params_a[0], params_a[2]]
        elif best_model_a == 'Flat':
            params_a = paramsFlat[i, 1]
            tau_a = [np.nan, np.nan]
        else: # when best_model_a is np.nan
            params_a = [np.nan, np.nan]
            tau_a = [np.nan, np.nan]
            
        # Append row data
        data.append([
            cluster, best_model_q, params_q, tau_q, AI_q, 
            best_model_a, params_a, tau_a, AI_a, state_separation
        ])
    
    # Create DataFrame
    columns = [
        'clusters', 'best_model_q', 'params_q', 'tau_q', 'AI_q',
        'best_model_a', 'params_a', 'tau_a', 'AI_a', 'state_separation'
    ]
    df = pd.DataFrame(data, columns=columns)
    return df
