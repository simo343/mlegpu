""" PSF generation and precise localization of single molecules in 3D"""

#%%

# core
from math import dist

from math import dist

import numpy as np
import pandas as pd

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as patches

# plotly
import dash
from dash import dcc, html, Input, Output, State
from dash.dash_table import DataTable
# from dash_extensions.snippets import send_data_frame
# from dash_extensions import Download
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from io import BytesIO # used for export image in dash as .svg


# jax
import jax
import jax.numpy as jnp
from jax import vmap
from jax.numpy.linalg import solve,norm
from jax import config
import jax.scipy as jsp
config.update("jax_enable_x64", False)

# skimage
from skimage.feature import peak_local_max
from sklearn.linear_model import LinearRegression
from functools import partial
import copy
from sklearn.cluster import DBSCAN, OPTICS

# scipy
import scipy.io
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import mat73

# functions
import sys
import importlib
import SMLM
from SMLM import *
import SMLM_functions_core
from SMLM_functions_core import *



#%% INDEX OF CONTENT



#   DATA PROCESSING FUNCTIONS
#   - trunc
#   - zernikes_decomposition


#   DATA FILTERING FUNCTIONS
#   - gaussian
#   - dash_plot
#   - dash_filtering
#   - filter_results


#   SYNTHETIC DATA GENERATION FUNCTIONS
#   - show_aberr
#   - plot_aberr
#   - generate_random_aberr
#   - generate_spiral
#   - generate_npc
#   - generate_image_photons


#%%




#------------------------------------------------
#           DATA PROCESSING FUNCTIONS           #
#------------------------------------------------


#  - trunc
#  - zernikes_decomposition



def trunc(x, decimals=2):
    return np.trunc(x * 10 ** decimals) / 10 ** decimals


from scipy.interpolate import BSpline, LSQUnivariateSpline, make_smoothing_spline


def modeset_decomposition(phase, modeset, z_modes=None, pupil=None, plot=False, figsize=(18,3)):
    '''"
    Finds the Zernike coefficients of a phase image using a given set of Zernike modes.
    pupil: you can restrict the pupil where the zernikes are defined, for example to ignore SAF contribution. If None, the whole NA is considered.
    '''
    Nk = phase.shape[0]
    n_modes = modeset.shape[0]
    if z_modes is None: z_modes = np.arange(n_modes) + 1

    # Flatten phase and basis arrays
    phase_flat = phase.flatten()
    modeset_flat = modeset.reshape(n_modes, -1)
    
    # Compute zernike_amplitudes via vectorized dot products using orthogonality
    numerators = modeset_flat @ phase_flat
    denominators = np.sum(modeset_flat**2, axis=1)
    modes_amplitude = np.divide(numerators, denominators, out=np.zeros_like(numerators), where=denominators!=0)
    
    # Reconstruct the fitted phase using broadcasting
    phase_fitted = np.sum(modeset * modes_amplitude[:, None, None], axis=0)

    if plot:
        # Compute the difference and RMS values on the pupil region (nonzero in pupil)
        phase_diff = phase - phase_fitted
        sum_pupil = np.sum(pupil > 0)
        rms_phase = np.sqrt(np.sum(phase**2) / sum_pupil)
        rms_fit   = np.sqrt(np.sum(phase_fitted**2) / sum_pupil)
        rms_diff  = np.sqrt(np.sum(phase_diff**2) / sum_pupil)
        
        # Global intensity range for plotting
        global_vals = np.concatenate((phase.flatten(), phase_fitted.flatten()))
        v_min, v_max = global_vals.min(), global_vals.max()
        
        # Plot original, fitted and difference phases.
        plt.figure(figsize=(14,4))
        plt.subplot(131)
        plt.imshow(phase, cmap='Spectral', vmin=v_min, vmax=v_max)
        plt.title('Phase to fit'); plt.colorbar()
        plt.text(0.98, 0.98, f"RMS: {rms_phase:.2f} rad", ha='right', va='top', transform=plt.gca().transAxes)
        plt.subplot(132)
        plt.imshow(phase_fitted, cmap='Spectral', vmin=v_min, vmax=v_max)
        plt.title('Phase fitted'); plt.colorbar()
        plt.text(0.98, 0.98, f"RMS: {rms_fit:.2f} rad", ha='right', va='top', transform=plt.gca().transAxes)
        plt.subplot(133)
        plt.imshow(phase_diff, cmap='Spectral', vmin=v_min, vmax=v_max)
        plt.title('Phase difference'); plt.colorbar()
        plt.text(0.98, 0.98, f"RMS: {rms_diff:.2f} rad", ha='right', va='top', transform=plt.gca().transAxes)
        plt.show()
        
        # Plot mode amplitudes.
        a_max = np.max(np.abs(modes_amplitude))
        plt.figure(figsize=figsize)
        plt.plot(modes_amplitude, '.')
        plt.title('Mode amplitudes [rad]')
        plt.xticks(ticks=range(len(z_modes)), labels=z_modes)
        plt.ylim(-(a_max + 0.5), a_max + 0.5)
        plt.grid()
        plt.show()

    return modes_amplitude, phase_fitted







def fit_signal(results, h):
    ''' Fits the signal vs z with an exponential decay and plots the results.'''
    '''Returns the predicted signal values to use for the CRLB calculation.'''

    def exp_decay(x, I0, decay_rate, offset):
        return I0 * np.exp(- decay_rate * x) + offset

    z = np.array(results['z'])
    points = np.array(results['signal'])
    x_axis = np.linspace(h.z_min * 1e9, h.z_max * 1e9, h.data.shape[2])

    lower_bounds = [1e2, 0, -1e3]
    upper_bounds = [1e5, 1e-1, 1e3]
    popt, _ = curve_fit(exp_decay, z.flatten(), points, bounds = (lower_bounds, upper_bounds), maxfev=10000)
    a, b, c = popt
    
    plt.plot(x_axis, exp_decay(x_axis, a, b, c), color='red')
    plt.scatter(z, points, s=0.01)
    plt.xlabel('Z / nm')
    plt.ylabel('Signal / photons')
    plt.title('Fit Signal vs Z')
    plt.show()

    return exp_decay(x_axis, a, b, c)












def filter_results(data, raw_data=None,
                   frame_range=(1, np.inf),
                   x_range=(-np.inf, np.inf), 
                   y_range=(-np.inf, np.inf), 
                   z_range=(-np.inf, np.inf),
                   signal_range=(1, np.inf), 
                   bg_range=(0, np.inf), 
                   LLR_range=(1, np.inf), 
                   SNR_range=(1, np.inf),
                   x_pxl=10, 
                   y_pxl=10, 
                   print_fraction=True,
    ):
    """Filter the results DataFrame based on the given ranges."""

    if isinstance(data, pd.DataFrame):  # for pandas DataFrame
        frame, x, y, z, s, bg, LLR, SNR = data['frame'], data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR'], data['SNR']
    elif isinstance(data, (np.ndarray, jnp.ndarray)):  # for NumPy array
        df = pd.DataFrame(data, columns=['frame', 'x', 'y', 'z', 'signal', 'bg', 'LLR', 'SNR'])
        frame, x, y, z, s, bg, LLR, SNR = df['frame'], df['x'], df['y'], df['z'], df['signal'], df['bg'], df['LLR'], df['SNR']


    # Initialize indices as a boolean array with the same length as results
    indices = np.ones(len(data), dtype=bool)

    if raw_data is None:  # Filtering without pixel units when raw_results is not provided
        indices &= (
            frame.between(frame_range[0], frame_range[1]) &
            x.between(x_range[0], x_range[1]) &
            y.between(y_range[0], y_range[1]) &
            z.between(z_range[0], z_range[1]) &
            s.between(signal_range[0], signal_range[1]) &
            bg.between(bg_range[0], bg_range[1]) &
            LLR.between(LLR_range[0], LLR_range[1]) &
            SNR.between(SNR_range[0], SNR_range[1])
        )
    else:  # Filtering with pixel units
        min_x, max_x = np.min(raw_data['x']), np.max(raw_data['x']) # Pre-filtering for pixelation artifacts
        prefilter_indices = (raw_data['x'].between(min_x + 0.1, max_x - 0.1) &
                             raw_data['y'].between(min_x + 0.1, max_x - 0.1))
        prefilt_raw = raw_data[prefilter_indices] # filter out the very edges of the fitted molecules
        mean_x, mean_y = np.mean(prefilt_raw['x']), np.mean(prefilt_raw['y'])

        indices &= prefilter_indices  # Apply the prefilter indices to the main indices
        indices &= (
            frame.between(frame_range[0], frame_range[1]) &
            x.between(x_range[0], x_range[1]) &
            y.between(y_range[0], y_range[1]) &
            z.between(z_range[0], z_range[1]) &
            s.between(signal_range[0], signal_range[1]) &
            bg.between(bg_range[0], bg_range[1]) &
            LLR.between(LLR_range[0], LLR_range[1]) &
            SNR.between(SNR_range[0], SNR_range[1]) &
            raw_data['x'].between(mean_x - x_pxl, mean_x + x_pxl) &
            raw_data['y'].between(mean_y - y_pxl, mean_y + y_pxl)
        )
        filt_raw = raw_data[indices]
    filt_data = data[indices]

    if print_fraction:  # Fraction in %
        frac_filt = np.round((1 - len(filt_data) / len(data)) * 100)
        print('Localizations:', len(filt_data), '/', len(data), '\t |', frac_filt, '% filtered out.')

    if raw_data is None:
        return filt_data
    else:
        return filt_data, filt_raw, indices














#%%


#------------------------------------------------
#      SYNTHETIC DATA GENERATION FUNCTIONS      #
#------------------------------------------------

# - show_aberr
# - plot_aberr
# - generate_random_aberr
# - generate_spiral
# - generate_npc
# - generate_image_photons



def plot_data_and_deconvolved_model(h, images, aberr, v, planes):
    """ Compare experimental images with model-generated images for each plane and visualize the error """

    N_planes = len(planes)
    fig, axes = plt.subplots(3, N_planes, figsize=(2.5 * N_planes, 2.5 * 3), gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    fig.suptitle("Compare Data with Deconvolved Model", fontsize=14)

    for plane in range(N_planes):
        data = images[plane][0]
        model = h.generate_image_bead_pos(aberr, v[plane][0], data, h.planes[plane])
        err = data - model
        vmin, vmax = np.min(data), np.max(data)

        # Row 0: Data
        ax_data = axes[0, plane]
        ax_data.imshow(data, cmap=parula, vmin=vmin, vmax=vmax)
        ax_data.set_xticks([]), ax_data.set_yticks([])
        ax_data.set_title(f"Plane {planes[plane]:.0f}", fontsize=14)
        if plane == 0:
            ax_data.set_ylabel("Data", fontsize=14)

        # Row 1: Model
        ax_model = axes[1, plane]
        ax_model.imshow(model, cmap=parula, vmin=vmin, vmax=vmax)
        ax_model.set_xticks([]), ax_model.set_yticks([])
        if plane == 0:
            ax_model.set_ylabel("Model", fontsize=14)

        # Row 2: Error
        ax_err = axes[2, plane]
        ax_err.imshow(err, cmap=parula, vmin=vmin, vmax=vmax)
        ax_err.set_xticks([]), ax_err.set_yticks([])
        if plane == 0:
            ax_err.set_ylabel("Error", fontsize=14)
            ax_err.text(5, data.shape[0] - 5, 'Score:', fontsize=14, ha='left', va='bottom', color='white')

        # print score
        llr_score = compute_LLR(model, data) / (h.Nx-4)**2
        ax_err.text(data.shape[1] - 5, data.shape[0] - 5, f"{llr_score:.3f}", color='white', fontsize=14, ha='right', va='bottom')
    plt.show()




# def plot_aberr_zernike(noll_index, aberr, figsize=(5,3)):
#     '''plot the amplitudes of the zernike modes for all given data'''
#     max_value = np.max(np.abs(aberr).max()) + 0.2
#     x_ticks = np.arange(len(aberr))
#     plt.figure(figsize=figsize)
#     plt.plot(x_ticks, aberr, 'o')
#     plt.xticks(x_ticks, np.int_(noll_index))
#     plt.xlabel('Zernike mode (Noll index)')
#     plt.ylabel('Magnitude (rad)')
#     plt.title('Zernike Aberrations', fontsize=16)
#     plt.axhline(y=0, color='black', linewidth=0.5)
#     plt.ylim(-max_value, max_value)
#     plt.grid()
#     plt.show()


def plot_modes(noll_index, voltages, aberr, plot_square=False, figsize=(5,3), title='Aberrations', range=None):
    '''plot the amplitudes of the selected Zernike modes against the voltages for DPP caracterization'''

    index = noll_index - 2
    #colors = colors(np.linspace(0, 1, len(voltages)))

    plt.figure(figsize=figsize)
    plt.plot(voltages**2, aberr[:, index], 'o-', label=noll_index) if plot_square else plt.plot(voltages, aberr[:, index], 'o-', label=noll_index)
    plt.xlabel('V^2') if plot_square else plt.xlabel('Voltage (V)')
    plt.ylabel('Magnitude (rad)')
    x_ticks = np.array(voltages**2) if plot_square else np.array(voltages)
    xtick_labels = [f'{np.sqrt(x):.0f}' for x in x_ticks] if plot_square else voltages
    plt.xticks(x_ticks, xtick_labels)
    plt.title(f'{title}')
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.ylim(range) if range is not None else None
    plt.legend()
    plt.grid()
    plt.show()


def plot_phases(h, aberr, voltages, LLRs, figsize=(15,8), vmin=None, vmax=None):
    '''Plot phases results for dpp deconvolution'''

    fig, axs = plt.subplots(nrows=3, ncols=len(voltages), figsize=figsize)
    for i, v in enumerate(voltages):
        '''plot deconvolved phase'''
        phase0 = jnp.sum(h.modeset*aberr[i][:, jnp.newaxis, jnp.newaxis], axis=0)
        ax0 = axs[0][i].imshow(phase0, cmap='Spectral'); plt.colorbar(ax0)
        axs[0][0].set_ylabel('deconvolved', fontsize=14)

        '''plot phase difference actuator vs flat'''
        zernikes_diff = aberr[i]-aberr[0]
        phase = jnp.sum(h.modeset*zernikes_diff[:, jnp.newaxis, jnp.newaxis], axis=0)
        ax1 = axs[1][i].imshow(phase, cmap='Spectral', vmin=vmin, vmax=vmax); plt.colorbar(ax1)
        axs[1][0].set_ylabel('difference', fontsize=14)

        '''plot wrapped phase difference'''
        phase_wrapped = jnp.angle(jnp.exp(1j*phase))
        ax2 = axs[2][i].imshow(phase_wrapped, vmin=-np.pi, vmax=np.pi, cmap='hsv'); plt.colorbar(ax2)

        '''make plot nice'''
        axs[0][i].set_title(f'{v} V / LLR: {trunc(LLRs[i],2)}', fontsize=12)
        axs[2][0].set_ylabel('wrapped', fontsize=14)
        
        for row in range(3):
            axs[row][i].set_xticks([])
            axs[row][i].set_yticks([])
    plt.tight_layout()
    plt.show()



def plot_data_and_model(h, images, aberr, v_best, planes, score, volt, actuator, figsize=(20, 5)):
    """Plots data and model images for specified planes in a two-row layout."""
    N_images = len(images)

    plt.figure(figsize=figsize)
    for i in range(N_images):
        data = images[i][0]
        vmin, vmax = np.min(data), np.max(data)
        model = h.generate_image_bead_pos(aberr, v_best[i][0], images[i][0], h.planes[i])

        # Plot data
        plt.subplot(2, N_images, i + 1)
        plt.imshow(data, cmap=parula, vmin=vmin, vmax=vmax)
        plt.xticks([]), plt.yticks([])
        plt.ylabel('Data', fontsize=14) if i==0 else None
        plt.title(f'{planes[i]} nm')

        # Plot model
        plt.subplot(2, N_images, N_images + i + 1)
        plt.imshow(model, cmap=parula, vmin=vmin, vmax=vmax)
        plt.xticks([]), plt.yticks([])
        plt.ylabel('Model', fontsize=14) if i==0 else None
        plt.title(f'{planes[i]} nm')

    plt.suptitle(f'Actuator {actuator} at {volt} V  -   score: {score:.3f}')
    plt.show()

def plot_data_and_model_zernike(h, images, aberr, v_best, planes, score, amp, zernike, figsize=(20, 5)):
    """Plots data and model images for specified planes in a two-row layout."""
    N_planes = len(planes)

    fig, axes = plt.subplots(2, N_planes, figsize=(2. * N_planes, 4), gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    fig.suptitle(f'Mode {zernike} at {amp} um  -   score: {score:.3f}', fontsize=14)

    for plane in range(N_planes):
        data = images[plane][0]
        vmin, vmax = np.min(data), np.max(data)
        model = h.generate_image_bead_pos(aberr, v_best[plane][0], images[plane][0], h.planes[plane])
        
        ax_data = axes[0, plane]
        im_data = ax_data.imshow(data, cmap=parula, vmin=vmin, vmax=vmax), ax_data.set_xticks([]), ax_data.set_yticks([])
        ax_data.set_title(f"Plane {planes[plane]:.0f}", fontsize=10)
        if plane == 0:
            ax_data.set_ylabel("Data", fontsize=12)
        
        ax_model = axes[1, plane]
        im_model = ax_model.imshow(model, cmap=parula, vmin=vmin, vmax=vmax), ax_model.set_xticks([]), ax_model.set_yticks([])
        if plane == 0:
            ax_model.set_ylabel("Model", fontsize=12)
    plt.show()



def plot_data_and_model_act(h, images, aberr, g, s, score, volt, actuator, figsize=(20, 5)):
    """Plots data and model images for specified planes in a two-row layout."""

    N_images = len(images)
    planes = [-800, -400,    0,  400,  800]
    images = np.array(images)

    g_amp, g_shape = g

    plt.figure(figsize=figsize)
    for i in range(N_images):
        data = images[i][0]
        vmin, vmax = np.min(data), np.max(data)
        model = h.generate_image_lorentz(aberr, g_amp, g_shape, s[i][0], data, planes[i]*1e-9)

        # Plot data
        plt.subplot(2, N_images, i + 1)
        plt.imshow(data, cmap=parula, vmin=vmin, vmax=vmax)
        plt.xticks([]), plt.yticks([])#, plt.colorbar()
        plt.ylabel('Data', fontsize=14) if i==0 else None
        plt.title(f'{planes[i]} nm')

        # Plot model
        plt.subplot(2, N_images, N_images + i + 1)
        plt.imshow(model, cmap=parula, vmin=vmin, vmax=vmax)
        plt.xticks([]), plt.yticks([])#, plt.colorbar()
        plt.ylabel('Model', fontsize=14) if i==0 else None
        plt.title(f'{planes[i]} nm')

    plt.suptitle(f'Actuator {actuator} at {volt} V')#  -   score: {score:.3f}')
    plt.show()



def plot_phase_results(phase_deconv, phase_0, actuator, volt, score_zero, score_best, figsize=(14, 4)):
    '''plot phase results'''

    vmin, vmax = -np.pi, np.pi
    phase_diff = phase_deconv - phase_0
    plt.figure(figsize=figsize)

    plt.subplot(131)
    plt.imshow(phase_deconv, cmap='Spectral')#, vmin=vmin, vmax=vmax)
    plt.xticks([]), plt.yticks([]), plt.colorbar(), plt.title(f'Result (score {score_best})')

    plt.subplot(132)
    plt.imshow(phase_0, cmap='Spectral')#, vmin=vmin, vmax=vmax)
    plt.xticks([]), plt.yticks([]), plt.colorbar(), plt.title(f'Flat phase (score {score_zero})')

    plt.subplot(133)
    # plt.imshow(phase_diff, cmap='hsv', vmin=vmin, vmax=vmax)
    plt.imshow(phase_diff, cmap='Spectral')#, vmin=vmin, vmax=vmax)
    plt.xticks([]), plt.yticks([]), plt.colorbar(), plt.title('difference')
    
    plt.suptitle(f'Actuator {actuator} at {volt} V')
    # plt.suptitle(f'Zernike {actuator} at {volt} um')

    plt.show()


def plot_deconvolved_phase(phase, RMS, score, figsize=(12, 4)):
    """ Plot unwrapped and wrapped phase side by side with RMS and score annotation """
    plt.figure(figsize=figsize)

    plt.subplot(121)
    plt.imshow(phase, cmap='Spectral')
    plt.xticks([]), plt.yticks([])
    plt.title('Unwrapped phase')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(jnp.angle(jnp.exp(1j*phase)), cmap='hsv')
    plt.xticks([]), plt.yticks([])
    plt.title('Wrapped phase')
    plt.colorbar()

    plt.suptitle(f'Deconvolved phase \n RMS: {RMS:.2f} rad - score: {score}', fontsize=13)
    plt.show()








def plot_aberr(noll_index, aberr1, aberr2, figsize = (10,3)) :
    "Plot the results from the aberr retrieval"

    max_value = np.max([np.abs(aberr1).max(), np.abs(aberr2).max()]) + 0.3

    plt.figure(figsize=figsize)
    plt.plot(np.arange(len(aberr1)), aberr1, 'o', label='Bead Aberrations')
    plt.plot(np.arange(len(aberr1)), aberr2, 'o', label='Deconvolved Aberrations')
    plt.xticks(np.arange(len(aberr1)), np.int_(noll_index))
    plt.xlabel('Zernike mode (Noll index)')
    plt.ylabel('Magnitude (rad)')
    plt.title(f'Aberrations')
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.ylim(-max_value, max_value)
    plt.grid()
    plt.legend()
    plt.show()


def generate_random_aberr(noll_index, norm=0.5, psf='cyl_psf', sph_percent=None):
    """ Generate a random aberration vector with the given Noll indices """

    """ Radial order:"""
    """  0th:  1 """
    """  1st:  2, 3 """
    """  2nd:  4, 5, 6 """
    """  3rd:  7, 8, 9,10 """
    """  4th: 11,12,13,14,15 """
    """  5th: 16,17,18,19,20,21 """
    """  6th: 22,23,24,25,26,27,28 """
    """  7th: 29,30,31,32,33,34,35,36 """
    """  8th: 37,38,39,40,41,42,43,44,45 """
    """  9th: 46,47,48,49,50,51,52,53,54,55 """
    """ 10th: 56,57,58,59,60,61,62,63,64,65,66 """
    """ 11th: 67,68,69,70,71,72,73,74,75,76,77,78 """
    """ 12th: 79,80,81,82,83,84,85,86,87,88,89,90,91 """
    """ 13th: 92,93,94,95,96,97,98,99,100,101,102,103,104,105 """
    """ 14th: 106,107,108,109,110,111,112,113,114,115,116,117,118,119,120 """
    """ 15th: 121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136 """

    '''Spherical aberrations: 11, 22, 37, 56, 79. Only on even Zernike orders'''
    
    

    # Normalize with optional spherical percentage constraint
    def normalize_with_ratio(aberr, mask, target_norm, sph_percent):
        if sph_percent is None:
            return aberr * (target_norm / np.linalg.norm(aberr))
        p = sph_percent / 100.0
        p = np.clip(p, 0.0, 1.0)
        sph = aberr.copy()
        sph[~mask] = 0.0
        non_sph = aberr - sph
        sph_rms = target_norm * np.sqrt(p)
        non_sph_rms = target_norm * np.sqrt(1.0 - p)
        sph_norm = np.linalg.norm(sph)
        non_sph_norm = np.linalg.norm(non_sph)
        if sph_norm > 1e-12 and sph_rms > 0:
            sph *= (sph_rms / sph_norm)
        else:
            sph[:] = 0.0
        if non_sph_norm > 1e-12 and non_sph_rms > 0:
            non_sph *= (non_sph_rms / non_sph_norm)
        else:
            non_sph[:] = 0.0
        out = sph + non_sph
        total_norm = np.linalg.norm(out)
        if total_norm > 1e-12:
            out *= (target_norm / total_norm)
        return out
    
    noll_index = np.asarray(noll_index)
    spherical_mask = np.isin(noll_index, [11, 22, 37, 56, 79])

    # base random vector, with damping of higher orders
    rnd_aberr = np.random.uniform(-1, 1, len(noll_index))
    rnd_aberr[:4] = 0.0  # no piston/tip/tilt/defocus

    thresholds = [4, 7, 11, 16, 22, 29, 37, 46, 56]
    for th in thresholds:
        rnd_aberr[noll_index >= th] *= 0.5

    # initial normalization (before cyl)
    rnd_aberr = normalize_with_ratio(rnd_aberr, spherical_mask, norm, sph_percent)

    # add cylindrical mode (Noll index 6) and renormalize with same split
    if psf == 'cyl_psf':
        print('Norm before cyl mode:', np.round(np.linalg.norm(rnd_aberr), 3))
        cyl2_idx = np.where(noll_index == 6)[0]
        if len(cyl2_idx) > 0:
            i = cyl2_idx[0]
            rnd_aberr[i] = rnd_aberr[i] * 0.2 + 1.0

        rnd_aberr = normalize_with_ratio(rnd_aberr, spherical_mask, norm, sph_percent)
        print('Norm after cyl mode:', np.round(np.linalg.norm(rnd_aberr), 3))

    return rnd_aberr




def generate_spiral_biplane(h, cam,
                    s = 2000,
                    bg = 100,
                    N_mol = 8,
                    N_img = 1000,
                    z0 = 0e-9,
                    dz = 10e-9,
                    R = 80e-9,
                    on_fraction=0.1,
                    generate_image_method = 'interpolation',
                    biplane = False,
                    z_biplane = 400):
    
    """
    Generate syntethic SMLM image stack of a spiral of molecules.
    if biplane = True, two stacks are generated with a z_shift between them.
    """

    if generate_image_method == 'interpolation' :
        fun = gen_img_ph_interp
    elif generate_image_method == 'fft' :
        fun = gen_img_ph_fft
    else:
        raise ValueError("generate_image_method must be 'interpolation' or 'fft'")

    I_sum = np.zeros((N_img, h.Nx, h.Nx)) # images are added to this stack
    N_on = int(round(N_img*on_fraction)) #number of images with the molecule on
    I_on = np.zeros((N_on, h.Nx, h.Nx)) # images with molecules

    if biplane:
        I_sum_2 = I_sum.copy()
        I_on_2 = I_on.copy()
        s /= 2.

        # generate the second PSF defocused by z_shift respect to the first one
        h2 = psf(z_min=h.z_min * 1e9,
                 z_max=h.z_max * 1e9,
                 z_defocus=h.z_defocus * 1e9 + z_biplane,
                 wavelength=h.wavelength * 1e9,
                 NA=h.NA,
                 Nx=h.Nx,
                 ux=h.ux * 1e9,
                 focal_len=h.focal_len * 1e9,
                 RI=h.RI,
                 uz=h.uz * 1e9,
                 Z_modes=h.Z_modes,
                 Z_magn=h.Z_magn,
                 os=h.os)


    for m in range(1, N_mol+1): # iterate over the molecules

        Image = np.zeros((N_img, h.Nx, h.Nx))
        idx = np.random.choice(N_img, N_on, replace=False) #randomly choose N_on indices

        # x, y, z positions
        x = R*np.cos(m*2*np.pi/N_mol) #molecule positions are distributed on a circle
        y = R*np.sin(m*2*np.pi/N_mol)
        z = z0 + (m-1)*dz
        
        I_on = fun(h, [x, y, z, s, 0], N_img = N_on) # generate the image
        Image[idx,:,:] = I_on
        I_sum += Image #add to the other molecule images

        if biplane : # repeat with second PSF
            Image_2 = np.zeros((N_img, h.Nx, h.Nx))
            I_on_2 = fun(h2, [x, y, z, s, 0], N_img = N_on)  
            Image_2[idx,:,:] = I_on_2
            I_sum_2 += Image_2         
        
        
    I_sum += bg # add bg to all images. Molecules are generated with bg = 0
    I_sum += np.random.normal(0, 1, I_sum.shape) #add noise
    I_sum = I_sum * cam.QE / cam.amp  # from photons to counts
    I_sum += cam.baseline
    I_sum[I_sum < cam.baseline] = cam.baseline

    if biplane : 
        I_sum_2 += bg
        I_sum_2 += np.random.normal(0, 1, I_sum_2.shape)
        I_sum_2 = I_sum_2 * cam.QE / cam.amp
        I_sum_2 += cam.baseline
        I_sum_2[I_sum_2 < cam.baseline] = cam.baseline
        I_sum = np.concatenate((I_sum, I_sum_2), axis = 2)
    np.random.shuffle(I_sum) # shuffle the molecule images

    return np.uint16(np.round(I_sum))




def generate_spiral(h, cam,
                    s = 2000,
                    bg = 100,
                    N_mol = 8,
                    N_img = 1000,
                    z0 = 0e-9,
                    dz = 10e-9,
                    R = 80e-9,
                    on_fraction=0.1,
                    generate_image_method = 'fft',
                    ph_distribution = 'constant',
                    double_blinking=True):
    
    """
    Generate syntethic SMLM image stack of a spiral of molecules.
    if biplane = True, two stacks are generated with a z_shift between them.
    """

    if generate_image_method == 'interpolation' :
        fun_img = gen_img_ph_interp
    elif generate_image_method == 'fft' :
        fun_img = gen_img_ph_fft
    else:
        raise ValueError("generate_image_method must be 'interpolation' or 'fft'")
    

    I_sum = np.zeros((N_img, h.Nx, h.Nx)) # images are added to this stack
    
    '''For cycle over the spiral molecules'''
    for m in range(N_mol): # iterate over the molecules

        Image = np.zeros((N_img, h.Nx, h.Nx))

        if double_blinking:
            N_on = int(round(N_img*on_fraction)) #number of images with the molecule on
            idx = np.random.choice(N_img, N_on, replace=False)
        else:
            N_on = N_img//N_mol
            idx = np.arange(N_on) + m*N_on

        # x, y, z, s
        x = R*np.cos((m+1)*2*np.pi/N_mol) #molecule positions are distributed on a circle
        y = R*np.sin((m+1)*2*np.pi/N_mol)
        z = z0 + m*dz
        signal = get_photon_distribution(ph_distribution, mean_photons=s, n_samples=N_on)  # signal per molecule

        I_on = fun_img(h, [x, y, z, signal, 0], N_img = N_on) # generate the images
        Image[idx,:,:] = I_on # place the N_on molecule images at the randomly chosen indices
        I_sum += Image #add to the other molecule images

    '''Photons'''
    I_sum += bg # add bg to all images. Images are generated without to not mess with probabilistic on/off switching.

    '''Electrons'''
    I_sum *= cam.QE  # convert photons to electrons
    I_sum = np.random.poisson(I_sum).astype(np.float64) # poissonian noise, computed on detected photons = incident photons * QE
    I_sum += np.random.normal(0, cam.readnoise, I_sum.shape) #add readout noise in electrons: 0.7 e- rms for Orca Fusion

    '''Digital Counts'''
    I_sum /= cam.amp  # convert electrons to digital counts
    I_sum += cam.baseline
    I_sum = np.round(I_sum) # rounding to integer counts
    I_sum = np.clip(I_sum, 0, cam.maxcounts) # clipping to camera max counts
    I_sum = I_sum.astype(np.uint16) # converting to unsigned 16-bit integer

    np.random.shuffle(I_sum) # shuffle for randomness
    return I_sum


def get_photon_distribution(ph_distribution, mean_photons=1000, n_samples=10000):
    """
    Generate photon count distribution based on specified type.

    ph_distribution : str
        Type of distribution: 'constant', 'exponential', 'normal', or 'gamma2' or 'gamma3'
    mean_photons : float
        Target mean photon count (default: 1000)
    n_samples : int
        Number of samples to generate (default: 10000)
    """
    
    if ph_distribution == 'constant':
        ph_dist = np.full(n_samples, mean_photons)
    
    elif ph_distribution == 'normal':
        ph_dist = np.random.normal(loc=mean_photons, scale=mean_photons / 3, size=n_samples)
        ph_dist = np.maximum(ph_dist, 0)  # Ensure no negative photon counts

    elif ph_distribution == 'exponential':
        ph_dist = np.random.gamma(shape=1, scale=mean_photons, size=n_samples)

    elif ph_distribution == 'gamma2':
        ph_dist = np.random.gamma(shape=2, scale=mean_photons/2., size=n_samples)

    elif ph_distribution == 'gamma3':
        ph_dist = np.random.gamma(shape=3, scale=mean_photons/3., size=n_samples)
    
    else:
        ValueError(f"Unknown distribution type: {ph_distribution}. Choose between: constant, exponential, normal, gamma2, gamma3")
    
    return ph_dist





# GENERATE SINGLE IMAGES

def gen_img_ph_interp(h, param, N_img=1):
    """generating a realistic camera image from a PSF model"""

    dx, dy, dz, s, bg = param
    dx /= h.ux/h.os
    dy /= h.ux/h.os
    dz = (dz - h.z_min) / h.uz 
    
    aberr = jnp.fft.ifftshift(jnp.sum(h.modeset * h.Z_magn[:, jnp.newaxis, jnp.newaxis], axis = 0))
    defocus,_ = h.calc_defocus( - h.z_defocus, h.RI[-1]) # defocus in immersive medium
    exp_jph = jnp.exp(1j*jnp.real(defocus + aberr)) * h.apodization; # phase
    PSF,_ = h.generate_psf(h.z_vec, exp_jph) # calculate 3D psf   

    itp = jax.image.scale_and_translate(PSF, 
                                        shape = (h.Nx_os, h.Nx_os, 1), #shape of images to be interpolated
                                        spatial_dims = (0, 1, 2), 
                                        scale = jnp.array([1, 1, 1]), 
                                        translation = jnp.array([dx, dy, -dz]),#.clip(v, 0, n-1),
                                        method = 'lanczos5')[...,0]
    
    I_binned = binning(itp, h.os) #binning the interpolated image to the original size
    Image = bg + s * I_binned # the bg input is 0. So no bg added here.
    I_stack = np.repeat(Image[np.newaxis, :, :], N_img, axis=0)
    return I_stack


def gen_img_ph_fft(h, param, N_img=1):
    """generating a realistic camera image from a PSF model"""

    if h.os != 1:
        raise ValueError("fft method only implemented for os=1. Also increasing os does not change anything within fft image generation.")

    dx, dy, dz, s, bg = param
    dx /= h.ux/h.os
    dy /= h.ux/h.os
    dz = (dz - h.z_min) / h.uz # in units of uz (psf slices)

    aberr = jnp.fft.ifftshift(jnp.sum(h.modeset * h.Z_magn[:, jnp.newaxis, jnp.newaxis], axis = 0))
    defocus,_ = h.calc_defocus( - h.z_defocus, h.RI[-1]) # defocus in immersive medium
    exp_jph = jnp.exp(1j*jnp.real(defocus + aberr)) * h.apodization; # phase
    
    z = ((dz - h.d2) * h.uz + h.z_vec[0])  # z position of emitter relative to coverslip
    fxy = h.ux /  (2 * jnp.pi / (h.Nk * h.uk))

    z_factor = jnp.exp(1j * (h.k1 * h.CTHETA1.real * z + 1j * h.k1 * h.CTHETA1.imag * jnp.clip(z, 0, None) +
                                            2 * np.pi * h.Kx/(h.uk * h.Nk) * fxy * dy +
                                            2 * np.pi * h.Ky/(h.uk * h.Nk) * fxy * dx)) 

    E_BFP = h.E_BFP[:,0] / h.k1 * z_factor #make the fields 3 dimensional
    E_fields = h.czt2(jnp.fft.fftshift(E_BFP * exp_jph, axes = (1, 2)))#[:, h.shift:-h.shift, h.shift:-h.shift] # fields in the camera. 
    I = jnp.sum(E_fields.real**2 + E_fields.imag**2, axis=0)
    I_BFP = jnp.sum((E_BFP.real**2 + E_BFP.imag**2), axis=0, where=h.apodization[None])
    I /= jnp.sum(I_BFP) # total intensity in the camera, normalized to intensity in the BFP

    itp = h.blurr_image(I) if h.sigma_blurring != 0 else I
    I_binned = binning(itp, h.os) if h.os != 1 else itp #binning the interpolated image to the original size

    # Image = bg + s * I_binned  # the bg input is 0. So no bg added here.
    # I_stack = np.repeat(Image[np.newaxis, :, :], N_img, axis=0)

    s = np.full(N_img, s) if np.asarray(s).ndim==0 else s  # repeat scalar to match N_img
    I_stack = bg + s[:, None, None] * I_binned[None, :, :]

    return I_stack



def bg_mask(N_crop, image_portion=0.25):
    """
    Creates a mask at the radial edges of the image to evaluate background.
     - image_portion = [0,1], is the fraction of the radius (1 = R_max = N_crop/2).
    """ 
    yy, xx = np.indices((N_crop, N_crop)) 
    center = np.array([N_crop // 2, N_crop // 2])
    r = np.hypot(xx - center[1], yy - center[0])
    rmax = r.max()
    mask_border = r >= ((1-image_portion) * rmax)
    return mask_border


# %%
