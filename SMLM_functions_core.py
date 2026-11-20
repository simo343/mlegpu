""" PSF generation and precise localization of single molecules in 3D"""

#%%

# core
import numpy as np
import pandas as pd
import gc

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
from io import BytesIO


# jax
import jax
import jax.numpy as jnp
from jax import vmap
from jax.numpy.linalg import solve,norm
from jax import config
import jax.scipy as jsp
config.update("jax_enable_x64", False)
from functools import partial

# skimage
from skimage.feature import peak_local_max
from sklearn.linear_model import LinearRegression
import copy

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
import SMLM_functions_data_processing
from SMLM_functions_data_processing import *



#%% INDEX OF CONTENT


#   CORE PSF FUNCTIONS
#   - Fisher
#   - add_CRLB
#   - create_coord
#   - load_PSF
#   - load_aberr
#   - combine_PSFs
#   - binning


#   PRELOC FUNCTIONS
#   - prepare_images_for_fitting
#   - draw_peaks
#   - crop_images
#   - filter_biplane_double_peaks
#   - preloc_setup
#   - preloc_and_crop


#   FIT FUNCTIONS
#   - estimate_v0
#   - perform_fit
#   - lm_poisson
#   - compute_LLR
#   - fit_step
#   - add_preloc_positions


#   PLOT FUNCTIONS
#   - show_prelocs
#   - show_results_3d
#   - show_results_2d
#   - show_results_1d
#   - show_results_hist
#   - print_results
#   - filter_results


#   ZERNIKE FUNCTIONS
#   - zernike_noll
#   - zernike_nm
#   - zernikeRadialFunc
#   - zernIndex


# EIGENBASIS FUNCTIONS
#   - orthogonal_eigenbasis

#%%




#------------------------------------------------
#                CORE FUNCTIONS                 #
#------------------------------------------------

#   - Fisher
#   - add_CRLB
#   - create_coord
#   - load_PSF
#   - load_aberr
#   - combine_PSFs
#   - binning





#@partial(jax.jit, static_argnames=('h'))
def Fisher(h, cam, s, bg, z_slice = 'all'):
    ''' 
    Computes the Fisher information matrix. If z_slice = 'all' : whole PSF. Otherwise only at the given z position.
    model:              I = bg + s * PSF(x,y,z)     
    gradient:           dI/dx = s * d(PSF)/dx
                        dI/ds = PSF
                        dI/dbg = 1
    matrix elements:    FI_ij = integral(dI/di * dI/dj)      i,j = variables: x, y, z, s, bg
    '''
    
    s *= cam.QE; bg *= cam.QE  # we consider photons in the BFP, thus we multiply by the QE

    # EMCCDs suffer from excess noise: superpoissonian statistics. this is taked into account by dividing the signal and bg by a factor of 2
    if cam.EM_gain != 1:
        s /= 2; bg /= 2

    if z_slice == 'all':
        # Take all z_slices of the PSF
        Nz = h.data.shape[2]
        data = jnp.stack([binning(h.data[:,:,m], h.os) for m in range(Nz)], axis=-1)
        Dy_px, Dx_px, Dz_px = jnp.gradient(data)   # y = row, x = column

    else: 
        # If z is a number, take only one z slice of the PSF
        Nz = 1
        data_temp = jnp.stack([binning(h.data[:,:,m], h.os) for m in jnp.arange(z_slice - 1, z_slice + 2, 1)], axis=-1)
        
        data = data_temp[..., 1] # use 1 slice to compute the gradient in x and y
        Dy_px, Dx_px = jnp.gradient(data)   # y = row, x = column
        Dz_px = jnp.gradient(data_temp, axis=2)[..., 1] # use 3 slices to compute the gradient in z

    Dx = s * Dx_px / h.ux       # Dx = dI/dx = s * d(PSF)/dx
    Dy = s * Dy_px / h.ux       # Dy = dI/dy = s * d(PSF)/dy
    Dz = s * Dz_px / h.uz       # Dz = dI/dz = s * d(PSF)/dz
    Ds = data                   # Ds = dI/ds = PSF
    Db = jnp.ones_like(Dx)      # Db = dI/db = 1
    Di = [Dx, Dy, Dz, Ds, Db]

    # noise = variance. 
    # Variance = (standard deviation) ** 2 of signal + background + readnoise
    noise = jnp.array(s * Ds + bg + cam.readnoise**2) # for poissonians, standard deviation is sqrt(s), sqrt(bg) and readnoise
    
    FI = jnp.zeros((5, 5, Nz))  # initializing Fisher information matrix 
    for i in range(5):
        for j in range(i, 5):
            if i == j: 
                FI = FI.at[i, j, :].set(jnp.sum(Di[i]**2 / noise, axis=(0, 1)))  # Diagonal elements
            else: 
                FI_ij = jnp.sum(Di[i] * Di[j] / noise, axis=(0, 1))  # Off-diagonal elements
                FI = FI.at[j, i, :].set(FI_ij)
                FI = FI.at[i, j, :].set(FI_ij)
    return FI




def add_CRLB(h, cam, results, show_progress=True):
    ''' Adds the CRLB for x, y and z to the localization results DataFrame.'''

    n = len(results)
    if n == 0:
        return results.copy()

    z_min, z_max, z_vec = np.round(h.z_min * 1e9), np.round(h.z_max * 1e9), np.round(h.z_vec * 1e9)
    z_vals, signals, bgs = results['z'].to_numpy(), results['signal'].to_numpy(), results['bg'].to_numpy()
    z_idx = np.argmin(np.abs(z_vec[:, None] - z_vals[None, :]),axis=0) # For each z (in nm), find the index of the closest slice in the PSF

    # Pre‑allocate
    crlb_x = np.full(n, np.nan, dtype=float)
    crlb_y, crlb_z = crlb_x.copy(), crlb_x.copy()

    # Progress interval (at least 1 to avoid div0)
    interval = max(1, n // 10)
    if show_progress:
        print(f"--- Computing CRLB for {n} localizations ---")

    for i in range(n):
        if show_progress and (i + 1) % interval == 0:
            pct = int((i + 1) / n * 100)
            print(f"  {pct}%")
        z = z_vals[i]
        if not (z_min < z < z_max): # outside PSF range → leave as NaN
            continue

        # compute CRLB for this localization
        crlb = h.calc_CRLB(cam, s=signals[i], bg=bgs[i], z_slice=z_idx[i])
        crlb_x[i] = crlb[0][0] * 1e9
        crlb_y[i] = crlb[1][0] * 1e9
        crlb_z[i] = crlb[2][0] * 1e9
        
    out = results.copy()
    out['CRLB x'] = np.round(crlb_x, 1)
    out['CRLB y'] = np.round(crlb_y, 1)
    out['CRLB z'] = np.round(crlb_z, 1)

    if show_progress:
        print("--- Completed ---")

    return out



def create_coord(N, u, mode):  
    '''
    creates axis with N entries (if N=[Nx Ny], a 2D grid is created)
    u...spacing between two neighbouring entries
    mode...'exact' or 'FFT'
    if 'exact' is chosen: zero point is exactly in the middle, i.e. for even grid
    sizes between the two middle pixels
    if 'FFT' is chosen: zero point is the middle pixel for odd grid sizes and
    the pixel to the lower right of the exact centre for even grid sizes.
    '''
    if isinstance(N, int):
        N = (N, N)

    if isinstance(u, (float, int)):
        u = (u, u)
 
    if mode == 'exact':
        x = np.linspace(-(N[0] - 1) / 2, (N[0] - 1) / 2, N[0]) * u[0]
        y = np.linspace(-(N[1] - 1) / 2, (N[1] - 1) / 2, N[1]) * u[1]
    elif mode == 'FFT':
        x = (np.arange(-N[0] // 2 + 1, N[0] // 2 + 1)) * u[0]
        y = (np.arange(-N[1] // 2 + 1, N[1] // 2 + 1)) * u[1]

    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    mask = R**2 <= ((min(N) / 2 * u[0]))**2

    return X, Y, R, mask



def load_PSF(filename):
    """ Load PSF.mat files and construct the classes objective, camera, and PSF """
    """ The PSF stack is imported from matlab, not generated in python """

    PSF = scipy.io.loadmat(filename)['PSF'][0][0]

    wavelength, z_defocus, RI_sample, Nx, ux, uz, os = (
        PSF[i][0][0] * 1e9 if i in [1, 2, 5, 6] else PSF[i][0][0] for i in range(1, 8))

    data, cam, obj = PSF[8], PSF[9][0,0], PSF[10][0,0]

    pixsize, QE, baseline, amp, EM_gain, readnoise = (
        cam[i][0][0] * (1e9 if i == 1 else 1) for i in range(1, 7))

    M, NA, RI = (obj[i][0][0] for i in range(1, 4))
    Ts_coefs, Tp_coefs = (obj[i][0] for i in range(4, 6))

    Z_modes, Z_magn, T_fit_coefs = (
        PSF[11][0,0][i][0] for i in range(3))
    
    cam = Camera(pixsize=pixsize, QE=QE, baseline=baseline, EM_gain=EM_gain, amp=amp, readnoise=readnoise)
    obj = Objective(M=M, NA=NA, RI=RI, Ts_coefs=Ts_coefs, Tp_coefs=Tp_coefs)

    Nk = 128
    z_max = (data.shape[2] - 1) * uz
    h = psf(NA=NA, wavelength=wavelength, z_defocus=z_defocus, RI=[RI_sample, RI], Nk=Nk, Nx=Nx, ux=ux, uz=uz, z_max=z_max, Z_modes=Z_modes, Z_magn=Z_magn, Ts_coefs=Ts_coefs, Tp_coefs=Tp_coefs, T_fit_coefs = T_fit_coefs, os=os)
    h.data = data

    return cam, obj, h


def load_aberr(filename, mat_version = None):
    """Load aberr.mat files"""
    if mat_version == '7.3' : 
        aberr = mat73.loadmat(filename)['aberr']
        Z_modes, Z_magn, T_fit_coefs = aberr['Z_modes'], aberr['Z_magn'], aberr['T_coefs']
    else:
        aberr = scipy.io.loadmat(filename)['aberr'][0][0]
        Z_modes, Z_magn, T_fit_coefs = (aberr[i][0] for i in range(3))

    return Z_modes, Z_magn, T_fit_coefs


def combine_PSFs(h1,h2):
    ''' Concatenates 2 psf in one along the x axis for biplane imaging'''
    h = copy.copy(h1)
    h.data = jnp.concatenate((h1.data, h2.data), axis = 1)
    h.z_defocus = [h1.z_defocus, h2.z_defocus]
    h.aberr = [h1.aberr, h2.aberr]
    return h



def binning(image, bin_size):
    """ Bins an image or array of images by summing pixel values in blocks of size bin_size x bin_size"""

    H, W = image.shape[-2], image.shape[-1]
    H_b = H // bin_size
    W_b = W // bin_size

    cropped = image[..., :H_b * bin_size, :W_b * bin_size]
    new_shape = image.shape[:-2] + (H_b, bin_size, W_b, bin_size)
    reshaped = cropped.reshape(new_shape)
    binned = jnp.sum(reshaped, axis=(-1, -3))
    return binned


    height, width = image.shape[:2]
    binned_height, binned_width = height // bin_size, width // bin_size

    # Reshape the image into a 2D array of blocks and sum pixel values in each block
    superpixels = jnp.sum(image[:binned_height * bin_size, :binned_width * bin_size].reshape(
        binned_height, bin_size, binned_width, bin_size), axis=(1, 3))

    return superpixels



#%%



#------------------------------------------------
#               PRELOC FUNCTIONS                #
#------------------------------------------------

#   - prepare_images_for_fitting
#   - find_peaks
#   - draw_peaks
#   - crop_images
#   - filter_biplane_double_peaks
#   - preloc_setup
#   - preloc_and_crop




# def cts2phs(I_counts, camera):#, file_format='tif'):
#     ''' Convert counts to photons'''
#     # baseline = 100 if file_format == 'dcimg' else camera.baseline #dcimg default baseline is 100
#     I_photons = np.asarray(I_counts, dtype=np.uint16)
#     I_photons = np.clip(I_photons, camera.baseline, 2**16-1)             # clip pixel values to 16-bit range
#     I_photons = np.asarray((I_photons - camera.baseline) * camera.amp / camera.QE, dtype=np.float32)  # subtract baseline, convert counts to electrons, convert electrons to photons
#     I_photons = np.maximum(I_photons, 0.01)                 # set minimum pixel values to very small photons
#     return I_photons


def cts2phs(DATA_counts, camera, inplace=False):
    """Convert counts -> photons. Set inplace=True to modify numpy arrays in-place for memory efficiency."""

    gain = camera.amp / camera.QE
    baseline = camera.baseline
    max_val = 2**16 - 1 # maximum pixel value for a 16-bit sensor (Orca Flash 4.0)
    min_val = 0.01  # minimum value to avoid zero photons
    
    def _convert_numpy(arr: np.ndarray) -> np.ndarray:
        tmp = (np.clip(arr, camera.baseline, max_val).astype(np.float32) - camera.baseline) * gain
        return np.maximum(tmp, min_val)

    # Convert single jax array -> jax float32   
    def _convert_jnp(arr):
        tmp = (jnp.clip(arr, camera.baseline, max_val).astype(jnp.float32) - camera.baseline) * gain
        return jnp.maximum(tmp, min_val)

    def _convert_inplace(arr):
        if arr.dtype != np.float32:                    # Convert dtype if needed
            arr = arr.astype(np.float32)
        np.clip(arr, baseline, max_val, out=arr)       # Clip in-place
        arr -= baseline                                # Subtract baseline in-place
        arr *= gain                                    # Apply gain in-place
        np.maximum(arr, min_val, out=arr)              # Apply minimum in-place
        return arr

    if isinstance(DATA_counts, list):                  # Case: list of arrays
        for i in range(len(DATA_counts)):
            arr = DATA_counts[i]
            if isinstance(arr, jnp.ndarray):           # JAX array
                DATA_counts[i] = _convert_jnp(arr)     # Use JAX converter
            elif isinstance(arr, np.ndarray):          # NumPy array
                if inplace:
                    DATA_counts[i] = _convert_inplace(arr)  # Modify original
                else:
                    DATA_counts[i] = _convert_numpy(arr)    # Create new
                    del arr                            # Free memory immediately
            gc.collect()                               # Force garbage collection
        return DATA_counts


    elif isinstance(DATA_counts, jnp.ndarray):         # Case: single JAX array
        return _convert_jnp(DATA_counts)               # Use JAX converter
    
    elif isinstance(DATA_counts, np.ndarray):          # Case: single NumPy array
        if inplace:
            return _convert_inplace(DATA_counts)       # Modify original
        else:
            return _convert_numpy(DATA_counts)         # Create new
        
    else:
        raise TypeError(f"Unsupported data type: {type(DATA_counts)}")
    


def find_peaks(img, sigma, min_dist, N_crop):
    ''' Find the peaks in the image using the convolution with a gaussian filter and the local maxima algorithm.'''
    conv_img = gaussian_filter(img, sigma=sigma)
    min_dist = max(min_dist, N_crop // 2 + 2)
    peaks = peak_local_max(conv_img, min_distance=min_dist).astype(int)  # xy pixel coordinates (x=columns, y=rows)
    return conv_img, peaks



#@partial(jax.jit, static_argnames=('N_crop'))
def draw_peaks(peaks, N_crop, offset=0, peak_color=None):
    ''' Draw peaks on the preloc image with color given by peak_color '''

    peak_color = ['white' for _ in range(len(peaks))] if peak_color is None else peak_color
    for i, peak in enumerate(peaks) :
        rectangle = patches.Rectangle(
            (peak[1] + offset - N_crop / 2, peak[0] - N_crop / 2), # peak[1] = row, peak[0] = column
            N_crop, N_crop, color=peak_color[i], fill=False)
        plt.gca().add_patch(rectangle)
        plt.scatter(peak[1] + offset, peak[0], color='red', s=1)




#@partial(jax.jit, static_argnames=('N_crop', 'thresh'))
def crop_images(image, tmp_peaks, N_crop, threshold_min=0, threshold_max=0, threshold_rel=0, threshold_rel_max=1e5):
    ''' Crop the molecules found with find_peaks'''
    
    peaks, cropped_images, SBR, tmp_SBR, peak_color = [], [], [], [], []
    for i, peak in enumerate(tmp_peaks):
        row_start, col_start = np.subtract(peak, N_crop // 2)
        row_end, col_end = row_start + N_crop, col_start + N_crop

        tmp_cropped_image = image[row_start:row_end, col_start:col_end] # row / column
        min_val, max_val = np.min(tmp_cropped_image), np.max(tmp_cropped_image)
        tmp_SBR.append((max_val / min_val) if min_val != 0 else np.inf) # SBR = max / min

        if (tmp_SBR[i] >= threshold_rel             # max/min > threshold_rel
            and tmp_SBR[i] <= threshold_rel_max     # max/min < threshold_rel_max
            and min_val >= threshold_min            # min > threshold_min
            and max_val >= threshold_max):          # max > threshold_max

            peaks.append(peak)
            cropped_images.append(tmp_cropped_image)
            SBR.append(np.round((max_val / min_val), 2))
            peak_color.append('white')
        else:
            peak_color.append('red')

    return peaks, cropped_images, SBR, tmp_SBR, peak_color




def merge_double_peaks(peaks_left, peaks_right, SBR_left, SBR_right):
    """ only for Biplane: merges the peaks that are present in both channels. It uses the position of the channel with higher SBR"""
    
    if not peaks_left or not peaks_right: # if there are no peaks in one of the stacks or both are empty
        return peaks_left if peaks_left else peaks_right

    max_dist = 3  # Set the maximum distance threshold
    distances = cdist(peaks_left, peaks_right)  # point by point distance between the 2 stacks
    indices = np.where(distances < max_dist)  # find the indices of the peaks that are closer than max_dist
    idx_left = [indices[0][idx] for idx in range(len(indices[0])) if SBR_left[indices[0][idx]] <= SBR_right[indices[1][idx]]]
    idx_right = [indices[1][idx] for idx in range(len(indices[0])) if SBR_left[indices[0][idx]] > SBR_right[indices[1][idx]]]
    return np.concatenate((np.delete(peaks_left, idx_left, axis=0), np.delete(peaks_right, idx_right, axis=0)), axis=0)






def preloc_setup(image, sigma, min_dist, threshold_min, threshold_max, threshold_rel, N_crop, figsize=(8,6), vmin=None, vmax=None, biplane=False):
    ''' Find good parameters for the preloc'''
    
    if N_crop % 2 == 0 : # use odd N_crop to have a symmetric cropping around the central pixel
        print('Use N_crop odd to have a symmetric cropping around the central pixel.')
        return

    # initials
    min_dist = max(min_dist, N_crop // 2 + 2)  # Minimum distance between peaks and image border
    long_axis = image.shape[1] # x axis = number of columns
    short_axis = long_axis // 2 if biplane else long_axis
       
    plt.figure(figsize=figsize)
    print('Image min / max: ', np.min(image), '/', np.max(image))
    print('SBR ratio of temporary prelocs (in descending order):', )

    # peaks left channel (if not biplane left channel = only channel)
    img_left = image[:, :short_axis]
    conv_img_left, tmp_peaks_left = find_peaks(img_left, sigma, min_dist, N_crop)
    peaks_left, _ , SBR_left, tmp_SBR_left, peak_color_left = crop_images(img_left, tmp_peaks_left, N_crop, threshold_min, threshold_max, threshold_rel)
    
    if biplane :
        # peaks right channel
        img_right = image[:, short_axis:]
        conv_img_right, tmp_peaks_right = find_peaks(img_right, sigma, min_dist, N_crop)
        peaks_right, _ , SBR_right, tmp_SBR_right, peak_color_right = crop_images(img_right, tmp_peaks_right, N_crop, threshold_min, threshold_max, threshold_rel)
        conv_img = np.concatenate((conv_img_left, conv_img_right), axis=1)
        peaks = merge_double_peaks(peaks_left, peaks_right, SBR_left, SBR_right) # delete the same peaks present in both channels
        
        draw_peaks(tmp_peaks_right, N_crop, offset=short_axis, peak_color=peak_color_right) # plot right stack peaks
        plt.axvline(x=short_axis-1, color='black', linestyle='-', linewidth=0.5) # draw a line between the 2 stacks
        print('L channel:', sorted(tmp_SBR_left, reverse=True))
        print('R channel:', sorted(tmp_SBR_right, reverse=True))

        # plot the convoluted image
        plt.imshow(conv_img, cmap=parula,vmin=vmin,vmax=vmax)
        plt.title('Convoluted Image'); plt.tight_layout(); plt.colorbar()
        draw_peaks(tmp_peaks_left, N_crop, offset=0, peak_color = peak_color_left); plt.show()
        
        # plot the image with the good molecules
        plt.figure(figsize=figsize)
        draw_peaks(peaks, N_crop)
        draw_peaks(peaks, N_crop, offset=short_axis) # plot right stack peaks
        plt.axvline(x=short_axis-1, color='black', linestyle='-', linewidth=0.5) # draw a line between the 2 stacks
        plt.title(f"{len(peaks)} Molecule{'s' if len(peaks) != 1 else ''} Detected")
        plt.imshow(image, cmap=parula,vmin=vmin,vmax=vmax); plt.tight_layout(); plt.colorbar(); plt.show()

    else : # single plane 
        conv_img = conv_img_left
        peaks = peaks_left
        print(sorted(tmp_SBR_left, reverse=True))

        plt.subplot(1,2,1)
        plt.imshow(conv_img, cmap=parula,vmin=vmin,vmax=vmax)
        plt.title('Smoothed Image'); plt.colorbar()
        draw_peaks(tmp_peaks_left, N_crop, offset=0, peak_color = peak_color_left)
        
        plt.subplot(1,2,2)
        draw_peaks(peaks, N_crop)
        plt.imshow(image, cmap=parula,vmin=vmin,vmax=vmax)
        plt.title('Image'); plt.colorbar()
        plt.suptitle(f"{len(peaks)} Molecule{'s' if len(peaks) != 1 else ''} Detected")
        plt.tight_layout(), plt.show()


  

#@partial(jax.jit, static_argnames=('sigma', 'min_dist', 'thresh', 'N_crop'))
def preloc_and_crop(stack, sigma=1, min_dist=15, threshold_min=0, threshold_max=0, threshold_rel=1.5, N_crop=17, biplane=False, show_progress=True):
    ''' Performs the preloc: returns the preloc table and the cropped images of molecules'''

    # initials
    min_dist = max(min_dist, N_crop // 2 + 2)  # Minimum distance between peaks and image border
    long_axis = stack[0].shape[1]
    short_axis = long_axis // 2 if biplane else long_axis
    peaks, cropped_images = [],[]

    if show_progress: # print progress
        print('--- Preloc started ---')
        print('# frames:', '\t',len(stack))
        print('Analizing ... ')
        print(' 0 - 10  %')
        size, counter = len(stack) // 10, 10

    for m, img in enumerate(stack): # loop over frames

        if show_progress:
            if (m + 1) % size == 0:
                if counter < 90 :
                    counter += 10
                    print(counter - 10, '-', counter, ' %')
                elif counter == 90 :
                    print('90 - 100 %')
                    counter += 10

        # peaks left channel (if not biplane left channel = only channel)
        img_left = img[:, : short_axis]
        _, tmp_peaks_left = find_peaks(img_left, sigma, min_dist, N_crop)

        if biplane :
            # peaks right channel
            img_right = img[:, short_axis:]
            _, tmp_peaks_right = find_peaks(img_right, sigma, min_dist, N_crop)

            # merge  peaks
            peaks_left, _ , SBR_left, _,_ = crop_images(img_left, tmp_peaks_left, N_crop, threshold_min, threshold_max, threshold_rel)  
            peaks_right, _ , SBR_right, _,_ = crop_images(img_right, tmp_peaks_right, N_crop, threshold_min, threshold_max, threshold_rel)
            tmp_peaks = merge_double_peaks(peaks_left, peaks_right, SBR_left, SBR_right)
            
            #crop molecules from both channels
            if len(tmp_peaks) == 0 : continue # if there are no peaks in both stacks skip the frame
            _, cropped_images_left,_,_,_ = crop_images(img_left, tmp_peaks, N_crop)
            _, cropped_images_right,_,_,_ = crop_images(img_right, tmp_peaks, N_crop)
            tmp_cropped_images = np.concatenate((cropped_images_left, cropped_images_right), axis=2)

        else: # single plane
            tmp_peaks, tmp_cropped_images , _, _, _ = crop_images(img_left, tmp_peaks_left, N_crop, threshold_min, threshold_max, threshold_rel)

        # frame, x coordinate (columns), y coordinate(rows)
        peaks.extend((m + 1, peak[1], peak[0]) for peak in tmp_peaks)  
        cropped_images.extend(tmp_cropped_images)

    print(len(peaks))


    data = np.hstack((np.arange(1,len(peaks)+1)[:, None], np.array(peaks)))  # Adding molecule index column
    table = pd.DataFrame(data, columns=["id", "frame", "x", "y"])
    print('--- Preloc completed ---') if show_progress else None
    print('# molecules: ', '\t', len(table)) if show_progress else None

    return table, np.array(cropped_images)






















#%%





#------------------------------------------------
#                FIT FUNCTIONS                  #
#------------------------------------------------

#   - estimate_v0
#   - perform_fit
#   - lm_poisson
#   - compute_LLR
#   - fit_step
#   - add_preloc_positions




@partial(jax.jit, static_argnames=('h'))
def estimate_v0(images, h, z_start=None):
    '''initial fit guess '''  

    v0 = jnp.zeros((images.shape[0], 5))
    z0 = h.data.shape[2]//2 if z_start is None else z_start
    bg0 = images.min(axis=(1,2))
    s0 = (images - bg0[:,jnp.newaxis,jnp.newaxis]).sum(axis=(1,2))

    v0 = v0.at[:,:2].set(0)         # x0, y0 = 0
    v0 = v0.at[:,2].set(z0)         # z0 = (z_max-z_min) / 2, or z_start
    v0 = v0.at[:,3].set(s0)         # s0 = sum(I_photons - bg)
    v0 = v0.at[:,4].set(bg0)        # bg0 = min(I_photons)
    return v0



def perform_fit(I_photons, preloc_table, h, batch_size = 1000, z_start=None, factor_up = 7, factor_down=9, lmbda=1., maxiter = 100,  show_progress = False):
    ''' Performs the fit'''
    
    lm_poisson_vmap = jax.jit(jax.vmap(lambda x,v0,lmbda: lm_poisson(h, x, v0, tol=1e-1, maxiter=maxiter, lmbda=lmbda, factor_up=factor_up, factor_down=factor_down)))
    
    N_img = I_photons.shape[0]
    results_list, fit_results_list = [], []
    tot_iterations = N_img//batch_size
    iteration_counter = 0

    for i in range(tot_iterations+1): 

        if show_progress:
            iteration_counter +=1
            if iteration_counter == 1 :
                print("--- Precise fit ---")
                print('# of Batches :', tot_iterations+1)
                print('Progress:')
                print(iteration_counter) #, '\t/\t', tot_iterations+1)
            else: 
                print(iteration_counter) #, '\t/\t', tot_iterations+1)
    
        if i == (tot_iterations):
            sl_batch = slice(i*batch_size,None)
        else:
            sl_batch = slice(i*batch_size,(i+1)*batch_size)

        I_photons_jnp = jnp.array(I_photons[sl_batch])           # images
        v0_jnp = estimate_v0(I_photons_jnp, h) if z_start is None else estimate_v0(I_photons_jnp, h, z_start)                   # initial guess
        lmbda_vmap = jnp.ones(I_photons_jnp.shape[0]) * lmbda     # 
        est_precise_jnp, LLR_jnp = lm_poisson_vmap(I_photons_jnp, v0_jnp, lmbda_vmap)
        est_precise, LLR = np.asarray(est_precise_jnp),np.asarray(LLR_jnp)      
        results, fit_results = add_preloc_positions(est_precise, LLR, preloc_table[sl_batch], h)
        results_list.append(results), fit_results_list.append(fit_results)

    results = pd.concat(results_list)
    fit_results = pd.concat(fit_results_list)

    if show_progress: print("--- Fit completed ---")

    return results, fit_results



@partial(jax.jit, static_argnames=('h','maxiter'))
def lm_poisson(h, x, v0, tol=1e-3, maxiter=100, lmbda=1e-0, factor_up=7, factor_down=9):  
    """LM algorithm adapted to Poissonian MLE; see suppl. mat of :
    Laurence, Ted A., and Brett A. Chromy. "Efficient maximum likelihood estimator fitting of histograms." Nature methods 7.5 (2010): 338-339."""
    est_precise = v0.copy()

    LLR = jnp.inf
    def body_fun(i,val):
        
        est_precise,lmbda,LLR = val
        return fit_step(h, x, est_precise, tol, lmbda, factor_up, factor_down)
    
    est_precise, lmbda, LLR = jax.lax.fori_loop(0, maxiter, body_fun, (est_precise,lmbda,LLR))
    return est_precise, LLR


# def compute_LLR(model, data) :
#     """Computes image by image poissonian log likelihood ratio (LLR)"""

#     positive_model = jnp.where(model <= 0, 1., model)    # replace log(0) = -inf with log(1) = 0
#     positive_data = jnp.where(data <= 0, 1., data)       # replace log(0) = -inf with log(1) = 0
        
#     if len(model.shape) == 3 : # stack of images
#         return 2*jnp.sum((model - data * jnp.log(positive_model)) - (data - data * jnp.log(positive_data)), axis = (1, 2)) #log-likehood ratio for fit quality check
#     elif len(model.shape) == 2 : # single image
#         return 2*jnp.sum((model - data * jnp.log(positive_model)) - (data - data * jnp.log(positive_data))) #log-likehood ratio for fit quality check

def compute_LLR(model, data):
    """Computes image by image poissonian log likelihood ratio (LLR)"""
    model = jnp.clip(model, 1e-9)  # Prevent log(0)
    data = jnp.clip(data, 1e-9)    # Prevent log(0)
    
    term1 = model - data * jnp.log(model)
    term2 = data - data * jnp.log(data)
    
    if model.ndim == 3:
        return 2 * jnp.sum(term1 - term2, axis=(-1, -2))
    else:
        return 2 * jnp.sum(term1 - term2)



def compute_least_squares(model, data):
    """calculates the least square error (Gaussian likelihood) for a given model and recorded data"""
    if len(model.shape) == 3 : # stack of images
        return jnp.sum(((model - data)**2) , axis = (1, 2))
    elif len(model.shape) == 2 : # single image
        return jnp.sum((model - data)**2)


#@partial(jax.jit, static_argnames=('h',))
def fit_step(h, x, a, tol, lmbda, factor_up, factor_down):
    ''' Fit: Levenberg-Marquardt algorithm'''
    ''' (J*J + lambda)delta = J*(x-f)'''
    ''' alpha = J*J + lambda'''
    ''' beta = J*(x-f)'''

    lmbda_min = 1e-5
    lmbda_max = 1e5
    n = len(a)

    f_a = h.f(a) # f evaluated at point a : f is the interpolation of the image with the PSF 
    J = jax.jacfwd(h.f)(a).reshape(-1,n) # Jacobian of f at point a
    
    tmp = x / jnp.square(f_a)
    JTJ = jnp.einsum('kj,kl->jl',J,J * tmp.flatten()[:,jnp.newaxis])
    JTr = -((1. - x / f_a).flatten().reshape(1,-1) @ J).flatten()  # gradient
    diag_JTJ = jnp.eye(n) * JTJ
    a_step  = solve(JTJ + lmbda * diag_JTJ, JTr) # calculated step. isresult of the linear system alpha * delta = beta
    f_a_new = h.f(a + a_step) # new function evaluation with updated parameters
    LLR = compute_LLR(f_a , x)
    LLR_new = compute_LLR(f_a_new , x)

    f_a_J = f_a.reshape(-1) + J @ a_step
    f_a_J_log = jnp.where(f_a_J <= 0, 1., f_a_J)
    
    f_a_log = jnp.where(f_a <= 0, 1., f_a).reshape(-1)
    rho_denom = 2. * (x.reshape(-1) * jnp.log(f_a_J_log) - x.reshape(-1) * jnp.log(f_a_log) - J @ a_step).sum()
    
    rho_denom = jnp.where(rho_denom == 0., jnp.inf, rho_denom)
    rho = (LLR - LLR_new) / rho_denom    
    cond = rho > tol
    P_lmbda = lambda l: jnp.clip(l, lmbda_min, lmbda_max)
    
    lmbda = jnp.where(cond, P_lmbda(lmbda / factor_down), P_lmbda(lmbda * factor_up))  # Decrease lambda if the model is closer to the minimum
                                                                                       # Increase lambda if the model is further from the minimum
    a = jnp.where(cond, a + a_step, a)          # do not update a

    # Clip the parameters. After the fit iterations are conlcuded.
    a = a.at[0:2].set(jnp.clip(a[0:2], -10*h.os, 10*h.os))      # clip x, y to +/- 10 pixels
    a = a.at[2].set(jnp.clip(a[2], 0, h.data.shape[2]))         # clip z at the PSF Z edges
    a = a.at[3:5].set(jnp.clip(a[3:5], 0.001 , jnp.inf))        # clip signal and bg to positive values

    LLR_ud = jnp.where(cond, LLR_new, LLR) # update the LLR
        
    return a, lmbda, LLR_ud



def add_preloc_positions(est_precise, LLR, preloc_table, h):
    """
    Adds prelocs positions to fit results.
    results: fit results with prelocs. space units are nanometers.
    raw_results: fit results without prelocs. space units are pixels (x,y) and PSF steps (z). Used for visualization and filtering.
    """
    results = pd.DataFrame(data = { 'id': preloc_table['id'],
                                    'frame': preloc_table['frame'], 
                                    'x': (est_precise[:, 1] + np.round(preloc_table["x"]*h.os)) * h.ux/h.os * 1e9, # nm  x: columns
                                    'y': (est_precise[:, 0] + np.round(preloc_table["y"]*h.os)) * h.ux/h.os * 1e9, # nm  y: rows
                                    'z': (est_precise[:, 2] * h.uz + h.z_min ) * 1e9, # nm
                                    'signal': est_precise[:, 3], # photon number
                                    'bg': est_precise[:, 4], # photon number
                                    'LLR': LLR})
    
    raw_results = pd.DataFrame(data = { 'id': preloc_table['id'], 
                                        'frame': preloc_table['frame'], 
                                        'x' : est_precise[:,1] / h.os, # pixels, x: columns
                                        'y' : est_precise[:,0] / h.os, # pixels, y: rows
                                        'z' : est_precise[:,2]}) # uz steps 

    return results, raw_results







#------------------------------------------------
#           DECONVOLUTION FUNCTIONS             #
#------------------------------------------------




def retrieve_phase(h, par, images, bounds, lrs, stop_counter, _print=False):
    ''' Performs phase retrieval. The method used is pre-defined in the class. par, images, bounds, and lrs must be consistent.'''

    '''Initialize'''
    try:
        N_planes = images.shape[0] # number of planes
        N_mols = images.shape[1]   # number of molecules (cropped images) per plane
        N_crop = images.shape[-1]  # image crop size
    except (AttributeError, TypeError):
        N_planes = len(images)
        N_mols = images[0].shape[0]
        N_crop = images[0].shape[-1]
    N_pixels = N_planes * N_mols * N_crop ** 2 # number of pixels in the image stack
   
    it, maxiter, t1, i_stop = 0, 1e5, 1, 0
    score_new, score_best = 1e4, 1e4
    par_new, par_old, par_best = par, par, par
    print('- It - ', '\t', '- score - ', '\t', '- sg -', '\t', '- bg -', '\t', '- stop -') if _print else None

    while it < maxiter:
        '''print'''
        if h.method == 'multiplane' or h.method == 'multiplane2':
            print(' ', it, '\t\t' , trunc(score_new, 3),   '  ','\t ' , trunc(par_new[1][0][0][3],2), '\t\t', trunc(par_new[1][0][0][4],2), '   \t   ', i_stop) if _print else None
        elif h.method == 'multiplane_bead_pos':
            print(' ', it, '\t\t' , trunc(score_new, 3),   '  ','\t ' , trunc(par_new[1][0][0][0],2), '\t\t', trunc(par_new[1][0][0][1],2), '   \t   ', i_stop) if _print else None

        '''update'''
        it += 1
        LLR_new, par_new, par_old, t1 = h.update(par_new, par_old, lrs, bounds, t1, images) # update parameters
        score_new = float(LLR_new / N_pixels) # update score
        '''save best result or stop'''
        if score_new < score_best:
            par_best, score_best = par_new, score_new # update best parameters and score
        else:
            i_stop += 1
            if i_stop > stop_counter:
                break
    '''save results'''
    print('score:', trunc(score_best, 4)) if _print else None

    return par_best, score_best




def retrieve_actuator(h, par, const, bounds, lrs, stop_counter):
    ''' Performs the actuator deconvolution with lorentzians'''

    '''Initialize'''
    images = const[1]
    N_voltages = images.shape[0] # number of voltages per actuator
    N_planes = images.shape[1] # number of planes
    N_mols = images.shape[2]  # number of images per plane
    N_crop = images.shape[-1]  # image crop size
    N_pixels = N_voltages * N_planes * N_mols * N_crop ** 2 # number of pixels in the image stack

    it, maxiter, t1, i_stop = 0, 1e5, 1, 0
    score_new, score_best = 1e4, 1e4
    par_new, par_old, par_best = par, par, par

    while it < maxiter :
        '''update'''
        it += 1
        LLR_new, par_new, par_old, t1 = h.update(par_new, par_old, lrs, bounds, t1, const) # update parameters
        score_new = float(LLR_new / N_pixels) # update score
        '''save best result or stop'''
        if score_new < score_best:
            score_best, par_best = score_new, par_new
        else: 
            i_stop += 1
            if i_stop > stop_counter:
                break

    return par_best, score_best









#%%




#------------------------------------------------
#                PLOT FUNCTIONS                 #
#------------------------------------------------

# - show_prelocs
# - show_molecules
# - show_results_3d
# - show_results_2d
# - show_results_1d
# - show_results_time
# - show_results_hist
# - show_pixelation



def show_prelocs(I_photons, n=32, figsize=(8, 5)):
    """Show some random cropped molecules."""
    idx = np.random.randint(0, I_photons.shape[0], n)
    rows, cols, m = int(np.ceil(n / 8)), 8, 0
    fig, ax = plt.subplots(rows, cols, figsize=figsize, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    for i in range(rows):
        for j in range(cols):
            if m < n:
                ax[i, j].imshow(I_photons[idx[m], :, :], cmap=parula)
                ax[i, j].axis('off')
                m += 1
    plt.show()


def show_molecules(images, filter_indices, results, raw_results=None, title='id', n=16):
    """Randomly shows good molecules and filtered molecules. Choose which values to show"""

    if isinstance(title, int): # If title is an integer, use the corresponding column
        title = results.columns[title] if 0 <= title < len(results.columns) else results.columns[0]
    else:
        title = title if title in results.columns else results.columns[0]
    if n % 4 != 0: n = n + 4 - n % 4  # Make sure n is a multiple of 4

    # divide images and results table into 2 groups: good and filtered
    good_images, bad_images = images[filter_indices], images[~filter_indices]

    # generate random indices and select images to plot, with respective titles
    good_idx = np.random.choice(good_images.shape[0], n, replace=len(good_images) < n)
    bad_idx = np.random.choice(bad_images.shape[0], n, replace=len(bad_images) < n)
    good_images_plot, bad_images_plot = good_images[good_idx], bad_images[bad_idx]

    good_x, bad_x = raw_results[filter_indices]['x'], raw_results[~filter_indices]['x']
    good_y, bad_y = raw_results[filter_indices]['y'], raw_results[~filter_indices]['y']
    good_sg, bad_sg = results[filter_indices]['signal'], results[~filter_indices]['signal']
    good_bg, bad_bg = results[filter_indices]['bg'], results[~filter_indices]['bg']

    if title in ['x', 'y']:
        print('Header: (x, y) / pixels')
        good_1_plot, bad_1_plot = good_x.iloc[good_idx], bad_x.iloc[bad_idx]
        good_2_plot, bad_2_plot = good_y.iloc[good_idx], bad_y.iloc[bad_idx]

    elif title in ['signal', 'bg']:
        print('Header: (signal, bg) / photons')
        good_1_plot, bad_1_plot = good_sg.iloc[good_idx], bad_sg.iloc[bad_idx]
        good_2_plot, bad_2_plot = good_bg.iloc[good_idx], bad_bg.iloc[bad_idx]

    else:
        if title in ['id', 'frame', 'LLR']:
            print('Header:', title)
        else:
            print('Header:', title, '/ nm')

        good_results, bad_results = results[filter_indices][title], results[~filter_indices][title]
        good_results_plot, bad_results_plot = good_results.iloc[good_idx], bad_results.iloc[bad_idx]

    rows, cols, r, s = int(n / 4), 9, 0, 0
    fig, axes = plt.subplots(rows, cols, figsize=(10, 1.3 * rows))

    for i in range(rows):
        for j in range(4):  # First 4 columns for good molecules
            if r < n:
                axes[i, j].imshow(good_images_plot[r], cmap=parula)
                axes[i, j].axis('off')

                sig_val = results[filter_indices]['signal'].iloc[good_idx[r]]
                llr_val = results[filter_indices]['LLR'].iloc[good_idx[r]]

                axes[i, j].text(1, 25, f"S: {sig_val:.0f}", fontsize=10, color='black', ha='left', va='baseline')
                axes[i, j].text(1, 32, f"LLR: {llr_val:.2f}", fontsize=10, color='black', ha='left', va='baseline')
                if title in ['x', 'y', 'signal', 'bg']:
                    axes[i, j].set_title(f"({good_1_plot.iloc[r]:.1f}, {good_2_plot.iloc[r]:.1f})", fontsize=8)
                else:
                    axes[i, j].set_title(f"{good_results_plot.iloc[r]:.0f}", fontsize=8)
                r += 1
        if j == 3:  # Skip the 5th column (index 4)
            axes[i, 4].axis('off') # leave a space between good and filtered molecules
    for i in range(rows):
        for j in range(5, 9):  # Last 4 columns for bad molecules
            if s < n:
                axes[i, j].imshow(bad_images_plot[s], cmap=parula)
                axes[i, j].axis('off')

                sig_val = results[~filter_indices]['signal'].iloc[bad_idx[s]]
                llr_val = results[~filter_indices]['LLR'].iloc[bad_idx[s]]
                axes[i, j].text(1, 25, f"S: {sig_val:.0f}", fontsize=10, color='black', ha='left', va='baseline')
                axes[i, j].text(1, 32, f"LLR: {llr_val:.2f}", fontsize=10, color='black', ha='left', va='baseline')

                if title in ['x', 'y', 'signal', 'bg']:
                    axes[i, j].set_title(f"({bad_1_plot.iloc[s]:.1f}, {bad_2_plot.iloc[s]:.1f})", fontsize=8)
                else:
                    axes[i, j].set_title(f"{bad_results_plot.iloc[s]:.0f}", fontsize=8)                
                s += 1

    fig.suptitle(" ", fontsize=16)
    fig.text(0.23, 0.95, "Good Molecules", ha='center', fontsize=18)
    fig.text(0.777, 0.95, "Filtered Molecules", ha='center', fontsize=18)
    plt.tight_layout()
    plt.show()




def show_results_3d(data, marker_size=1.5): 
    ''' Show the 3d scatter plot with plotly'''

    if isinstance(data, pd.DataFrame):  # for pandas DataFrame
        x, y, z, s, bg, LLR = data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR']
    elif isinstance(data, (np.ndarray, jnp.ndarray)):  # for NumPy array
        data = pd.DataFrame(data, columns=['frame', 'x', 'y', 'z', 'signal', 'bg', 'LLR'])
        # x, y, z, s, bg, LLR = data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR']

    fig = plt.figure()
    fig = px.scatter_3d(data, x='x', y='y', z='z', color='z',
                        opacity=1, size_max=0.001, width=800, height=800)
    fig.update_traces(marker_size=marker_size)
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=True), #False),
            yaxis=dict(showticklabels=True), #False),
            zaxis=dict(showticklabels=True), #False),
        ),
    )
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Save as SVG",
                        method="restyle",  # No direct method; use custom script instead
                        args=[],
                        execute=True,
                        name="image"
                    )
                ],
            )
        ]
    )


    # fig.write_image("image.svg")
    fig.show()




def show_results_2d(data, marker_size=0.5):
    """Display 2D scatter plots for X/Y, X/Z, and Y/Z."""

    if isinstance(data, pd.DataFrame):  # for pandas DataFrame
        x, y, z, s, bg, LLR = data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR']
    elif isinstance(data, (np.ndarray, jnp.ndarray)):  # for NumPy array
        df = pd.DataFrame(data, columns=['frame', 'x', 'y', 'z', 'signal', 'bg', 'LLR'])
        x, y, z, s, bg, LLR = df['x'], df['y'], df['z'], df['signal'], df['bg'], df['LLR']

    plt.figure(figsize=(8, 5)); fs = 12

    def create_subplot(subplot_num, rowspan, colspan, title, x_data, y_data):
        ax = plt.subplot2grid((2, 3), subplot_num, rowspan=rowspan, colspan=colspan)
        ax.set_title(title, fontsize=fs)
        ax.scatter(x_data, y_data, s=marker_size)
        if subplot_num == (0, 0):
            ax.invert_yaxis()  # Invert y-axis for X/Y plot
            ax.set_aspect('equal', adjustable='box')  # Ensure equal scaling for both axes

    create_subplot((0, 0), 2, 2, "X / Y",x, y)
    create_subplot((0, 2), 1, 1, "X / Z", x, z)
    create_subplot((1, 2), 1, 1, "Y / Z", y, z)

    plt.suptitle('Scatter Plot [nm]', fontsize=fs + 2)
    plt.tight_layout()
    plt.show()


def show_results_xy(data, show_labels=True, figsize=(10,10), marker_size=0.5):

    if isinstance(data, pd.DataFrame):  # for pandas DataFrame
        x, y, z, s, bg, LLR = data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR']
    elif isinstance(data, (np.ndarray, jnp.ndarray)):  # for NumPy array
        df = pd.DataFrame(data, columns=['frame', 'x', 'y', 'z', 'signal', 'bg', 'LLR'])
        x, y, z, s, bg, LLR = df['x'], df['y'], df['z'], df['signal'], df['bg'], df['LLR']

    width, height = figsize[0] * 100, figsize[1] * 100
    fig = px.scatter(x = x, y = y, color = z, opacity = 1)
    fig.update_traces(marker_size=marker_size)
    fig.update_layout(width=int(width), 
                    height=int(height),
                    xaxis=dict(
                        zeroline=False, 
                        showgrid=False, 
                        showticklabels=show_labels,
                        scaleanchor='y',  # Ensure same scale on both axes
                        scaleratio=1  # Ratio to be the same for both axes
                    ),
                    yaxis=dict(
                        zeroline=False,
                        autorange='reversed',
                        showgrid=False,
                        showticklabels=show_labels
                    ),
                    plot_bgcolor='black', 
                    paper_bgcolor='black'
                        )
    
    # fig.write_image("image.svg")
    fig.show()



def show_results_1d(data, raw_data=None, h=None, marker_size=0.5):
    """Display 1D plots for Δx, Δy, z, Photon number, Background, and LLR."""

    if isinstance(data, pd.DataFrame):  # for pandas DataFrame
        x, y, z, s, bg, LLR = data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR']
    elif isinstance(data, (np.ndarray, jnp.ndarray)):  # for NumPy array
        df = pd.DataFrame(data, columns=['frame', 'x', 'y', 'z', 'signal', 'bg', 'LLR'])
        x, y, z, s, bg, LLR = df['x'], df['y'], df['z'], df['signal'], df['bg'], df['LLR']

    x_axis = np.arange(len(x))
    plt.figure(figsize=(8, 5)); fs = 12
    
    def create_subplot(subplot_num, title, data, x_axis, marker_size, fs):
        plt.subplot(2, 3, subplot_num)
        plt.title(title, fontsize=fs)
        plt.plot(x_axis, data, '.', ms=marker_size)
        plt.text(0.95, 0.95, f"μ: {np.mean(data):.2f}", ha='right', va='top', transform=plt.gca().transAxes)
        plt.text(0.95, 0.85, f"σ: {np.std(data):.2f}", ha='right', va='top', transform=plt.gca().transAxes)

    if raw_data is not None:
        create_subplot(1, "Δx (pixels)",    raw_data['x'],   x_axis, marker_size, fs)
        create_subplot(2, "Δy (pixels)",    raw_data['y'],   x_axis, marker_size, fs)
    else : 
        create_subplot(1, "mod(Δx) (pixels)",    np.mod(x, h.ux * 1e9 / h.os),   x_axis, marker_size, fs)
        create_subplot(2, "mod(Δy) (pixels)",    np.mod(y, h.ux * 1e9 / h.os),   x_axis, marker_size, fs)

    create_subplot(3, "z (nm)",         z,      x_axis, marker_size, fs)
    create_subplot(4, "Photon number",  s,      x_axis, marker_size, fs)
    create_subplot(5, "Background",     bg,     x_axis, marker_size, fs)
    create_subplot(6, "LLR",            LLR,    x_axis, marker_size, fs)

    plt.suptitle('Localizations', fontsize=fs + 2)
    plt.tight_layout()
    plt.show()




def show_results_time(data, raw_data=None, h=None, exp_time = None, marker_size=0.5):
    """Display 1D plots vs frame of for Δx, Δy, z, Photon number, Background, and LLR."""

    if isinstance(data, pd.DataFrame):  # for pandas DataFrame
        frame, x, y, z, s, bg, LLR = data['frame'], data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR']
    elif isinstance(data, (np.ndarray, jnp.ndarray)):  # for NumPy array
        df = pd.DataFrame(data, columns=['frame', 'x', 'y', 'z', 'signal', 'bg', 'LLR'])
        frame, x, y, z, s, bg, LLR = df['frame'], df['x'], df['y'], df['z'], df['signal'], df['bg'], df['LLR']

    time = frame * exp_time *1e-3/60 if exp_time else frame # time in minutes if exp_time is given, else time in frames

    plt.figure(figsize=(8, 5)); fs = 12
    
    def create_subplot(subplot_num, title, data, x_axis, marker_size, fs):
        plt.subplot(2, 3, subplot_num)
        plt.title(title, fontsize=fs)
        plt.scatter(x_axis, data, s=marker_size)
        plt.text(0.95, 0.95, f"μ: {np.mean(data):.2f}", ha='right', va='top', transform=plt.gca().transAxes)
        plt.text(0.95, 0.85, f"σ: {np.std(data):.2f}", ha='right', va='top', transform=plt.gca().transAxes)

    if raw_data is not None:
        create_subplot(1, "Δx (pixels)",    raw_data['x'],   time, marker_size, fs)
        create_subplot(2, "Δy (pixels)",    raw_data['y'],   time, marker_size, fs)
    else : 
        create_subplot(1, "mod(Δx) (pixels)",    np.mod(x, h.ux * 1e9 / h.os),   time, marker_size, fs)
        create_subplot(2, "mod(Δy) (pixels)",    np.mod(y, h.ux * 1e9 / h.os),   time, marker_size, fs)

    create_subplot(3, "z (nm)",         z,      time, marker_size, fs)
    create_subplot(4, "Photon number",  s,      time, marker_size, fs)
    create_subplot(5, "Background",     bg,     time, marker_size, fs)
    create_subplot(6, "LLR",            LLR,    time, marker_size, fs)

    plt.suptitle('Localizations vs time (minutes)' if exp_time else 'Localizations vs frames', fontsize=fs + 2)

    plt.tight_layout()
    plt.show()




def show_results_hist(data, raw_data=None, h=None, bins=20):
    """Display histograms for Δx, Δy, z, Photon number, Background, and LLR."""

    if isinstance(data, pd.DataFrame):  # for pandas DataFrame
        x, y, z, s, bg, LLR = data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR']
    elif isinstance(data, (np.ndarray, jnp.ndarray)):  # for NumPy array
        df = pd.DataFrame(data, columns=['frame', 'x', 'y', 'z', 'signal', 'bg', 'LLR'])
        x, y, z, s, bg, LLR = df['x'], df['y'], df['z'], df['signal'], df['bg'], df['LLR']
    
    if h is not None:
        LLR /= (h.N_crop ** 2)

    plt.figure(figsize=(8, 5));  fs = 12

    def create_subplot(subplot_num, title, data, bins, fs):
        plt.subplot(2, 3, subplot_num)
        plt.title(title, fontsize=fs)
        plt.hist(data, bins)
        plt.text(0.95, 0.95, f"μ: {np.mean(data):.2f}", ha='right', va='top', transform=plt.gca().transAxes)
        plt.text(0.95, 0.85, f"σ: {np.std(data):.2f}", ha='right', va='top', transform=plt.gca().transAxes)

    if raw_data is not None:
        create_subplot(1, "Δx (pixels)", raw_data['x'], bins, fs)
        create_subplot(2, "Δy (pixels)", raw_data['y'], bins, fs)
    else : 
        create_subplot(1, "mod(Δx) (pixels)", np.mod(x, h.ux * 1e9 / h.os), bins, fs)
        create_subplot(2, "mod(Δy) (pixels)", np.mod(y, h.ux * 1e9 / h.os), bins, fs)

    create_subplot(3, "z (nm)",         z,      bins, fs)
    create_subplot(4, "Photon number",  s,      bins, fs)
    create_subplot(5, "Background",     bg,     bins, fs)
    create_subplot(6, "LLR",            LLR,    bins, fs)

    plt.suptitle('Histograms', fontsize=fs+3)
    plt.tight_layout()
    plt.show()




def show_pixelation(h, results, bins_plotted=1, bins=100):
    
    plt.figure(figsize=(10, 4))
    mod_x = np.mod(results['x'], bins_plotted * h.ux * 1e9 / h.os)
    mod_y = np.mod(results['y'], bins_plotted * h.ux * 1e9 / h.os)
    mod_z = np.mod(results['z'], bins_plotted * h.uz * 1e9)

    for i, (data, axis) in enumerate(zip([mod_x, mod_y, mod_z], ['X', 'Y', 'Z']), 1):
        plt.subplot(1, 3, i)
        plt.hist(data, bins=bins)
        plt.title(axis)

    plt.suptitle('Pixelation', fontsize=15)
    plt.show()













#%%


#------------------------------------------------
#             ZERNIKE POLYNOMIALS               #
#------------------------------------------------

# - zernike_noll
# - zernike_nm
# - zernikeRadialFunc
# - zernIndex


def zernike_noll(j, N, rot=0):
    """
      Creates the Zernike polynomial with mode index j,
      where j = 1 corresponds to piston.
      Args:
         j (int): The noll j number of the zernike mode
         N (int): The diameter of the zernike more in pixels
         rot (float): Rotates the Zernike mode by rot radians around  its centre. Defaults to zero.
      Returns:
         ndarray: The Zernike mode
    """
    n, m = zernIndex(j)
    return zernike_nm(n, m, N, rot)


def zernike_nm(n, m, N, rot=0):
    """
      Creates the Zernike polynomial with radial index, n, and  azimuthal index, m.
      Args:
         n (int): The radial order of the zernike polynomial
         m (int): The azimuthal order of the zernike polynomial
         N (int): The diameter of the zernike polynomial in pixels
         rot (float): Rotates the zernike by rot radians around its centre. Defaults to 0.
      Returns:
         ndarray: The Zernike polynomial
    """
    coords = np.fft.fftfreq(N, 1/N) /(N/2) # here is the new definition of the coordinate grid
    if N%2==0:                          # NEW: centering pupil also for even N and odd N
        coords += np.diff(coords)[0]/2  # NEW: shift coordinate system if even N to center Zernikes
    X, Y = np.meshgrid(coords, coords)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    if m==0:
        Z = np.sqrt(n+1)*zernikeRadialFunc(n, 0, R)
    else:
        if m > 0: # j is even
            Z = np.sqrt(2*(n+1)) * zernikeRadialFunc(n, m, R) * np.cos((m*theta)+rot)
        else:   #i is odd
            m = abs(m)
            Z = np.sqrt(2*(n+1)) * zernikeRadialFunc(n, m, R) * np.sin((m*theta)+rot)

    # clip
    Z = Z*np.less_equal(R, 1.0)

    # pupil
    mask = (X**2 + Y**2 <= N) # new!!

    return np.fft.fftshift(Z*mask) # new!!


def zernikeRadialFunc(n, m, r):
     """
     Fucntion to calculate the Zernike radial function
     Parameters:
         n (int): Zernike radial order
         m (int): Zernike azimuthal order
         r (ndarray): 2-d array of radii from the centre the array
     Returns:
         ndarray: The Zernike radial function
     """

     R = np.zeros(r.shape)
     # Can cast the below to "int", n,m are always *both* either even or odd
     for i in range(0, int((n - m) / 2) + 1):

         R += np.array(r**(n - 2 * i) * (((-1)**(i)) *
                        scipy.special.factorial(n - i)) /
                        (scipy.special.factorial(i) *
                        scipy.special.factorial(int(0.5 * (n + m) - i)) *
                        scipy.special.factorial(int(0.5 * (n - m) - i))),
                        dtype='float')
     return R


def zernIndex(j):
     """
     Find the [n,m] list giving the radial order n and azimuthal order
     of the Zernike polynomial of Noll index j.
     Parameters:
         j (int): The Noll index for Zernike polynomials
     Returns:
         list: n, m values
     """
     n = int((-1.+np.sqrt(8*(j-1)+1))/2.)
     p = (j-(n*(n+1))/2.)
     k = n%2
     m = int((p+k)/2.)*2 - k
     if m!=0:
         if j%2==0:
             s=1
         else:
             s=-1
         m *= s
     return [n, m]


def lorentzian_2d(grid, g):
    '''Create a 2D Lorentzian to model the DPP actuators influence function'''

    X, Y = grid # grid of coordinates with size Nk x Nk (BFP size). (0,0) is the center of the pupil.
    I0, x0, y0, gamma_x, gamma_y, phi, offset = g

    x_rot = (X - x0) * jnp.cos(phi) + (Y - y0) * jnp.sin(phi)
    y_rot = -(X - x0) * jnp.sin(phi) + (Y - y0) * jnp.cos(phi)
    r2 = (x_rot**2 / gamma_x**2) + (y_rot**2 / gamma_y**2)

    lorentzian = I0 / (1 + r2) + offset

    return lorentzian


def gaussian_2d(grid,  g):
    '''Create a 2D gaussian to model the DPP actuators influence function'''

    X, Y = grid # grid of coordinates with size Nk x Nk (BFP size). (0,0) is the center of the pupil.
    A,  x0, y0, sigma_x, sigma_y, phi = g

    x_rot = (X - x0) * jnp.cos(phi) + (Y - y0) * jnp.sin(phi)
    y_rot = -(X - x0) * jnp.sin(phi) + (Y - y0) * jnp.cos(phi)

    gaussian = A * jnp.exp(-((x_rot**2) / (2 * sigma_x**2) + (y_rot**2) / (2 * sigma_y**2)))

    return gaussian


#%%

#------------------------------------------------
#             ORTHOGONALIZE EIGENBASIS               #
#------------------------------------------------

# - orthogonal_eigenbasis
# - orthogonalize_IF



def prepare_modeset(modeset, Nk):
    '''This function pulish the modeset by normalizing each mode to RMS=1 and binning or interpolating to the desired size Nk.'''

    N = modeset.shape[-1]
    pupil = zernike_noll(1, Nk)

    if N==Nk:
        pass
    elif N > Nk and N % Nk == 0:
        bin_factor = N // Nk
        modeset = jnp.mean(modeset.reshape(len(modeset), Nk, bin_factor, Nk, bin_factor), axis=(-3, -1))
    else:
        # Interpolate to target size Nk using bilinear interpolation
        modeset = jnp.stack([
            jax.image.resize(m, (Nk, Nk), method="linear") for m in modeset
        ])

    stds = np.array([np.std(modeset[i][pupil > 0]) for i in range(modeset.shape[0])])
    stds = np.where(stds == 0, 1, stds) # Avoid divide-by-zero if pupil mask empty
    modeset /= stds[:, None, None] # Normalize each 2D frame

    return modeset


def orthogonal_eigenbasis(N, data_path, normalize=False, IF_provided=False):
    '''Returns an orthogonal eigenbasis.
    
    Parameters:
        N (int):            number of pixels acorss the pupil
        data_path (str):    path and filename of the data that is loaded (the data can be the influence functions of the device or the orthogonalize eigenbasis)
        IF_provided (bool): states whether data contains the influence functions or the orthogonalized eigenbasis

    Returns:
        modeset of orthogonalized eigenbasis
    '''

    # Create pupil mask
    coords = np.fft.fftfreq(N, 1/N) /(N/2) # here is the new definition of the coordinate grid
    if N%2==0:                          # centering pupil also for even N and odd N
        coords += np.diff(coords)[0]/2  # shift coordinate system if even N to center Zernikes
    X, Y = np.meshgrid(coords, coords)
    mask = (X**2 + Y**2 <= 1) # fftshifted pupil

    # Load data & interpolate/extrapolate
    if data_path == None:
        raise ValueError("Please provide a path to the data of the influence functiosn or the orthogonalized basis!") 
    if IF_provided:
        IF = jnp.load(data_path)
        OEB = orthogonalize_IF(IF)
    else:
        OEB = jnp.load(data_path)

    # match size of OEB to requested diameter
    if OEB.shape[-1] == N:
        pass
    elif OEB.shape[-1] > N and OEB.shape[-1] % N == 0:
        bin_factor = OEB.shape[-1] // N
        OEB = jnp.mean(OEB.reshape(len(OEB), N, bin_factor, N, bin_factor), axis=(-3, -1))
    else:
        raise ValueError("The pupil diameter of the data must be chosen such that N evenly divides the pupil diameter for correct binning.")
    
    # Normalize Basis to
    if normalize:

        M = jnp.sum(OEB[:, np.newaxis]*OEB[np.newaxis, ...], axis=(-2, -1))
        EVs = np.diag(M)

        OEB /= EVs[:, jnp.newaxis, jnp.newaxis] # normalize to 1
        OEB *= np.sqrt(np.sum(mask)) # scale with the number of pixels inside the pupil
    
    return OEB * np.fft.fftshift(mask)
    


def orthogonalize_IF(IF):
    '''Compute the orthogonbal eigenbasis from the Influence functions'''
    # To be done

    raise ValueError("This function is still in progress")
    return None

#%%
# Parula (matlab) colormap      
_parula_rgb_data = [
[0.2422, 0.1504, 0.6603],[0.2444, 0.1534, 0.6728],[0.2464, 0.1569, 0.6847],[0.2484, 0.1607, 0.6961],
[0.2503, 0.1648, 0.7071],[0.2522, 0.1689, 0.7179],[0.2540, 0.1732, 0.7286],[0.2558, 0.1773, 0.7393],
[0.2576, 0.1814, 0.7501],[0.2594, 0.1854, 0.7610],[0.2611, 0.1893, 0.7719],[0.2628, 0.1932, 0.7828],
[0.2645, 0.1972, 0.7937],[0.2661, 0.2011, 0.8043],[0.2676, 0.2052, 0.8148],[0.2691, 0.2094, 0.8249],
[0.2704, 0.2138, 0.8346],[0.2717, 0.2184, 0.8439],[0.2729, 0.2231, 0.8528],[0.2740, 0.2280, 0.8612],
[0.2749, 0.2330, 0.8692],[0.2758, 0.2382, 0.8767],[0.2766, 0.2435, 0.8840],[0.2774, 0.2489, 0.8908],
[0.2781, 0.2543, 0.8973],[0.2788, 0.2598, 0.9035],[0.2794, 0.2653, 0.9094],[0.2798, 0.2708, 0.9150],
[0.2802, 0.2764, 0.9204],[0.2806, 0.2819, 0.9255],[0.2809, 0.2875, 0.9305],[0.2811, 0.2930, 0.9352],
[0.2813, 0.2985, 0.9397],[0.2814, 0.3040, 0.9441],[0.2814, 0.3095, 0.9483],[0.2813, 0.3150, 0.9524],
[0.2811, 0.3204, 0.9563],[0.2809, 0.3259, 0.9600],[0.2807, 0.3313, 0.9636],[0.2803, 0.3367, 0.9670],
[0.2798, 0.3421, 0.9702],[0.2791, 0.3475, 0.9733],[0.2784, 0.3529, 0.9763],[0.2776, 0.3583, 0.9791],
[0.2766, 0.3638, 0.9817],[0.2754, 0.3693, 0.9840],[0.2741, 0.3748, 0.9862],[0.2726, 0.3804, 0.9881],
[0.2710, 0.3860, 0.9898],[0.2691, 0.3916, 0.9912],[0.2670, 0.3973, 0.9924],[0.2647, 0.4030, 0.9935],
[0.2621, 0.4088, 0.9946],[0.2591, 0.4145, 0.9955],[0.2556, 0.4203, 0.9965],[0.2517, 0.4261, 0.9974],
[0.2473, 0.4319, 0.9983],[0.2424, 0.4378, 0.9991],[0.2369, 0.4437, 0.9996],[0.2311, 0.4497, 0.9995],
[0.2250, 0.4559, 0.9985],[0.2189, 0.4620, 0.9968],[0.2128, 0.4682, 0.9948],[0.2066, 0.4743, 0.9926],
[0.2006, 0.4803, 0.9906],[0.1950, 0.4861, 0.9887],[0.1903, 0.4919, 0.9867],[0.1869, 0.4975, 0.9844],
[0.1847, 0.5030, 0.9819],[0.1831, 0.5084, 0.9793],[0.1818, 0.5138, 0.9766],[0.1806, 0.5191, 0.9738],
[0.1795, 0.5244, 0.9709],[0.1785, 0.5296, 0.9677],[0.1778, 0.5349, 0.9641],[0.1773, 0.5401, 0.9602],
[0.1768, 0.5452, 0.9560],[0.1764, 0.5504, 0.9516],[0.1755, 0.5554, 0.9473],[0.1740, 0.5605, 0.9432],
[0.1716, 0.5655, 0.9393],[0.1686, 0.5705, 0.9357],[0.1649, 0.5755, 0.9323],[0.1610, 0.5805, 0.9289],
[0.1573, 0.5854, 0.9254],[0.1540, 0.5902, 0.9218],[0.1513, 0.5950, 0.9182],[0.1492, 0.5997, 0.9147],
[0.1475, 0.6043, 0.9113],[0.1461, 0.6089, 0.9080],[0.1446, 0.6135, 0.9050],[0.1429, 0.6180, 0.9022],
[0.1408, 0.6226, 0.8998],[0.1383, 0.6272, 0.8975],[0.1354, 0.6317, 0.8953],[0.1321, 0.6363, 0.8932],
[0.1288, 0.6408, 0.8910],[0.1253, 0.6453, 0.8887],[0.1219, 0.6497, 0.8862],[0.1185, 0.6541, 0.8834],
[0.1152, 0.6584, 0.8804],[0.1119, 0.6627, 0.8770],[0.1085, 0.6669, 0.8734],[0.1048, 0.6710, 0.8695],
[0.1009, 0.6750, 0.8653],[0.0964, 0.6789, 0.8609],[0.0914, 0.6828, 0.8562],[0.0855, 0.6865, 0.8513],
[0.0789, 0.6902, 0.8462],[0.0713, 0.6938, 0.8409],[0.0628, 0.6972, 0.8355],[0.0535, 0.7006, 0.8299],
[0.0433, 0.7039, 0.8242],[0.0328, 0.7071, 0.8183],[0.0234, 0.7103, 0.8124],[0.0155, 0.7133, 0.8064],
[0.0091, 0.7163, 0.8003],[0.0046, 0.7192, 0.7941],[0.0019, 0.7220, 0.7878],[0.0009, 0.7248, 0.7815],
[0.0018, 0.7275, 0.7752],[0.0046, 0.7301, 0.7688],[0.0094, 0.7327, 0.7623],[0.0162, 0.7352, 0.7558],
[0.0253, 0.7376, 0.7492],[0.0369, 0.7400, 0.7426],[0.0504, 0.7423, 0.7359],[0.0638, 0.7446, 0.7292],
[0.0770, 0.7468, 0.7224],[0.0899, 0.7489, 0.7156],[0.1023, 0.7510, 0.7088],[0.1141, 0.7531, 0.7019],
[0.1252, 0.7552, 0.6950],[0.1354, 0.7572, 0.6881],[0.1448, 0.7593, 0.6812],[0.1532, 0.7614, 0.6741],
[0.1609, 0.7635, 0.6671],[0.1678, 0.7656, 0.6599],[0.1741, 0.7678, 0.6527],[0.1799, 0.7699, 0.6454],
[0.1853, 0.7721, 0.6379],[0.1905, 0.7743, 0.6303],[0.1954, 0.7765, 0.6225],[0.2003, 0.7787, 0.6146],
[0.2061, 0.7808, 0.6065],[0.2118, 0.7828, 0.5983],[0.2178, 0.7849, 0.5899],[0.2244, 0.7869, 0.5813],
[0.2318, 0.7887, 0.5725],[0.2401, 0.7905, 0.5636],[0.2491, 0.7922, 0.5546],[0.2589, 0.7937, 0.5454],
[0.2695, 0.7951, 0.5360],[0.2809, 0.7964, 0.5266],[0.2929, 0.7975, 0.5170],[0.3052, 0.7985, 0.5074],
[0.3176, 0.7994, 0.4975],[0.3301, 0.8002, 0.4876],[0.3424, 0.8009, 0.4774],[0.3548, 0.8016, 0.4669],
[0.3671, 0.8021, 0.4563],[0.3795, 0.8026, 0.4454],[0.3921, 0.8029, 0.4344],[0.4050, 0.8031, 0.4233],
[0.4184, 0.8030, 0.4122],[0.4322, 0.8028, 0.4013],[0.4463, 0.8024, 0.3904],[0.4608, 0.8018, 0.3797],
[0.4753, 0.8011, 0.3691],[0.4899, 0.8002, 0.3586],[0.5044, 0.7993, 0.3480],[0.5187, 0.7982, 0.3374],
[0.5329, 0.7970, 0.3267],[0.5470, 0.7957, 0.3159],[0.5609, 0.7943, 0.3050],[0.5748, 0.7929, 0.2941],
[0.5886, 0.7913, 0.2833],[0.6024, 0.7896, 0.2726],[0.6161, 0.7878, 0.2622],[0.6297, 0.7859, 0.2521],
[0.6433, 0.7839, 0.2423],[0.6567, 0.7818, 0.2329],[0.6701, 0.7796, 0.2239],[0.6833, 0.7773, 0.2155],
[0.6963, 0.7750, 0.2075],[0.7091, 0.7727, 0.1998],[0.7218, 0.7703, 0.1924],[0.7344, 0.7679, 0.1852],
[0.7468, 0.7654, 0.1782],[0.7590, 0.7629, 0.1717],[0.7710, 0.7604, 0.1658],[0.7829, 0.7579, 0.1608],
[0.7945, 0.7554, 0.1570],[0.8060, 0.7529, 0.1546],[0.8172, 0.7505, 0.1535],[0.8281, 0.7481, 0.1536],
[0.8389, 0.7457, 0.1546],[0.8495, 0.7435, 0.1564],[0.8600, 0.7413, 0.1587],[0.8703, 0.7392, 0.1615],
[0.8804, 0.7372, 0.1650],[0.8903, 0.7353, 0.1695],[0.9000, 0.7336, 0.1749],[0.9093, 0.7321, 0.1815],
[0.9184, 0.7308, 0.1890],[0.9272, 0.7298, 0.1973],[0.9357, 0.7290, 0.2061],[0.9440, 0.7285, 0.2151],
[0.9523, 0.7284, 0.2237],[0.9606, 0.7285, 0.2312],[0.9689, 0.7292, 0.2373],[0.9770, 0.7304, 0.2418],
[0.9842, 0.7330, 0.2446],[0.9900, 0.7365, 0.2429],[0.9946, 0.7407, 0.2394],[0.9966, 0.7458, 0.2351],
[0.9971, 0.7513, 0.2309],[0.9972, 0.7569, 0.2267],[0.9971, 0.7626, 0.2224],[0.9969, 0.7683, 0.2181],
[0.9966, 0.7740, 0.2138],[0.9962, 0.7798, 0.2095],[0.9957, 0.7856, 0.2053],[0.9949, 0.7915, 0.2012],
[0.9938, 0.7974, 0.1974],[0.9923, 0.8034, 0.1939],[0.9906, 0.8095, 0.1906],[0.9885, 0.8156, 0.1875],
[0.9861, 0.8218, 0.1846],[0.9835, 0.8280, 0.1817],[0.9807, 0.8342, 0.1787],[0.9778, 0.8404, 0.1757],
[0.9748, 0.8467, 0.1726],[0.9720, 0.8529, 0.1695],[0.9694, 0.8591, 0.1665],[0.9671, 0.8654, 0.1636],
[0.9651, 0.8716, 0.1608],[0.9634, 0.8778, 0.1582],[0.9619, 0.8840, 0.1557],[0.9608, 0.8902, 0.1532],
[0.9601, 0.8963, 0.1507],[0.9596, 0.9023, 0.1480],[0.9595, 0.9084, 0.1450],[0.9597, 0.9143, 0.1418],
[0.9601, 0.9203, 0.1382],[0.9608, 0.9262, 0.1344],[0.9618, 0.9320, 0.1304],[0.9629, 0.9379, 0.1261],
[0.9642, 0.9437, 0.1216],[0.9657, 0.9494, 0.1168],[0.9674, 0.9552, 0.1116],[0.9692, 0.9609, 0.1061],
[0.9711, 0.9667, 0.1001],[0.9730, 0.9724, 0.0938],[0.9749, 0.9782, 0.0872],[0.9769, 0.9839, 0.0805]]
from matplotlib.colors import ListedColormap
cmaps = {}
cmaps['parula'] = ListedColormap(_parula_rgb_data, name='parula')
parula = cmaps['parula']

# %%
