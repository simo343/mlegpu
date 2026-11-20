""" PSF generation and precise localization of single molecules in 3D"""

#%%

# core
import time
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
from io import BytesIO


# jax
import jax
import jax.numpy as jnp
from jax import vmap
from jax.numpy.linalg import solve,norm
from jax import config
import jax.scipy as jsp
# config.update("jax_enable_x64", False)

# skimage
from skimage.feature import peak_local_max
from sklearn.linear_model import LinearRegression
from functools import partial
import copy

# scipy
import scipy.io
import scipy.optimize as opt
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import mat73


# functions
import sys
import importlib
import SMLM_functions_core
from SMLM_functions_core import *
import SMLM_functions_data_processing
from SMLM_functions_data_processing import *


#%% INDEX OF CONTENT

# CAMERA CLASS

# OBJECTIVE CLASS

# PSF CLASS

    #   INTERPOLATION
    #   - f
    #   - f_single
    #   - f_biplane
    #   - generate_image

    # FIELD CALCULATION
    #   - parameters
    #   - fresnel_coefs
    #   - BFP_fields
    #   - czt2_coefs
    #   - czt2
    #   - generate_psf
    #   - gaussian_kernel
    #   - blurr_image

    # PUPIL FUNCTIONS
    #   - zernikes
    #   - eigenbasis
    #   - aberrations
    #   - calc_defocus
    #   - calc_SA
    #   - RI_mismatch_correction
    #   - apodization
    #   - calc_CRLB

    #   PLOT FUNCTIONS
    #   - show_slice_xy
    #   - show_projection
    #   - show_pupil
    #   - show_CRLB


# CHILDREN PSF CLASSES

#   - PSF_FFT
    #   - generate_image_fft

#   - PSF_DECONVOLUTION
    #  - compute_err
    #  - grad_fun
    #  - fwd_multiplane
    #  - fwd_multiplane_bead
    #  - fwd_multiplane_pxl
    #  - generate_image_bead
    #  - generate_image_dpp
    #  - actuator_IF
    #  - DPP






#%%



#------------------------------------------------
#                 CAMERA CLASS                  #
#------------------------------------------------



class Camera:
    def __init__(self, pixsize = 6500, QE = 0.75, baseline = 100, amp = 0.24, EM_gain = 1, readnoise = 1, bitdepth = 16):
        '''
        # Formula to get Photon number from counts:
        Img[photons] = (Img[counts] - baseline[counts]) * amp[electrons/count] / QE[electrons/photon] 

        Be Aware: amp is named Conversion Factor by Hamamatsu and Sensitivity by Andor.
        it is the number of electrons that you need to make 1 count.
        ''' 
        self.pixsize = pixsize                  # pixel size in nm
        self.QE = QE                            # conversion factor from electrons to photons
        self.baseline = baseline                # dark offset, in counts
        self.EM_gain = EM_gain                  # electrons multiplicative factor. for non EMCCDs is 1
        self.amp = amp / EM_gain                # conversion factor from counts to electrons, normalized by EM_gain
        self.readnoise = readnoise / EM_gain    # in electrons rms, normalized by EM_gain
        self.bitdepth = bitdepth                # camera bitdepth
        self.maxcounts = 2**bitdepth - 1        # maximum counts given by bitdepth







#------------------------------------------------
#                OBJECTIVE CLASS                #
#------------------------------------------------



class Objective:
    def __init__(self, M = 60, NA = 1.49, RI = 1.518, focal_len = 3e6, Ts_coefs = [1], Tp_coefs = [1]):
        self.M = M
        self.NA = NA
        self.RI = RI
        self.focal_len = focal_len      # nm, f_obj = f_tubelens / M
        self.Ts_coefs = Ts_coefs
        self.Tp_coefs = Tp_coefs





#------------------------------------------------
#                   PSF CLASS                   #
#------------------------------------------------



class psf:
    def __init__(self, 
                 NA = 1.49, 
                 wavelength = 670,          # nm
                 z_defocus = 400,           # nm
                 RI = [1.33, 1.518],        # sample, immersion medium
                 Nk = 128, 
                 Nx = 17, 
                 ux = 108.3,                # nm
                 uz = 5,                    # nm
                 focal_len = 3e6,           # nm
                 z_min = 0,                 # nm
                 z_max = 300,               # nm
                 Z_modes = [5, 6, 7, 8, 11],# noll indices
                 Z_magn = [0, 0, 0, 0, 0],  # noll amplitudes, in rad
                 Zernike_modeset = True,    # If false use another modeset
                 modeset_array = None,      # array with custom modeset
                 Ts_coefs = [1],
                 Tp_coefs = [1],
                 T_fit_coefs = [1],
                 d2 = 0.,                   # nm
                 os = 1,
                 sigma = 0.,                # nm
                 interp_method = 'lanczos5',
                 correct_RI_mismatch = 'no'):
        
        '''initialize psf base parameters'''
        self.NA = NA                                # NA of objective
        self.RI = RI                                # list of RIs: [sample, (intermediate layer, when present), immersion medium]
        self.wavelength = wavelength * 1e-9         # vacuum wavelength of peak emission in m
        self.focal_len = focal_len * 1e-9           # objective focal lenght in m
        self.Nk = Nk                                # grid size in k space. Must be a power of 2 for computational efficiency.

        '''initialize psf size parameters'''
        self.z_min = z_min * 1e-9                   # PSF start z value. Distance from coverslip in m
        self.z_max = z_max * 1e-9                   # PSF end z value. Distance from coverlip in m
        self.uz = uz * 1e-9                         # PSF z step size in m
        self.Nx = Nx                                # PSF pixel size in real space. Should be 4 pixels larger than the images to fit for better interpolation.
        self.ux = ux * 1e-9                         # PSF effective pixel size in m: sensor pixel size / objective magnification
        self.os = os                                # oversampling. integer.

        '''initialize psf properties'''
        self.d2 = d2 * 1e-9                         # intermediate layer thickness in m
        self.z_defocus = z_defocus * 1e-9           # distance between the coverslip and the nominal objective focal plane. in m
        self.sigma_blurring = sigma * os / ux       # gaussian sigma to blur the PSF. in units of camera effective pixel size * os.
        self.Z_modes = jnp.array(Z_modes)           # aberration Noll indices
        self.Z_magn = jnp.array(Z_magn)             # aberration amplitude (radians).
        self.zernikes() if Zernike_modeset else self.custom_modeset(np.array(modeset_array))            # creates the modeset basis: modeset and modeset_fftshift
        self.Ts_coefs = jnp.array(Ts_coefs)         # objective s apodization coefficients
        self.Tp_coefs = jnp.array(Tp_coefs)         # objective p apodization coefficients
        self.T_fit_coefs = jnp.array(T_fit_coefs)   # field transmission
        self.interp_method = interp_method          # interpolation method: tricubic or lanczos5

        ''' PSF functions (keep this order) '''
        self.parameters()           # computes parameters needed in the following functions
        self.apodization()          # computes the objective field amplitude transmission

        self.fresnel_coefs()        # computes Fresnel coefficients
        self.czt2_coefs()           # computes parameters needed in the 2d chirp z-transform
        self.gaussian_kernel()      # create the gaussian kernel for the PSF blurring

        ''' Electric fields in the BFP. [E_BFP shape: (6, len(z_vec), Nk, Nk)]'''
        self.Exx, self.Eyx = self.BFP_fields([jnp.pi/2,0])             # emitting dipole oriented along x
        self.Exy, self.Eyy = self.BFP_fields([jnp.pi/2, jnp.pi/2])     # emitting dipole oriented along y
        self.Exz, self.Eyz = self.BFP_fields([0,0])                    # emitting dipole oriented along z
        self.E_BFP = jnp.array([self.Exx, self.Eyx, self.Exy, self.Eyy, self.Exz, self.Eyz])[..., jnp.newaxis].transpose((0, 3, 1, 2)) # newaxis needed for z

        ''' PUPIL contributions: aberrations and objective defocus '''
        self.pupil_aberrations = self.aberrations()  # Pupil phase given by aberrations (Defocus Noll idx = 4 not included). fft shifted
        self.obj_defocus,_ = self.calc_defocus( - self.z_defocus, self.RI[2]) # pupil phase given by objective nominal defocus (in immersion medium). fft shifted
        # Phase correction of defocus due to RI mismatch. fft shifted.
        self.calc_SA( - self.z_defocus)  # This term is implicitly inlcuded in the Axelrod model. Here we calculate it explicitly for visualization, RI correction or zernike decomposition.
        self.RI_mismatch_correction(correct_RI_mismatch)  # apply spherical aberration correction if desired: tot_defocus = defocus + (SA or SA_pure or 0). fft shifted

        '''Generate the PSF'''
        self.exp_jphase = jnp.exp(1j*jnp.real(self.tot_defocus + self.pupil_aberrations)) * self.apodization # pupil phase that creates the PSF. fft shifted.
        self.data, self.data_fields = self.generate_psf(self.z_vec, self.exp_jphase) #calculate the 3D PSF with blurring




    
        



    # FIELD CALCULATION
    #   - parameters
    #   - fresnel_coefs
    #   - BFP_fields
    #   - czt2_coefs
    #   - czt2
    #   - generate_psf
    #   - gaussian_kernel
    #   - blurr_image


    def parameters(self):
        '''This function computes the parameters needed in the psf class '''
        
        # Refractive Indices of: sample, intermediate layer, coverslip         
        if len(self.RI) == 1:
            self.RI = [self.RI[0], self.RI[0], self.RI[0]] # medium 1 (sample, typ water), 2 (intermediate layer, when present), 3 (immersion medium=glass)
        elif len(self.RI) == 2:
            self.RI = [self.RI[0], self.RI[1], self.RI[1]] 

        # array of z slices: is the emitters z positions. used to calculate the PSF
        self.z_vec = jnp.linspace(self.z_min, self.z_max, int((self.z_max - self.z_min) / self.uz + 1)) # meters
        self.dz = self.z_vec[1] - self.z_vec[0]  # z step size in m

        '''k-space parameters'''
        self.k0 = 2*jnp.pi/self.wavelength                                  # Wavevector in vacuum
        self.k1, self.k2, self.k3 = [self.k0 * RI for RI in self.RI]        # Wavevectors in media 1, 2 and 3
        self.uk = 2*self.k0 * self.NA / self.Nk                             # BFP pixel size
        #k_vec = (jnp.fft.fftfreq(self.Nk, 1/self.Nk)+0.5)*self.uk
        k_vec = (jnp.fft.fftfreq(self.Nk, 1/self.Nk))*self.uk
        self.Kx, self.Ky = jnp.meshgrid(k_vec, k_vec)
        self.Kr = jnp.sqrt(self.Kx**2 + self.Ky**2)

        x = jnp.arange(-self.Nk//2, self.Nk//2) + 1
        self.gauss_grid = jnp.meshgrid(x, x) # grid used to generate gaussians

        '''parameters for the pupil plane'''
        self.N_pad = jnp.round(2*jnp.pi/self.uk/self.ux)  # number of pixels to pad the BFP fields to meet the resolution ux
        # self.pupil = jnp.fft.ifftshift((zernike_noll(1, self.Nk))).astype(int) # pupil defined by zernike 1
        self.pupil = self.Kr <= (self.k0*self.NA)       # UAF + SAF pupil mask (in fourier space)
        self.pupil_UAF = self.Kr <= (self.k0 * self.NA * (self.RI[0] / self.RI[2])) # UAF pupil mask (in fourier space)

        '''parameters for interpolation'''
        self.N_crop = self.Nx - 4                           # size of images to interpolate
        self.N_crop_os = self.N_crop * self.os              # images size in the oversampled PSF
        self.shift = ((self.Nx-self.N_crop)//2)*self.os     # shift in x and y between the PSF and the image center
        


    #@partial(jax.jit, static_argnums=(0,))
    def fresnel_coefs(self):
        ''' 
        This function computes those Fresnel coefficients that are the same for all the emitters. 
            We consider here the system shared properties:
            - RI: refractive indices of the three media: [RI_sample, RI_intermediate, RI_coverslip];
            - d2: thickness of intermediate layer (medium 2) in meters; (if present)
            - T: field transmission; two pages: for s and p-pol: T=[T_s, T_p];
            We do not consider here the individual emitters properties:
            - dipole: [theta, phi] = polar and azimuthal angles of emitting dipole;
            - mu: magnitude of dipole;
            - z: z position of the emitter dipole;
        '''

        T_s, T_p = self.T[:, :, 0], self.T[:, :, 1]

        # Coordinates in the objective pupil: Eq. 17.
        self.PHI3 = jnp.arctan2(self.Kx, self.Ky)
        #self.THETA1 = jnp.arccos(jnp.sqrt(1-(self.RI(2)/self.RI(0)*self.Kr/k3)**2))   #angle in medium 1
        #self.THETA2 = jnp.arccos(jnp.sqrt(1-(self.RI(2)/self.RI(1)*self.Kr/k3)**2))   #angle in medium 2
        self.THETA3 =jnp.arcsin(self.pupil * self.Kr / self.k3)  #angle in medium 3

        'Snell s law: Eq. 4.'
        # Cos(theta) becomes imaginary when the angle is larger than the critical angle: this is the exponential decay of the evanescent wave.
        self.CTHETA1 = jnp.sqrt(0j + 1 - (self.Kr / self.k1)**2) #this is required to estimate the z_factor
        self.CTHETA2 = jnp.sqrt(0j + 1 - (self.Kr / self.k2)**2)
        self.CTHETA3 = jnp.cos(self.THETA3)

        'Fresnel-coefs: Eq. 3.'

        'transission and reflection coefficients from sample to intermediate layer (medium 1 to medium 2)'
        tp12 = 2 * self.RI[0] * self.CTHETA1 / (self.RI[0] * self.CTHETA2 + self.RI[1] * self.CTHETA1)
        ts12 = 2 * self.RI[0] * self.CTHETA1 / (self.RI[0] * self.CTHETA1 + self.RI[1] * self.CTHETA2)
        rp12 = (self.RI[1] * self.CTHETA1 - self.RI[0] * self.CTHETA2) / (self.RI[0] * self.CTHETA2 + self.RI[1] * self.CTHETA1)
        rs12 = (self.RI[0] * self.CTHETA1 - self.RI[1] * self.CTHETA2) / (self.RI[0] * self.CTHETA1 + self.RI[1] * self.CTHETA2)

        'transission and reflection coefficients from intermediate layer to coverslip (medium 2 to medium 3)'
        tp23 = 2 * self.RI[1] * self.CTHETA2 / (self.RI[1] * self.CTHETA3 + self.RI[2] * self.CTHETA2)
        ts23 = 2 * self.RI[1] * self.CTHETA2 / (self.RI[1] * self.CTHETA2 + self.RI[2] * self.CTHETA3)
        rp23 = (self.RI[2] * self.CTHETA2 - self.RI[1] * self.CTHETA3) / (self.RI[1] * self.CTHETA3 + self.RI[2] * self.CTHETA2)
        rs23 = (self.RI[1] * self.CTHETA2 - self.RI[2] * self.CTHETA3) / (self.RI[1] * self.CTHETA2 + self.RI[2] * self.CTHETA3)

        'From them, we calculate the transmission and reflection Fresnel coefs for the three-layer system: Eq. 12.'
        self.tp = (tp12 * tp23 * jnp.exp(1j * self.k2 * self.d2 * self.CTHETA2)) / (1 + rp12 * rp23 * jnp.exp(2j * self.k2 * self.d2 * self.CTHETA2))
        self.ts = (ts12 * ts23 * jnp.exp(1j * self.k2 * self.d2 * self.CTHETA2)) / (1 + rs12 * rs23 * jnp.exp(2j * self.k2 * self.d2 * self.CTHETA2))

        'Prefactor C (without z_factor): (the only constant where focal_len plays a role). Eq. 11.'
        C = (self.k3**2 * jnp.exp(1j * self.k3 * self.focal_len) * self.CTHETA3) / self.focal_len / self.RI[0] * jnp.exp(-1j * self.k3 * self.d2 * self.CTHETA3 * 0) # correction to put z=0 at the coverslip-intermediate layer interface
        self.C = C / self.k3**1  # NOTE: re-normalization here, to prevent large numbers

        'Apodization of the objective: Eq. 16.'
        # rotation of rays by their angle theta3, such that they are all parallel to the optical axis
        self.Apo_p = 1 / jnp.sqrt(self.CTHETA3) * self.pupil * T_p  # Apodization of the objective lens, T_p is measured field transmission
        self.Apo_s = 1 / jnp.sqrt(self.CTHETA3) * self.pupil * T_s  # Apodization of the objective lens, T_s is measured field transmission



    #@partial(jax.jit, static_argnums=(0,))
    def BFP_fields(self, dipole, mu=1):
        '''
        Calculates the BFP fields (Ex,Ey) of an arbitrary oriented dipole at a distance z from the coverslip.
        It includes also near-field effects (SAF emission). Based on the paper of Axelrod, J. of Microscopy 2012.
            Arguments:
            -  dipole:      [theta, phi] = polar and azimuthal angles of emitting dipole.
            -  mu:          magnitude of dipole: molecule intensity.
        '''

        theta_mu, phi_mu = dipole # orientation of dipole emitter
    
        'Dipole projections onto directions s, p and z: Eq. 13.'
        mu_p = mu * jnp.sin(theta_mu) * jnp.cos(phi_mu - self.PHI3)
        mu_s = mu * jnp.sin(theta_mu) * jnp.sin(phi_mu - self.PHI3)
        mu_z = mu * jnp.cos(theta_mu)

        'Field magnitudes in layer 3 (pre-objective zone), along the s, p and z-axes (see paper for axes definitions). Eq. 10.'
        E3p = self.C * self.tp * self.CTHETA3 * (mu_p / self.RI[2] + mu_z * jnp.sin(self.THETA3) /  self.RI[0] / self.CTHETA1)
        E3s = self.C * self.ts * (mu_s / self.RI[2] / self.CTHETA1)
        E3z = self.C * self.tp * jnp.sin(self.THETA3) * (mu_p / self.RI[2] + mu_z * jnp.sin(self.THETA3) /  self.RI[0] / self.CTHETA1)

        'Influence of the objective: rotation of rays by their angle theta3, such that they are all parallel to the optical axis'
        E_BFP_p = (E3p * self.CTHETA3 + E3z * jnp.sin(self.THETA3)) * self.Apo_p
        E_BFP_s = E3s * self.Apo_s  # s-polarization remains unchanged by this rotation

        'Coordinate transform into x-and y-polarization -> fields in the back focal plane of objective'
        E_BFP_x = ((jnp.cos(self.PHI3)) * E_BFP_p - (jnp.sin(self.PHI3)) * E_BFP_s)  
        E_BFP_y = ((jnp.sin(self.PHI3)) * E_BFP_p + (jnp.cos(self.PHI3)) * E_BFP_s)

        return E_BFP_x, E_BFP_y




    def czt2_coefs(self):
        """ This function computes all the terms needed for the 2d chirp z-transform that are common to all the emitters. """
        
        ux_des = self.ux / self.os # desired resolution in nm: effective pixel size in the image plane (camera sensor)
        self.Nx_os = self.Nx * self.os # x-y size of oversampled PSF

        self.N_pad = 2 ** int(jnp.ceil(jnp.log2(self.Nx_os + self.Nk - 1)))
        self.pad_left = int(jnp.ceil((self.N_pad - self.Nk)/2))
        self.pad_right = (self.N_pad - self.Nk) // 2
        self.start_idx = int(jnp.ceil((self.N_pad - self.Nx_os)/2))
        self.end_idx = self.start_idx + self.Nx_os
        x = jnp.arange(-self.N_pad // 2 , self.N_pad // 2)

        ux_fft = 2 * jnp.pi / (self.Nk * self.uk)  # FFT resolution without padding
        r = ux_des / ux_fft  # Required r-factor to meet the desired resolution at the given grid size N: this is the advantage of czt2 transform
        self.alpha = r / self.Nk # alpha = ux * uk / os / 2 / pi
        self.kernel = jnp.exp(-1j * self.alpha * jnp.pi * x[:, jnp.newaxis]**2) * jnp.exp(-1j * self.alpha * jnp.pi * x**2)
        self.fft_kernel = jnp.fft.fft2(jnp.fft.ifftshift(self.kernel))




    def czt2(self, E_in):
        """ 2D Chirp Z-Transform. Used to compute the Fourier transform of the BFP: propagates the fields from the BFP to the camera."""
        'The benefit of the chirp transform is that the size of the pupil function is not dependent on the pixel'
        'size of the camera and a pupil size of 128 x 128 pixels can represent a reasonable sampling in the pupil function.'
         
        E_in = jnp.conj(E_in) #note that czt2 seems to invert the phase compared to fft2! this is a quick fix for now.
        # NOTE: the final conjugation seems to be necessary (compared with Nicolas simulations), but so far, I don't know why!
        
        f1 = jnp.pad(E_in, ((0,0), (self.pad_left, self.pad_right), (self.pad_left, self.pad_right)), mode='constant')
        f1 *= jnp.conj(self.kernel[jnp.newaxis, :,:])
        f1 = jnp.fft.fft2(jnp.fft.ifftshift(f1, axes=(1,2)), axes = (1,2))
        f1 *= self.fft_kernel
        f1 = jnp.fft.fftshift(jnp.fft.ifft2(f1, axes=(1,2)), axes=(1,2)) * self.alpha * jnp.conj(self.kernel)
        E_out = f1[:, self.start_idx:self.end_idx, self.start_idx:self.end_idx] #crop the image to match Nx*os

        
        # f1 = jnp.pad(E_in, ((0,0), (self.pad_left, self.pad_right), (self.pad_left, self.pad_right)), mode='constant')
        # f1 *= jnp.conj(self.kernel[jnp.newaxis, :,:])
        # f1 = jnp.fft.fft2(jnp.fft.ifftshift(f1, axes=(1,2)), axes = (1,2))
        # f1 *= self.fft_kernel
        # f1 = jnp.fft.fftshift(jnp.fft.ifft2(f1, axes=(1,2)), axes=(1,2)) * self.alpha * jnp.conj(self.kernel)
        # E_out = f1[:, self.start_idx:self.end_idx, self.start_idx:self.end_idx] #crop the image to match Nx*os

        return E_out





    # INTERPOLATION FUNCTION f
    #   - f
    #   - f_single
    #   - f_biplane
    #   - generate_image
    
    #@partial(jax.jit, static_argnums=(0,))
    def f(self, v):  # f = inteporlation function
        if isinstance(self.z_defocus, list):
            return self.f_biplane(v)
        else:
            return self.f_single(v)

    
    @partial(jax.jit, static_argnums=(0,))
    def f_single(self,v):
        I_hires = jax.image.scale_and_translate(self.data, #data=psf, I = image high resolution
                                                shape = (self.N_crop_os, self.N_crop_os, 1), #shape of images to be interpolated
                                                spatial_dims = (0, 1, 2),
                                                scale = jnp.array([1, 1, 1]),
                                                translation = jnp.array([v[0]-self.shift, v[1]-self.shift, -v[2]]), # y, x, z
                                                method = self.interp_method)[...,0]
        if self.os > 1:
            I_hires = binning(I_hires, self.os) # downsample the interpolated image

        return v[4] + v[3] * I_hires # bg + signal * image
        

    @partial(jax.jit, static_argnums=(0,))
    def f_biplane(self,v):
        half_PSF = self.data.shape[1] // 2
        
        I_hires_left = jax.image.scale_and_translate(self.data[:, 0 : half_PSF, :],
                                        shape = (self.N_crop_os, self.N_crop_os, 1),
                                        spatial_dims = (0, 1, 2),
                                        scale = jnp.array([1, 1, 1]),
                                        translation = jnp.array([v[0]-self.shift, v[1]-self.shift, -v[2]]),
                                        method = self.interp_method)[...,0]
                                        
        I_hires_right = jax.image.scale_and_translate(self.data[:, half_PSF : , :], 
                                        shape = (self.N_crop_os, self.N_crop_os, 1),
                                        spatial_dims = (0, 1, 2),
                                        scale = jnp.array([1, 1, 1]),
                                        translation = jnp.array([v[0]-self.shift, v[1]-self.shift, -v[2]]),
                                        method = self.interp_method)[...,0]
        if self.os > 1:
            I_hires_left = binning(I_hires_left, self.os)
            I_hires_right = binning(I_hires_right, self.os)

        return v[4] + jnp.concatenate((v[3]/2.*I_hires_left, v[3]/2.*I_hires_right), axis=1)  


    def generate_image_fft(self, a, v, M=None, z_def=None, PD=0.):

        sig_scale = 1. if M is None else ((M - v[4]).sum())
        z_def = self.z_defocus if z_def is None else z_def # if z_def is not given, use the one in the class
        
        aberr = jnp.fft.ifftshift(jnp.sum(self.modeset * a[:, jnp.newaxis, jnp.newaxis], axis = 0) + PD)
        defocus,_ = self.calc_defocus( - z_def, self.RI[-1]) # defocus in immersive medium
        exp_jph = jnp.exp(1j*jnp.real(defocus + aberr)) * self.apodization # phase

        z = ((v[2] - self.d2) * self.uz + self.z_vec[0])  # z position of emitter relative to coverslip
        fxy = self.ux /  (2 * jnp.pi / (self.Nk * self.uk))
        z_factor = jnp.exp(1j * (self.k1 * self.CTHETA1.real * z + 1j * self.k1 * self.CTHETA1.imag * jnp.clip(z, 0, None) +
                                             2 * np.pi * self.Kx/(self.uk * self.Nk) * fxy * v[1] +
                                             2 * np.pi * self.Ky/(self.uk * self.Nk) * fxy * v[0])) 

        E_BFP = self.E_BFP[:,0] / self.k1 * z_factor #make the fields 3 dimensional
        E_fields = self.czt2(jnp.fft.fftshift(E_BFP * exp_jph, axes = (1, 2)))[:, self.shift:-self.shift, self.shift:-self.shift] # fields in the camera.            
        I = jnp.sum(E_fields.real**2 + E_fields.imag**2, axis=0)
        I_BFP = jnp.sum((E_BFP.real**2 + E_BFP.imag**2), axis=0, where=self.apodization[None])
        I /= jnp.sum(I_BFP) # total intensity in the camera, normalized to intensity in the BFP

        return self.blurr_image(I) * sig_scale * v[3] + v[4] if self.sigma_blurring != 0 else I * sig_scale * v[3] + v[4]



    def generate_psf(self, emitter_z_pos, exp_jph):
        """ 
        This function calculates the intensity image at the sensor of a pointlike emitter at a given z position.
        z = 0 is at the coverslip-sample interface, with z positive moving in the sample.
        """
        '''z dependent phase factor'''
        # Field propagation in the sample. (only factor where z position is considered)
        # z_factor = exp(i*z*k1*cos(theta1)) = exp(i*z*k1*sqrt(1-(kr/k1)^2) ). k1 is the sample's wavevector.
        z = (emitter_z_pos[jnp.newaxis, : , jnp.newaxis, jnp.newaxis] - self.d2)
        z_factor = jnp.exp(1j * self.k1 * (self.CTHETA1.real * z + 1j * self.CTHETA1.imag * jnp.clip(z, 0, None)))
        #z_factor = jnp.exp(1j * self.k1 * self.CTHETA1 * ) # correction to change z=0 to the coverslip-intermediate layer interface
        E_BFP_z = self.E_BFP * z_factor / self.k1 #make the fields 3 dimensional
        #print(jnp.linalg.norm(E_BFP_z))
        '''PSF'''
        # I = sum(|E|**2) over all fields, for every z slice. I == PSF.
        # the czt2 function takes a BFP field (dim = Nk) and returns an ovesampled PSF slice (dim = Nx*os)
        E_fields = jnp.array([self.czt2(jnp.fft.fftshift(field * exp_jph, axes = (1,2))) for field in E_BFP_z]) # fields in the camera.                                
        I = jnp.sum(E_fields.real**2 + E_fields.imag**2, axis=0).transpose(1,2,0) # total intensity in the camera, not normalized.
        I_BFP = jnp.sum(E_BFP_z.real**2 + E_BFP_z.imag**2, axis=0, where = self.apodization) #total intensity in the BFP, used below in the normalization.
        I /= jnp.sum(I_BFP, axis = (1,2)) # total intensity in the camera, normalized to intensity in the BFP
        
        '''Electric_fields'''
        E_fields = E_fields.transpose(0, 2, 3, 1) / jnp.sqrt(jnp.sum(I_BFP, axis=(1,2))[None])
    
        '''Check intensity normalization'''
        # Uncheck this lines to plot the total normalized intensity in every z slide (normalization: BFP = 100% of photons -> sensor = 70%-95%, depending on cropping and defocus).
        # print(' - z slice (nm) - \t - I (cam / BFP) -')
        # [print(' ', f'{np.round(value  * 1e9):.0f}', '\t\t', ' ',  jnp.sum(I, axis = (0,1))[i]) for i, value in enumerate(z)]
        
        # '''BLURRING'''
        if self.sigma_blurring != 0: 

            vmap_blurring = jax.vmap(lambda z_slice: self.blurr_image(z_slice), in_axes=2, out_axes=2)
            I = vmap_blurring(I)
        
        return I, E_fields

    @partial(jax.jit, static_argnums=(0,))
    def generate_image(self, a, v=[0, 0, 0, 1, 0], M=None, z_def=None, PD=0.):
        """
        Generates a molecule image given a set of zernikes a=[a1,a2,...], a parameter vector v=[x,y,z,s,bg], an image M and the defocus.
        The Image M is used to calibrate the photon numbers. If it is not given, the photon number may be not accurate
        If defocus is not given, the one in the class is used
        """


        sig_scale = 1 if M is None else (M - v[4]).sum()
        z_def = self.z_defocus if z_def is None else z_def # if z_def is not given, use the one in the class

        aberr = jnp.fft.ifftshift(jnp.sum(self.modeset * a[:, jnp.newaxis, jnp.newaxis], axis = 0) + PD)
        defocus,_ = self.calc_defocus( - z_def, self.RI[-1]) # defocus in immersive medium
        exp_jph = jnp.exp(1j*jnp.real(defocus + aberr)) * self.apodization; # phase
        PSF,_ = self.generate_psf(self.z_vec, exp_jph) # calculate 3D psf
        image = jax.image.scale_and_translate(PSF, 
                                              shape = (self.N_crop_os, self.N_crop_os, 1), #shape of image to be interpolated
                                              spatial_dims = (0, 1, 2), 
                                              scale = jnp.array([1, 1, 1]), 
                                              translation = jnp.array([v[0]-self.shift, v[1]-self.shift, -v[2]]), # y, x, z
                                              method = self.interp_method)[...,0]
        if self.os > 1:
            image = binning(image, self.os)

        return v[4] + sig_scale * v[3] * image

    

    
    

    def gaussian_kernel(self):
        """Generate the gaussian kernel to blur the PSF. blurring given by sigma."""
        size = int(2 * jnp.ceil(3 * self.sigma_blurring) + 1)
        x = jnp.arange(-size // 2 + 1, size // 2 + 1)
        kernel = jnp.exp(-0.5 * (x / self.sigma_blurring) ** 2)
        kernel = kernel / jnp.sum(kernel)
        self.gauss_kernel = jnp.outer(kernel, kernel)
        


    def blurr_image(self, image):
        """Blurr the image with a Gaussian filter. Intensity does not changes."""
        padded_image = jnp.pad(image, ((2*self.shift, 2*self.shift), (2*self.shift, 2*self.shift)), mode='edge') # pad to remove border artifcats when blurring
        blurred = jsp.signal.convolve(padded_image, self.gauss_kernel, mode='same', method='fft')[2*self.shift:-2*self.shift, 2*self.shift:-2*self.shift]
        normalization_factor = jnp.sum(image) / jnp.sum(blurred) # normalize to initial image intensity
        return blurred * normalization_factor





    # PUPIL FUNCTIONS
    #   - zernikes
    #   - eigenbasis
    #   - aberrations
    #   - calc_defocus
    #   - calc_SA
    #   - RI_mismatch_correction
    #   - apodization
    #   - calc_CRLB



    def zernikes(self):
        ''' Creates the basis set of Zernike polynomials for the given Noll indices. '''
        n_modes = len(self.Z_modes)
        modeset = np.ones((n_modes, self.Nk, self.Nk)) # initialize modeset basis stack (np array here)
        for (n,zernike) in enumerate(self.Z_modes):
           modeset[n,:,:] = zernike_noll(zernike, self.Nk) 
        self.modeset = jnp.asarray(modeset)
        #self.modeset_fft = jnp.fft.ifftshift(self.modeset)


    def custom_modeset(self, modeset):
        ''' Creates the basis set of orthogonalized eigenbasis. '''
        modeset = prepare_modeset(modeset, self.Nk)
        self.modeset = jnp.asarray(modeset)


    def aberrations(self):
        ''' Create pupil aberrations phase given the Noll basis modeset. '''
        aberr = jnp.sum(self.modeset * self.Z_magn[:, jnp.newaxis, jnp.newaxis], axis = 0)
        return jnp.fft.ifftshift(aberr)



    def calc_defocus(self, z_defocus, RI):
        '''
        Generate spherical pupil defocus for a given nominal focal plane z_defocus and refractive index RI.
        z_defocus is the distance between the coverslip and the nominal objective focal plane (the one that you set with the Z piezo).
        IMPORTANT: the PSF distrotiona due to a sample with a different RI are implicitly included in the BFP E_fields though the fresnel coefficients.
            - RI = immersion medium. Used to generate the PSF.
            - RI = sample. Added to correct the PSF focus if desired, through the functions calc_SA and RI_mismatch_correction.
        '''
        defocus =  + z_defocus * jnp.sqrt(0j + (self.k0 * RI)**2 - self.Kr**2) * self.pupil # plus sign here in front of z_defocus
        relative_defocus = defocus - jnp.mean(defocus[self.pupil]) #subtracting offset, for offset free integration
        return defocus, relative_defocus



    def calc_SA(self, z_defocus): 
        '''
        Calculates the spherical aberrations induced from the refractive for an objective with a given nominal defocus z_defocus.
        SA: corrects the PSF distortion and focus position: effective focus and nominal focus will coincide.
        SA_pure: restores only the PSF distortion: effective focus and nominal focus will NOT coincide.
        '''
        _, self.def_sample = self.calc_defocus(z_defocus, self.RI[0])               # defocus in sample (RI = buffer)
        _, def_imm_medium = self.calc_defocus(z_defocus, self.RI[2])                # defocus in immersion medium (RI = oil)
        self.SA = (self.def_sample - def_imm_medium)   #* self.pupil        # this is the full aberration caused by the refractive index mismatch
        self.SA_pure = (self.SA - (jnp.sum(self.SA[self.pupil] * self.def_sample[self.pupil]) /
                                   jnp.sum(self.def_sample[self.pupil]**2)) * self.def_sample) * self.pupil #pure spherical aberration -> the defocus term is removed
    




    def RI_mismatch_correction(self, correct_RI_mismatch):
        '''Sets the desired spherical aberration correction'''
        correction_options = {'yes': self.SA, 'pure': self.SA_pure, 'no': 0}
        if correct_RI_mismatch not in correction_options:
            print('--- WARNING --- ')
            print('The RI fluid/coverslip mismatch generates spherical aberrations that move the objective focal position and distort the PSF shape')
            print('Do you want to correct the PSF? type one:')
            print(' - yes:   introduces counter spherical aberrations that restore the PSF shape and the focus z position')
            print(' - pure:  introduces counter spherical aberrations that restore only the PSF shape')
            print(' - no:    simulates the real PSF with spherical aberrations (default option)')
            print('---------------')
            print()
            self.tot_defocus = self.obj_defocus
        else:
            self.tot_defocus = self.obj_defocus + correction_options[correct_RI_mismatch]




    def apodization(self) :
        """ Consider amplitude apodization transmitted by the objective and from the pupil to the camera"""

        _, _, R, pupil = create_coord(self.Nk, 2 / self.Nk, 'exact')

        # Polarization dependent Field transmission from Sample to BFP. 
        Ts = jnp.polyval(self.Ts_coefs, R) * pupil
        Ts = jnp.array(Ts) / jnp.max(Ts)
        Tp = jnp.polyval(self.Tp_coefs, R) * pupil
        Tp = jnp.array(Tp) / np.max(Tp)
        self.T = jnp.fft.ifftshift(jnp.dstack((Ts, Tp)))

        # Effective power transmission from Pupil to Camera.
        T_fit = jnp.polyval(self.T_fit_coefs, R) * pupil
        T_fit = jnp.array(T_fit) / np.max(T_fit)
        #T_fit[T_fit<0] = 0
        self.apodization = jnp.fft.ifftshift(T_fit)




    def calc_CRLB(self, cam, s = 2000, bg = 100, z_slice = 'all'):
        ''' Computes the CRLB of the whole PSF if z = 'all'. Otherwise only at the given z position.'''

        if isinstance(self.z_defocus, list): #biplane imaging
            #split the PSF model into the two channels, calculate Fisher Info for each channel and sum them
            h1 = copy.deepcopy(self)
            h1.data = self.data[:,0:self.data.shape[1] // 2,:]  # take left psf
            h2 = copy.deepcopy(self)
            h2.data = self.data[:,self.data.shape[1] // 2:,:]  # take right psf
            
            FI = Fisher(h1, cam, s/2, bg/2) + Fisher(h2, cam, s/2, bg/2) #calculating Fisher information matrix

        else: #single channel imaging

            FI = Fisher(self, cam, s, bg, z_slice) #calculating Fisher information matrix

        F_inv = jnp.linalg.inv(FI.transpose(2,0,1))  # calculate the inverse of the Fisher information matrix
        CRLB = jnp.array([jnp.sqrt(F_inv[:, i, i]) for i in range(5)]) # calculate Cramer-Rao lower bound
        
        return CRLB






    # PLOT FUNCTIONS
    #   - show_slice_xy
    #   - show_projection
    #   - show_pupil
    #   - show_CRLB


    def show_slice_xy(self):
        """PLot PSF Z slices with imshow"""

        biplane = isinstance(self.z_defocus, list)
        plt.figure(figsize=(6,8)) if biplane else plt.figure(figsize=(10,3)); 
        fs = 12 

        y_step = (self.Nx-1) / 4
        x_step = (2*self.Nx-1) / 4 if biplane else y_step
        y_ticks = ((2 * np.arange(5) * y_step + 1) * self.os -1) / 2
        x_ticks = ((2 * np.arange(5) * x_step + 1) * self.os -1) / 2
        y_ticklabels = (np.arange(5) * y_step + 1).astype(int) # tick labels
        x_ticklabels = 2 * y_ticklabels  if biplane else y_ticklabels # tick labels   

        z_defocus_str = f'{self.z_defocus[0]*1e9:.0f} and {self.z_defocus[1]*1e9:.0f}' if biplane else f'{self.z_defocus*1e9:.0f}'
        plt.suptitle(f'Objective focused at {z_defocus_str} nm', fontsize=fs)

        for i, z_pos in enumerate([0, int(self.data.shape[2] // 2 + 1), int(self.data.shape[2] - 1)]):
            plt.subplot(3, 1, i + 1) if biplane else plt.subplot(1, 3, i + 1)
            z_label = (self.z_min + (self.z_max - self.z_min) * i / 2) * 1e9
            plt.title(f"{z_label:.0f} nm", fontsize=fs)
            plt.imshow(self.data[..., z_pos] * self.os**2, cmap=parula)
            plt.xticks(x_ticks, x_ticklabels)
            plt.yticks(y_ticks, y_ticklabels)
            plt.xlabel('x / pixels') if not biplane or i == 2 else None
            plt.ylabel('y / pixels') if biplane or i == 0 else None
            plt.colorbar()
        plt.tight_layout()
        plt.show()



    def show_projection(self):
        biplane = isinstance(self.z_defocus, list)
        plt.figure(figsize=(8,6)); fs = 12
        x_limit, y_limit = 400., 140.

        projection_y = np.vstack([np.sum(self.data[:, i * self.data.shape[1]//2:(i+1) * self.data.shape[1]//2]*self.os, axis=1) for i in range(2)]) if biplane else np.sum(self.data*self.os, axis=1)
        projection_x = np.sum(self.data * self.os, axis=0)
        projections = [projection_x, projection_y]
        titles = ['X projection', 'Y projection']
        norm = Normalize(vmin=np.min(projections), vmax=np.max(projections))

        step = (self.Nx-1) / 4
        y_step = y_limit / self.Nx
        y_ticks = y_step * (np.arange(5) * step + 0.5)  # y tick positions
        y_ticklabels = (np.arange(5) * step + 1).astype(int)  # tick labels
        y_ticklabels = 2 * y_ticklabels if biplane else y_ticklabels  # tick labels

        x_ticks = np.arange(5) * x_limit / 4
        x_ticklabels = [int(((self.z_max - self.z_min) * i / 4 + self.z_min) * 1e9) for i in range(5)]

        # plot projections
        for i, (title, projection) in enumerate(zip(titles, projections), 1):
            plt.subplot(2, 1, i)
            plt.imshow(projection, norm=norm, extent=[0, x_limit, y_limit, 0], cmap=parula)
            plt.title(title, fontsize=fs)
            plt.ylabel('x / pixels') if i == 1 else plt.ylabel('y / pixels')
            plt.xticks(x_ticks, x_ticklabels)
            plt.yticks(y_ticks, y_ticklabels)
            # plt.colorbar()

        z_defocus_str = f'{self.z_defocus[0]*1e9:.0f} and {self.z_defocus[1]*1e9:.0f}' if biplane else f'{self.z_defocus*1e9:.0f}'
        plt.suptitle(f'Objective focused at {z_defocus_str} nm', fontsize=fs)
        plt.xlabel('z / nm')
        plt.tight_layout()
        plt.show()




    def show_pupil(self, pupil='SAF'):

        pupil = self.pupil_UAF if pupil == 'UAF' else self.pupil # for pupil far from the coverslip: UAF, else: SAF
        plt.figure(figsize=(10, 3)); fs = 12

        biplane = isinstance(self.z_defocus, list)
        titles = ["Aberrations L channel", "Aberrations R channel", "Apodization"] if biplane else ["Aberrations", "Defocus", "Apodization"]
        cmaps = ['hsv', 'hsv', 'gray'] if biplane else ['hsv', 'plasma', 'gray']

        first_plot = np.real(jnp.fft.fftshift(self.pupil_aberrations[0] * pupil)) if biplane else np.real(jnp.fft.fftshift(self.pupil_aberrations * pupil))
        second_plot = np.real(jnp.fft.fftshift(self.pupil_aberrations[1] * pupil)) if biplane else np.real(jnp.fft.fftshift(jnp.real(-self.tot_defocus) * pupil))
        third_plot = jnp.fft.fftshift(self.apodization * pupil)
        pupils = [first_plot, second_plot, third_plot]

        for i, (title, cmap, pupil) in enumerate(zip(titles, cmaps, pupils), 1):
            plt.subplot(1, 3, i)
            plt.title(title, fontsize=fs)
            plt.imshow(pupil, cmap=cmap)
            plt.xticks([]); plt.yticks([])
            plt.colorbar()
        plt.suptitle(f'Pupil  ', fontsize=fs+3)
        plt.tight_layout()
        plt.show()




    def show_SA(self):
        plt.figure(figsize=(10, 3)); fs = 12

        titles = ["Real defocus", "RIM corrected", "RIM corrected (pure)"]
        first_plot = np.real(jnp.fft.fftshift(jnp.real(-self.obj_defocus) * self.pupil))
        second_plot = np.real(jnp.fft.fftshift(jnp.real(-(self.obj_defocus + self.SA) * self.pupil_UAF)))
        third_plot = np.real(jnp.fft.fftshift(jnp.real(-(self.obj_defocus + self.SA_pure)) * self.pupil_UAF))

        pupils = [first_plot, second_plot, third_plot]
        vmin, vmax = np.min([np.min(pupil) for pupil in pupils]), np.max([np.max(pupil) for pupil in pupils])

        for i, (title, pupil) in enumerate(zip(titles, pupils), 1):
            plt.subplot(1, 3, i)
            plt.title(title, fontsize=fs)
            # plt.imshow(pupil, cmap='plasma')
            plt.imshow(pupil, cmap='plasma', vmin=vmin, vmax=vmax)
            plt.colorbar()
        plt.suptitle(f'Defocus', fontsize=fs+3)
        plt.tight_layout()
        plt.show()




    def show_CRLB(self, cam, s=2000, bg=100, ylim = False, export=False):
        x_axis = self.z_vec * 1e9  # z range
        CRLB = self.calc_CRLB(cam, s, bg) * 1e9

        if not isinstance(s, (int, float)): # get  signal and background just for plotting. CRLB have been computed with effective s and bg.
            s /= cam.QE if cam.EM_gain == 1  else  cam.QE / 2
            bg /= cam.QE if cam.EM_gain == 1 else cam.QE / 2

        fig, ax1 = plt.subplots()

        # plot CRLB
        ax1.plot(x_axis, CRLB[0], linestyle='-', label=r'σ$_{x}$')
        ax1.plot(x_axis, CRLB[1], linestyle='-', label=r'σ$_{y}$')
        ax1.plot(x_axis, CRLB[2], label=r'σ$_{z}$')
        ax1.set_xlim(min(x_axis), max(x_axis))
        ax1.set_ylim(0, ylim) if ylim else None
        ax1.set_xlabel('z / nm')
        ax1.set_ylabel('σ / nm')
        ax1.grid()

        defocus_str = f'Objectives focal planes: {self.z_defocus[0]*1e9:.0f} and {self.z_defocus[1]*1e9:.0f} nm' if isinstance(self.z_defocus, list) else f'Objective focal plane: {self.z_defocus*1e9:.0f} nm'
        fig.suptitle(f'Theoretical CRLB')
        fig.text(0.28, 0.85, defocus_str, fontsize=12, color='black', rotation=0)
        fig.text(0.28, 0.8, f'Mean Signal: {np.mean(s):.0f} / molecule', fontsize=12)
        fig.text(0.28, 0.75, f'Mean Background: {np.mean(bg):.0f} / pixel', fontsize=12, color='black', rotation=0)

        # Check if s is a scalar
        if not isinstance(s, (int, float)):
            # Adding the second y-axis if s is not a scalar
            ax2 = ax1.twinx()
            ax2.plot(x_axis, s, color='red', label='s', linestyle='--')
            ax2.plot(x_axis, bg, color='black', label='bg', linestyle='--')
            ax2.set_ylabel('signal / ph', color='black')
            ax2.tick_params(axis='y', labelcolor='black')

            # Legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        else:
            # Legend for only the primary y-axis
            ax1.legend(loc='best')

        plt.tight_layout()
        plt.show()

        if export:
            return CRLB[0], CRLB[1], CRLB[2]







# ---------------------- PSF CHILDREN CLASSES ----------------------


# PSF class for fft interpolation

class psf_fft(psf):
    '''Children psf class for FFT interpolation'''

    def __init__ (self, psf_instance):

        self.__dict__.update(psf_instance.__dict__)     # Initialize attributes from the psf instance
        self.shift_fft = (self.Nx - self.N_crop) // 2   # shift in x and y between the PSF and the image center in Fourier space (before os)

    def generate_image_fft(self, param):

        x, y, z, s, bg = param
        exp_xy = jnp.exp(1j * (self.Kx * y * self.ux + self.Ky * x * self.ux))
        exp_z = jnp.exp(1j * self.k1 * self.CTHETA1 * z * self.uz)
        Ez = self.E_BFP * exp_z # pupil * exp_z
        I_tot = jnp.sum(jnp.abs(Ez * self.apodization)**2, axis=0)
        E_fields = jnp.array([self.czt2(jnp.fft.fftshift(field * self.exp_jphase * exp_xy, axes = (-1,-2))) for field in Ez])#.sum(axis=0)
        I_field = jnp.sum(jnp.abs(E_fields)**2, axis=0)
        _nrm = jnp.where(jnp.sum(I_tot) == 0, 0, 1./jnp.sum(I_tot)) # normalization factors
        itp = (I_field * _nrm )[0] # normalized image
        itp = self.blurr_image(itp) if self.sigma_blurring != 0 else itp # blurr image

        I_binned = binning(itp, self.os)
        Image = bg + s * I_binned
        
        return Image
    
    def f(self, v):

        return self.generate_image_fft(v)[self.shift_fft:-self.shift_fft, self.shift_fft:-self.shift_fft]







# PSF class for deconvolution

class psf_deconvolution(psf):
    '''Children psf class for gradient descent deconvolution'''
        
    def __init__(self,
                 psf_instance,
                 method = 'multiplane',
                 z_defocus = 0,
                 planes = 0,
                 photon_statistics = 'poissonian',
                 IF_width = 1e4):
        
        self.__dict__.update(psf_instance.__dict__)         # Initialize attributes from the psf instance
        self.method = method                                # method for the deconvolution: 'multiplane
        self.photon_statistics = photon_statistics          # photons statistics for the error function: 'poissonian' or 'gaussian'
        self.z_defocus = z_defocus * 1e-9                   # defocus in m
        self.planes = jnp.array((z_defocus + np.array(planes)) * 1e-9) # distance between planes for multiplane deconvolution. in m
        self.n_planes = self.planes.shape[0]                # number of planes
        self.IF_width = IF_width * 1e-9                     # width of the DPP influence function. in m

        '''Initials'''
        self.update = get_update(self.fwd_fun())            # update function. to call inside the gradient descent while loop


        '''DECONVOLUTION METHODS'''
        '''multiplane'''            # fits aberr and positions. zernike-wise.
        '''multiplane_bead'''       # fits aberr and positions. zernike-wise with constraint positions between planes.
        '''multiplane_bead_pos'''   # fits aberr. zernike-wise with constraint positions = [0,0,0] between planes.
        '''multiplane_pxl'''        # fits only aberr. pixel-wise.
        '''multiplane_pxl_bead'''   # fits aberr and positions. pixel-wise with constraint positions between planes.



    # FUNCTIONS FOR DECONVOLUTION
    #  - compute_err
    #  - grad_fun
    #  - fwd_multiplane
    #  - fwd_multiplane_bead
    #  - fwd_multiplane_pxl
    #  - generate_image_bead
    #  - generate_image_dpp
    #  - actuator_IF
    #  - DPP



    ### Optimization Processes---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def compute_err(self, model, data):
        '''Returns the error function used in the deconvolution'''

        if self.photon_statistics == 'poissonian':  # for low photon numbers (SMLM data)
            return compute_LLR(model, data)
        elif self.photon_statistics == 'gaussian':    # for high photon numbers (Beads data)
            return compute_least_squares(model, data)
        else:
            raise ValueError(f"Unsupported photon statistics: {self.photon_statistics}. Choose 'poissonian' for LLR or 'gaussian' for Least Squares.")


    def fwd_fun(self):
        '''Choose the error function used in the deconvolution'''

        # zernike-wise retrieval
        if self.method == 'multiplane': # fit aberrations with zernikes
            return self.fwd_multiplane_smlm
        if self.method == 'multiplane_bead': # fit positions pixel wise
            return self.fwd_multiplane_bead
        if self.method == 'multiplane_bead_pos': # fit positions with zernikes
            return self.fwd_multiplane_bead_pos
        
        # lorentz-wise retrieval
        if self.method == 'multiplane_lorentz': # use lorentzian to fit the actuator
            return self.fwd_multiplane_lorentz
        if self.method == 'multiplane_multivoltage': # use lorentzian but vmap over the voltages
            return self.fwd_multiplane_multivoltage
        
        # pixel-wise retrieval
        elif self.method == 'multiplane_pxl': # change BFP pixel values to fit the aberrations
            return self.fwd_multiplane_pxl
        elif self.method == 'multiplane_pxl_bead': # here the positions among planes are fixed
            return self.fwd_multiplane_pxl_bead
        else:
            raise ValueError(f"Unsupported method: {self.method}. Choose 'multiplane', 'multiplane_bead', 'multiplane_bead_pos', 'multiplane_pxl', 'multiplane_pxl_bead'.")
    




    '''--------------------------------'''
    ''' Zernike-wise retrieval methods '''
    '''--------------------------------'''


    # @partial(jax.jit, static_argnums=(0,))
    # def fwd_multiplane_smlm(self, par_l, const_l):
    #     '''
    #     Computes the cumulative error for all the images in all the planes, using the same aberrations a.
    #     This method is used for SMLM data, where molecules are uncorrelated (position in different planes are not constrained).
    #     - a: set of common zernike aberrations.
    #     - v_v_l: list of arrays with the bead positions and signal and bg in each plane. independent among different planes.
    #     '''
    #     a, v_v_l = par_l      # set of updated variables
    #     M_v_l = const_l       # set of constants variables (not updated)
    #     error = 0             # initialize error
    #     for i in range(self.n_planes): # iterate over the planes
    #         error += self.compute_err(jax.vmap(lambda v, M: self.generate_image(a, v, M, self.planes[i]))(v_v_l[i], M_v_l[i]), M_v_l[i]).sum()
    #     return error


    @partial(jax.jit, static_argnums=(0,))
    def fwd_multiplane_smlm(self, par_l, const_l):
        '''
        Computes the cumulative error for all the images in all the planes, using the same aberrations a.
        This method is used for SMLM data, where molecules are uncorrelated (position in different planes are not constrained).
        - a: set of common zernike aberrations.
        - v_v_l: list of arrays with the bead positions and signal and bg in each plane. individual in each plane.
        '''

        a, v_v_l = par_l      # set of updated variables
        M_v_l = const_l       # set of constants variables (not updated)

        # vmap over molecules
        def fwd_v_molecule(a, v_m, M_m, pl): # len(v_m) = len(M_m) == n molecules in that plane
            return self.compute_err(jax.vmap(lambda v_m, M: self.generate_image(a, v_m, M, pl))(v_m, M_m), M_m)

        # vmap over planes
        def fwd_v_plane(a, v_p, M_p, pl_p): # len(v_p) = len(M_p) = len(pl_p) == n planes
            return jax.vmap(lambda v_m, M_m, pl: fwd_v_molecule(a, v_m, M_m, pl))(v_p, M_p, pl_p)
        
        return fwd_v_plane(a, v_v_l, M_v_l, self.planes).sum()



    @partial(jax.jit, static_argnums=(0,))
    def fwd_multiplane_bead(self, par_l, const_l):
        '''
        Computes the cumulative error for all the images in all the planes, using the same aberrations a.
        This method is used for bead data, where molecules are correlated (position in different planes are constrained).
         - a: set of common zernike aberrations.
         - pos_v: bead position array. different among different beads but constraint among the same bead at different planes.
         - ph_v_l: signal and bg list of arrays. independent among different beads and planes.
        '''
        a, pos_v, ph_v_l = par_l    # set of updated variables
        M_v_l = const_l             # set of constants variables (not updated)
        error = 0                   # initialize error
        for i in range(self.n_planes): # iterate over the planes
            error += self.compute_err(jax.vmap(lambda p, ph, M: self.generate_image_bead(a, p, ph, M, self.planes[i]))(pos_v, ph_v_l[i], M_v_l[i]), M_v_l[i]).sum()
        return error
    

    @partial(jax.jit, static_argnums=(0,))
    def fwd_multiplane_bead_pos(self, par_l, const_l):
        '''
        Computes the cumulative error for all the images in all the planes, using the same pupil aberrations and dpp phase
         - pos is the bead position array. different among different beads but constraint among the same bead at different planes.
         - ph_v_l is the signal and bg list of arrays. independent among different beads and planes.
        '''
        a, ph_v_l = par_l           # set of updated variables
        M_v_l = const_l             # set of constants variables (not updated)
        error = 0                   # initialize error
        for i in range(self.n_planes): # iterate over the planes
            error += self.compute_err(jax.vmap(lambda ph, M: self.generate_image_bead_pos(a, ph, M, self.planes[i]))(ph_v_l[i], M_v_l[i]), M_v_l[i]).sum()
        return error






    @partial(jax.jit, static_argnums=(0,))
    def generate_image_bead(self, a, pos=[0, 0, 0], ph=[1, 0], M=None, z_def=None):
        """
        Generates a molecule image given a set of zernikes a=[a1,a2,...], and two parameter vectors: pos=[x,y,z] and  ph=[s,bg], an image M and the defocus
        If Image M is not given, the photon number may be not accurate
        If defocus is not given, the one in the class is used
        """

        sig_scale = 1 if M is None else (M - M.min()).sum()
        z_def = self.z_defocus if z_def is None else z_def # if z_def is not given, use the one in the class

        aberr = jnp.fft.ifftshift(jnp.sum(self.modeset * a[:, jnp.newaxis, jnp.newaxis], axis = 0))
        defocus,_ = self.calc_defocus( - z_def, self.RI[-1]) # defocus in immersive medium
        exp_jph = jnp.exp(1j*jnp.real(defocus + aberr)) * self.apodization; # phase
        PSF,_ = self.generate_psf(self.z_vec, exp_jph) # calculate 3D psf
        image = jax.image.scale_and_translate(PSF, 
                                            shape = (self.N_crop_os, self.N_crop_os, 1), #shape of image to be interpolated
                                            spatial_dims = (0, 1, 2), 
                                            scale = jnp.array([1, 1, 1]), 
                                            translation = jnp.array([pos[0]-self.shift, pos[1]-self.shift, -pos[2]]), # y, x, z
                                            method = self.interp_method)[...,0]
        if self.os > 1:
            image = binning(image, self.os)

        return ph[1] + sig_scale * ph[0] * image
    

    @partial(jax.jit, static_argnums=(0,))
    def generate_image_bead_pos(self, a, ph=[1, 0], M=None, z_def=None):
        """
        Generates a molecule image given a set of zernikes a=[a1,a2,...], and one parameter vectors ph=[s,bg], an image M and the defocus
        Position is set to [0,0,0]. Position is fitted using zernikes [2,3,4].
        If Image M is not given, the photon number may be not accurate
        If defocus is not given, the one in the class is used
        """

        sig_scale = 1 if M is None else (M - M.min()).sum()
        z_def = self.z_defocus if z_def is None else z_def # if z_def is not given, use the one in the class

        pos = [0,0,0]
        aberr = jnp.fft.ifftshift(jnp.sum(self.modeset * a[:, jnp.newaxis, jnp.newaxis], axis = 0))
        defocus,_ = self.calc_defocus( - z_def, self.RI[-1]) # defocus in immersive medium
        exp_jph = jnp.exp(1j*jnp.real(defocus + aberr)) * self.apodization; # phase
        PSF,_ = self.generate_psf(self.z_vec, exp_jph) # generate 3D psf
        image = jax.image.scale_and_translate(PSF, 
                                            shape = (self.N_crop_os, self.N_crop_os, 1), #shape of image to be interpolated
                                            spatial_dims = (0, 1, 2), 
                                            scale = jnp.array([1, 1, 1]), 
                                            translation = jnp.array([pos[0]-self.shift, pos[1]-self.shift, -pos[2]]), # y, x, z
                                            method = self.interp_method)[...,0]
        if self.os > 1:
            image = binning(image, self.os)

        return ph[1] + sig_scale * ph[0] * image


    '''--------------------------------'''









    '''--------------------------------'''
    ''' Lorentz-wise retrieval methods '''
    '''--------------------------------'''

    @partial(jax.jit, static_argnums=(0,))
    def fwd_multiplane_lorentz(self, par_l, const_l):
        '''
        Computes the cumulative error for all the images in all the planes, using the same pupil aberrations and dpp phase
         - pos is the bead position array. different among different beads but constraint among the same bead at different planes.
         - ph_v_l is the signal and bg list of arrays. independent among different beads and planes.
        '''
        g, ph_v_l = par_l           # set of updated variables
        a, M_v_l = const_l             # set of constants variables (not updated)
        
        g_amp = g[0]
        g_rest = g[1:]
        error = 0                   # initialize error
        for i in range(self.n_planes): # iterate over the planes
            error += self.compute_err(jax.vmap(lambda ph, M: self.generate_image_lorentz(a, g_amp, g_rest, ph, M, self.planes[i]))(ph_v_l[i], M_v_l[i]), M_v_l[i]).sum()
        return error

    @partial(jax.jit, static_argnums=(0,))
    def fwd_multiplane_multivoltage(self, par_l, const_l):
        '''
        Computes the cumulative error for all the images in all the planes, using the same pupil aberrations and dpp phase
         - pos is the bead position array. different among different beads but constraint among the same bead at different planes.
         - ph_v_l is the signal and bg list of arrays. independent among different beads and planes.
        '''
        g, ph_v_l = par_l       # set of updated variables
        a, M_v_l = const_l      # set of constants variables (not updated)        
        g_amp, g_shape = g      # split g into g_amp and g_shape

        # vmap over molecules (1 per plane)
        def fwd_v_molecule(a, g_amp, g_shape, ph_m, M_m, pl):
            return self.compute_err(jax.vmap(lambda ph, M: self.generate_image_lorentz(a, g_amp, g_shape, ph, M, pl))(ph_m, M_m), M_m)
        
        # vmap over planes (5 per voltage)
        def fwd_v_plane(a, g_amp, g_shape, ph_mp, M_mp, pl_p):
            return jax.vmap(lambda ph_m, M_m, pl: fwd_v_molecule(a, g_amp, g_shape, ph_m, M_m, pl))(ph_mp, M_mp, pl_p)
        
        # vmap over voltages (3 or 5 per actuator)
        def fwd_v_voltage(a, g_amp_v, g_shape, ph_mpv, M_mpv, pl_p):
            return jax.vmap(lambda g_amp, ph_mp, M_mp: fwd_v_plane(a, g_amp, g_shape, ph_mp, M_mp, pl_p))(g_amp_v, ph_mpv, M_mpv).sum()

        return fwd_v_voltage(a, g_amp, g_shape, ph_v_l, M_v_l, self.planes)


    @partial(jax.jit, static_argnums=(0,))
    def generate_image_lorentz(self, a, g_amp, g_shape, ph=[1, 0], M=None, z_def=None):
        """
        Generates a molecule image given a set of zernikes a=[a1,a2,...], and two parameter vectors: pos=[0,0,0] and  ph=[s,bg], an image M and the defocus
        """

        sig_scale = (M - M.min()).sum()
        aberr = jnp.fft.ifftshift(jnp.sum(self.modeset * a[:, jnp.newaxis, jnp.newaxis], axis = 0))
        g = jnp.array([g_amp[0], g_shape[0], g_shape[1], g_shape[2], g_shape[3], g_shape[4], g_amp[1]])
        actuator = jnp.fft.fftshift(lorentzian_2d(self.gauss_grid, g)) * self.pupil
        defocus,_ = self.calc_defocus( - z_def, self.RI[-1]) # defocus in immersive medium
        exp_jph = jnp.exp(1j*jnp.real(defocus + aberr + actuator)) * self.apodization; # phase
        PSF,_ = self.generate_psf(self.z_vec, exp_jph) # calculate 3D psf
        image = jax.image.scale_and_translate(PSF, 
                                            shape = (self.N_crop_os, self.N_crop_os, 1), #shape of image to be interpolated
                                            spatial_dims = (0, 1, 2), 
                                            scale = jnp.array([1, 1, 1]),
                                            translation = jnp.array([-self.shift,-self.shift, 0]), # y, x, z
                                            method = self.interp_method)[...,0]
        if self.os > 1:
            image = binning(image, self.os)
        return ph[1] + sig_scale * ph[0] * image
    

    '''--------------------------------'''








    '''------------------------------'''
    ''' Pixel-wise retrieval methods '''
    '''------------------------------'''

    @partial(jax.jit, static_argnums=(0,))
    def fwd_multiplane_pxl(self, par_l, const_l):
        '''Computes the cumulative error for all the images in all the planes, using the same pupil aberrations and dpp phase.'''
        dpp_phase = par_l           # set of variables that are updated
        M_v_l, v_v_l = const_l      # set of constants that are not updated
        error = 0                   # initialize error
        for i in range(self.n_planes): # iterate over the planes
            error += self.compute_err(jax.vmap(lambda v, M: self.generate_image_pxl(dpp_phase, v, M, self.planes[i]))(v_v_l[i], M_v_l[i]), M_v_l[i]).sum()
        return error


    @partial(jax.jit, static_argnums=(0,))
    def fwd_multiplane_pxl_bead(self, par_l, const_l): #
        '''
        Computes the cumulative error for all the images in all the planes, using the same pupil aberrations and dpp phase
         - pos is the bead position array. different among different beads but constraint among the same bead at different planes.
         - ph_v_l is the signal and bg list of arrays. independent among different beads and planes.
        '''        
        dpp_phase, pos_v, ph_v_l = par_l    # set of variables that are updated
        M_v_l = const_l                     # set of constants that are not updated
        error = 0                           # initialize error
        for i in range(self.n_planes): # iterate over the planes
            error += self.compute_err(jax.vmap(lambda p, ph, M: self.generate_image_pxl_bead(dpp_phase, p, ph, M, self.planes[i]))(pos_v, ph_v_l[i], M_v_l[i]), M_v_l[i]).sum()
        return error
    

    @partial(jax.jit, static_argnums=(0))
    def generate_image_pxl(self, dpp_phase=None, v=[0, 0, 0, 1, 0], M=None, z_def=None):
        """
        generates a molecule image given a set of zernikes, parameter vector v=[x,y,z,s,bg], an image M and the defocus
            Arguments:
            - dpp_phase:    fftshifted phase landscape of the dpp (shape=(Nk, Nk)), if None the phase is set to zero for the whole pupil
            - v:            molecule position: [x, y, z, signal, background] (in psf units)
            - M:            images. Used to scale the interpolated photon numbers. If None the fit works but the photons numbers are wrong.
            - z_def:        defocus (m)            
        """

        sig_scale = 1 if M is None else (M - M.min()).sum()
        z_def = self.z_defocus if z_def is None else z_def # if z_def is not given, use the one in the class
        
        dpp = jnp.zeros(self.DPP_phase.shape) if dpp_phase is None else dpp_phase
        dpp = dpp * self.pupil
        defocus,_ = self.calc_defocus( - z_def, self.RI[-1])    # defocus in immersive medium
        exp_jph = jnp.exp(1j*jnp.real(defocus + self.pupil_aberrations + dpp)) * self.apodization               
        PSF,_ = self.generate_psf(self.z_vec, exp_jph)
        image = jax.image.scale_and_translate(PSF, 
                                            shape = (self.N_crop_os, self.N_crop_os, 1),
                                            spatial_dims = (0, 1, 2),
                                            scale = jnp.array([1, 1, 1]), 
                                            translation = jnp.array([v[0]-self.shift, v[1]-self.shift, -v[2]]), # y, x, z
                                            method = self.interp_method)[...,0]
        if self.os > 1:
            image = binning(image, self.os)

        return v[4] + sig_scale * v[3] * image

    @partial(jax.jit, static_argnums=(0))
    def generate_image_pxl_bead(self, dpp_phase=None, pos=[0, 0, 0], ph=[1, 0], M=None, z_def=None):
        """
        generates a molecule image given a set of zernikes, parameter vector v=[x,y,z,s,bg], an image M and the defocus
            Arguments:
            - dpp_phase:    fftshifted phase landscape of the dpp (shape=(Nk, Nk)), if None the phase is set to zero for the whole pupil
            - v:            molecule position: [x, y, z, signal, background] (in psf units)
            - M:            images. Used to scale the interpolated photon numbers. If None the fit works but the photons numbers are wrong.
            - z_def:        defocus (m)            
        """

        sig_scale = 1 if M is None else (M - M.min()).sum()
        z_def = self.z_defocus if z_def is None else z_def # if z_def is not given, use the one in the class
        
        dpp = jnp.zeros(self.DPP_phase.shape) if dpp_phase is None else dpp_phase
        dpp = dpp * self.pupil
        defocus,_ = self.calc_defocus( - z_def, self.RI[-1])    # defocus in immersive medium
        exp_jph = jnp.exp(1j*jnp.real(defocus + self.pupil_aberrations + dpp)) * self.apodization               
        PSF,_ = self.generate_psf(self.z_vec, exp_jph)
        image = jax.image.scale_and_translate(PSF, 
                                            shape = (self.N_crop_os, self.N_crop_os, 1),
                                            spatial_dims = (0, 1, 2),
                                            scale = jnp.array([1, 1, 1]), 
                                            translation = jnp.array([pos[0]-self.shift, pos[1]-self.shift, -pos[2]]), # y, x, z
                                            method = self.interp_method)[...,0]
        if self.os > 1:
            image = binning(image, self.os)

        return ph[1] + sig_scale * ph[0] * image
    

    '''------------------------------'''




def get_update(fun): # fun(par, *args, **kwargs)
    @jax.jit
    def update(par, x_old, lrs, bounds, t1, *args, **kwargs):
        '''Nesterov Gradient descent update with momentum and bounds'''
        _, grads = jax.value_and_grad(fun, argnums=0)(par, *args, **kwargs) # get gradients
        x = jax.tree_map(lambda p, lrs, g, bmin, bmax: jnp.clip(p - lrs * g, bmin, bmax), par, lrs, grads, bounds[0], bounds[1]) # intermediate step
        t2 = 0.5 * (1 + jnp.sqrt(1 + 4 * t1**2)) # momentum update (Nesterof)
        par_ud = jax.tree_map(lambda x, xo: x + (t1-1)/t2 * (x - xo), x, x_old) # par update with new momentum
        err_ud = fun(par_ud, *args, **kwargs) # update error
        return err_ud, par_ud, x, t2
    return update





# %%
