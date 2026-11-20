"""This file demonstrates the computatation of a single molecule image"""
"""Alexander Jesacher, 2025-06-10"""


#%% import libraries and defining functions

import numpy as np
from functools import partial
import jax.numpy as jnp
import jax
import aotools as ao
from matplotlib.pyplot import * 
from scipy.optimize import minimize  # Imported for potential future optimization tasks

zero_on_layer_top = False  #if True, the z=0 point is defined on top of the intermediate layer; if False, it is defined on top of the coverlip


def calc_dipole_BFP(dipole, Nk, RI, NA, lam, d_layer, z):
    
    """""
    This function calculates the BFP-fields (Ex,Ey) of an arbitrary oriented dipole in
    the focal point of an objective lens; 
    -) An intermediate layer between coverglass and specimen can be assumed as well
    -) includes also near-field effects (SAF emission)
    -) based on the paper of Axelrod, J. of Microscopy 2012
    N...grid size
    lambda_0...vacuum wavelength
    NA...numerical aperture of objective
    RI...vector of refractive indices RI=[RI_specimen, RI_intermed., RI_immoil]
    dipole...[theta phi]   polar and azimuthal anlges of dipole
    optional arguments: (assumed as zero if not defined)
    1) d_layer...thickness of intermediate layer (medium 2) in meter
    2) z...distance of dipole above the coverslip
    3) f...focal length of objective (m) 
    4) mu...magnitude of dipole 
    """
        
    f = 3e-3 #focal length of objective; not very critical, just a scaling factor here
    mu = 1 #dipole magnitude (also not very critical, just a scaling factor)
     
    theta_mu = dipole[0] #polar angle of dipole
    phi_mu = dipole[1] #azimuthal angle of dipole
    k0 = 2 * jnp.pi / lam #wavevector in vacuum
      
    uk = 2*k0* NA / Nk  #pixel unit in k-space
    k_vec = jnp.fft.fftfreq(Nk,1/Nk)*uk
    Ky, Kx = jnp.meshgrid(k_vec, k_vec) #k-space coordinates
    
    PHI3 = jnp.arctan2(Ky, Kx) #azimuthal angle in medium 3 (immersion medium)
    Kr = jnp.sqrt(Kx**2 + Ky**2) #radial wavevector component in medium 3
    mask_UAF = Kr <= (k0 * NA * (RI[0] / RI[2]))  # BPF-zone containing UAF-light
    pupil = Kr <= (k0 * NA)  # circular pupil of objective

    k1 = k0 * RI[0]  # Wavevectors in media 1 (typ. water), 2 (intermediate layer), and 3 (immers. medium)
    k2 = k0 * RI[1]
    k3 = k0 * RI[2]

    THETA3 = jnp.arcsin(pupil * Kr / k3)                        #angle in medium 3

    # Fresnel-coefs  %Eq. 3 of the paper
    CTHETA1 = jnp.sqrt(0j + 1 - (Kr / k1)**2)
    CTHETA2 = jnp.sqrt(0j + 1 - (Kr / k2)**2)
    CTHETA3 = jnp.cos(THETA3)

    tp12 = 2 * RI[0] * CTHETA1 / (RI[0] * CTHETA2 + RI[1] * CTHETA1)
    tp23 = 2 * RI[1] * CTHETA2 / (RI[1] * CTHETA3 + RI[2] * CTHETA2)
    ts12 = 2 * RI[0] * CTHETA1 / (RI[0] * CTHETA1 + RI[1] * CTHETA2)
    ts23 = 2 * RI[1] * CTHETA2 / (RI[1] * CTHETA2 + RI[2] * CTHETA3)
    rp12 = (RI[1] * CTHETA1 - RI[0] * CTHETA2) / (RI[0] * CTHETA2 + RI[1] * CTHETA1)
    rp23 = (RI[2] * CTHETA2 - RI[1] * CTHETA3) / (RI[1] * CTHETA3 + RI[2] * CTHETA2)
    rs12 = (RI[0] * CTHETA1 - RI[1] * CTHETA2) / (RI[0] * CTHETA1 + RI[1] * CTHETA2)
    rs23 = (RI[1] * CTHETA2 - RI[2] * CTHETA3) / (RI[1] * CTHETA2 + RI[2] * CTHETA3)  
    
    # From them, we calculate Fresnel coefs for the three-layer system
    tp = (tp12 * tp23 * jnp.exp(1j * k2 * d_layer * CTHETA2)) / (1 + rp12 * rp23 * jnp.exp(2j * k2 * d_layer * CTHETA2))
    ts = (ts12 * ts23 * jnp.exp(1j * k2 * d_layer * CTHETA2)) / (1 + rs12 * rs23 * jnp.exp(2j * k2 * d_layer * CTHETA2))

    # Dipole projections onto directions s, p and z:
    mu_p = mu * jnp.sin(theta_mu) * jnp.cos(phi_mu - PHI3)
    mu_s = mu * jnp.sin(theta_mu) * jnp.sin(phi_mu - PHI3)
    mu_z = mu * jnp.cos(theta_mu)

    # Calcualting prefactor C: (the only constant where f plays a role)
    if zero_on_layer_top: #this keeps the z=0 point on top of the intermediate layer
        z_factor = jnp.exp(1j * k1 * CTHETA1 * z)
        C = (k3**2 * jnp.exp(1j * k3 * f) * CTHETA3) / f / RI[0] * jnp.exp(-1j * k3 * d_layer * CTHETA3) *  z_factor  #original axelrod line  
    else: #this keeps the z=0 point on top of the coverlip (underneath the intermediate layer)
        z_factor = jnp.exp(1j * k1 * CTHETA1 * (z - d_layer)) #Note: here we deviate from Axelrod: we define the z=0 point always on top of the coverlip, i.e. underneath the (optional) intermediate layer
        C = (k3**2 * jnp.exp(1j * k3 * f) * CTHETA3) / f / RI[0] * jnp.exp(-1j * k3 * 0 * CTHETA3) *  z_factor #Alex: I've set d_layer=0; effectively removed the back-propagation
    
    C = C / k3**1  # NOTE: re-normalization here, to prevent large numbers

    # Field magnitudes in layer 3 (pre-objective zone), along the s, p and z-axes (see paper for axes definitions)
    E3p = C * tp * CTHETA3 * (mu_p / RI[2] + mu_z * jnp.sin(THETA3) / RI[0] / CTHETA1) #the additional factor RI[0] in the denominator is mentioned in an erratum (https://doi.org/10.1111/jmi.12173)
    E3s = C * ts * (mu_s / RI[2] / CTHETA1)
    E3z = C * tp * jnp.sin(THETA3) * (mu_p / RI[2] + mu_z * jnp.sin(THETA3) / RI[0] / CTHETA1)

    # Influence of the objective: rotation of rays by their angle theta3, such that they are all parallel to the optical axis
    Apo_p = 1 / jnp.sqrt(CTHETA3) * pupil 
    Apo_s = 1 / jnp.sqrt(CTHETA3) * pupil 
    E_BFP_p = (E3p * CTHETA3 + E3z * jnp.sin(THETA3)) * Apo_p
    E_BFP_s = E3s * Apo_s  # s-polarization remains unchanged by this rotation

    # Coordinate transform into x-and y-polarization -> fields in the back focal plane of objective
    E_BFP_x = ((jnp.cos(PHI3)) * E_BFP_p - (jnp.sin(PHI3)) * E_BFP_s)  
    E_BFP_y = ((jnp.sin(PHI3)) * E_BFP_p + (jnp.cos(PHI3)) * E_BFP_s)

    return E_BFP_x, E_BFP_y, mask_UAF, pupil

def calc_defocus(lam, Kx, Ky, pupil, z, RI):
    """create spherical defocus for a given z-shift and refractive index"""
    k0 = 2*jnp.pi/lam
    defocus = +z * jnp.sqrt(0j + (k0*RI)**2 - Kx**2 - Ky**2) * pupil
    mean_defocus = jnp.mean(jnp.where(pupil, defocus, 0))
    return defocus - mean_defocus #subtracting mean value

def calc_molecule_image(NA, lam0, uk, Nx, Nk, padval, d_layer, RI_layer, z_def):
    
    #--------------define some fixed parameters
    if zero_on_layer_top: #this keeps the z=0 point on top of the intermediate layer
        z = 0 #in this case the z-position of the molecule should also always be zero
    else:
        z = d_layer  #distance of dipole above the coverslip (set to thickness of intermediate layer)
    #--------------------------------------------------
    
    k_vec = jnp.fft.fftfreq(Nk,1/Nk)*uk
    Ky, Kx = jnp.meshgrid(k_vec, k_vec) 
    RI = [1.33, RI_layer, 1.518]  # RI of specimen, intermediate layer, immersion oil
    
    #calculating the intensity image of an isotropic, pointlike emitter"""
    Exx, Eyx, _, pupil = calc_dipole_BFP([jnp.pi/2,0], Nk, RI, NA, lam0, d_layer, z)
    Exy, Eyy,_,_ = calc_dipole_BFP([jnp.pi/2, jnp.pi/2], Nk, RI, NA, lam0, d_layer, z)
    Exz, Eyz,_,_ = calc_dipole_BFP([0,0], Nk, RI, NA, lam0, d_layer, z)

    aberr = jnp.exp(1j*jnp.real(calc_defocus(lam0, Kx, Ky, pupil, z_def, RI[2]))) #calculating defocus phase defined by z_defocus;

    E_BFP = jnp.zeros((6,Nk,Nk), dtype = jnp.complex64)

    E_BFP = E_BFP.at[0,:,:].set(Exx*aberr)
    E_BFP = E_BFP.at[1,:,:].set(Eyx*aberr)
    E_BFP = E_BFP.at[2,:,:].set(Exy*aberr)
    E_BFP = E_BFP.at[3,:,:].set(Eyy*aberr)
    E_BFP = E_BFP.at[4,:,:].set(Exz*aberr)
    E_BFP = E_BFP.at[5,:,:].set(Eyz*aberr)
    E_BFP = jnp.fft.fftshift(E_BFP, axes = (1,2))

    E_BFP_pad = jnp.pad(E_BFP, ((0,0),(padval,padval),(padval,padval)), mode='constant', constant_values=0)

    I = jnp.fft.fftshift(jnp.sum(jnp.abs(jnp.fft.fft2(jnp.fft.ifftshift(E_BFP_pad), axes = (1,2)))**2, axis = 0))/N_pad**2 #calculating the image in the focal plane
    I /= jnp.sum(I)  #normalization with respect to total energy (i.e. UAF + SAF)
  
    #---cropping to the size of the recorded molecule image----    
    x_range = slice((N_pad - Nx)//2, (N_pad + Nx)//2)
    I_focus = I[x_range, x_range]
    
    return I_focus, E_BFP



# %% defining user parameters and calulating an exemplary molecule image + precision for thickness and RI measurements

d_layer = 0e-9    #thickness of intermediate layer
RI_layer = 1.52     #refractive index of intermediate layer
z_defocus = 0e-9 #defocus setting of objective lens with respect to the sample; neg. values move the focus into the direction of the sample;  

NA = 1.5 #numerical aperture of objective lens
Nx = 25  #size of molecule image in pixels
Nk = 128   #size of BFP image in pixels
os = 1.0 #PSF oversampling factor; factor at which PSFs are spatiall oversampled in comparison to the camera pixel grid. For us, 1.0 is fine. 
ux = 108e-9  #effective camera pixel size in meters
lam0 = 670e-9 #vacuum emission wavelength 
k0 = 2*jnp.pi/lam0
uk = 2*k0* NA / Nk  #pixel unit in k-space 
N_pad = int(jnp.round(2*jnp.pi/(ux/os * uk)))#   %2*pi/(N_pad * ux) = uk --> N_pad = 2*pi/(ux*uk)
padval = (N_pad-Nk)//2

# if zero_on_layer_top: #if z=0 is defined at the top of the intermediate layer, the defocus for imaging should be independent of the layer thickness d_layer
#     z_def = z_defocus
# else: #in this case z=0 is defined on top of the coverlip (underneath the intermediate layer)
#     z_def = z_defocus - d_layer  #neg. values move the focus up(!);

I_molecule, E_BFP = calc_molecule_image(NA, lam0, uk, Nx, Nk, padval, d_layer, RI_layer, z_defocus) #calculates the molecule image and the six BFP fields

figure()
title("camera image of single molecule")
imshow(I_molecule)


# %%
