"""Fisher information maximization to find optimal plane spacing for "blind deconvolution" SMLM"""

#%%
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # repo root
from SMLM import * 
from matplotlib.pyplot import *
from itertools import combinations_with_replacement
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

# Set JAX to use CPU
#jax.config.update('jax_platform_name', 'cpu')

RI = [1.33, 1.518]
NA = 1.5
M = 60 #magnification of microscope
lam0 = 670e-9 #wavelength of light in vacuum

#calculate PSF: we choose optimal imaging conditions for TIRF 
Z_modes = range(2,23) #we include tip/tilt/defocus
#Z_modes = np.array([11,22])
Z_magn = np.zeros(len(Z_modes))
Z_magn[4] = 0.16 * 2*np.pi #optimal astigmatism for 3D SMLM

h = psf(NA = NA, 
                 wavelength = lam0*1e9,                 # nm
                 z_defocus = 00,           # nm
                 RI = RI,        # sample, coverslip
                 Nk = 128,                   #grid size in fourier space
                 Nx = 20, 
                 ux = 100,                  # nm
                 uz = 10,                    # nm
                 focal_len = 3e6,           # nm
                 z_min = 0,                 # nm
                 z_max = 300,               # nm
                 Z_modes = Z_modes,
                 Z_magn = Z_magn,
                 Ts_coefs = [1],
                 Tp_coefs = [1],
                 T_fit_coefs = [1],
                 d2 = 0,                    # nm
                 os = 1)# %%   #note that the pixel size in h.data is h.ux/h.os

cam = Camera(pixsize = 6500, QE = 0.75, baseline = 100, amp = 0.24, EM_gain = 1, readnoise = 1)
imshow(h.data[:,:,0]);# colorbar();
axis("off")
show()

@jax.jit
def calc_model_wrapper(a_dyn, a_stat, z_defocus): 
    a = a_dyn
    v = a_stat
    
    #a = a_stat[4::] #choose here for which variables you want to have the Fisher info calculated; these parameters are in a_dyn, the rest in a_stat
    #v = jnp.array((a_dyn[0], a_stat[1], a_dyn, a_stat[2], a_stat[3]))
    #v = jnp.concatenate([a_dyn[0:3], a_stat[0:2]])
    #a = a_stat[2::]
    
    return h.generate_image(a, v, z_def = z_defocus) # calc_model(h,a,v,z_defocus)

def calc_Fisher(a_dyn, a_stat, z_defocus):
    """returns the Fisher information matrix for a given PSF and parameter set p = (a,v)"""
    
    I_mol = calc_model_wrapper(a_dyn, a_stat, z_defocus)
    J = jax.jacfwd(calc_model_wrapper, argnums = 0) #calclate all gradients
    #vec = jnp.concatenate([g_v, g_a], axis = 2) 
    vec = J(a_dyn, a_stat, z_defocus) #calculate Jacobian with respct to aberrations (a) and typical parameters (v)
    vec /= jnp.sqrt(I_mol[:,:,jnp.newaxis])

    FI = jnp.sum(1e-2*vec[..., jnp.newaxis] * vec[:,:, jnp.newaxis, :], axis = (0,1))

    return FI, I_mol
           
#@partial(jax.jit, static_argnums=(0,))
#def calc_model(h, a, v, z_subset, phi_aberr = 0):
    """produces a molecule image given a set of zernikes and a parameter vector v=[x,y,z,sig,bg]
    the defocus value can be chosen as well (in meters)"""
    
    #a,v = p #split the input parameter vector into zernike coefficients and the rest
     
    #aberr = (h.modeset * a[:, jnp.newaxis, jnp.newaxis]).sum(axis=0) # pupil aberrations defined via Zernikes
    #defocus,_ = h.calc_defocus(-z_subset, h.RI[1]) #user-defined defocus
    #exp_jph = jnp.exp(1j*jnp.real(defocus + aberr + phi_aberr)) * h.apodization; # exponential of pupil phase.
    
    PSF = h.generate_image(a, v, z_def = z_subset) # calculate 3D psf
  
  
    
    I_hires = jax.image.scale_and_translate(PSF, 
                                            shape = (h.Nx_os, h.Nx_os, 1), 
                                            spatial_dims = (0, 1, 2), 
                                            scale = jnp.ones(3), 
                                            translation = jnp.array([v[0], v[1], -v[2]]), # clip(v, 0, n-1)
                                            method = h.interp_method)[...,0]
    if h.os > 1:
        I_hires = binning(I_hires, h.os)
    return v[4] + v[3] * I_hires 

@jax.jit
def calc_image_stack(a, v, z_subset, a_aberr = 0): 
    """calculates multi-plane image stack for a given set of defocus planes, of zernike coefficients and a parameter vector v=[x,y,z,sig,bg]"""
    I = jnp.zeros((len(z_subset), h.Nx-4, h.Nx-4))
    for m in range(len(z_subset)):
        tmp = h.generate_image(a + a_aberr, v[m], z_def = z_subset[m])
        I = I.at[m].set(tmp) #calc_model(h, a, v[m], z_subset[m], phi_aberr = phi_aberr))
    return I

#@partial(jax.jit, static_argnums=(5,6,7,))
def execute_gradient_descent(a_est, v, z_defocus, I_exp, vel, momentum, learning_rate, num_iterations):
    LLR = jnp.zeros(num_iterations)
    for i in range(num_iterations):
        a_est, vel, LLR_step = update_step(a_est, v, z_defocus, I_exp, vel, momentum, learning_rate)
        LLR = LLR.at[i].set(LLR_step)
    return a_est, LLR

@jax.jit
def update_step(a_est, v, z_defocus, I_exp, vel, momentum, learning_rate):
    """update step of gradient descent"""
     # Compute the gradient with respect to aberrations
    gradient = error_grad(a_est, v, z_defocus, I_exp)
    vel = momentum  * vel + learning_rate  * gradient #update velocity
    a_est = jnp.clip(a_est - vel, a_min = -2, a_max = 2)
    LLR = calc_error(a_est, v, z_defocus, I_exp)
    return a_est, vel, LLR

@jax.jit
def calc_error(a, v, z_defocus, I_exp): 
    I_simu = calc_image_stack(a, v, z_defocus)
    error =  2*jnp.sum((I_simu - I_exp * jnp.log(I_simu)) - (I_exp - I_exp * jnp.log(I_exp)))
    #error = jnp.sum((jnp.real(I_simu) - jnp.real(I_exp))**2)

    # Compute the smoothed TV norm
    #v_norm_Z = smooth_tv_norm(Z_magn)
    #tv_norm_T = smooth_tv_norm(T_fit_coefs)

    return error# + lambda_Z * tv_norm_Z + lambda_T * tv_norm_T

# Function to compute the gradient
error_grad = jax.grad(calc_error, argnums=0)


def show_image_stack(I, suptitle = ""):
    fig, ax = subplots(I.shape[0], I.shape[1], figsize = (4,7))
    
    if I.shape[1] > 1:
        for i in range(I.shape[1]):
            if I.shape[0] > 1:
                ax[0,i].set_title("z = " + str(int(np.round(1e9*z_defocus[planes[i]]))) + " nm")
            else: 
                ax[i].set_title("z = " + str(int(np.round(1e9*z_defocus[planes[i]]))) + " nm")
            
            if I.shape[0] > 1:
                for m in range(I.shape[0]):
                    ax[m,i].imshow(I[m,i,::])
                    ax[m,i].axis("off")
            else: 
                ax[i].imshow(I[0,i,::])
                ax[i].axis("off")
    else:
        if I.shape[0] > 1:
            for m in range(I.shape[0]):
                ax[m].imshow(I[m,0,::])
                ax[m].set_title("plane " + str(int(np.round(1e9*z_defocus[planes[0]]))))
                ax[m].axis("off")
        else:
            imshow(I[0,0,::])
            axis("off")
                
    fig.suptitle(suptitle)
    show()

def sum_FI_matrices(indices, weights):
    return jnp.sum(FI_vec[indices]*weights[:,jnp.newaxis, jnp.newaxis], axis=0)

def create_random_aberr(h, N_trials, a_free, sigma_free, a_tot_spher, a_tot_nonspher):
    """creates a random aberration map with a given standard deviation"""
    pupil = np.fft.fftshift(h.pupil)[np.newaxis,:,:]
    
    #----define a free-form aberration phase mask----
    
    phi = np.random.rand(N_trials, h.Nk, h.Nk) * 2*np.pi - np.pi
    phi_smooth = gaussian_filter(phi, sigma=sigma_free, axes = (1,2))
    phi_smooth -= np.mean(phi_smooth * pupil)
    phi_smooth *= pupil
    phi_smooth /= np.sqrt(np.sum(phi_smooth**2, axis = (1,2))/np.sum(pupil))[:, np.newaxis, np.newaxis]
    phi_smooth *= a_free
    
    a_free = Zernike_decomposition(phi_smooth, h)
    
    #----define an additional total aberration vector----
    
    idx_spher = np.where(np.isin(Z_modes, [4, 11, 22]))[0]
    idx_nonspher = np.delete(np.arange(len(Z_modes)), np.where(np.isin(np.arange(len(Z_modes)), idx_spher))[0])   
    a_tot = np.random.rand(N_trials, len(Z_modes)) - 0.5    
    a_tot[:,idx_nonspher] /= np.linalg.norm(a_tot[:,idx_nonspher], axis = 1)[:,np.newaxis] #normalize to unit vector
    a_tot[:,idx_spher] /= np.linalg.norm(a_tot[:,idx_spher], axis = 1)[:,np.newaxis] #normalize to unit vector
    a_tot[:,idx_spher] *= a_tot_spher
    a_tot[:,idx_nonspher] *= a_tot_nonspher
    
    phi = phase_from_Zernikes(h, a_tot)
    
    phi_aberr_GT = phi + phi_smooth
    a_GT = a_tot  + a_free
    
    fig, ax = subplots(2,N_trials)
    for m in range(N_trials):
        ax[0,m].set_title("trial " + str(m))
        ax[0,m].imshow(phi_aberr_GT[m]); 
        ax[0,m].axis("off")
        ax[1,m].bar(Z_modes, a_GT[m,:], color = "b", label = "mean residual aberrations")  
        
    suptitle("GT aberrations")
    show()
    
    
    return a_GT, phi_aberr_GT

def calc_experimental_image_stack(h, a, v, z_subset, a_free):
    """calculate aberrated experimental image stack for a given set of defocus planes and an aberration"""
    """the aberration can be defined via two different ways: 
    A...by setting a_tot_spher and a_tot_nonspher to the desired values
    B...by providing an additional random aberration phase mask phi_aberr, defined via a_free and sigma_free"""
    
    N_trials = a_free.shape[0]   
    #calculate experimental data (ground truth)
    I_exp = jnp.zeros((N_trials, len(z_subset), h.Nx-4, h.Nx-4))
    for m in range(N_trials):       
        I_exp = I_exp.at[m].set(np.random.poisson(calc_image_stack(a, v, z_subset, a_aberr = a_free[m])))

    return I_exp

def Zernike_decomposition(phi, h):
    """decompose a given phase mask into Zernike modes"""
    a = jnp.zeros((phi.shape[0], h.modeset.shape[0]))
    
    for n in range(a.shape[0]):
        for m in range(a.shape[1]):
            a = a.at[n,m].set(jnp.sum(h.modeset[m] * phi[n])/jnp.sum(h.pupil))
    return a

def phase_from_Zernikes(h, a_Zernike):
    """create a phase mask from Zernike modes"""
    phi = jnp.zeros((a_Zernike.shape[0], h.Nk, h.Nk))
    
    for m in range(a_Zernike.shape[0]):
        a_tmp = a_Zernike[m]
        phi = phi.at[m].set((h.modeset * a_tmp[:, jnp.newaxis, jnp.newaxis]).sum(axis=0))
    
    return phi# pupil aberrations defined via Zernikes

# %% 
"""calcuating Fisher information of aberration magnitudes for a single PSF """

sig = 2000
bg = 20

a = jnp.zeros(len(h.Z_modes))
idx = np.where(np.array(Z_modes) == 6)[0][0]  #find index that contains astigmatism
a = a.at[idx].set(+0.16 * 2*jnp.pi) #optimal astigmatism for 3D SMLM
v = jnp.array([0.0, 0.0, 0.0, sig, bg])

#----here, chose which parameters you want to have in the Fisher info matrix (-> a_dyn) and which in the model (-> a_stat
#----IMPORTANT: also change the Fisher info calculation function accordingly ("calc_model_wrapper")
#a_dyn = jnp.array(v[2])   #choose here for which variables you want to have the Fisher info calculated; these parameters are in a_dyn, the rest in a_stat
#a_stat = jnp.concatenate([jnp.array([v[0], v[1], v[3], v[4]]), a])
#a_dyn = jnp.array(v[0:3])
#a_stat = jnp.concatenate([jnp.array([v[4], v[5]]), a])
a_dyn = a.copy()
a_stat = v.copy()

z_defocus = jnp.array([00e-9])

FI, I_mol = calc_Fisher(a_dyn, a_stat, z_defocus)
D = jnp.linalg.det(FI)

print(FI[0,0])

#print(jnp.sum(FI))
print("determinant = ", D)
imshow(FI); colorbar(); #title("FI matrix for aberrations")
axis("off")
show()

#%% 

"""calculate Fisher matrices for aberration magnitudes for all z-planes"""

z_defocus = jnp.arange(-1000e-9, 1000e-9, 100e-9)
Nz = len(z_defocus)
FI_vec = jnp.zeros((len(z_defocus), len(a_dyn), len(a_dyn)))
FI_inv_vec = jnp.zeros((len(z_defocus), len(a_dyn), len(a_dyn)))
D_FI = jnp.zeros(len(z_defocus))

for m,defocus in enumerate(z_defocus): 
    FI, _ = calc_Fisher(a_dyn, a_stat, defocus)
    D_FI = D_FI.at[m].set(jnp.linalg.det(FI))
    FI_vec = FI_vec.at[m,:,:].set(FI)
    FI_inv_vec = FI_inv_vec.at[m,:,:].set(jnp.linalg.inv(FI))

#plot(z_defocus*1e9, jnp.sum(FI_vec, axis = (1,2))); xlabel("defocus / nm"); ylabel("Fisher info"); title("total Fisher info"); show()
plot(z_defocus*1e9, D_FI); xlabel("defocus / nm"); ylabel("determinant"); title("determinant of FI matrix"); show()

import ipywidgets as widgets

def browse_images(images):
    n = len(images)
    def view_image(i):
        plt.imshow(images[i], cmap=plt.cm.gray, vmin = FI_vec.min(), vmax = FI_vec.max())
        plt.title('Image: %s' % i)
        plt.show()
    widgets.interact(view_image, i=(0,n-1))

browse_images(FI_vec)


#plot(z_defocus*1e9, FI_inv_vec[:,1,1]); xlabel("defocus / nm"); ylabel("std of z est."); title("delta z"); show();
#%%

""" Find the optimal combination of planes by maximizing the determinant of the Fisher information matrix
 fully exhaustive search"""

N_planes = 4# Number of planes to choose in each combination
include_fixed = True #should an additional plane at a fixed position be included?
energy_weight = 1 #relative weight of the energy in the fixed plane
z_fixed = 00 #position of fixed plane in nm

rank = 1 #choose the rank of the plane-tuple

# Generate all combinations of planes out of Nz
Nz = len(FI_vec)  # Example total number of planes
comb_indices = np.array(list(combinations_with_replacement(range(Nz), N_planes)))
weights = jnp.ones((len(comb_indices), comb_indices.shape[1]))

if include_fixed:
    #-----indlcuding a fixed plane: replace first index with index of fixed plane-----
    idx = np.where(jnp.round(z_defocus*1e9) == z_fixed) #choose the first plane to be the one at defocus = 
    idx_array = np.full(comb_indices.shape[0], idx[0])
    comb_indices = jnp.concatenate([comb_indices, idx_array[:, jnp.newaxis]], axis=1)
    weights = jnp.ones((len(comb_indices), comb_indices.shape[1]))
    weights = weights.at[:,0].set(energy_weight) #increase the weight of a particular plane  

# Vectorize the summation over all combinations
FI_sums = jax.vmap(sum_FI_matrices)(comb_indices, weights)

# Compute determinants in a vectorized manner
D_vals = jax.vmap(jnp.linalg.det)(FI_sums)

idx_list = jnp.argsort(D_vals)[::-1]
print(str(rank) + "-optimal planes:", jnp.round(z_defocus[comb_indices[idx_list[rank-1]]]*1e9))
print("opt. indices:", comb_indices[idx_list[rank-1]])
print("value of determinant:", D_vals[idx_list[rank-1]])    

plot(D_vals); xlabel("combination index"); ylabel("determinant"); title("determinant of FI matrix"); show()
FI_tot = sum_FI_matrices(comb_indices[jnp.argmax(D_vals)], weights[0,:])
imshow(FI_tot); colorbar(); title("FI matrix for optimal planes")

#%% find the Determinant for a particular combination of planes

D = np.zeros(10)
for d in range(10):
    
    planes = [-100*d, 0, 100*d]

    idx = np.zeros(len(planes), dtype = jnp.int32)
    tmp = np.zeros((len(a_dyn), len(a_dyn)))
    for m in range(len(planes)):
        idx[m] = jnp.where(np.round(z_defocus*1e9) == planes[m])[0][0]
        tmp += FI_vec[idx[m],::]
        D[d] = jnp.linalg.det(tmp)
    
plot(np.arange(10)*100,D)    
xlabel("plane distance / nm")
ylabel("det(FI) / a.u.")
grid("on")
print("determinant at planes:")
print(np.round(z_defocus[idx]*1e9))
print("D = ", D)    


#%% 

"""---------------VERIFICATION PART--------------------------
simulating a multi-plane image and trying to retrieve aberrations"""

N_trials = 5 #how many repetition of the simu for averaging

#-------define aberrations that should be applied to the "experimental data"
#method A....define total spherical and non-spherical aberration magnitudes
a_tot_spher = 0.5
a_tot_nonspher = 0.5
#method B...define a free-form aberration given by total magnitude a_free and spat. frequ. filter sigma_free
a_free = 0
sigma_free = 30 #blur kernel size - the smaller the finer the spatial frequ. structure

a_GT, phi_aberr_GT = create_random_aberr(h, N_trials, a_free, sigma_free, a_tot_spher, a_tot_nonspher)

#%% run gradient descent for each trial in sequence

# choosing planes to record images and simulating the images
rank = 1 #which plane set do you want to use (according to ranking)
planes = comb_indices[idx_list[rank-1]]
#
planes = np.array([10,10,10]) #only central plane
planes = np.array([6,10,14]) #manually choose planes

print("planes chosen: ", np.round(z_defocus[np.array(planes)]*1e9))

energy_weight = 1 #enregy amplificatoin factor in the last plane (=fixed plane) with respectd to the aux. planes
sig = 10000*3/len(planes) #energy in aux. planes
bg = 100*3/len(planes) #background in aux. planes

#--------------------------------------------------------------------------
v = jnp.zeros((len(planes), 5))
v = v.at[:,3].set(sig)
v = v.at[:,4].set(bg)

if include_fixed:
    v = v.at[-1,3].set(energy_weight * sig) #increase the energy in the fixed plane
    v = v.at[-1,4].set(energy_weight * bg) #increase the energy in the fixed plane

a_est = a.copy() #init. aberration estimate

I_exp = calc_experimental_image_stack(h, a, v, z_defocus[planes], a_GT)
show_image_stack(I_exp, suptitle = "experimental data")

#------------Gradient descent parameters-----------------
num_iterations = 200
learning_rate  = .2/ (sig*energy_weight) #learning rate
momentum  = .5 #momentum parameter
vel = 0  # Initialize update velocity

# Vectorize execute_gradient_descent over the 0th dimension of I_exp
execute_gradient_descent_vmap = jax.vmap(execute_gradient_descent, in_axes=(None, None, None, 0, None, None, None, None))

# Now call the vectorized function; this will run execute_gradient_descent on each 3D-stack in I_exp in parallel
a_est_vmap, LLR_vmap = execute_gradient_descent_vmap(a_est, v, z_defocus[planes], I_exp, vel, momentum, learning_rate, num_iterations)
LLR_mean = jnp.mean(LLR_vmap, axis = 0) 
LLR_std = jnp.std(LLR_vmap, axis = 0)

#--------ERROR ANALYSIS-------------

I_simu = I_exp.copy()
for m in range(N_trials):
    I_simu = I_simu.at[m].set(calc_image_stack(a_est_vmap[m,:], v, z_defocus[planes]))
show_image_stack(I_simu, suptitle = "estimated PSF")

fig, ax = subplots()
ax.plot(LLR_mean)
xlabel("iteration")
ylabel("mean LLR (ribbon = std)")
chi2_exp = I_exp[0].size - len(a)   #expecetancy value of chi2 distribution
ax.axhline(chi2_exp, color = "green", linestyle = "--") #
fill_between(np.arange(num_iterations), LLR_mean - LLR_std, LLR_mean + LLR_std, color = "lightblue", alpha = 0.5)
print("mean final LLR value = ", LLR_mean[-1])

figure(figsize = (6,3))
bar(Z_modes, a_GT[0,:], color = "b", label = "mean residual aberrations")
bar(Z_modes, a_est_vmap[0,:] - a, color = "r", width = 0.4, label = "mean residual aberrations")
title("example GT (blue) + estimated aberrations (red)")
xlabel("Zernike mode (Noll)")
ylabel("aberration magnitude / rad RMS")
show()

phi_aberr_est = phase_from_Zernikes(h, a_est_vmap - a)
fig, ax = subplots(2,N_trials, figsize=(5,2))
for m in range(N_trials):
    ax[0,m].imshow(phi_aberr_GT[m], vmin = -np.pi, vmax = np.pi); 
    ax[1,m].imshow(phi_aberr_est[m],  vmin = -np.pi, vmax = np.pi); 
    ax[0,m].axis("off")
    ax[1,m].axis("off")
suptitle("GT (line 1) and estimated masks")
axis("off")
show()

print("-------------------------------------------------")
print("planes chosen: ", np.round(z_defocus[np.array(planes)]*1e9))

#----calculate spherical and non-spherical errors separately-------
idx_spher = np.where(np.isin(Z_modes, [4, 11, 22]))[0]
idx_nonspher = np.delete(np.arange(len(Z_modes)), np.where(np.isin(np.arange(len(Z_modes)), idx_spher))[0])
delta_spher = a_est_vmap[:,idx_spher] - a_GT[:,idx_spher]
delta_nonspher = a_est_vmap[:,idx_nonspher] - a_GT[:,idx_nonspher]
error_spher = np.linalg.norm(delta_spher, axis = 1)
error_nonspher = np.linalg.norm(delta_nonspher, axis = 1)
print("mean error (spher) = " + str(np.round(error_spher.mean(),2)) + " +- " + str(np.round(error_spher.std(),2)) + " rad")
print("mean error (non-spher) = " + str(np.round(error_nonspher.mean(),2)) + " +- " + str(np.round(error_nonspher.std(),2)) + " rad")

#----calculate phase error of phase mask----
pupil = np.fft.fftshift(h.pupil)
delta_phi = phi_aberr_GT - phi_aberr_est
error_phi = np.sqrt(np.sum((delta_phi * pupil[np.newaxis,:,:])**2, axis = (1,2))/np.sum(pupil))
print("mean std-error phase mask = " + str(np.round(error_phi.mean(),2)) + " +- " + str(np.round(error_phi.std(),2)) + " rad")



#%%

# idx_spher = np.where(np.isin(Z_modes, [4, 11, 22]))[0]
# idx_nonspher = np.delete(np.arange(len(Z_modes)), np.where(np.isin(np.arange(len(Z_modes)), idx_spher))[0])
# error_est_spher = np.sqrt(np.sum(delta_aberr[:, idx_spher]**2, axis = 1)/N_trials).mean()
# error_est_nonspher = np.sqrt(np.sum(delta_aberr[:, idx_nonspher]**2, axis = 1)/N_trials).mean()

# figure()
# bar(np.arange(2), [error_est_spher, error_est_nonspher], label = "mean residual aberrations")
# title("mean relative error: sphericals (left); non-sphericals (right)")
# xlabel("Zernike mode (Noll)")
# ylabel("aberration magnitude / rad RMS")
# show()

# error_mean = np.linalg.norm(np.mean(aberr - a_est_vmap, axis =0))
# error_std = np.linalg.norm(np.std(aberr - a_est_vmap, axis = 0))
# print("mean remaining aberration magnitude = ", error_mean)
# print("std = ", error_std)



# %%
