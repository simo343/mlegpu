"""phase retrieval: finding aberrations from a z-stack of a single bead"""

#%%
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # repo root
from SMLM import * 
from PIL import Image #for tif stacks
from jax import grad
from jax import grad, jit
import jax.numpy as jnp
import ipywidgets as widgets
from tifffile import imwrite, imread


@jax.jit
def calc_bead_stack(Z_magn, T_fit_coefs):
    aberr = jnp.exp(1j*jnp.sum(h.modeset * Z_magn[:,jnp.newaxis, jnp.newaxis], axis = 0))

    #---apodization using polynomial T-fit coefficients----
    h.T_fit_coefs = jnp.append(T_fit_coefs, 1) #append a "1" for the offest (this must be one)
    h.calc_apodization()
    h.apodization = jnp.clip(h.apodization, a_min = 0, a_max = 10) #clip negative values 
    Apo = jnp.fft.fftshift(h.apodization)

    #---apodization using Zernike modes----
    #Apo = 1 + jnp.sum(h.modeset * T_fit_coefs[:,jnp.newaxis, jnp.newaxis], axis = 0)
    #Apo = jnp.clip(Apo, a_min = 0, a_max = 10) #clip negative values 

    fields = jnp.array([h.czt2(field * Apo * aberr) for field in E_BFP_tot])                                      
    I_simu = jnp.sum(jnp.abs(fields)**2, axis=0)
    I_simu = blur(I_simu, FG)
    I_simu /= jnp.sum(I_simu, axis = (1,2))[:,jnp.newaxis, jnp.newaxis] #normalize each z-slice individually

    return I_simu, jnp.angle(aberr), Apo

@jax.jit
def calc_error(Z_magn, T_fit_coefs, I_exp, gamma): 
    I_simu,_,_ = calc_bead_stack(Z_magn, T_fit_coefs)
    error = jnp.sum((jnp.real(I_simu**gamma) - jnp.real(I_exp**gamma))**2)

    # Compute the smoothed TV norm
    tv_norm_Z = smooth_tv_norm(Z_magn)
    tv_norm_T = smooth_tv_norm(T_fit_coefs)

    return error + lambda_Z * tv_norm_Z + lambda_T * tv_norm_T


def compare_stacks(I_simu, I_exp): 
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(jnp.sum(I_exp, 2), extent = [0, h.ux*h.Nx*1e6, 0, dz*N_z*1e6], cmap = "jet")
    ax[0].set_xlabel("x (um)")
    ax[0].set_ylabel("z (um)")
    ax[1].imshow(jnp.sum(I_simu, 2), extent = [0, h.ux*h.Nx*1e6, 0, dz*N_z*1e6], cmap = "jet")
    ax[1].set_xlabel("x (um)")
    ax[1].set_ylabel("z (um)")

def load_exp_stack(path): 
    I_exp = imread(path).astype("float")
    I_exp -= jnp.min(I_exp, axis = (1,2))[:, jnp.newaxis, jnp.newaxis] #background subtraction
    #I_exp -= jnp.median(I_exp[:,:,0], axis = 1)[:, jnp.newaxis, jnp.newaxis] #background subtraction (median filter
    I_exp /= jnp.sum(I_exp, axis = (1,2))[:, jnp.newaxis, jnp.newaxis] #normalize each slice individually
    return I_exp

def gauss(Nx, ux, sigma): 
    x = (jnp.arange(Nx) - jnp.floor(Nx/2))*ux
    X,Y = jnp.meshgrid(x,x); 
    G = jnp.exp(-(X**2 + Y**2)/(2*(sigma*1e-9)**2))
    FG = jnp.real(jnp.fft.fft2(jnp.fft.ifftshift(G)))
    return G, FG

@jax.jit
def blur(I_simu, FG): 
    F_I = jnp.fft.fft2(jnp.fft.ifftshift(I_simu, axes = (1,2)))
    return jnp.real(jnp.fft.fftshift(jnp.fft.ifft2(F_I * FG), axes = (1,2)))

def browse_images(images):
    n = images.shape[0]
    def view_image(i):
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title('Slice number: %s' % i)
        plt.axis('off')
        plt.show()
    widgets.interact(view_image, i=(0,n-1))

@jax.jit
def smooth_tv_norm(x, beta=1e-3):
    """Compute the smoothed TV norm and its gradient."""
    diff = jnp.diff(x)
    norm = jnp.sqrt(diff**2 + beta)
    return jnp.sum(norm)

#%% user parameters

path = "./PSF stacks/bead1_green_channel.tif"
dz = 0.2e-6 #axial interspacing (in meter) between two successive bead images
NA = 1.5
lam = 580 #wavelength (in nm)
ux = 160 #lateral pixel size (in nanometer)
sigma = 150    #Gauss-width for blurring the simulation (in nm)
gamma = .5 #gamma (exponent used for error calculation)
z_strength = 0.4 #relative strength of the z-dipole compared to the x- and y-dipolse (choose in [0,1])
RI_fluid = 1.33 #RI of medium containing the beads (water or mounting medium)

#%% initialize 

I_exp = load_exp_stack(path)
N_z = I_exp.shape[0]
z_vec = (np.arange(N_z) - N_z//2)*dz 

#define modes to be included into the fit
Z_modes = jnp.arange(2,57)
Z_magn = jnp.zeros_like(Z_modes).astype("float32")
Z_magn = Z_magn.at[2].set(0) #pre-set defocus

T_fit_coefs = jnp.zeros_like(Z_modes).astype("float32") #initial guess for T-fit coefficients
#T_fit_coefs = jnp.array([0,0,0,0]).astype("float") #initial guess for T-fit coefficients

#calculate PSF
h = psf(NA = NA, 
                 wavelength = lam,                 # nm
                 z_defocus = 00,           # nm
                 RI = [RI_fluid, 1.518],        # sample, coverslip
                 Nk = 128,                   #grid size in fourier space
                 Nx = I_exp.shape[1], 
                 ux = ux,                  # nm
                 uz = 5,                    # nm
                 focal_len = 3e6,           # nm
                 z_min = 0,                 # nm
                 z_max = 300,               # nm
                 Z_modes = Z_modes,
                 Z_magn =  Z_magn,
                 Ts_coefs = [1],
                 Tp_coefs = [1],
                 T_fit_coefs = jnp.append(T_fit_coefs, 1),
                 d2 = 0,                    # nm
                 os = 1)# %%


G, FG = gauss(h.Nx, h.ux, sigma) #calc. Gaussian blurring kernel 

#pre-calculate defocus stack (complex field)
D_stack = jnp.zeros((N_z, h.Nk, h.Nk), dtype = "complex64")
for m in range(N_z):
    D_stack = D_stack.at[m].set(jnp.exp(1j*h.calc_defocus(z_vec[m], h.RI[1])[0]))

E_BFP_tot = jnp.fft.fftshift(h.E_BFP * D_stack[jnp.newaxis, ::], axes = (2,3))  #all BFP fields, for all z-planes and dipole contributions
E_BFP_tot *= jnp.array([1,1,1,1,z_strength, z_strength])[:,jnp.newaxis, jnp.newaxis, jnp.newaxis]

I_simu,_,_ = calc_bead_stack(Z_magn, T_fit_coefs)
compare_stacks(I_simu, I_exp)


# Function to compute the gradient
error_grad = grad(calc_error, argnums=(0,1))

# Gradient descent parameters
v_Z = jnp.zeros_like(Z_magn) #velocity for Zernikes
v_T = jnp.zeros_like(T_fit_coefs) #velocity for T-fit coefficients
learning_rate_Z = 1e-2
learning_rate_T = 1e-2
momentum_Z = 0.2 #momentum parameter
momentum_T = 0.2 
num_iterations = 5000

lambda_Z = 1e-16 #strength of the TV norm for Zernikes
lambda_T = 1e-16 #strength of the TV norm for T-fit coefficients

print("Error = ", calc_error(Z_magn, T_fit_coefs, I_exp, gamma))


#%% execute gradient descent 

error = []

for i in range(num_iterations):
   
    # Compute the gradient
    gradient = error_grad(Z_magn, T_fit_coefs, I_exp, gamma)
    # Update I_simu and Z_magn using the gradient
    
    #---gradient step for the Zernikes (phase)-----
    v_Z = momentum_Z * v_Z + learning_rate_Z * gradient[0]
    Z_magn = jnp.clip(Z_magn - v_Z, a_min = -3, a_max = 3)

    #---gradient step for the T-fit coefficients (amplitude)-----
    v_T = momentum_T * v_T + learning_rate_T * gradient[1]
    T_fit_coefs = jnp.clip(T_fit_coefs - v_T, a_min = -1, a_max = 1)

    error = np.append(error, calc_error(Z_magn, T_fit_coefs, I_exp, gamma))
    if i%20 == 0:
        print(num_iterations - i)
        plt.bar(h.Z_modes, Z_magn)
        plt.show()
        print("error = ", error[-1])

print("done")

#%% show results

Z_magn_est = Z_magn.copy()
T_fit_coefs_est = T_fit_coefs.copy()
print("Optimal Z_magn:", Z_magn_est)
plt.bar(h.Z_modes, Z_magn_est)

I_simu, phase, amplitude = calc_bead_stack(Z_magn_est, T_fit_coefs)
compare_stacks(I_simu, I_exp)

plt.figure()
plt.plot(error)
plt.xlabel("Iteration")
plt.ylabel("Error")

phase_centered = jnp.sum(h.modeset[3::] * Z_magn[3::,jnp.newaxis, jnp.newaxis], axis = 0)
fig, ax = plt.subplots(1,2)
phaseimage = ax[0].imshow(phase_centered, cmap = "hsv")
ax[0].set_title("Phase")
ampimage = ax[1].imshow(amplitude, cmap = "gray")
ax[1].set_title("Amplitude")
fig.colorbar(phaseimage, ax=ax[0])
fig.colorbar(ampimage, ax=ax[1])


#%% browse images 

browse_images(jnp.concatenate((I_exp, I_simu), axis=2))

#%% saving resulting tif-stack
imwrite('PSF_fit.tif', jnp.concatenate((I_exp, I_simu), axis=2))

