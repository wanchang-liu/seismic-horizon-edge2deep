import numpy as np
import scipy.signal
import scipy.ndimage
import pyseistr as ps
import scipy.ndimage as ndimage


def compute_structure_tensor(input_npy, sigma=1.0):
    """
    Compute the structure tensor of the input data using gradients in all three directions.

    Args:
        input_npy (numpy.ndarray): The input 3D seismic data (image stack).
        sigma (float): The standard deviation for the Gaussian filter, controlling the scale of the tensor.

    Returns:
        numpy.ndarray: The structure tensor matrix.
    """
    # Compute the gradients in the x, y, and z directions using Gaussian filters
    gradient_x = ndimage.gaussian_filter(input_npy, sigma, order=[1, 0, 0])
    gradient_y = ndimage.gaussian_filter(input_npy, sigma, order=[0, 1, 0])
    gradient_z = ndimage.gaussian_filter(input_npy, sigma, order=[0, 0, 1])

    # Compute the elements of the structure tensor
    J11 = gradient_x ** 2
    J22 = gradient_y ** 2
    J33 = gradient_z ** 2
    J12 = gradient_x * gradient_y
    J13 = gradient_x * gradient_z
    J23 = gradient_y * gradient_z

    # Construct the structure tensor matrix (3x3 matrix for each voxel)
    J = np.array([[[J11, J12, J13], [J12, J22, J23], [J13, J23, J33]]])

    return J


def moving_window(data, window, func):
    wrapped = lambda region: func(region.reshape(window))
    return scipy.ndimage.generic_filter(data, wrapped, window)


def marfurt_semblance(region):
    region = region.reshape(-1, region.shape[-1])
    ntraces, nsamples = region.shape

    square_of_sums = np.sum(region, axis=0) ** 2
    sum_of_squares = np.sum(region ** 2, axis=0)
    sembl = square_of_sums.sum() / sum_of_squares.sum()
    return sembl / ntraces


def gersztenkorn_eigenstructure(region):
    region = region.reshape(-1, region.shape[-1])

    cov = region.dot(region.T)
    vals = np.linalg.eigvalsh(cov)
    return vals.max() / vals.sum()


def compute_coherence2(seismic):
    marfurt = moving_window(seismic, (3, 3, 9), marfurt_semblance)

    return marfurt


def compute_coherence3(seismic):
    gersztenkorn = moving_window(seismic, (3, 3, 9), gersztenkorn_eigenstructure)

    return gersztenkorn


def StructureOrientedFiltering(input_npy, niter=10, kappa=20, gamma=0.1, step=(1., 1., 1.), sigma=1.0, option=2):
    """
    Apply Anisotropic Diffusion Guided Filtering using structure tensor and coherence for noise reduction.

    Args:
        input_npy (numpy.ndarray): The input 3D seismic data (image stack).
        niter (int): The number of iterations for the diffusion process.
        kappa (float): The diffusivity coefficient, controlling the sensitivity to gradients.
        gamma (float): The time step for the diffusion update.
        step (tuple): The step size in the x, y, and z directions (spacing between pixels/voxels).
        sigma (float): The standard deviation for the Gaussian filter to compute structure tensor.
        option (int): The diffusion model option (1 for exponential, 2 for rational diffusion).

    Returns:
        numpy.ndarray: The filtered output 3D seismic data (image stack).
    """
    output_npy = input_npy.copy()  # Copy the input image stack for processing

    # Initialize internal variables for the gradients and the diffusion coefficients
    deltaD = np.zeros_like(output_npy)  # Gradient in the inline direction
    deltaS = deltaD.copy()  # Gradient in the crossline direction
    deltaE = deltaD.copy()  # Gradient in the time direction
    UD = deltaD.copy()  # Update matrix for inline direction diffusion
    NS = deltaD.copy()  # Update matrix for crossline direction diffusion
    EW = deltaD.copy()  # Update matrix for time direction diffusion
    gD = np.ones_like(output_npy)  # Diffusion coefficient in the inline direction
    gS = gD.copy()  # Diffusion coefficient in the crossline direction
    gE = gD.copy()  # Diffusion coefficient in the time direction

    # Compute the structure tensor for the input data
    J = compute_structure_tensor(output_npy, sigma)

    # Compute the coherence map based on the structure tensor
    coherence = compute_coherence2(input_npy)

    # Iterate to perform the anisotropic diffusion process
    for ii in range(niter):
        # Calculate the gradients along the three directions (inline, crossline, and time)
        deltaD[:-1, :, :] = np.diff(output_npy, axis=0)  # Gradient in the inline direction
        deltaS[:, :-1, :] = np.diff(output_npy, axis=1)  # Gradient in the crossline direction
        deltaE[:, :, :-1] = np.diff(output_npy, axis=2)  # Gradient in the time direction

        # Select the diffusion model based on the option
        if option == 1:
            # Exponential diffusion (classic anisotropic diffusion)
            gD = np.exp(-(deltaD / kappa) ** 2.) / step[0]
            gS = np.exp(-(deltaS / kappa) ** 2.) / step[1]
            gE = np.exp(-(deltaE / kappa) ** 2.) / step[2]
        elif option == 2:
            # Rational diffusion (robust to noise and sharp edges)
            gD = 1. / (1. + (deltaD / kappa) ** 2.) / step[0]
            gS = 1. / (1. + (deltaS / kappa) ** 2.) / step[1]
            gE = 1. / (1. + (deltaE / kappa) ** 2.) / step[2]

        # Modify the diffusion coefficients based on coherence
        gD *= coherence
        gS *= coherence
        gE *= coherence

        # Update the diffusion matrices with the gradients and coefficients
        D = gD * deltaD
        E = gE * deltaE
        S = gS * deltaS

        # Compute the diffusion update for each direction
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:, :, :] -= D[:-1, :, :]
        NS[:, 1:, :] -= S[:, :-1, :]
        EW[:, :, 1:] -= E[:, :, :-1]

        # Update the image stack with the diffusion result
        output_npy += gamma * (UD + NS + EW)

    return output_npy


# Construct Structure-Oriented Mean Filtering
def StructureOrientedMeanFiltering(cmpn, r1=2, r2=2, eps=0.01, order=2):
    [dipi, dipx] = ps.dip3dc(cmpn)  # Calculate dip angles

    mean_filter = ps.somean3dc(cmpn, dipi, dipx, r1, r2, eps, order)  # Construct structure-oriented mean filter

    return mean_filter


# Construct Structure-Oriented Median Filtering
def StructureOrientedMedianFiltering(cmpn, r1=2, r2=2, eps=0.01, order=2):
    [dipi, dipx] = ps.dip3dc(cmpn)  # Calculate dip angles

    median_filter = ps.somf3dc(cmpn, dipi, dipx, r1, r2, eps, order)  # Construct structure-oriented median filter

    return median_filter
