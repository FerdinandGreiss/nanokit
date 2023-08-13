"""
Script with functions to find and analyze expression spots in single-molecule
fluorescence movies. It was usually used in combination with a script for lateral 
drift correction, converting a 2-color stack of dimension (2*T, H, W) into two drift corrected
movies (T, H, W) with name "{path-to-movie}-driftCorrected-ch{1 and 2}.tif"

After you define a path to the movie to be analyzed ({path-to-movie}-driftCorrected-ch2.tif), 
it will try to find protein and DNA spots in a movie with dimensions [Time, Height, Width]. 
The movie shape is therefore (T, H, W). CH1 is usually the DNA channel, and CH2 the protein 
channel (if changed, REVERSE_CHANNEL = True)

It will generate the following files/directories inside the movie folder:
1) Figures in "fig-output" folder
2) Data in "data-output" folder with name "intensity_measures.hdf5"
    * Intensity measures is a HDF file format with the following entries:
        Signal: Signal fluorescent signal of expression spots over time (n, T)
        Background: Background fluorescent signal of expression spots over time (n, T)
        x-position: x positions for the expression spots (n)
        y-position: y positions for the expression spots (n)
3) DNA spots in a CSV file as "dna-spots.csv"
4) Protein spot in a CSV file as "{Movie Name}-trajs.csv":
    * The protein spots can be visualized on the tiff movie using the amazing quot GUI: 
    https://github.com/alecheckert/quot

Author: Ferdinand Greiss, Weizmann Institute of Science, 2023
"""
import os
import warnings

import h5py
import matplotlib.patches as patches
import matplotlib.pylab as plt
import numba
import numpy as np
import pandas as pd
import tifffile as tiff
import trackpy as tp

from matplotlib import rc
from microfilm import microplot
from scipy import ndimage as ndi
from scipy import signal
from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix
from scipy.stats import sigmaclip
from sklearn.cluster import DBSCAN

#fontsize: 20 is good for publications
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size':12})

@numba.jit(nopython=True, nogil=True, cache=False)
def local_maxima(frame, box):
    """ Finds pixels with maximum value within a region of interest """
    Y, X = frame.shape
    maxima_map = np.zeros(frame.shape, np.uint8)
    box_half = int(box / 2)
    box_half_1 = box_half + 1
    for i in range(box_half, Y - box_half_1):
        for j in range(box_half, X - box_half_1):
            local_frame = frame[
                i - box_half: i + box_half + 1,
                j - box_half: j + box_half + 1,
            ]
            flat_max = np.argmax(local_frame)
            i_local_max = int(flat_max / box)
            j_local_max = int(flat_max % box)
            if (i_local_max == box_half) and (j_local_max == box_half):
                maxima_map[i, j] = 1
    y, x = np.where(maxima_map)
    return y, x


@numba.jit(nopython=True, nogil=True, cache=False)
def gradient_at(frame, y, x, i):
    gy = frame[y + 1, x] - frame[y - 1, x]
    gx = frame[y, x + 1] - frame[y, x - 1]
    return gy, gx


@numba.jit(nopython=True, nogil=True, cache=False)
def net_gradient(frame, y, x, box, uy, ux):
    box_half = int(box / 2)
    ng = np.zeros(len(x), dtype=np.float32)
    for i, (yi, xi) in enumerate(zip(y, x)):
        for k_index, k in enumerate(range(yi - box_half, yi + box_half + 1)):
            for l_index, m in enumerate(
                range(xi - box_half, xi + box_half + 1)
            ):
                if not (k == yi and m == xi):
                    gy, gx = gradient_at(frame, k, m, i)
                    ng[i] += (
                        gy * uy[k_index, l_index] + gx * ux[k_index, l_index]
                    )
    return ng


@numba.jit(nopython=True, nogil=True, cache=False)
def identify_in_image(image, minimum_ng, box):
    """Identify local maxima in an image.

    Args:
        image (np.ndarray): The image to search for local maxima in.
        minimum_ng (float): The minimum gradient required for a pixel to be considered a local maximum.
        box (int): The size of the box to use for computing the gradients.

    Returns:
        A tuple of three arrays: the y and x coordinates of the local maxima, and their corresponding gradients.
    """
    y, x = local_maxima(image, box)
    box_half = int(box / 2)
    # Now comes basically a meshgrid
    ux = np.zeros((box, box), dtype=np.float32)
    uy = np.zeros((box, box), dtype=np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = np.sqrt(ux ** 2 + uy ** 2)
    ux /= unorm
    uy /= unorm
    ng = net_gradient(image, y, x, box, uy, ux)
    positives = ng > minimum_ng
    y = y[positives]
    x = x[positives]
    ng = ng[positives]
    return y, x, ng


def identify_in_frame(frame, minimum_ng, box, roi=None, transform_fnc=None):
    if roi is not None:
        frame = frame[roi[0][0]: roi[1][0], roi[0][1]: roi[1][1]]
    if transform_fnc is not None:
        frame = transform_fnc(frame)
    image = np.float32(frame)  # otherwise numba goes crazy
    y, x, net_gradient = identify_in_image(image, minimum_ng, box)
    if roi is not None:
        y += roi[0][0]
        x += roi[0][1]
    return y, x, net_gradient


def identify_by_frame_number(movie, minimum_ng, box, frame_number, roi=None, transform_fnc=None):
    frame = movie[frame_number]
    y, x, net_gradient = identify_in_frame(frame, minimum_ng, box, roi, transform_fnc)
    frame = frame_number * np.ones(len(x))
    return np.rec.array(
        (frame, x, y, net_gradient),
        dtype=[("frame", "i"), ("x", "i"), ("y", "i"), ("net_gradient", "f4")],
        )
    
def identify(movie, minimum_ng, box, roi=None, transform_fnc=None):
    if len(movie.shape) > 2:
        identifications = [
                identify_by_frame_number(movie, minimum_ng, box, i, roi, transform_fnc) 
                for i in range(len(movie))
                ]
        return np.hstack(identifications).view(np.recarray)
    else:
        y, x, net_gradient = identify_in_frame(movie, minimum_ng, box, roi, transform_fnc)
        frame = np.zeros(len(x))
        return np.rec.array(
            (frame, x, y, net_gradient),
            dtype=[("frame", "i"), ("x", "i"), ("y", "i"), ("net_gradient", "f4")],
            )


def round_mask(y, x, dia, dim):
    """Create a round mask centered at a specific location in a 2D array.

    Args:
        y (int): The y coordinate of the center of the mask.
        x (int): The x coordinate of the center of the mask.
        dia (float): The diameter of the mask.
        dim (Tuple[int, int]): The dimensions of the 2D array to create the mask for.

    Returns:
        A 2D boolean array representing the round mask.
    """
    yind, xind = np.indices(dim)
    return np.sqrt((yind-y)**2 + (xind-x)**2) < dia/2


def get_inner_outer_mask(y, x, dia, shape):
    """Create two binary masks representing an inner and outer disk centered at a specific location in a 2D array.

    Args:
        y (int): The y coordinate of the center of the disks.
        x (int): The x coordinate of the center of the disks.
        dia (float): The diameter of the inner disk.
        shape (Tuple[int, int]): The dimensions of the 2D array to create the disks for.

    Returns:
        A tuple of two 2D boolean arrays, representing the inner and outer disks, respectively.
    """
    H, W = shape
    inner_disk = round_mask(y, x, dia, (H, W))
    disk = round_mask(y, x, dia*2, (H, W))
    outer_disk = np.bitwise_xor(inner_disk, disk)
    return inner_disk, outer_disk


def estimate_IOM(I, y, x, dia=4):
    """Estimate the integrated intensity IO and the background intensity 
    for a circular region of interest in a 2D image.

    Args:
        I (np.ndarray): A 2D numpy array representing the image.
        y (int): The y coordinate of the center of the circular region of interest.
        x (int): The x coordinate of the center of the circular region of interest.
        dia (int): The diameter of the circular region of interest. Default is 4.

    Returns:
        A tuple of two float values, representing the estimated IOD and the estimated background intensity, respectively.
    """
    H, W = I.shape
    inner_disk, outer_disk = get_inner_outer_mask(y, x, dia, (H, W))

    # See https://www.pnas.org/content/117/1/60 for reference
    M = np.median(sigmaclip(I[outer_disk], low=2, high=2)[0])

    return np.sum(I[inner_disk]-M), M


def estimate_IOM_movie(movie, y, x, dia=4):
    """Estimate the integrated intensity IO and the background intensity for a circular region of interest in a movie.

    Args:
        movie (np.ndarray): A 3D numpy array representing the movie (T, H, W).
        y (int): The y coordinate of the center of the circular region of interest.
        x (int): The x coordinate of the circular region of interest.
        dia (int): The diameter of the circular region of interest. Default is 4.

    Returns:
        A tuple of two numpy arrays, representing the estimated IOD and the estimated background intensity, respectively.
    """
    _, H, W = movie.shape
    inner_disk, outer_disk = get_inner_outer_mask(y, x, dia, (H, W))

    # See https://www.pnas.org/content/117/1/60 for reference
    M = np.array([np.median(sigmaclip(img, low=2, high=2)[0]) for img in movie[:, outer_disk]])

    return np.sum(movie[:, inner_disk]-M[:,None], axis=-1), M


def estimate_IOMs_movie(movie, y, x, dia=4, box=31):
    """Estimate the integrated intensity IO and the background intensities for circular regions of interest in a movie.

    Args:
        movie (np.ndarray): A 3D numpy array representing the movie.
        y (np.ndarray): A 1D numpy array containing the y coordinates of the centers of the circular regions of interest.
        x (np.ndarray): A 1D numpy array containing the x coordinates of the circular regions of interest.
        dia (int): The diameter of the circular region of interest. Default is 4.
        box (int): The size of the square box centered at each (x, y) point to extract the region of interest. Default is 31.

    Returns:
        A 2D numpy array of shape (T, N), where T is the number of frames in the movie and N is the number of circular regions of interest. 
        Each element (t, n) of the array represents the estimated IO for the n-th circular region of interest in the t-th frame of the movie.
    """
    T, _, _ = movie.shape
    I0s = np.zeros((T, len(x)))
    for n, (x, y) in enumerate(zip(x, y)):
        region = get_region(movie, x, y, box)
        I0, _ = estimate_IOM_movie(region, box//2, box//2, dia=dia)
        I0s[:, n] = I0
    return I0s


def estimate_IOM_for_spots(movie, spots, dia=4, box=31):
    I0s = np.zeros(len(spots))
    for n, (t, x, y) in enumerate(zip(spots.frame, spots.x, spots.y)):
        region = get_region(movie, x, y, box, t)
        I0, _ = estimate_IOM(region, box//2, box//2, dia=dia)
        I0s[n] = I0
    return I0s


def spots_on_frame(spots, movie, frame=0, roi=None, clusters=False):
    T, _, _ = movie.shape
    
    frame = frame if frame < T else -1

    test_spots = spots[spots.frame==frame]
    vmin, vmax = np.percentile(movie[frame], (1, 99.9))
    plt.figure(figsize=(4,4), dpi=150)
    plt.imshow(
        movie[frame], 
        cmap=plt.cm.gray, 
        interpolation='None', 
        vmin=vmin,
        vmax=vmax)
    plt.scatter(test_spots.x, test_spots.y, s=15, facecolors='none', edgecolors='b')
    plt.scatter(test_spots.x_sub, test_spots.y_sub, marker='x', s=15, c='r')

    if roi is not None:
        y1, x1 = roi[0]
        y2, x2 = roi[1]
        rect = patches.Rectangle((x1, y1), width=x2-x1, height=y2-y1, linewidth=1, edgecolor='g', facecolor='none')
        plt.gca().add_patch(rect)

    if clusters:
        dx = spots.groupby("cluster").x.apply(lambda i: i.max() - i.min())
        dy = spots.groupby("cluster").y.apply(lambda i: i.max() - i.min())
        xm = spots.groupby("cluster").x.mean()
        ym = spots.groupby("cluster").y.mean()
        for y, x, h, w in zip(ym, xm, dy, dx):
            rect = patches.Rectangle((x-w/2, y-h/2), width=w, height=h, linewidth=0.5, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
    plt.show()

def rs(psf_image):
    """
    Localize the center of a PSF using the radial 
    symmetry method.
    Originally conceived by the criminally underrated
    Parasarathy R Nature Methods 9, pgs 724–726 (2012).
    args
    ----
        psf_image : 2D ndarray, PSF subwindow
    returns
    -------
        float y estimate, float x estimate
    """
    # Get the size of the image frame and build
    # a set of pixel indices to match
    N, M = psf_image.shape
    N_half = N // 2
    M_half = M // 2
    ym, xm = np.mgrid[:N-1, :M-1]
    ym = ym - N_half + 0.5
    xm = xm - M_half + 0.5 
    
    # Calculate the diagonal gradients of intensities across each
    # corner of 4 pixels
    dI_du = psf_image[:N-1, 1:] - psf_image[1:, :M-1]
    dI_dv = psf_image[:N-1, :M-1] - psf_image[1:, 1:]
    
    # Smooth the image to reduce the effect of noise, at the cost
    # of a little resolution
    fdu = ndi.uniform_filter(dI_du, 3)
    fdv = ndi.uniform_filter(dI_dv, 3)
    #fdu = dI_du
    #fdv = dI_dv
    
    dI2 = (fdu ** 2) + (fdv ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = -(fdv + fdu) / (fdu - fdv)
        
    # For pixel values that blow up, instead set them to a very
    # high float
    m[np.isinf(m)] = 9e9
    
    b = ym - m * xm

    sdI2 = dI2.sum()
    ycentroid = (dI2 * ym).sum() / sdI2
    xcentroid = (dI2 * xm).sum() / sdI2
    w = dI2 / np.sqrt((xm - xcentroid)**2 + (ym - ycentroid)**2)

    # Correct nan / inf values
    w[np.isnan(m)] = 0
    b[np.isnan(m)] = 0
    m[np.isnan(m)] = 0

    # Least-squares analytical solution to the point of 
    # maximum radial symmetry, given the slopes at each
    # edge of 4 pixels
    wm2p1 = w / ((m**2) + 1)
    sw = wm2p1.sum()
    smmw = ((m**2) * wm2p1).sum()
    smw = (m * wm2p1).sum()
    smbw = (m * b * wm2p1).sum()
    sbw = (b * wm2p1).sum()
    det = (smw ** 2) - (smmw * sw)
    xc = (smbw*sw - smw*sbw)/det
    yc = (smbw*smw - smmw*sbw)/det

    # Adjust coordinates so that they're relative to the
    # edge of the image frame
    yc = (yc + (N + 1) / 2.0) - 1
    xc = (xc + (M + 1) / 2.0) - 1

    return yc, xc


def get_subpixels_in_frame(frame, spots, box=9):
    """
    Given a 2D array `frame` and a Pandas DataFrame `spots` containing x and y coordinates, this function extracts a 
    square region of size `box` centered at each spot and returns the subpixel coordinates of the spot in the 
    corresponding region. The subpixel coordinates are relative to the top-left corner of the region and are returned 
    as a 2xN array, where N is the number of spots.
    
    Parameters
    ----------
    frame : numpy.ndarray
        A 2D numpy array representing the image frame.
    spots : pandas.DataFrame
        A Pandas DataFrame containing x and y columns representing the spot coordinates.
    box : int, optional
        An integer representing the size of the square region to extract around each spot. Default is 9.
    
    Returns
    -------
    numpy.ndarray
        A 2xN numpy array containing the subpixel coordinates of each spot in the corresponding region.
    """
    H, W 
    H, W = frame.shape
    subpixels = np.zeros((2,len(spots)), dtype=np.float_)
    indexer = box // 2
    for n, (x, y) in enumerate(zip(spots.x, spots.y)):
        xl = x-indexer if x-indexer >= 0 else 0
        xh = x+indexer if x+indexer < W else W
        yl = y-indexer if y-indexer >= 0 else 0
        yh = y+indexer if y+indexer < H else H-1

        s_i = frame[yl:yh, xl:xh]
        yc, xc = rs(s_i)
        subpixels[0, n] = yc + yl + 1
        subpixels[1, n] = xc + xl + 1
    return subpixels


def get_subpixels(movie, spots, box=9):
    """
    Extracts subpixels of spots in a movie.

    Args:
        movie (np.ndarray): A 3D array representing a movie with dimensions (frames, height, width).
        spots (pd.DataFrame): A pandas DataFrame with columns for frame, x, and y positions.
        box (int): An integer specifying the size of the box used to extract subpixels. Default is 9.

    Returns:
        np.ndarray: A 2D array with dimensions (2, N) where N is the number of spots. The first row contains the
        y-coordinates of the subpixels and the second row contains the x-coordinates of the subpixels.
    """
    _, H,
    _, H, W = movie.shape
    subpixels = np.zeros((2,len(spots)), dtype=np.float_)
    indexer = box // 2
    for n, (t, x, y) in enumerate(zip(spots.frame, spots.x, spots.y)):
        xl = x-indexer if x-indexer >= 0 else 0
        xh = x+indexer if x+indexer < W else W
        yl = y-indexer if y-indexer >= 0 else 0
        yh = y+indexer if y+indexer < H else H-1

        s_i = movie[t, yl:yh, xl:xh]
        yc, xc = rs(s_i)
        subpixels[0, n] = yc + yl + 1
        subpixels[1, n] = xc + xl + 1
    return subpixels


def get_region(movie, x, y, box, t=None):
    """
    Returns a region of interest (ROI) from a given movie at the specified (x, y) location.
    
    Args:
        movie (numpy.ndarray): A 3D numpy array representing the movie.
        x (float): The x-coordinate of the center of the ROI.
        y (float): The y-coordinate of the center of the ROI.
        box (int): The length of the side of the square ROI in pixels.
        t (int, optional): The index of the frame to return the ROI for. If None, returns ROIs for all frames. Defaults to None.
    
    Returns:
        Union[numpy.ndarray, Tuple[int, numpy.ndarray]]: If t is None, returns a 3D numpy array representing the ROI for all frames. 
                                                            Otherwise, returns a tuple of the frame index and a 2D numpy array representing the ROI for that frame.
    """
    x, y = int(np.rint(x)), int(np.rint(y))
    _, H, W = movie.shape
    indexer = box // 2
    xl = x-indexer if x-indexer >= 0 else 0
    xh = x+indexer if x+indexer < W else W
    yl = y-indexer if y-indexer >= 0 else 0
    yh = y+indexer if y+indexer < H else H-1
    if t is None:
        return movie[:, yl:yh, xl:xh]
    return movie[t, yl:yh, xl:xh]


def colocalize(target, reference, max_dist=1.5):
    """
    Colocalizes spots between two images.

    Args:
        target (ndarray): A 2D numpy array with shape (n, 2) containing n spots with
            their respective x and y coordinates.
        reference (ndarray): A 2D numpy array with shape (m, 2) containing m spots with
            their respective x and y coordinates.
        max_dist (float): The maximum distance allowed for a spot in the target image to
            be considered as colocalized with a spot in the reference image. Default is 1.5.

    Returns:
        positive (ndarray): A 2D numpy array with shape (k, 2) containing k spots from the 
            target image that were colocalized with spots from the reference image.
        negative (ndarray): A 2D numpy array with shape (l, 2) containing l spots from the 
            reference image that were not colocalized with any spot from the target image.
        expression_spots (ndarray): A 1D numpy boolean array with shape (n,) containing True 
            for spots in the target image that were colocalized with spots in the reference 
            image, and False otherwise.
    """

    Nref = len(reference)
    new_spots = np.zeros((Nref, 3))
    expression_spots = np.zeros(len(target), dtype=np.bool_)

    for i, (xi, yi) in enumerate(reference):
        dist = []
        for _, (xj, yj) in enumerate(target):
            dist.append( np.sqrt((xi-xj)**2 + (yi-yj)**2) )
        min_dist_idx = np.argmin(dist)
        if dist[min_dist_idx] > max_dist:
            new_spots[i, 0] = xi
            new_spots[i, 1] = yi
        else:
            new_spots[i, 0] = target[min_dist_idx, 0].copy()
            new_spots[i, 1] = target[min_dist_idx, 1].copy()
            new_spots[i, 2] = 1
            expression_spots[min_dist_idx] = True

    positive = new_spots[new_spots[:,-1]==1,:2]
    negative = new_spots[new_spots[:,-1]==0,:2]
    return positive, negative, expression_spots 


def pandas_colocalize(expr_spots, dna_positions, threshold=4):
    """
    expr_spots: Pandas dataframe with columns x_sub, y_sub, and cluster
    dna_positions: dim(N, 2) with col1: x - col2: y
    """
    expr_spots = expr_spots.copy()
    expr_spots["colocalized"] = np.zeros(len(expr_spots), dtype=np.bool_)

    # In case the list for dna_positions is empty, all expr_spots do not colocalize
    if len(dna_positions) == 0:
        return expr_spots

    cluster_nos = np.unique(expr_spots.cluster)
    for cluster_no in cluster_nos:
        idx = expr_spots.cluster == cluster_no
        _spots = expr_spots[idx]
        xm, ym = _spots.x_sub.mean(), _spots.y_sub.mean()
        dist = np.sqrt((xm - dna_positions[:,0])**2 + (ym - dna_positions[:,1])**2 )
        min_dist_idx = np.argmin(dist)
        if dist[min_dist_idx] < threshold:
            expr_spots.colocalized[idx] = True
    return expr_spots


def label_cluster(cluster):
    all_trajs = np.sort(np.asarray(list(set(cluster.trajectory))))
    labels = []
    for traj in cluster.trajectory:
        labels.append(np.where(traj==all_trajs)[0][0])
    return labels


def get_data_from_hdf(path):
    """
    Read and return data stored in an HDF5 file.

    Args:
        path (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary containing the data stored in the HDF5 file.
    """
    with h5py.File(path, "r") as f:
        # Get the data
        data = {}
        for key in f.keys():
            data[key] = f[key][:]
    return data


def make_montage(region, no_images, save_path=None):
    if not isinstance(region, (list, tuple)):
        regions = (region,)
    else:
        regions = region

    width = 1.5
    height = 1.5
    plt.figure(figsize=(width*no_images, height*len(regions)))
    for j, _region in enumerate(regions):
        T, _, _ = _region.shape
        _vmin, _vmax = np.percentile(_region, (1, 99.9))
        for i, time in enumerate(np.linspace(5, T, no_images, dtype=int, endpoint=False)):
            plt.subplot2grid((len(regions), no_images), (j, i))
            plt.imshow(
                _region[time],
                interpolation='None',
                cmap=plt.cm.gray,
                vmin=_vmin,
                vmax=_vmax)
            plt.xticks([])
            plt.yticks([])
            plt.box(False)
        #plt.tight_layout()
    plt.subplots_adjust(wspace=0.075, hspace=0.075)
    if save_path is not None:
        plt.savefig(save_path, dpi=250)
    plt.show()

if __name__=="__main__":
    # Get movie
    PATH_TO_MOVIE = "{path-to-movie-file}-driftCorrected-ch2.tif"

    # If you have a good image for DNA localization (improved by manually picking the well-focused images and maybe averaged to reduce noise)
    PATH_TO_DNA_IMAGE = "image-of-dna-spots.tif"
    # otherwise define as None
    PATH_TO_DNA_IMAGE = None

    # If PATH_TO_DNA_IMAGE = None, the script will take the DNA channel and check spots from an average image 
    # taken within the time period given here
    TEMP_REGION_FOR_DNA = (7, 20)

    # I assume the first frame is DNA. If not REVERSE_CHANNEL = True
    REVERSE_CHANNEL = False

    # Define the dt between frames of the same color
    SECONDS_PER_FRAME = 4 #seconds
    minutes_per_frame = SECONDS_PER_FRAME/60

    # Output folder for figures
    OUTPUT_FOLDER = "fig-output"

    # Output folder and file name for data
    DATA_OUTPUT_FOLDER = "data-output"
    TRAJECTORIES_FILE_NAME = 'intensity_measures.hdf5'

    ##############################################
    # From here everything is done automatically #
    ##############################################
    movie = tiff.imread(PATH_TO_MOVIE)
    T, H, W = movie.shape

    PARENT_DIR, FILE_NAME = os.path.split(PATH_TO_MOVIE)
    FILE_NAME = FILE_NAME.split(".")[0]

    new_folder = os.path.join(PARENT_DIR, OUTPUT_FOLDER)
    if not os.path.isdir(new_folder):
        os.mkdir(new_folder)

    green_PATH_TO_MOVIE = PATH_TO_MOVIE.replace('ch2', 'ch1')
    green_movie = tiff.imread(green_PATH_TO_MOVIE)

    if REVERSE_CHANNEL:
        _tmp = movie.copy()
        movie = green_movie.copy()
        green_movie = _tmp.copy()

    # Define roi where no fiducial marker is found
    gap = 100
    y1, x1 = gap, gap
    y2, x2 = H-gap, W-gap
    roi = ((y1, x1), (y2, x2))

    spots = pd.DataFrame(identify(movie, 20000, 7, roi=roi))

    sub = get_subpixels(movie, spots, box=9)
    spots["x_sub"] = sub[1,:]
    spots["y_sub"] = sub[0,:]

    green_image = green_movie[TEMP_REGION_FOR_DNA[0]:TEMP_REGION_FOR_DNA[1]].mean(axis=0)
    if PATH_TO_DNA_IMAGE is not None:
        green_image = tiff.imread(PATH_TO_DNA_IMAGE)

    # Standard => 20000
    dna_spots = pd.DataFrame(identify(green_image, 5000, 7, roi=roi))

    dna_locs_sub = get_subpixels_in_frame(green_image, dna_spots, box=9)
    dna_spots["x"] = dna_locs_sub[1,:]
    dna_spots["y"] = dna_locs_sub[0,:]

    dna_spots.to_csv(os.path.join(PARENT_DIR, "dna-spots.csv"))

    # Typical eps => 2.6
    linked = tp.link(
        spots, 
        search_range=2.6, 
        pos_columns=["x_sub", "y_sub"], 
        memory=8)

    # Filter stubs (Typical length threshold = 10)
    linked = linked.groupby("particle").filter(lambda x: len(x) > 10)

    # Typical eps => 0.25, min_samples=20
    X = np.vstack((linked.x_sub, linked.y_sub)).T
    clusters = DBSCAN(eps=0.25
                    , min_samples=20).fit(X)
    linked["cluster"] = clusters.labels_
    masked_spots = linked[linked.cluster >= 0]

    print("\nParent directory: {0}".format(PARENT_DIR))
    print("File: {0}\n".format(FILE_NAME))
    print("{0:0.1f}% spots do not go into a cluster".format((clusters.labels_<0).sum()/len(clusters.labels_)*100))
    print("No. of clusters: {0} and trajectories: {1}".format(np.sum(np.unique(clusters.labels_)>=0), len(np.unique(linked.particle))))

    masked_spots["intensity"] = estimate_IOM_for_spots(movie, masked_spots, dia=3)

    spots_on_frame(masked_spots, movie, frame=-1, roi=roi, clusters=True)

    # Generate trajectories within clusters (saved in "trajectory" integer)
    masked_spots = masked_spots.rename(columns={"particle": "trajectory"})

    # Colocalize expression spots with dna spots (saved in "colocalized" boolean)
    reference = np.asarray([dna_spots.x, dna_spots.y]).T
    expr_spots = pandas_colocalize(masked_spots, reference, 4)

    save_properties = expr_spots.copy()
    save_properties = save_properties.rename(columns={"x":"x_int", "y":"y_int", "x_sub":"x", "y_sub":"y", "trajectory": "linked"})#, "trajectory": "traj"})
    save_properties = save_properties.rename(columns={"cluster": "trajectory"})#, "trajectory": "traj"})
    save_properties.to_csv(os.path.join(PARENT_DIR, FILE_NAME + "-trajs.csv"))

    ###################################
    # Plot expression spots on images # 
    ###################################

    plt.figure(figsize=(8,6))
    plt.subplot2grid((3,4), (0, 0), colspan=3, rowspan=3)

    _coloc = np.sum(expr_spots.groupby('cluster').colocalized.any())
    _all_clusters = len(expr_spots.groupby('cluster'))

    _vmin, _vmax = np.percentile(green_image, (3, 99.9))
    plt.imshow(green_image, cmap='gray_r', interpolation='None', vmin=_vmin, vmax=_vmax)
    plt.title("{0:0.0f}% ({1}/{2})\nprotein clusters found on DNA".format(_coloc/_all_clusters*100, _coloc, _all_clusters))
    plt.scatter(
        dna_spots.x, 
        dna_spots.y, 
        label='DNA spots', 
        color='k',
        facecolors='None')
    plt.scatter(
        expr_spots[expr_spots.colocalized==False].groupby("cluster").x_sub.mean(), 
        expr_spots[expr_spots.colocalized==False].groupby("cluster").y_sub.mean(), 
        marker="s", 
        facecolors='none', 
        edgecolor='w',
        label='HT: Not colocalized',
        alpha=0.5)
    plt.scatter(
        expr_spots[expr_spots.colocalized==True].groupby("cluster").x_sub.mean(), 
        expr_spots[expr_spots.colocalized==True].groupby("cluster").y_sub.mean(), 
        label='HT: Colocalized', 
        marker="*",
        facecolors='none', 
        color='#FF0002')
    plt.axis('equal')
    plt.legend(loc='best')
    plt.xlabel('X-position (pixel)')
    plt.ylabel('Y-position (pixel)')
    plt.subplot2grid((3,4), (0, 3))
    nr_spots = expr_spots.groupby("frame").size()
    plt.plot(nr_spots.index*minutes_per_frame, nr_spots, "k-", linewidth=1)
    plt.xlabel('Time (min)')
    plt.ylabel('#Spots')
    plt.tick_params(axis='both', which='both', direction='in', bottom=1, top=1, left=1, right=1)
    plt.subplot2grid((3,4), (1, 3))
    _example = np.unique(expr_spots.cluster)[-1]
    _example_cluster = expr_spots[expr_spots.cluster == _example].sort_values("frame")
    plt.plot(_example_cluster.x_sub, _example_cluster.y_sub, 'k-', alpha=0.5)
    _labels = np.array(label_cluster(_example_cluster))
    _labels = _labels/_labels.max() if _labels.max() > 0 else _labels
    plt.scatter(_example_cluster.x_sub, _example_cluster.y_sub, color=plt.cm.rainbow(_labels))
    plt.axis("equal")
    _wind = 1
    plt.ylim([_example_cluster.y_sub.min()-_wind, _example_cluster.y_sub.max()+_wind])
    plt.xlim([_example_cluster.x_sub.min()-_wind, _example_cluster.x_sub.max()+_wind])
    plt.xlabel('X-position (pixel)')
    plt.ylabel('Y-position (pixel)')
    plt.tick_params(axis='both', which='both', direction='in', bottom=1, top=1, left=1, right=1)
    plt.subplot2grid((3,4), (2, 3))
    _example = np.unique(expr_spots.cluster)[1]
    _example_cluster = expr_spots[expr_spots.cluster == _example].sort_values("frame")
    plt.plot(_example_cluster.x_sub, _example_cluster.y_sub, 'k-', alpha=0.5)
    _labels = np.array(label_cluster(_example_cluster))
    _labels = _labels/_labels.max() if _labels.max() > 0 else _labels
    plt.scatter(_example_cluster.x_sub, _example_cluster.y_sub, color=plt.cm.rainbow(_labels))
    plt.axis("equal")
    _wind = 1
    plt.ylim([_example_cluster.y_sub.min()-_wind, _example_cluster.y_sub.max()+_wind])
    plt.xlim([_example_cluster.x_sub.min()-_wind, _example_cluster.x_sub.max()+_wind])
    plt.xlabel('X-position (pixel)')
    plt.ylabel('Y-position (pixel)')
    plt.tick_params(axis='both', which='both', direction='in', bottom=1, top=1, left=1, right=1)
    plt.tight_layout()
    plt.savefig(os.path.join(PARENT_DIR, OUTPUT_FOLDER, "localization-statistics.pdf"), dpi=250)
    plt.show()

    ####################################
    # Plot double color overlap images # 
    ####################################

    _width = 50
    interest_spots = expr_spots[expr_spots.colocalized==True]

    _counts, _frame = np.histogram(interest_spots.frame, bins=np.arange(0, T, 1))
    if np.any(_counts>0):
        max_coloc_frame = _frame[np.argmax(_counts)]
        max_spots = interest_spots[interest_spots.frame == max_coloc_frame]
        x_center, y_center = int(max_spots.x_sub.mean()), int(max_spots.y_sub.mean())

        hl, hh = y_center-_width, y_center+_width
        wl, wh = x_center-_width, x_center+_width
        _green = green_movie[3:25, hl:hh, wl:wh].mean(axis=0).astype(np.uint16)
        _red = movie[max_coloc_frame, hl:hh, wl:wh]

        if not REVERSE_CHANNEL:
            microim1 = microplot.microshow(images=[_green,], cmaps=['pure_cyan'])
            plt.close()
            microim2 = microplot.microshow(images=[_red,], cmaps=['pure_magenta'])
            plt.close()
            microim3 = microplot.microshow(images=[_red,_green], cmaps=['pure_magenta', 'pure_cyan'])
            plt.close()
            micropanel = microplot.Micropanel(rows=1, cols=3, figsize=[8,3])
            micropanel.add_element(pos=[0,0], microim=microim1)
            micropanel.add_element(pos=[0,1], microim=microim2)
            micropanel.add_element(pos=[0,2], microim=microim3)
        else:
            microim1 = microplot.microshow(images=[_red,], cmaps=['pure_cyan'])
            plt.close()
            microim2 = microplot.microshow(images=[_green,], cmaps=['pure_magenta'])
            plt.close()
            microim3 = microplot.microshow(images=[_green,_red], cmaps=['pure_magenta', 'pure_cyan'])
            plt.close()
            micropanel = microplot.Micropanel(rows=1, cols=3, figsize=[8,3])
            micropanel.add_element(pos=[0,0], microim=microim1)
            micropanel.add_element(pos=[0,1], microim=microim2)
            micropanel.add_element(pos=[0,2], microim=microim3)

        micropanel.savefig(os.path.join(PARENT_DIR, OUTPUT_FOLDER, "localization-overlay.pdf"), bbox_inches='tight', pad_inches=0, dpi=250)

    ####################################################
    # Measure signal in centered sub-region to
    # reduce effects from inhomogenous TIRF excitation
    ####################################################

    interest_spots = expr_spots.copy()

    radius = 80
    xcenter, ycenter = interest_spots.x.mean(), interest_spots.y.mean()
    idx = np.sqrt((interest_spots.x-xcenter)**2 + (interest_spots.y-ycenter)**2) < radius
    interest_spots = interest_spots[idx]

    # Remove spots that are very close for intensity measurements
    x_mean = interest_spots.groupby("cluster").x_sub.mean()
    y_mean = interest_spots.groupby("cluster").y_sub.mean()

    xy_mean = np.vstack((x_mean, y_mean)).T
    dm = distance_matrix(xy_mean, xy_mean)
    dm[np.identity(len(dm), dtype=np.bool_)] = 1e4
    idx = ~np.any(dm < 3, axis=1)
    masked_clusters = x_mean.index[idx]
    interest_spots = interest_spots[interest_spots.cluster.isin(masked_clusters)]

    spots_on_frame(interest_spots, movie, frame=-1, clusters=True)

    x_mean = interest_spots.groupby("cluster").x_sub.mean().to_numpy()
    y_mean = interest_spots.groupby("cluster").y_sub.mean().to_numpy()
    time_start = interest_spots.groupby("cluster").frame.min().to_numpy()

    all = estimate_IOMs_movie(movie, y_mean, x_mean, dia=3)

    normed_all = (all / all[:10,:].mean(axis=0)[None, :])-1
    normed_all = all
    filtered = signal.medfilt(normed_all, kernel_size=(5,1))

    _xmin, _xmax = -0.5e4, 0.5e5

    plt.figure(figsize=(7, 7))
    plt.subplot(321)
    plt.plot(filtered[:, :20], '-', alpha=0.8)
    plt.xlabel("Time (frame)")
    plt.ylabel("Normed intensity (a.u.)")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.ylim([filtered.min()*0.8, filtered.max()*1.2])
    plt.subplot(322)
    bins = np.linspace(_xmin, _xmax, 35)
    density, bins = np.histogram(filtered.flatten(), bins=bins, density=True)
    density, bins = np.histogram(np.concatenate([f[i:] for f, i in zip(filtered.T, time_start)]), bins=bins, density=True)
    unity_density = density / density.sum()
    plt.hist(
        bins[:-1], 
        bins, 
        weights=unity_density,
        color="r",
        log=0,
        label="-DNA")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylim([0, 0.55])
    plt.xlabel("Pooled signals (a.u.)")
    plt.ylabel("Probability")
    plt.tick_params(axis='both', which='both', direction='in', bottom=1, top=1, left=1, right=1)

    plt.subplot(323)
    bins = np.linspace(_xmin, _xmax, 60)
    density, bins = np.histogram(interest_spots[interest_spots.colocalized==False].intensity, bins=bins, density=True)
    unity_density = density / density.sum()
    plt.hist(
        bins[:-1], 
        bins, 
        weights=unity_density,
        color="r",
        log=0,
        zorder=1,
        label="-DNA")

    density, bins = np.histogram(interest_spots[interest_spots.colocalized==True].intensity, bins=bins, density=True)
    unity_density = density / density.sum()
    plt.hist(
        bins[:-1], 
        bins, 
        weights=unity_density,
        color="k",
        histtype='step',
        label='+DNA',
        zorder=0)
    plt.xlabel("Spot signals (a.u.)")
    plt.ylabel("Probability")
    plt.legend(loc='best', frameon=False)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    traj_max_coloc = interest_spots[interest_spots.colocalized==True].groupby("trajectory").intensity.max()
    traj_max_nocoloc = interest_spots[interest_spots.colocalized==False].groupby("trajectory").intensity.max()

    plt.subplot(324)

    bins = np.logspace(np.log10(_xmin), np.log10(_xmax), 50)
    bins = np.linspace(_xmin, _xmax, 50)
    density, bins = np.histogram(np.concatenate([traj_max_nocoloc, traj_max_coloc]), bins=bins, density=True)
    unity_density = density / density.sum()
    plt.hist(
        bins[:-1], 
        bins, 
        weights=unity_density,
        color="r",
        log=0,
        alpha=1,
        label='+/- DNA',
        zorder=1)

    density, bins = np.histogram(traj_max_coloc, bins=bins, density=True)
    unity_density = density / density.sum()
    plt.hist(
        bins[:-1], 
        bins, 
        weights=unity_density,
        histtype='step',
        color="k",
        log=0,
        label='+ DNA',
        alpha=1,
        zorder=0)
    plt.legend(loc='best', frameon=False)
    plt.xlabel("Max. traj. signal (a.u.)")
    plt.ylabel("Probability")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #lt.xscale('log')
    plt.ylim([0, 0.4])

    traj_intensity = interest_spots.groupby("trajectory").intensity.mean().to_numpy()
    traj_x_mean = interest_spots.groupby("trajectory").x_sub.mean().to_numpy()
    traj_y_mean = interest_spots.groupby("trajectory").y_sub.mean().to_numpy()

    plt.subplot(325)
    dist = np.sqrt((traj_x_mean.mean() - traj_x_mean)**2 + (traj_y_mean.mean() - traj_y_mean)**2)
    plt.plot(dist, traj_intensity, 'k.', alpha=1)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Distance from center (pixel)")
    plt.ylabel("Mean traj. intensity (a.u.)")
    plt.ylim([_xmin, _xmax])
    plt.tight_layout()
    plt.savefig(os.path.join(PARENT_DIR, OUTPUT_FOLDER, "intensity-information.pdf"), dpi=250)
    plt.show()


    ################################################################
    # Plot individual spots with cluster and trajectory information
    # * Debugging
    # * Make montage or movie
    # * Data visualization
    ################################################################

    interest_spots = expr_spots.copy()
    with_spot_classfier_and_labels = False

    # Pick example trajectory
    picked_locations = [(353, 343), (387, 358), (379, 434), (436,404)]

    _ymin, _ymax = -0.5e4, 1.35e5

    no_plots = 10
    xy_samples = []
    for b in [True, False, None]:
        if b is True:
            name = 'wDNA'
        elif b is False:
            name = 'woDNA'

        cluster_id = np.unique(interest_spots[interest_spots.colocalized==b].cluster)
        if len(cluster_id) == 0 and (b is not None):
            continue

        idx = -1
        if no_plots < len(cluster_id):
            choices = np.random.choice(cluster_id, size=no_plots)
        else: 
            choices = cluster_id

        if b is None:
            name = 'manual'
            _x, _y = interest_spots.groupby('cluster').x_sub.mean(), interest_spots.groupby('cluster').y_sub.mean()
            choices = []
            for picked_location in picked_locations:
                choices.append( np.argmin(np.sqrt((picked_location[1]-_x)**2 + (picked_location[0]-_y)**2)) )

        if len(choices)<5:
            row_height = len(choices)*1.8
        else:
            row_height = len(choices)*1.5

        plt.figure(figsize=(10, row_height))
        plt.suptitle(name)
        for id in choices:
            cluster = interest_spots[interest_spots.cluster == id]

            cluster_start, cluster_end = cluster.frame.min(), cluster.frame.max()

            _x, _y = cluster.x_sub.mean(), cluster.y_sub.mean()
            xy_samples.append((_x, _y))
            _signal, _M = estimate_IOM_movie(movie, _y, _x, dia=3)
            traj_times = []
            for traj_id in np.unique(cluster.trajectory):
                _traj = cluster[cluster.trajectory == traj_id]
                traj_times.append((_traj.frame.min(), _traj.frame.max()))

            idx += 1
            plt.subplot2grid((len(choices), 5), (idx, 0), colspan=3)
            
            if with_spot_classfier_and_labels:
                plt.axvspan(cluster_start*minutes_per_frame, cluster_end*minutes_per_frame, alpha=0.2)
                [plt.axvspan(f*minutes_per_frame, e*minutes_per_frame, alpha=0.2, color='r') for f,e in traj_times]
                plt.plot(np.arange(len(_M))*minutes_per_frame, _M, 'k--', linewidth=1)
                plt.plot(np.arange(len(_M))*minutes_per_frame, _signal+_M, linewidth=1, color='k', label="y:{0:0.0f} x:{1:0.0f}".format(_y, _x))
                plt.legend(loc='best', frameon=False)
            else:
                plt.plot(np.arange(len(_M))*minutes_per_frame, _signal+_M, linewidth=1, color='k')
                plt.ylim([_ymin, _ymax])
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)

            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.xlabel("Time (min)")
            plt.ylabel("Intensity (a.u.)")
            plt.subplot2grid((len(choices), 5), (idx, 3))
            plt.hist(_signal, bins=25, density=True, orientation="horizontal")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

            plt.subplot2grid((len(choices), 5), (idx, 4))
            box = 31
            plt.title("y:{0:0.0f} x:{1:0.0f}\n({2})".format(_y, _x, id))
            plt.imshow(
                get_region(movie, _x, _y, box, t=cluster_start),
                interpolation='None',
                cmap=plt.cm.gray
                )
            plt.xticks([])
            plt.yticks([])
            plt.scatter(box//2, box//2, s=150, facecolors='none', edgecolors='r')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.6)
        plt.savefig(os.path.join(PARENT_DIR, OUTPUT_FOLDER, "HT9spots-" + name + ".pdf"), dpi=250)
        plt.show()

    save_region = False
    if save_region:
        visual_output_folder = "visuals"
        new_folder = os.path.join(PARENT_DIR, visual_output_folder)
        if not os.path.isdir(new_folder):
            os.mkdir(new_folder)

        sample_id = 6
        pick_x, pick_y = xy_samples[sample_id]
        protein_region = get_region(movie, pick_x, pick_y, box=31)
        dna_region = get_region(green_movie, pick_x, pick_y, box=31)

        tiff.imsave(os.path.join(PARENT_DIR, visual_output_folder, "ProteinRegion-y{0:0.0f}-x{1:0.0f}.tif".format(pick_y, pick_x)), protein_region)
        tiff.imsave(os.path.join(PARENT_DIR, visual_output_folder, "DNARegion-y{0:0.0f}-x{1:0.0f}.tif".format(pick_y, pick_x)), dna_region)

        make_montage(protein_region, 20, save_path=os.path.join(PARENT_DIR, visual_output_folder, "DNARegion-y{0:0.0f}-x{1:0.0f}.pdf".format(pick_y, pick_x)))

    ###############
    # DATA SAVING #
    ###############

    interest_spots = expr_spots.copy()

    radius = 90
    xcenter, ycenter = interest_spots.x.mean(), interest_spots.y.mean()
    idx = np.sqrt((interest_spots.x-xcenter)**2 + (interest_spots.y-ycenter)**2) < radius
    interest_spots = interest_spots[idx]

    _ymin, _ymax = -1.0e4, 14e4

    save_data = True
    if save_data:
        new_folder = os.path.join(PARENT_DIR, DATA_OUTPUT_FOLDER)
        if not os.path.isdir(new_folder):
            os.mkdir(new_folder)

        cluster_ids = np.unique(interest_spots.cluster)
        print("Saving {0} locations to hdf file << {2} >>\nin sub-folder: << {1} >> ".format(len(cluster_ids), DATA_OUTPUT_FOLDER, TRAJECTORIES_FILE_NAME))

        T, H, W = movie.shape
        _signal = np.zeros((T, len(cluster_ids)))
        _background = np.zeros_like(_signal)
        _x_position = np.zeros((1, len(cluster_ids)))
        _y_position = np.zeros((1, len(cluster_ids)))
        for i, cluster_id in enumerate(cluster_ids):
            cluster = interest_spots[interest_spots.cluster == cluster_id]
            _x, _y = cluster.x_sub.mean(), cluster.y_sub.mean()
            _S, _M = estimate_IOM_movie(movie, _y, _x, dia=3)
            _signal[:, i] = _S
            _background[:, i] = _M
            _x_position[:, i] = _x
            _y_position[:, i] = _y

        plt.figure(figsize=(7,2))
        _min, _max = np.min(_signal, axis=-1), np.max(_signal, axis=-1)
        plt.fill_between(np.arange(len(_min))*minutes_per_frame, _min, _max, color='k', alpha=0.25, edgecolor='none')
        for _sample in np.random.randint(0, len(_signal.T), size=1):
            plt.plot(np.arange(len(_min))*minutes_per_frame, _signal[:,_sample], linewidth=1, color='k')
        plt.xlabel("Time (min)")
        plt.ylabel("Intensity (a.u.)")
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.ylim([_ymin, _ymax])
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.tight_layout()
        plt.savefig(os.path.join(PARENT_DIR, OUTPUT_FOLDER, "average-trajectories-{0}.pdf".format(_sample)), dpi=250)
        plt.show()

        
        with h5py.File(os.path.join(PARENT_DIR, DATA_OUTPUT_FOLDER, TRAJECTORIES_FILE_NAME), 'w') as hf:
            hf.create_dataset("signal", data=_signal)
            hf.create_dataset("background", data=_background)
            hf.create_dataset("x-position", data=_x_position)
            hf.create_dataset("y-position", data=_y_position)

    ##############################
    # Get trajectory statistics
    ##############################

    interest_spots = expr_spots.copy()

    # Plot trajectory length (should be expontial due to slow folding rate of HT9)
    length = interest_spots.groupby('trajectory').frame.apply(lambda f: f.max() - f.min())
    traj_end = interest_spots.groupby('trajectory').frame.max()
    # Here also remove trajs that only start at the end of the movie; 
    # probably not very likely, but could happen for low TXTL rates (e.g. high rifampicin).

    # Adjust traj lengths for trajs that have not ended by the end of the experiment
    length[traj_end >= len(movie)-2] = np.nan

    # Remove short trajectories from analysis
    min_length = 4
    length = length[length>min_length]

    mono_exp = lambda x, a, tau: a*np.exp(-x/tau)

    plt.figure(figsize=(8,3.5))
    plt.subplot(121)
    plt.title("No. of traj. {0}".format(len(length)))
    bins = np.arange(min_length, 250, 2)
    bins = np.linspace(0, 250, 50)
    density, bins = np.histogram(length, bins=bins, density=True)
    unity_density = density/density.sum()
    p, l, _ = plt.hist(
        bins[:-1], 
        bins, 
        weights=unity_density,
        color="r", 
        log=0)
    mid = (l[1:]-l[:-1])/2 + l[:-1]
    popt, pcov = curve_fit(mono_exp, mid, p, p0=[1, 10])
    perr = np.sqrt(np.diag(pcov))
    plt.plot(mid, mono_exp(mid, *popt), '-k',label='Tau={0:0.1f} frames'.format(popt[-1]))
    plt.xlabel("Traj. length (frames)")
    plt.ylabel("Probability")
    plt.legend(loc='best', frameon=False)