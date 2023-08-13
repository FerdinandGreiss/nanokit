#!/usr/bin/env python
"""
Script to correct movies for lateral drift using bright fiducial markers
that are statically bound to the surface of a microfluidic chip.

We were using Micro-Manager for image acquisition. The movie can be provided as 
 1. List of images in a folder: img_channel000_position000_time000000085_z000.tif
 2. Tiff image stack with dimensions (T, H, W)

The script assumes that there are two colors and produces two tiff movies with
 1. {Movie Name}-driftCorrected-ch1.tif
 2. {Movie Name}-driftCorrected-ch2.tif
in the same folder as the given movie. In addition, it will save a plot 
and csv file for the fiducial marker drift along xy, tx, and ty.

Some code was adapted from the amazing quot package:
    https://github.com/alecheckert/quot

The script was tested on Mac and Windows. 

Author: Ferdinand Greiss, Weizmann Institute of Science, 2023
"""

import glob
import os
import re

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tifffile as tiff
from lmfit import Parameters, minimize
from scipy import ndimage as ndi
from scipy.ndimage import shift as nd_shift
from tqdm import tqdm


def label_spots(binary_img, intensity_img=None, mode="max"):
    """
    Find continuous nonzero objects in a binary image,
    returning the coordinates of the spots.

    If the objects are larger than a single pixel,
    then to find the central pixel do
        1. use the center of mass (if mode == 'centroid')
        2. use the brightest pixel (if mode == 'max')
        3. use the mean position of the binary spot
            (if img_int is not specified)

    args
    ----
        binary_img      :   2D ndarray (YX), dtype bool
        intensity_img   :   2D ndarray (YX)
        mode            :   str, 'max' or 'centroid'

    returns
    -------
        2D ndarray (n_spots, 2), dtype int64,
            the Y and X coordinate of each spot

    """
    # Find and label every nonzero object
    img_lab, n = ndi.label(binary_img, structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    index = np.arange(1, n + 1)

    # Find the centers of the spots
    if intensity_img is None:
        positions = np.asarray(ndi.center_of_mass(binary_img, 
            labels=img_lab, index=index))
    elif mode == "max":
        positions = np.asarray(ndi.maximum_position(intensity_img,
            labels=img_lab, index=index))
    elif mode == "centroid":
        positions = np.asarray(ndi.center_of_mass(intensity_img,
            labels=img_lab, index=index))

    return positions.astype("int64")

def threshold_image(I, t=200.0, return_filt=False, mode='max'):
    """
    Take all spots in an image above a threshold *t*
    and return their approximate centers.

    If *return_filt* is set, the function also returns the 
    raw image and binary image. This is useful as a 
    back door when writing GUIs.

    args
    ----
        I           :   2D ndarray, image
        t           :   float, threshold
        return_filt :   bool
        mode        :   str, either 'max' or 'centroid'

    returns
    -------
        If *return_filt* is set:
        (
            2D ndarray, same as I;
            2D ndarray, binary image;
            2D ndarray, shape (n_spots, 2), the spot
                coordinates
        )
        else
            2D ndarray of shape (n_spots, 2), 
                the spot coordinates

    """
    I_bin = I > t
    pos = label_spots(I_bin, intensity_img=I, mode=mode)
    if return_filt:
        return I, I_bin, pos 
    else:
        return pos

def min_max(I, w=9, t=200.0, mode='constant', return_filt=False,
    **kwargs):
    """
    Use the difference between the local maximum and local minimum
    in square subwindows to identify spots in an image.

    args
    ----
        I           :   2D ndarray
        w           :   int, window size for test
        t           :   float, threshold for spot detection
        mode        :   str, behavior at boundaries (see scipy.ndimage
                        documentation)
        kwargs      :   to ndimage.maximum_filter/minimum_filter

    returns
    -------
        if return_filt:
        (
            2D ndarray, the post-convolution image;
            2D ndarray, the thresholded binary image;
            2D ndarray, shape (n_spots, 2), the y and x 
                coordinates of each spot
        )
        else
            2D ndarray, shape (n_spots, 2), the y and x
                coordinates of each spot

    """
    size = (w, w)
    I_filt = I - ndi.minimum_filter(I, size=size, mode=mode, **kwargs)

    # Set the probability of detection near the border to zero
    hw = w//2
    I_filt[:hw,:] = t-1
    I_filt[:,:hw] = t-1
    I_filt[-hw:,:] = t-1
    I_filt[:,-hw:] = t-1

    # Threshold the result
    return threshold_image(I_filt, t=t, return_filt=return_filt)

def gauss2D(x, y, cen_x, cen_y, sig_x, sig_y, a, offset):
    """
    Define a 2D Gaussian function for fitting
    """
    return a*np.exp(-(((cen_x-x)/sig_x)**2 + ((cen_y-y)/sig_y)**2)/2.0)+offset

def residuals(parameter, x, y, data):
    """
    Define a residual function between an experimental PSF and the 2D Gaussian
    """
    cen_x = parameter["centroid_x"].value
    cen_y = parameter["centroid_y"].value
    sigma_x = parameter["sigma_x"].value
    sigma_y = parameter["sigma_y"].value
    offset = parameter["background"].value
    amp = parameter["amplitude"].value

    return (data - gauss2D(x, y, cen_x, cen_y, sigma_x, sigma_y, amp, offset))

def fit(spots, image, x, y, std=1.0, window=9):
    """
    Fit all spots in image to a 2D Gaussian function and return a DataFrame
    with super-resolution values.
    """
    H, W = image.shape

    params = Parameters()
    params.add("amplitude",value=None)
    params.add("centroid_x",value=None)
    params.add("centroid_y",value=None)
    params.add("sigma_x",value=None)
    params.add("sigma_y",value=None)
    params.add("background",value=None)

    fit_results = {
        'amplitude': [],
        'centroid_x': [],
        'centroid_y': [],
        'sigma_x': [],
        'sigma_y': [],
        'err_amplitude': [],
        'err_centroid_x': [],
        'err_centroid_y': [],
        'err_sigma_x': [],
        'err_sigma_y': []}

    for cen_y, cen_x in spots:
        if cen_y < window or cen_x < window or cen_y + window >= H or cen_x + window >= W:
            continue

        up = cen_y - window//2
        down = cen_y + window//2
        left = cen_x - window//2
        right = cen_x + window//2

        region = image[up:down, left:right]
        min_count, max_count = region.min(), region.max()
        amp = max_count - min_count
        offset = min_count

        local_x = x[up:down, left:right]
        local_y = y[up:down, left:right]

        params["amplitude"].value = amp
        params["centroid_x"].value = cen_x
        params["centroid_y"].value = cen_y
        params["sigma_x"].value = std
        params["sigma_y"].value = std
        params["background"].value = offset

        fit = minimize(residuals, params, args=(local_x, local_y, region), max_nfev=100)
        
        if fit.success:
            for key in fit.params.keys():
                if key in fit_results.keys():
                    fit_results[key].append(fit.params[key].value)
                    fit_results["err_" + key].append(fit.params[key].stderr)

    return pd.DataFrame(fit_results)

def get_image_properties_from_file(name):
    """
    Get the information from image file names (generated from Micro-Manager)
    and store in dictionary.

    Image file names look like this:
    > img_channel000_position000_time000000085_z000.tif
    """
    properties  = {}
    string = os.path.splitext(name)[0]
    for element in string.split("_"):
        key = re.findall('([a-zA-Z ]*)\d*.*', element)[0]
        if len(key) < len(element):
            value = element[len(key):]
            properties[key] = int(value)
    return properties

def tracking(last_spots, next_spots, max_distance=5):
    """
    Rather stupid tracking algorithm that just finds the nearest neighbor
    """
    if len(next_spots) == 0:
        d = pd.DataFrame().reindex(columns=last_spots.columns)
        return d

    flagged, masked_spots, ids = [], [], []
    for _, last_spot in last_spots.iterrows():
        y, x = last_spot.centroid_y, last_spot.centroid_x
        yn, xn = next_spots.centroid_y, next_spots.centroid_x
        dr = np.sqrt((y-yn)**2+(x-xn)**2)
        idx = np.argmin(dr)
        if dr.iloc[idx] < max_distance and idx not in flagged:
            flagged.append(idx)
            ids.append(last_spot.id)
            masked_spots.append(next_spots.iloc[idx])

    masked_spots = pd.DataFrame(masked_spots)
    masked_spots["id"] = ids
    return masked_spots

if __name__ == "__main__":
    # Give the path to the movie for correction
    # Channel 1 should be 488 (DNA)
    # Channel 2 should be 647 (proteins)
    import sys
    
    if len(sys.arv) > 1:
        file = sys.argv[1]
    else:
        file = "path-to-movie-file.tif"
    
    # The THRESHOLD parameter should be fine, unless no (decrease THRESHOLD) or
    # too many (increase THRESHOLD to ~0.9) fiducial marker(s) were found
    # 0.5 is usually good with tetraspeck beads
    THRESHOLD = 0.5

    # If you want to remove the first few images for some reason, you can 
    # define the starting point here
    starting_point = 0

    PARENT_DIR, FILE_NAME = os.path.split(file)

    if os.path.isdir(file):
        img_files = glob.glob(os.path.join(file, "*.tif"))
        img_files_sorted = sorted(img_files, key=lambda e: get_image_properties_from_file(os.path.split(e)[-1])["time"])
        image_stack = np.array([tiff.imread(i) for i in img_files_sorted])
    else:
        image_stack = tiff.imread(file)

    ch1 = image_stack[::2,].copy()
    ch2 = image_stack[1::2,].copy()

    if len(ch1) < len(ch2):
        ch2 = ch2[:-1]
    elif len(ch1) > len(ch2):
        ch1 = ch1[:-1]

    ch1 = ch1[starting_point:]
    ch2 = ch2[starting_point:]

    T, image_height, image_width = ch1.shape

    index_y = np.arange(0, image_height)
    index_x = np.arange(0, image_width)
    y, x = np.meshgrid(index_x, index_y)
    x_grid = y
    y_grid = x

    
    threshold = THRESHOLD*ch1[-1].max()
    window = 10
    sigma = 2.0

    fitting_results = None
    first_particles = None
    for i, image in enumerate(tqdm(ch1[::-1])):

        spots = min_max(image, w=window, t=threshold)
        sub_pixels = fit(
            spots, 
            image, 
            x_grid, 
            y_grid, 
            std=sigma,
            window=window)

        sub_pixels["time"] = i*np.ones(len(sub_pixels), dtype=np.uint16)

        if first_particles is None:
            if len(spots) > 0:
                first_particles = sub_pixels.copy()
                first_particles["id"] = np.arange(len(sub_pixels))

                fitting_results = first_particles.copy()
                tqdm.write("{0} fiducial marker(s) found. Tracking in progress:".format(len(sub_pixels)))
            else:
                tqdm.write("No fiducial marker found.")
                break
        else:
            if len(sub_pixels) == 0:
                continue

            lastest = fitting_results[fitting_results.time==fitting_results.time.max()]
            sub_pixels = tracking(lastest, sub_pixels, max_distance=2)
            fitting_results = fitting_results.append(sub_pixels, ignore_index=True)

    fitting_results["time"] = len(ch1) - 1 - fitting_results["time"]
    fitting_results.to_csv(os.path.join(PARENT_DIR, "fiducials.csv"))

    drift_x, drift_y, drift_t = [], [], []
    for n in np.unique(fitting_results["id"]):
        fiducial = fitting_results[fitting_results["id"] == n]
        x, y = fiducial.centroid_x.to_numpy(), fiducial.centroid_y.to_numpy()
        t = fiducial.time.to_numpy()

        # Drift is calculated relative to first 5 frames
        drift_x.extend(list(x-x[t<5].mean()))
        drift_y.extend(list(y-y[t<5].mean()))
        drift_t.extend(t)

    # Remove very rare outliers from data
    idx = np.bitwise_and(np.array(drift_x) < 10, np.array(drift_y) < 10)
    drift_t = np.array(drift_t)[idx]
    drift_y = np.array(drift_y)[idx]
    drift_x = np.array(drift_x)[idx]

    # Smooth signal and reduce data points for individual frames
    smoothed_t = np.arange(T, dtype=float)
    smoothed_x = sm.nonparametric.lowess(drift_x, drift_t, frac=0.11, xvals=smoothed_t)
    smoothed_y = sm.nonparametric.lowess(drift_y, drift_t, frac=0.11, xvals=smoothed_t)

    plt.figure(figsize=(8,3))
    plt.subplot(131)
    plt.title("Spatial drift")
    plt.plot(drift_x, drift_y, 'k.')
    plt.plot(smoothed_x, smoothed_y, 'r-')
    plt.axis('equal')
    plt.xlabel("x-position")
    plt.ylabel("y-position")
    plt.subplot(132)
    plt.title("Temporal X-drift")
    plt.plot(drift_t, drift_x, 'k.')
    plt.plot(smoothed_t, smoothed_x, 'r-')
    plt.xlabel("Time")
    plt.ylabel("x-position")
    plt.subplot(133)
    plt.title("Temporal Y-drift")
    plt.plot(drift_t,drift_y, 'k.')
    plt.plot(smoothed_t, smoothed_y, 'r-')
    plt.xlabel("Time")
    plt.ylabel("y-position")
    plt.tight_layout()
    plt.savefig(os.path.join(PARENT_DIR, "fiducial-drift.png"), dpi=150)
    plt.show()

    def correct_image_stack(stack, drift_y, drift_x, verbose=True):
        drift_corrected = np.zeros_like(stack)
        for i, (image, dy, dx) in enumerate(tqdm(zip(stack, drift_y, drift_x), disable=not verbose)):
            drift_corrected[i] = nd_shift(image, (-dy, -dx))
        return drift_corrected

    ch1_corrected = correct_image_stack(ch1, smoothed_y, smoothed_x)
    ch2_corrected = correct_image_stack(ch2, smoothed_y, smoothed_x)

    new_name = os.path.split(PARENT_DIR)[-1]

    tiff.imsave(os.path.join(PARENT_DIR, new_name+"-driftCorrected-ch1.tif"), ch1_corrected)
    tiff.imsave(os.path.join(PARENT_DIR, new_name+"-driftCorrected-ch2.tif"), ch2_corrected)
    
