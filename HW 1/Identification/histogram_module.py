import numpy as np
from numpy import histogram as hist
from PIL import Image


#Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0,filteringpath)

import gauss_module


#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram

def normalized_hist(img_gray, num_bins):
    
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'


    steps = 255/num_bins
    bins = np.arange(0, 255, steps)
    
    # steps could have not reached 255, we should check
    if bins[-1] != 255: # if TRUE we append 255 as the last upper limit of the last bin 
        bins = np.append(bins, np.array([255]))
    
    # we could implement a Binary Search for this task/ List comprehension
    hists = [0] * len(bins)
    
    # For each row
    for pixel in img_gray:
        # for each pixel in that row
        for p in pixel:
            # for each bin we check the upper limit
            for index, b in enumerate(bins):
                
                # if pixel is smaller or the same to the upper limit of the bin then we add one to the final bin
                if p<=b:
                    hists[index] +=1
                    break 
    # The initial bin starts at 0 and ends and 0, that is why we delete it
    hists.pop(0)
    hists = np.array(hists)     

    # We normalize the histogram so that the sum of the values is equal to one
    n_pixels = img_gray.shape[0]*img_gray.shape[1]
    hists = hists / n_pixels
            
    
    return hists, bins


#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    

    #Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))
    # The bin size is:
    bin_size = 255 / num_bins
    
    # Now, we should create three arrays (RED, GREEN, BLUE), for each pixel in the image
    img_color_double_flatten = img_color_double.flatten()
    red = img_color_double_flatten[::3]
    green = img_color_double_flatten[1::3]
    blue = img_color_double_flatten[2::3]
    
    
    # At this point we have three arrays of dimension 1 x n.pixels
    # we should insert each pixel in one particular bin, we can do this by dividing each value by n.bins
    red_bin = (red/bin_size).astype(int)
    green_bin = (green/bin_size).astype(int)
    blue_bin = (blue/bin_size).astype(int)
    
    
    
    # Loop for each pixel i in the image = n. rows * n.col
    n_pixels = len(img_color_double)*len(img_color_double[0])
    
    for i in range(n_pixels):

        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        hists[red_bin[i], green_bin[i], blue_bin[i]] += 1

        pass


    #Normalize the histogram such that its integral (sum) is equal 1
    hists = hists / n_pixels

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    
    return hists



#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins ):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'
    
    
    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    
    # The bin size is:
    bin_size = 255 / num_bins
    
    # Now, we should create two arrays (RED, GREEN), for each pixel in the image
    # let's keep only red and green values
    img_color_double_flatten = img_color_double.flatten()
    red = img_color_double_flatten[::3]
    green = img_color_double_flatten[1::3]
    
    # At this point we have two arrays of dimension 1 x n.pixels
    # we should insert each pixel in one particular bin, we can do this by dividing each value by n.bins
    red_bin = (red/bin_size).astype(int)
    green_bin = (green/bin_size).astype(int)
    
    
    # Loop for each pixel i in the image = n. rows * n.col
    n_pixels = len(img_color_double)*len(img_color_double[0])
    
    for i in range(n_pixels):

        # Increment the histogram bin which corresponds to the R,G value of the pixel i
        hists[red_bin[i], green_bin[i]] += 1

        pass


    #Normalize the histogram such that its integral (sum) is equal 1
    hists = hists / n_pixels

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    
    return hists
    


#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)

def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'
    
    
    lower = -6
    upper = 6
    
    # We call the gaussderiv function in order to get the partial derivatives for x and y 
    # Cap the range of derivative values is in the range [-6, 6]
    Dx, Dy = gauss_module.gaussderiv(img_gray, 3)
    Dx = np.clip(Dx, lower, upper)
    Dy = np.clip(Dy, lower, upper) 
    
    # We stack them in order to simplify the iteration
    stacked_deriv = list(zip(Dx.reshape(-1), Dy.reshape(-1)))

    bin_size = (upper - lower)/num_bins
    
    # We fill up the list with bins of equal length
    bins = [lower for x in range(num_bins+1)]
    previous = lower
    for i in range(num_bins):
        bin = previous + bin_size
        bins[i+1] = bin
        previous = bin

    # defining a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    
    # filling the array's values with the frequencies of the 
    # pixels in the bins intervals
    for i in range(len(stacked_deriv)):

        Dxy = [0,0]
        for k in range(len(bins)):
            if bins[k-1] <= stacked_deriv[i][0] < bins[k]:
                Dxy[0] = k-1
            if bins[k-1] <= stacked_deriv[i][1] < bins[k]:
                Dxy[1] = k-1
                    
        hists[Dxy[0],Dxy[1]] += 1
    
    hists = hists/np.sum(hists)
    # Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists


def is_grayvalue_hist(hist_name):
    if hist_name == 'grayvalue' or hist_name == 'dxdy':
        return True
    elif hist_name == 'rgb' or hist_name == 'rg':
        return False
    else:
        assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
    if hist_name == 'grayvalue':
        return normalized_hist(img, num_bins_gray)
    elif hist_name == 'rgb':
        return rgb_hist(img, num_bins_gray)
    elif hist_name == 'rg':
        return rg_hist(img, num_bins_gray)
    elif hist_name == 'dxdy':
        return dxdy_hist(img, num_bins_gray)
    else:
        assert False, 'unknown distance: %s'%hist_name