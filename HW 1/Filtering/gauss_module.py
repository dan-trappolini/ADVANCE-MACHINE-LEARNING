# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
from scipy.ndimage import convolve1d as conv
from PIL import Image

"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""

def get_half_width(sigma):
    return round(3*sigma - 0.5)


def gauss(sigma):
    
    # w_hat = get_half_width(sigma)
    # w = 2*(w_hat) + 1
    # sums = 0 
    # Gx = []

    # for i in range(w):
    #     Gx.append(np.exp(-(i - w_hat) * (i - w_hat)/(2*sigma**2)))
    #     sums = sums + Gx[i]  
    # for i in range(w):
    #     Gx[i] = Gx[i] / sums
        
    # x = np.arange(int(-3*sigma),int(3*sigma) + 1, 1)
    
    # start -> lower limit  of the interval
    # end -> upper limit of the interval
    # interval -> is the steps in the interval
    start = int(-3*sigma)
    end = int(3*sigma)
    interval = 1
    
    # we then convert the list to a numpy array 
    x = np.array(range(start, end, interval))
    
    Gx = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-x**2/(2*sigma**2))
        
    return Gx, x



"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):
        
    # convolved2d takes as input a 2D (greyscale) array not RGB
    

    Gx, x = gauss(sigma)

    # since the gaussian filter is separable
    # (Fx convolved Fy) convolved I = Fx convolved (Fy convolved I)
    
    # Create 2 distinct filters from the gaussian. One horizontal and one vertical
    Fy = np.reshape(Gx, (1, x.size)) # for example if sigma = 4 =====> H = (1,25)
    Fx = np.reshape(Gx, (x.size, 1))   #                        =====> V = (25,1)

    
    # we first convolve each row with a 1D - filter
    first_convo = conv2(img, Fy, mode='same', boundary='fill') 

    # then convolve each column with a 1D - filter
    smooth_img = conv2(first_convo, Fx, mode='same', boundary='fill')


    smooth_img = Image.fromarray(smooth_img) # reconvert image to RGB (otherwise the image will be green...)


    return smooth_img



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):
    
    # start -> lower limit  of the interval
    # end -> upper limit of the interval
    # interval -> is the steps in the interval
    start = int(-3*sigma)
    end = int(3*sigma)
    interval = 1
    
    # we then convert the list to a numpy array 
    x = np.array(range(start, end, interval))
    
    Dx = -1/(np.sqrt(2*np.pi)*sigma**3)*x*np.exp(-x**2/(2*sigma**2))
    
    return Dx, x



def gaussderiv(img, sigma):
    
    sigma = int(sigma)

    img = gaussianfilter(img, sigma)

    Dx, x = gaussdx(sigma)
    
    # def func(x):
    #     return np.convolve(x, Dx, "same")
        
    
    #imgDx=np.apply_along_axis(func,axis=0, arr=img) #apply to all rows
    #imgDy=np.apply_along_axis(func,axis=1, arr=img) #apply to all columns  
    imgDx = conv(img, Dx, axis=1)
    imgDy = conv(img, Dx, axis=0)
    return imgDx, imgDy

