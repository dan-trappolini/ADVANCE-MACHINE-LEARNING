import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image


def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    # whether histogram function expects grayvalue or color image
    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)

    # form list of file images to list of histograms
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)

    D = np.zeros((len(model_images), len(query_images))) # container for distances
    best_match = [] # container for best match for each image

    for i in range(len(query_images)):
        query_hist_item = query_hists[i]
        for j in range(len(model_hists)):
            model_hist_item = model_hists[j]
            h1 = query_hist_item.copy()
            h2 = model_hist_item.copy()
            distance = dist_module.get_dist_by_name(h1, h2, dist_type) # distance between histogram
            D[j, i] = distance # save distance measure in correct position
        indexx = list(D[:, i]).index(min(D[:, i])) # find histogram with lower distance respect to the one in exam
        best_match.append(indexx)

    return np.array(best_match), D


def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    
    image_hist = []

    #We compute the histogram value for every image and we append it to our image_hist list

    for i in range(len(image_list)):
        img = np.array(Image.open(image_list[i]))
        img = img.astype('double')
        if hist_isgray:
            img = rgb2gray(img)
        hist = histogram_module.get_hist_by_name(img, num_bins, hist_type)
 
        if len(hist) == 2:
             hist = hist[0]
 
        image_hist.append(hist)

    return image_hist



# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    
    
    plt.figure()

    num_nearest = 5  # show the top-5 neighbors
    
    best_match, D = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
    for i in range(len(query_images)):

        #We sort the images according to the best_match function retrieving the num_nearest closest
        closest_images = sorted(range(len(D[:, i])), key=lambda k: D[:, i][k])[:num_nearest]

        #We construct the plot
        for j in range(1, num_nearest+2):

           
            plt.figure(num_nearest+1, figsize=(50, 50))
            #We want to show the closest images in the same row
            ax = plt.subplot(1, num_nearest+1, j)
            ax.title.set_size(40)
            plt.axis('off')

            if j == 1: #The query image
                ax.title.set_text('Q'+str(i))
                plt.imshow(np.array(Image.open(query_images[i])))
            else: #The closest images that we found
                
                ax.title.set_text('M'+str(round(D[closest_images[j-2], i], 2)))
                plt.imshow(np.array(Image.open(model_images[closest_images[j-2]])))
        plt.show()

    return


