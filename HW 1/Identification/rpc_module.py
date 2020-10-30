#%%
import numpy as np
import matplotlib.pyplot as plt

import histogram_module
import dist_module
import match_module

#Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Identification')
sys.path.insert(0,filteringpath)
# compute and plot the recall/precision curve
#
# D - square matrix, D(i, j) = distance between model image i, and query image j
#
# note: assume that query and model images are in the same order, i.e. correct answer for i-th query image is the i-th model image


def plot_rpc(D, plot_color):
    
    recall = []
    precision = []

    num_queries = D.shape[1]   # number of images in query
    num_images = D.shape[0]  # number of images in model

    assert(num_images == num_queries), 'Distance matrix should be a square matrix'
    
    labels = np.diag([1]*num_images)
    
    d = D.reshape(D.size)  # flatten distance matrix
    l = labels.reshape(labels.size)  # flatten labels
     
    sortidx = d.argsort()   # sorted distance matrix indexes
    d = d[sortidx]  # sorted flatten distance matrix 
    l = l[sortidx]  # sorted flatten labels 
    
    tp = 0
    
    for idt in range(len(d)):
        
        tp += l[idt]
        
        
        
        #Compute precision and recall values and append them to "recall" and "precision" vectors
        
        precision.append(tp/ (idt + 1))
        recall.append(tp/num_images)
    
    

    plt.plot([1-precision[i] for i in range(len(precision))], recall, plot_color+'-')



def compare_dist_rpc(model_images, query_images, dist_types, hist_type, num_bins, plot_colors):
    
    assert len(plot_colors) == len(dist_types), 'number of distance types should match the requested plot colors'

    for idx in range( len(dist_types) ):

        [best_match, D] = match_module.find_best_match(model_images, query_images, dist_types[idx], hist_type, num_bins)

        plot_rpc(D, plot_colors[idx])
    

    plt.axis([0, 1, 0, 1]);
    plt.xlabel('1 - precision');
    plt.ylabel('recall');
    
    # legend(dist_types, 'Location', 'Best')
    
    plt.legend( dist_types, loc='best')

