import numpy as np
import math

# Compute the intersection distance between histograms x and y
# Return 1 - hist_intersection, so smaller values correspond to more similar histograms
# Check that the distance range in [0,1]

def dist_intersect(x,y):
    
    # First part of the formula is sum(minimum(x_i, y_i)) for each i 
    min_intersection = np.minimum(x, y)
    
    
    # To normalize the result we should divide it by the number of total values of an histogram
    # Since we know that the size is the same for every image, it doesn't matter which one we pick
    intersection = np.true_divide(np.sum(min_intersection), np.sum(x))
    complement_intersection = 1-intersection

    # Check that everything works:
    if 0 <= complement_intersection <= 1:
        return complement_intersection
    else:
        print('An error has occured')



# Compute the L2 distance between x and y histograms
# Check that the distance range in [0,sqrt(2)]

def dist_l2(x,y):
    
    
    
    # L2 distance or Euclidean distance is measured by computing the sum of 
    # squared difference of the two arrays
    l2_list = np.square(x - y)
    l2 = np.sum(l2_list)
    
    # Check that everything works:
    if 0 <= l2 <= math.sqrt(2):
        return l2
    else:
        print('An error has occured')
    


# Compute chi2 distance between x and y
# Check that the distance range in [0,Inf]
# Add a minimum score to each cell of the histograms (e.g. 1) to avoid division by 0

def dist_chi2(x,y):
    
    # First of all we compute the divisor for each value 
    div = (x+1) + (y+1)
    
    # if divisor_i is equal to 0 change with 1 in order to perform division and avoid division by 0
    
    # After we compute the numerator of the equation, essentially it's the l2 distance
    chi2_num = np.square((x+1) - (y+1))
    
    # We divide the numerator by the denominator that we previously computed
    chi2_list = np.divide(chi2_num, div)
    chi2 = np.sum(chi2_list)
    # Check that everything works:
        
    
    if 0 <= chi2 <= math.inf:
        return chi2
    else:
        print('An error has occured')


def get_dist_by_name(x, y, dist_name):
    if dist_name == 'chi2':
        return dist_chi2(x,y)
    elif dist_name == 'intersect':
        return dist_intersect(x,y)
    elif dist_name == 'l2':
        return dist_l2(x,y)
    else:
        assert False, 'unknown distance: %s'%dist_name
  