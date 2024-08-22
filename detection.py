import xarray as xr
import matplotlib.pyplot as plt
import oceanspy as ospy
import numpy as np
from tqdm import tqdm

# def function for detection of surface anom creating eddies

def eddyDetection(SSH,Okubo_Weiss): # Test if area has negative Okubo-Weiss parameter, than checks for local max/min
    # df.argmax()/df.argmin()
    # Boundaries: X:[4,len(X)-1-4], Y:[4,len(Y)-1-4]

    mxbndX = len(SSH.X)-5
    mxbndY = len(SSH.Y)-5
    max_coord = []
    min_coord = []
    count = 0
    count_skip = 0
    T = mxbndY-3
    pbar = tqdm(total=T, desc="Generating Frames")
    for j in range(4, mxbndY + 1):
        pbar.update(1)
        for i in range(4, mxbndX + 1):
            count += 1
            if Okubo_Weiss[j, i] >= 0:  # Boundary conditions
                count_skip += 1
                continue
            center = SSH[j][i]
            neighbors = [
                SSH[j+4, i],
                SSH[j+3, i-3], SSH[j+3, i], SSH[j+3, i+3],
                SSH[j+2, i-2], SSH[j+2, i], SSH[j+2, i+2],
                SSH[j+1, i-1], SSH[j+1, i], SSH[j+1, i+1],
                SSH[j, i-4], SSH[j, i-3], SSH[j, i-2], SSH[j, i-1], center, SSH[j, i+1], SSH[j, i+2], SSH[j, i+3], SSH[j, i+4],
                SSH[j-1, i-1], SSH[j-1, i], SSH[j-1, i+1],
                SSH[j-2, i-2], SSH[j-2, i], SSH[j-2, i+2],
                SSH[j-3, i-3], SSH[j-3, i], SSH[j-3, i+3],
                SSH[j-4, i]
            ]

            # Convert neighbors to a numpy array
            neighbors = np.array(neighbors)

            # Find the index of the max/min value
            flat_index_max = np.argmax(neighbors)
            flat_index_min = np.argmin(neighbors)

            # Check if the center is the max/min value
            if flat_index_max == neighbors.size // 2:  # Center is at the middle of the list
                max_coord.append((j, i))
            elif flat_index_min == neighbors.size // 2:  # Center is at the middle of the list
                min_coord.append((j, i))
    print('Skiped windows %:',(count_skip/count)*100)      
    return max_coord, min_coord



def resolution(X,Y):
    # For each in grid is each Y, and each contains all of X 
    grid = []

    for i in range(len(Y)):
        pos_dist_min = []
        for j in range(len(X)-1):

            if i == 0 or j == 0 or i == len(Y)-1 or j == len(X):
                pos_dist_min.append(np.nan)
            else:
                dist1 = abs(X[j].values - X[j+1].values)*111*np.cos(np.deg2rad(Y[i].values))
                dist2 = abs(X[j].values - X[j-1].values)*111*np.cos(np.deg2rad(Y[i].values))

                if dist1 == dist2:
                    pos_dist_min.append(dist1)
                else:
                    try:
                        pos_dist_min.append(min(dist1,dist2))
                    except:
                        print(dist1,dist2)
                        pos_dist_min.append(min(dist1,dist2))
        pos_dist_min.append(np.nan)
        grid.append(pos_dist_min)

    return grid

def inner_eddy_region(eddy_center):
    pass