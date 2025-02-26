import xarray as xr
import matplotlib.pyplot as plt
import oceanspy as ospy
import numpy as np
from tqdm import tqdm

# def function for detection of surface anom creating eddies

def eddyDetection(SSH, Okubo_Weiss):
    """
    Locates local max/min of SSH based on negative Okubo-Weiss parameter of -0.2std
    """
    std_OW = Okubo_Weiss.std().item() * -0.2
    X = SSH.X.values
    Y = SSH.Y.values
    SSH = SSH.values
    Okubo_Weiss_np = Okubo_Weiss.values
    mxbndX = len(X) - 5
    mxbndY = len(Y) - 5
    max_coord = []
    min_coord = []
    count = 0
    count_skip = 0
    T = mxbndY - 3
    pbar = tqdm(total=T, desc="Generating Frames")

    for j in range(4, mxbndY + 1):
        pbar.update(1)
        for i in range(4, mxbndX + 1):
            count += 1
            if Okubo_Weiss_np[j, i] >= std_OW:  # Boundary conditions
                count_skip += 1
                continue

            center = SSH[j, i]
            neighbors = [
                SSH[j+4, i],
                SSH[j+3, i-3], SSH[j+3, i], SSH[j+3, i+3],
                SSH[j+2, i-2], SSH[j+2, i], SSH[j+2, i+2],
                SSH[j+1, i-1], SSH[j+1, i], SSH[j+1, i+1],
                SSH[j, i-4], SSH[j, i-3], SSH[j, i-2], SSH[j, i-1], SSH[j, i+1], SSH[j, i+2], SSH[j, i+3], SSH[j, i+4],
                SSH[j-1, i-1], SSH[j-1, i], SSH[j-1, i+1],
                SSH[j-2, i-2], SSH[j-2, i], SSH[j-2, i+2],
                SSH[j-3, i-3], SSH[j-3, i], SSH[j-3, i+3],
                SSH[j-4, i]
            ]
            

            if center > np.max(neighbors):
                max_coord.append([Y[j], X[i]])
            elif center < np.min(neighbors):
                min_coord.append([Y[j], X[i]])

    pbar.close()
    print('Skipped windows %:', (count_skip / count) * 100)
    return max_coord, min_coord