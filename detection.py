import xarray as xr
import matplotlib.pyplot as plt
import oceanspy as ospy
import numpy as np
from tqdm import tqdm

# def function for detection of surface anom creating eddies

def eddyDetection(SSH,Okubo_Weiss): # Test if area has negative Okubo-Weiss parameter, than checks for local max/min
    """
    Locates local max/min of SSH based on negative Okubo-Weiss parameter of -0.2std
    """
    std_OW = Okubo_Weiss.std()
    std_OW = std_OW*-0.2
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
            if Okubo_Weiss[j, i] >= std_OW:  # Boundary conditions 
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
    """
    DO NOT USE!!!\n
    Not correct!!!
    """
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

def inner_eddy_region(eta=xr.DataArray,eddy_center=list(),warm=False,cold=False):
    """
    Computes the inner region of a eddy \n
    Set ether warm or cold True to compute the region
    """
    eddies = xr.full_like(eta,fill_value=0)
    eddies = eddies.rename("EddyDetection")
    Eta = eta
    eta = eta.values
    def pos_X_search(eddy_location=list(),warm=False,cold=False):
        condXmax = 0
        condXmin = 0
        if warm:
            for i in range(0,100): # 100*2km radius
                # Test is extent is found and stop for loop
                if condXmax and condXmin:
                    continue

                min_X = eta[eddy_location[0]][eddy_location[1]-i]
                max_X = eta[eddy_location[0]][eddy_location[1]+i]

                # Check the change in SSH level from each point outwards from center
                change_min = min_X - eta[eddy_location[0]][eddy_location[1]-i-1]
                change_max = max_X - eta[eddy_location[0]][eddy_location[1]+i+1]

                # Test conditions if extent is reached
                if change_max < 0 and condXmax == False:
                    condXmax = eddy_location[1] + i + 1
                    print('Xmax: ',change_max)
                if change_min < 0 and condXmin == False:
                    condXmin = eddy_location[1] - i - 1
                    print('Xmin: ',change_min)
                
                
        
            if condXmax==False or condXmin==False:
                print('Error in X domain: No change in eta detected. To low extent etc')
            X_axis = [condXmin,condXmax]
                        
            return X_axis
        
        elif cold:
            for i in range(0,100): # 100*2km radius
                # Test is extent is found and stop for loop
                if condXmax and condXmin:
                    continue

                min_X = eta[eddy_location[0]][eddy_location[1]-i]
                max_X = eta[eddy_location[0]][eddy_location[1]+i]

                # Check the change in SSH level from each point outwards from center
                change_min = min_X - eta[eddy_location[0]][eddy_location[1]-i-1]
                change_max = max_X - eta[eddy_location[0]][eddy_location[1]+i+1]

                # Test conditions if extent is reached
                if change_max > 0 and condXmax == False:
                    condXmax = eddy_location[1] + i + 1
                if change_min > 0 and condXmin == False:
                    condXmin = eddy_location[1] - i - 1
                
        
            if condXmax==False or condXmin==False:
                print('Error in X domain: No change in eta detected. To low extent etc')
            X_axis = [condXmin,condXmax]
                        
            return X_axis


    def pos_XY_search(eddy_location=list(),warm=False,cold=False):
        condXmax = 0
        condXmin = 0
        condYmax = 0
        condYmin = 0
        if warm:
            for i in range(0,47): # 100*2km radius
                # Test is extent is found and stop for loop
                if condXmax and condXmin and condYmax and condYmin:
                    continue

                try:
                    min_X = eta[eddy_location[0]-i][eddy_location[1]-i]
                except:
                    pass
                try:
                    max_X = eta[eddy_location[0]+i][eddy_location[1]+i]
                except:
                    pass
                
                try:
                    min_Y = eta[eddy_location[0]+i][eddy_location[1]-i]
                except:
                    pass
                try:
                    max_Y = eta[eddy_location[0]-i][eddy_location[1]+i]
                except:
                    pass

                # Check the change in SSH level from each point outwards from center
                try:
                    change_minX = min_X - eta[eddy_location[0]-i-1][eddy_location[1]-i-1]
                except:
                    pass
                try:
                    change_maxX = max_X - eta[eddy_location[0]+i+1][eddy_location[1]+i+1]
                except:
                    pass
                try:
                    change_minY = min_Y - eta[eddy_location[0]+i+1][eddy_location[1]-i-1]
                except:
                    pass
                try:
                    change_maxY = max_Y - eta[eddy_location[0]-i-1][eddy_location[1]+i+1]
                except:
                    pass

                # Test conditions if extent is reached
                if change_maxX < 0 and condXmax == False:
                    condXmax = [eddy_location[1]+i+1,eddy_location[0]+i+1]
                if change_minX < 0 and condXmin == False:
                    condXmin = [eddy_location[1]-i-1,eddy_location[0]-i-1]
                if change_minY < 0 and condYmin == False:
                    condYmin = [eddy_location[1]-i-1,eddy_location[0]+i+1]
                if change_maxY < 0 and condYmax == False:
                    condYmax = [eddy_location[1]+i+1,eddy_location[0]-i-1]
                
                
        
            if condXmax==False or condXmin==False or condYmin==False or condYmax==False:
                print('Error in XY domain: No change in eta detected. To low extent etc')
            Axis = [condXmin,condXmax,condYmin,condYmax]
                        
            return Axis
        
        elif cold:
            for i in range(0,47): # 100*2km radius
                # Test is extent is found and stop for loop
                if condXmax and condXmin and condYmax and condYmin:
                    continue

                try:
                    min_X = eta[eddy_location[0]-i][eddy_location[1]-i]
                except:
                    pass
                try:
                    max_X = eta[eddy_location[0]+i][eddy_location[1]+i]
                except:
                    pass
                
                try:
                    min_Y = eta[eddy_location[0]+i][eddy_location[1]-i]
                except:
                    pass
                try:
                    max_Y = eta[eddy_location[0]-i][eddy_location[1]+i]
                except:
                    pass
                # Check the change in SSH level from each point outwards from center
                change_minX = min_X - eta[eddy_location[0]-i-1][eddy_location[1]-i-1]
                change_maxX = max_X - eta[eddy_location[0]+i+1][eddy_location[1]+i+1]
                change_minY = min_Y - eta[eddy_location[0]+i+1][eddy_location[1]-i-1]
                change_maxY = max_Y - eta[eddy_location[0]-i-1][eddy_location[1]+i+1]

                # Test conditions if extent is reached
                if change_maxX > 0 and condXmax == False:
                    condXmax = [eddy_location[1]+i+1,eddy_location[0]+i+1]
                if change_minX > 0 and condXmin == False:
                    condXmin = [eddy_location[1]-i-1,eddy_location[0]-i-1]
                if change_minY > 0 and condYmin == False:
                    condYmin = [eddy_location[1]-i-1,eddy_location[0]+i+1]
                if change_maxY > 0 and condYmax == False:
                    condYmax = [eddy_location[1]+i+1,eddy_location[0]-i-1]
                
                
        
            if condXmax==False or condXmin==False or condYmin==False or condYmax==False:
                print('Error in XY domain: No change in eta detected. To low extent etc')
            Axis = [condXmin,condXmax,condYmin,condYmax]
                        
            return Axis


    def pos_Y_search(eddy_location=list(),warm=False,cold=False):
        condYmax = 0
        condYmin = 0
        if warm:
            for i in range(0,47): # 47*2km radius
                # Test is extent is found and stop for loop
                if condYmax and condYmin:
                    continue
                try:
                    if Eta.Y[-1-i+eddy_location[0]].values == Eta.Y[0].values or Eta.Y[1+i+eddy_location[1]].values == Eta.Y[-1].values:
                        pass
                except:
                    try:
                        if Eta.Y[-1-i+eddy_location[0]].values == Eta.Y[0].values:
                            pass
                    except:
                        condYmin = True
                    try:
                        if Eta.Y[1+i+eddy_location[1]].values == Eta.Y[-1].values:
                            pass
                    except:
                        condYmax = True
                    if condYmax and condYmin:
                        continue
                    

                try:
                    min_Y = eta[eddy_location[0]-i][eddy_location[1]]
                except:
                    pass

                try:
                    max_Y = eta[eddy_location[0]+i][eddy_location[1]]
                except:
                    pass

                try:
                    # Check the change in SSH level from each point outwards from center
                    change_min = min_Y - eta[eddy_location[0]-i-1][eddy_location[1]]

                except:
                    pass

                try:
                    change_max = max_Y - eta[eddy_location[0]+i+1][eddy_location[1]]

                except:
                    pass
                
                # Test conditions if extent is reached
                if change_max < 0 and condYmax == False:
                    condYmax = eddy_location[0] + i + 1
                    print(change_max)
                if change_min < 0 and condYmin == False:
                    print(change_min)
                    condYmin = eddy_location[0] - i - 1
                
            if condYmax==False or condYmin==False:
                print('Error in Y domain: No change in eta detected. To low extent etc')
            X_axis = [condYmin,condYmax]
                        
            return X_axis
        
        elif cold:
            for i in range(0,47): # 47*2km radius
                if condYmax and condYmin:
                    continue
                if Eta.Y[-1-i+eddy_location[0]].values == 72.004763 or Eta.Y[1+i+eddy_location[1]].values == 72.995613:
                    if Eta.Y[-1-i+eddy_location[0]].values == 72.004763:
                        condYmin = True
                    if Eta.Y[1+i+eddy_location[1]].values == 72.995613:
                        condYmax = True
                    if condYmax and condYmin:
                        continue

                try:
                    min_Y = eta[eddy_location[0]-i][eddy_location[1]]
                except:
                    pass
                try:
                    max_Y = eta[eddy_location[0]+i][eddy_location[1]]
                except:
                    pass

                # Check the change in SSH level from each point outwards from center
                try:
                    change_min = min_Y - eta[eddy_location[0]-i-1][eddy_location[1]]
                except:
                    pass
                
                try:
                    change_max = max_Y - eta[eddy_location[0]+i+1][eddy_location[1]]
                except:
                    pass

                # Test conditions if extent is reached
                if change_max > 0 and condYmax == False:
                    condYmax = eddy_location[0] + i + 1
                elif change_min > 0 and condYmin == False:
                    condYmin = eddy_location[0] - i - 1
        
            if condYmax==False or condYmin==False:
                print('Error in Y domain: No change in eta detected. To low extent etc')
            X_axis = [condYmin,condYmax]
                        
            return X_axis


    def area_of_inner_eddy(threshold=float(),domainX=list(),domainY=list(),warm=False,cold=False):
        if warm:
            # Search domain error fix, sets to max extent of grid, if not, uses found domain
            if domainX == [0,0]:
                Xrange = np.arange(len(Eta.X))
            else:
                Xrange = np.arange(domainX)
            if domainY == [0,0]:
                Yrange = np.arange(len(Eta.Y))
            else:
                Yrange = np.arange(domainY)

            # Changing values in eddies dataset for inner region
            for j in Yrange:
                for i in Xrange:
                    if eddies[j][i] < threshold or eddies[j][i] == 2: # value 1 indicates area of inner warm eddy, add more for other layers
                        continue
                    eddies[j][i] = 1

        elif cold:
            # Search domain error fix, sets to max extent of grid, if not, uses found domain
            if domainX == [0,0]:
                Xrange = np.arange(len(eta.X))
            else:
                Xrange = np.arange(domainX)
            if domainY == [0,0]:
                Yrange = np.arange(len(eta.Y))
            else:
                Yrange = np.arange(domainY)

            # Changing values in eddies dataset for inner region
            for j in Yrange:
                for i in Xrange:
                    if eddies[j][i] > threshold or eddies[j][i] == 1: # value 2 indicates area of inner eddy, add more for other layers
                        continue
                    eddies[j][i] = 2

    if warm:
        domainX = pos_X_search(eddy_center,warm=True)
        domainY = pos_Y_search(eddy_center,warm=True)
        domainXY = pos_XY_search(eddy_center,warm=True)
        print('Domain of eddy: ',[domainX,domainY])
        print('Domain of eddy XY: ', domainXY)

        totDomain = [[],[]]
        if domainX[0]:
            totDomain[0].append(domainX[0])
        if domainX[1]:
            totDomain[0].append(domainX[1])
        if domainY[0]:
            totDomain[1].append(domainY[0])
        if domainY[1]:
            totDomain.append(domainY[1])
        for i in domainXY:
            if i[0]:
                totDomain[0].append(i[0])
            if i[1]:
                totDomain[1].append(i[1])

        print('TotDomain: ',[np.min(totDomain[0]),np.max(totDomain[0])],[np.min(totDomain[1]),np.max(totDomain[1])])
        
    elif cold:
        domainX = pos_X_search(eddy_center,cold=True)
        domainY = pos_Y_search(eddy_center,cold=True)
        domainXY = pos_XY_search(eddy_center,cold=True)
        print('Domain of eddy: ',[domainX,domainY])
        print('Domain of eddy XY: ', domainXY)