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
                max_coord.append([SSH.Y[j].item(), SSH.X[i].item()])
            elif flat_index_min == neighbors.size // 2:  # Center is at the middle of the list
                min_coord.append([SSH.Y[j].item(), SSH.X[i].item()])
    print('Skiped windows %:',(count_skip/count)*100)      
    return max_coord, min_coord


def resolution(X,Y): # Not in use!
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


def inner_eddy_region(eta=xr.DataArray,eddy_center=list(),warm=False,cold=False,eddiesDataset=None,use_eddyDataset=False): # Old version
    """
    Computes the inner region of a eddy \n
    Set ether warm or cold True to compute the region\n
    After first run of one eddy center, add the returned dataset as eddiesDataset!\n
    This workes for both cold and warm eddies, which are set to values 2 and 1 in dataset
    """


    if use_eddyDataset:
        eddies = eddiesDataset
    else:
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
                if -1-i+eddy_location[1] < 0 or 1+i+eddy_location[1] > 434:
                    if -1-i+eddy_location[1] < 0:
                        condXmin = True
                    if 1+i+eddy_location[1] > 434:
                        condXmax = True
                if condXmax and condXmin:
                    continue
                
                try:
                    min_X = eta[eddy_location[0]][eddy_location[1]-i]
                except:
                    pass
                try:
                    max_X = eta[eddy_location[0]][eddy_location[1]+i]
                except:
                    pass

                # Check the change in SSH level from each point outwards from center
                try:
                    change_min = min_X - eta[eddy_location[0]][eddy_location[1]-i-1]
                except:
                    pass
                try:
                    change_max = max_X - eta[eddy_location[0]][eddy_location[1]+i+1]
                except:
                    pass

                # Test conditions if extent is reached
                if change_max < 0 and condXmax == False:
                    condXmax = eddy_location[1] + i + 1
                if change_min < 0 and condXmin == False:
                    condXmin = eddy_location[1] - i - 1
                
                
        
            if condXmax==False or condXmin==False:
                print('Error in X domain: No change in eta detected. To low extent etc')
            X_axis = [condXmin,condXmax]
                        
            return X_axis
        
        elif cold:
            for i in range(0,100): # 100*2km radius
                # Test is extent is found and stop for loop
                if -1-i+eddy_location[1] < 0 or 1+i+eddy_location[1] > 434:
                    if -1-i+eddy_location[1] < 0:
                        condXmin = True
                    if 1+i+eddy_location[1] > 434:
                        condXmax = True
                if condXmax and condXmin:
                    continue
                
                try:
                    min_X = eta[eddy_location[0]][eddy_location[1]-i]
                except:
                    pass
                try:
                    max_X = eta[eddy_location[0]][eddy_location[1]+i]
                except:
                    pass

                # Check the change in SSH level from each point outwards from center
                try:
                    change_min = min_X - eta[eddy_location[0]][eddy_location[1]-i-1]
                except:
                    pass
                try:
                    change_max = max_X - eta[eddy_location[0]][eddy_location[1]+i+1]
                except:
                    pass

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
                if -1-i+eddy_location[1] < 0 or 1+i+eddy_location[1] > 434:
                    if -1-i+eddy_location[1] < 0:
                        condXmin = True
                    if 1+i+eddy_location[1] > 434:
                        condXmax = True
                if -1-i+eddy_location[0] < 0 or 1+i+eddy_location[0] > 46:
                    if -1-i+eddy_location[0] < 0:
                        condYmin = True
                    if 1+i+eddy_location[0] > 46:
                        condYmax = True

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
                # Test is extent is found and stop for loop
                if -1-i+eddy_location[1] < 0 or 1+i+eddy_location[1] > 434:
                    if -1-i+eddy_location[1] < 0:
                        condXmin = True
                    if 1+i+eddy_location[1] > 434:
                        condXmax = True
                if -1-i+eddy_location[0] < 0 or 1+i+eddy_location[0] > 46:
                    if -1-i+eddy_location[0] < 0:
                        condYmin = True
                    if 1+i+eddy_location[0] > 46:
                        condYmax = True

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
                if -1-i+eddy_location[0] < 0 or 1+i+eddy_location[0] > 46:
                    if -1-i+eddy_location[0] < 0:
                        condYmin = True
                    if 1+i+eddy_location[0] > 46:
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
                if change_min < 0 and condYmin == False:
                    condYmin = eddy_location[0] - i - 1
                
            if condYmax==False or condYmin==False:
                print('Error in Y domain: No change in eta detected. To low extent etc')
            X_axis = [condYmin,condYmax]
                        
            return X_axis
        
        elif cold:
            for i in range(0,47): # 47*2km radius
                if -1-i+eddy_location[0] < 0 or 1+i+eddy_location[0] > 46:
                    if -1-i+eddy_location[0] < 0:
                        condYmin = True
                    if 1+i+eddy_location[0] > 46:
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


    def area_of_inner_eddy(threshold=float(),domainX=list(),domainY=list(),warm=False,cold=False,eddies=eddies):
        if warm:
            # Search domain error fix, sets to max extent of grid, if not, uses found domain
            if domainX == [0,0]:
                Xrange = np.arange(len(Eta.X))
            else:
                Xrange = np.arange(domainX[0],domainX[1])
            if domainY == [0,0]:
                Yrange = np.arange(len(Eta.Y))
            else:
                Yrange = np.arange(domainY[0],domainY[1])

            # Changing values in eddies dataset for inner region
            for j in Yrange:
                for i in Xrange:
                    if eta[j][i] < threshold or eddies[j][i] == 2: # value 1 indicates area of inner warm eddy, add more for other layers
                        continue
                    eddies[j][i] = 1
            
            # Check for outliers
            test = eddies
            for j in Yrange:
                for i in Xrange:
                    if eddies[j][i] != 1: # value 1 indicates area of inner warm eddy, add more for other layers
                        continue
                    if eddies[j][i-1] !=1 and eddies[j][i+1] !=1:
                        test[j][i] = 0
                    if eddies[j-1][i] !=1 and eddies[j+1][i] !=1:
                        test[j][i] = 0
            eddies = test

        if cold:
            # Search domain error fix, sets to max extent of grid, if not, uses found domain
            if domainX == [0,0]:
                Xrange = np.arange(len(eta.X))
            else:
                Xrange = np.arange(domainX[0],domainX[1])
            if domainY == [0,0]:
                Yrange = np.arange(len(Eta.Y))
            else:
                Yrange = np.arange(domainY[0],domainY[1])

            # Changing values in eddies dataset for inner region
            for j in Yrange:
                for i in Xrange:
                    if eta[j][i] > threshold or eddies[j][i] == 1: # value 2 indicates area of inner eddy, add more for other layers
                        continue
                    eddies[j][i] = 2
            
            # Check for outliers
            test = eddies
            for j in Yrange:
                for i in Xrange:
                    if eddies[j][i] != 2: # value 2 indicates area of inner cold eddy, add more for other layers
                        continue
                    if eddies[j][i-1] !=2 and eddies[j][i+1] !=2:
                        test[j][i] = 0
                    if eddies[j-1][i] !=2 and eddies[j+1][i] !=2:
                        test[j][i] = 0
            eddies = test
        
        return eddies


    if warm:
        # Computes domain of eddy
        domainX = pos_X_search(eddy_center,warm=True)
        domainY = pos_Y_search(eddy_center,warm=True)
        domainXY = pos_XY_search(eddy_center,warm=True)
        # print('Domain of eddy: ',[domainX,domainY])
        # print('Domain of eddy XY: ', domainXY)

        totDomain = [[],[]]
        if domainX[0] != 0 and not isinstance(domainX[0], bool):
            totDomain[0].append(domainX[0])
        if domainX[1] != 0 and not isinstance(domainX[1], bool):
            totDomain[0].append(domainX[1])
        if domainY[0] != 0 and not isinstance(domainY[0], bool):
            totDomain[1].append(domainY[0])
        if domainY[1] != 0 and not isinstance(domainY[1], bool):
            totDomain.append(domainY[1])
        for i in domainXY:
            try:
                if i[0] != 0 and not isinstance(i[0], bool):
                    totDomain[0].append(i[0])
            except:
                continue
            if i[1] != 0 and not isinstance(i[1], bool):
                totDomain[1].append(i[1])
        
        # Find the maximum eta value all boundaries to use as threshold
        max_eta_boundary_coord = []
        max_eta_boundary_value = []
        for j in totDomain[1]:
            for i in totDomain[0]:
                max_eta_boundary_coord.append([i,j])
        if domainX[0] != 0 and not isinstance(domainX[0], bool):
            max_eta_boundary_coord.append([domainX[0],eddy_center[0]])
        if domainX[1] != 0 and not isinstance(domainX[1], bool):
            max_eta_boundary_coord.append([domainX[1],eddy_center[0]])
        if domainY[0] != 0 and not isinstance(domainY[0], bool):    
            max_eta_boundary_coord.append([eddy_center[1],domainY[0]])
        if domainY[1] != 0 and not isinstance(domainY[1], bool):
            max_eta_boundary_coord.append([eddy_center[1],domainY[1]])
        
        for i in max_eta_boundary_coord:
            try:
                max_eta_boundary_value.append(eta[i[1]][i[0]])
            except:
                continue
        try:
            threshold = np.max(max_eta_boundary_value)

            try:
                totDomain = [np.min(totDomain[0]),np.max(totDomain[0])],[np.min(totDomain[1]),np.max(totDomain[1])]
                dataset = area_of_inner_eddy(threshold=threshold,domainX=[totDomain[0][0],totDomain[0][1]],domainY=[totDomain[1][0],totDomain[1][1]],warm=True,eddies=eddies)
            except:
                dataset = area_of_inner_eddy(threshold=threshold,domainX=[0,0],domainY=[0,0],warm=True,eddies=eddies)
        except:
            print('Error in eddy: ',eddy_center,' Skiped')

        
    if cold:
        domainX = pos_X_search(eddy_center,cold=True)
        domainY = pos_Y_search(eddy_center,cold=True)
        domainXY = pos_XY_search(eddy_center,cold=True)
        # print('Domain of eddy: ',[domainX,domainY])
        # print('Domain of eddy XY: ', domainXY)

        totDomain = [[],[]]
        if domainX[0] != 0 and not isinstance(domainX[0], bool):
            totDomain[0].append(domainX[0])
        if domainX[1] != 0 and not isinstance(domainX[1], bool):
            totDomain[0].append(domainX[1])
        if domainY[0] != 0 and not isinstance(domainY[0], bool):
            totDomain[1].append(domainY[0])
        if domainY[1] != 0 and not isinstance(domainY[1], bool):
            totDomain.append(domainY[1])
        for i in domainXY:
            try:
                if i[0] != 0 and not isinstance(i[0], bool):
                    totDomain[0].append(i[0])
            except:
                continue
            if i[1] != 0 and not isinstance(i[1], bool):
                totDomain[1].append(i[1])
        
        # Find the minimum eta value all boundaries to use as threshold
        min_eta_boundary_coord = []
        min_eta_boundary_value = []
        for j in totDomain[1]:
            for i in totDomain[0]:
                min_eta_boundary_coord.append([i,j])
        if domainX[0] != 0 and not isinstance(domainX[0], bool):
            min_eta_boundary_coord.append([domainX[0],eddy_center[0]])
        if domainX[1] != 0 and not isinstance(domainX[1], bool):
            min_eta_boundary_coord.append([domainX[1],eddy_center[0]])
        if domainY[0] != 0 and not isinstance(domainY[0], bool):    
            min_eta_boundary_coord.append([eddy_center[1],domainY[0]])
        if domainY[1] != 0 and not isinstance(domainY[1], bool):
            min_eta_boundary_coord.append([eddy_center[1],domainY[1]])
        
        for i in min_eta_boundary_coord:
            try:
                min_eta_boundary_value.append(eta[i[1]][i[0]])
            except:
                continue
        try:
            threshold = np.min(min_eta_boundary_value)
            try:
                totDomain = [np.min(totDomain[0]),np.max(totDomain[0])],[np.min(totDomain[1]),np.max(totDomain[1])]
                dataset = area_of_inner_eddy(threshold=threshold,domainX=[totDomain[0][0],totDomain[0][1]],domainY=[totDomain[1][0],totDomain[1][1]],cold=True,eddies=eddies)
            except:
                dataset = area_of_inner_eddy(threshold=threshold,domainX=[0,0],domainY=[0,0],cold=True,eddies=eddies)
        except:
            print('Error in eddy: ',eddy_center,' Skiped')
    
    try:
        return dataset
    except:
        return eddies


def full_inner_eddy_region(eta=xr.DataArray,eddy_center=list(),warm=False,cold=False,eddiesDataset=xr.DataArray):
    """
    Computes the inner region of a eddy \n
    Set ether warm or cold True to compute the region\n
    After first run of one eddy center, add the returned dataset as eddiesDataset!\n
    This workes for both cold and warm eddies, which are set to values 2 and 1 in dataset
    """

    # Transform eddy centerpoints from coords to index position from eta array
    eddy_center = [np.argwhere(eta.Y.values == eddy_center[0])[0][0],np.argwhere(eta.X.values == eddy_center[1])[0][0]]

    eddies = eddiesDataset
    Eta = eta
    eta = eta.values


    def pos_X_search(eddy_location=list(),warm=False,cold=False):
        condXmax = 0
        condXmin = 0
        if warm:
            for i in range(0,100): # 100*2km radius
                # Test is extent is found and stop for loop
                if -1-i+eddy_location[1] < 0 or 1+i+eddy_location[1] > 434:
                    if -1-i+eddy_location[1] < 0:
                        condXmin = True
                    if 1+i+eddy_location[1] > 434:
                        condXmax = True
                if condXmax and condXmin:
                    continue
                
                try:
                    min_X = eta[eddy_location[0]][eddy_location[1]-i]
                except:
                    pass
                try:
                    max_X = eta[eddy_location[0]][eddy_location[1]+i]
                except:
                    pass

                # Check the change in SSH level from each point outwards from center
                try:
                    change_min = min_X - eta[eddy_location[0]][eddy_location[1]-i-1]
                except:
                    pass
                try:
                    change_max = max_X - eta[eddy_location[0]][eddy_location[1]+i+1]
                except:
                    pass

                # Test conditions if extent is reached
                if change_max < 0 and condXmax == False:
                    condXmax = eddy_location[1] + i
                if change_min < 0 and condXmin == False:
                    condXmin = eddy_location[1] - i
                
                
        
            if condXmax==False or condXmin==False:
                pass
                print('Error in X domain: No change in eta detected. To low extent etc')
            X_axis = [condXmin,condXmax]
                        
            return X_axis
        
        elif cold:
            for i in range(0,100): # 100*2km radius
                # Test is extent is found and stop for loop
                if -1-i+eddy_location[1] < 0 or 1+i+eddy_location[1] > 434:
                    if -1-i+eddy_location[1] < 0:
                        condXmin = True
                    if 1+i+eddy_location[1] > 434:
                        condXmax = True
                if condXmax and condXmin:
                    continue
                
                try:
                    min_X = eta[eddy_location[0]][eddy_location[1]-i]
                except:
                    pass
                try:
                    max_X = eta[eddy_location[0]][eddy_location[1]+i]
                except:
                    pass

                # Check the change in SSH level from each point outwards from center
                try:
                    change_min = min_X - eta[eddy_location[0]][eddy_location[1]-i-1]
                except:
                    pass
                try:
                    change_max = max_X - eta[eddy_location[0]][eddy_location[1]+i+1]
                except:
                    pass

                # Test conditions if extent is reached
                if change_max > 0 and condXmax == False:
                    condXmax = eddy_location[1] + i + 1
                if change_min > 0 and condXmin == False:
                    condXmin = eddy_location[1] - i - 1
                
        
            if condXmax==False or condXmin==False:
                pass
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
                if -1-i+eddy_location[1] < 0 or 1+i+eddy_location[1] > 434:
                    if -1-i+eddy_location[1] < 0:
                        condXmin = True
                    if 1+i+eddy_location[1] > 434:
                        condXmax = True
                if -1-i+eddy_location[0] < 0 or 1+i+eddy_location[0] > 46:
                    if -1-i+eddy_location[0] < 0:
                        condYmin = True
                    if 1+i+eddy_location[0] > 46:
                        condYmax = True

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
                    condXmax = [eddy_location[1]+i+1,eddy_location[0]+i]
                if change_minX < 0 and condXmin == False:
                    condXmin = [eddy_location[1]-i-1,eddy_location[0]-i]
                if change_minY < 0 and condYmin == False:
                    condYmin = [eddy_location[1]-i-1,eddy_location[0]+i]
                if change_maxY < 0 and condYmax == False:
                    condYmax = [eddy_location[1]+i+1,eddy_location[0]-i]
                
                
        
            if condXmax==False or condXmin==False or condYmin==False or condYmax==False:
                pass
                print('Error in XY domain: No change in eta detected. To low extent etc')
            Axis = [condXmin,condXmax,condYmin,condYmax]
                        
            return Axis
        
        elif cold:
            for i in range(0,47): # 100*2km radius
                # Test is extent is found and stop for loop
                # Test is extent is found and stop for loop
                if -1-i+eddy_location[1] < 0 or 1+i+eddy_location[1] > 434:
                    if -1-i+eddy_location[1] < 0:
                        condXmin = True
                    if 1+i+eddy_location[1] > 434:
                        condXmax = True
                if -1-i+eddy_location[0] < 0 or 1+i+eddy_location[0] > 46:
                    if -1-i+eddy_location[0] < 0:
                        condYmin = True
                    if 1+i+eddy_location[0] > 46:
                        condYmax = True

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
                if change_maxX > 0 and condXmax == False:
                    condXmax = [eddy_location[1]+i+1,eddy_location[0]+i]
                if change_minX > 0 and condXmin == False:
                    condXmin = [eddy_location[1]-i-1,eddy_location[0]-i]
                if change_minY > 0 and condYmin == False:
                    condYmin = [eddy_location[1]-i-1,eddy_location[0]+i]
                if change_maxY > 0 and condYmax == False:
                    condYmax = [eddy_location[1]+i+1,eddy_location[0]-i]
                
                
        
            if condXmax==False or condXmin==False or condYmin==False or condYmax==False:
                pass
                print('Error in XY domain: No change in eta detected. To low extent etc')
            Axis = [condXmin,condXmax,condYmin,condYmax]
                        
            return Axis


    def pos_Y_search(eddy_location=list(),warm=False,cold=False):
        condYmax = 0
        condYmin = 0
        if warm:
            for i in range(0,47): # 47*2km radius
                # Test is extent is found and stop for loop
                if -1-i+eddy_location[0] < 0 or 1+i+eddy_location[0] > 46:
                    if -1-i+eddy_location[0] < 0:
                        condYmin = True
                    if 1+i+eddy_location[0] > 46:
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
                    condYmax = eddy_location[0] + i
                if change_min < 0 and condYmin == False:
                    condYmin = eddy_location[0] - i
                
            if condYmax==False or condYmin==False:
                pass
                print('Error in Y domain: No change in eta detected. To low extent etc')
            X_axis = [condYmin,condYmax]
                        
            return X_axis
        
        elif cold:
            for i in range(0,47): # 47*2km radius
                if -1-i+eddy_location[0] < 0 or 1+i+eddy_location[0] > 46:
                    if -1-i+eddy_location[0] < 0:
                        condYmin = True
                    if 1+i+eddy_location[0] > 46:
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
                    condYmax = eddy_location[0] + i
                elif change_min > 0 and condYmin == False:
                    condYmin = eddy_location[0] - i
        
            if condYmax==False or condYmin==False:
                pass
                print('Error in Y domain: No change in eta detected. To low extent etc')
            X_axis = [condYmin,condYmax]
                        
            return X_axis


    def area_of_inner_eddy(threshold=float(),domainX=list(),domainY=list(),warm=False,cold=False,eddies=eddies):
        if warm:
            # Search domain error fix, sets to max extent of grid, if not, uses found domain
            if domainX == [0,0]:
                Xrange = np.arange(len(Eta.X))
            else:
                Xrange = np.arange(domainX[0],domainX[1])
            if domainY == [0,0]:
                Yrange = np.arange(len(Eta.Y))
            else:
                Yrange = np.arange(domainY[0],domainY[1])

            # Changing values in eddies dataset for inner region
            for j in Yrange:
                for i in Xrange:
                    if eta[j][i] < threshold or eddies[j][i] == 2: # value 1 indicates area of inner warm eddy, add more for other layers
                        continue
                    eddies[j][i] = 1
            
            # Check for outliers
            test = eddies
            for j in Yrange:
                for i in Xrange:
                    if eddies[j][i] != 1: # value 1 indicates area of inner warm eddy, add more for other layers
                        continue
                    if eddies[j][i-1] !=1 and eddies[j][i+1] !=1 and eddies[j][i-2] !=1 and eddies[j][i+2] !=1:
                        test[j][i] = 0
                    if eddies[j-1][i] !=1 and eddies[j+1][i] !=1 and eddies[j-2][i] !=1 and eddies[j+2][i] !=1:
                        test[j][i] = 0
            eddies = test

        elif cold:
            # Search domain error fix, sets to max extent of grid, if not, uses found domain
            if domainX == [0,0]:
                Xrange = np.arange(len(eta.X))
            else:
                Xrange = np.arange(domainX[0],domainX[1])
            if domainY == [0,0]:
                Yrange = np.arange(len(Eta.Y))
            else:
                Yrange = np.arange(domainY[0],domainY[1])

            # Changing values in eddies dataset for inner region
            for j in Yrange:
                for i in Xrange:
                    if eta[j][i] > threshold or eddies[j][i] == 1: # value 2 indicates area of inner eddy, add more for other layers
                        continue
                    eddies[j][i] = 2
            
            # Check for outliers
            test = eddies
            for j in Yrange:
                for i in Xrange:
                    if eddies[j][i] != 2: # value 2 indicates area of inner cold eddy, add more for other layers
                        continue
                    if eddies[j][i-1] !=2 and eddies[j][i+1] !=2 and eddies[j][i-2] !=2 and eddies[j][i+2] !=2:
                        test[j][i] = 0
                    if eddies[j-1][i] !=2 and eddies[j+1][i] !=2 and eddies[j-2][i] !=2 and eddies[j+2][i] !=2:
                        test[j][i] = 0
            eddies = test
        
        return eddies


    if warm:
        # Computes domain of eddy
        domainX = pos_X_search(eddy_center,warm=True)
        domainY = pos_Y_search(eddy_center,warm=True)
        domainXY = pos_XY_search(eddy_center,warm=True)
        # print('Domain of eddy: ',[domainX,domainY])
        # print('Domain of eddy XY: ', domainXY)

        totDomain = [[],[]]
        if domainX[0] != 0 and not isinstance(domainX[0], bool):
            totDomain[0].append(domainX[0])
        if domainX[1] != 0 and not isinstance(domainX[1], bool):
            totDomain[0].append(domainX[1])
        if domainY[0] != 0 and not isinstance(domainY[0], bool):
            totDomain[1].append(domainY[0])
        if domainY[1] != 0 and not isinstance(domainY[1], bool):
            totDomain.append(domainY[1])
        for i in domainXY:
            try:
                if i[0] != 0 and not isinstance(i[0], bool):
                    totDomain[0].append(i[0])
            except:
                continue
            if i[1] != 0 and not isinstance(i[1], bool):
                totDomain[1].append(i[1])
        
        # Find the maximum eta value all boundaries to use as threshold
        max_eta_boundary_coord = []
        max_eta_boundary_value = []
        for j in totDomain[1]:
            for i in totDomain[0]:
                max_eta_boundary_coord.append([i,j])
        if domainX[0] != 0 and not isinstance(domainX[0], bool):
            max_eta_boundary_coord.append([domainX[0],eddy_center[0]])
        if domainX[1] != 0 and not isinstance(domainX[1], bool):
            max_eta_boundary_coord.append([domainX[1],eddy_center[0]])
        if domainY[0] != 0 and not isinstance(domainY[0], bool):    
            max_eta_boundary_coord.append([eddy_center[1],domainY[0]])
        if domainY[1] != 0 and not isinstance(domainY[1], bool):
            max_eta_boundary_coord.append([eddy_center[1],domainY[1]])
        
        for i in max_eta_boundary_coord:
            try:
                max_eta_boundary_value.append(eta[i[1]][i[0]])
            except:
                continue
        try:
            threshold = np.max(max_eta_boundary_value)

            try:
                totDomain = [np.min(totDomain[0]),np.max(totDomain[0])],[np.min(totDomain[1]),np.max(totDomain[1])]
                dataset = area_of_inner_eddy(threshold=threshold,domainX=[totDomain[0][0],totDomain[0][1]],domainY=[totDomain[1][0],totDomain[1][1]],warm=True,eddies=eddies)
            except:
                dataset = area_of_inner_eddy(threshold=threshold,domainX=[0,0],domainY=[0,0],warm=True,eddies=eddies)
        except:
            print('Error in eddy: ',eddy_center,' Skiped')

        
    if cold:
        domainX = pos_X_search(eddy_center,cold=True)
        domainY = pos_Y_search(eddy_center,cold=True)
        domainXY = pos_XY_search(eddy_center,cold=True)
        # print('Domain of eddy: ',[domainX,domainY])
        # print('Domain of eddy XY: ', domainXY)

        totDomain = [[],[]]
        if domainX[0] != 0 and not isinstance(domainX[0], bool):
            totDomain[0].append(domainX[0])
        if domainX[1] != 0 and not isinstance(domainX[1], bool):
            totDomain[0].append(domainX[1])
        if domainY[0] != 0 and not isinstance(domainY[0], bool):
            totDomain[1].append(domainY[0])
        if domainY[1] != 0 and not isinstance(domainY[1], bool):
            totDomain.append(domainY[1])
        for i in domainXY:
            try:
                if i[0] != 0 and not isinstance(i[0], bool):
                    totDomain[0].append(i[0])
            except:
                continue
            if i[1] != 0 and not isinstance(i[1], bool):
                totDomain[1].append(i[1])
        
        # Find the minimum eta value all boundaries to use as threshold
        min_eta_boundary_coord = []
        min_eta_boundary_value = []
        for j in totDomain[1]:
            for i in totDomain[0]:
                min_eta_boundary_coord.append([i,j])
        if domainX[0] != 0 and not isinstance(domainX[0], bool):
            min_eta_boundary_coord.append([domainX[0],eddy_center[0]])
        if domainX[1] != 0 and not isinstance(domainX[1], bool):
            min_eta_boundary_coord.append([domainX[1],eddy_center[0]])
        if domainY[0] != 0 and not isinstance(domainY[0], bool):    
            min_eta_boundary_coord.append([eddy_center[1],domainY[0]])
        if domainY[1] != 0 and not isinstance(domainY[1], bool):
            min_eta_boundary_coord.append([eddy_center[1],domainY[1]])
        
        for i in min_eta_boundary_coord:
            try:
                min_eta_boundary_value.append(eta[i[1]][i[0]])
            except:
                continue
        try:
            threshold = np.min(min_eta_boundary_value)
            try:
                totDomain = [np.min(totDomain[0]),np.max(totDomain[0])],[np.min(totDomain[1]),np.max(totDomain[1])]
                dataset = area_of_inner_eddy(threshold=threshold,domainX=[totDomain[0][0],totDomain[0][1]],domainY=[totDomain[1][0],totDomain[1][1]],cold=True,eddies=eddies)
            except:
                dataset = area_of_inner_eddy(threshold=threshold,domainX=[0,0],domainY=[0,0],cold=True,eddies=eddies)
        except:
            print('Error in eddy: ',eddy_center,' Skiped')
    
    try:
        return dataset
    except:
        return eddies


def full_inner_eddy_region_v2(eta=xr.DataArray,eddy_center=list(),warm=False,cold=False,eddiesDataset=xr.DataArray):
    """
    Computes the inner region of a eddy \n
    Set ether warm or cold True to compute the region\n
    After first run of one eddy center, add the returned dataset as eddiesDataset!\n
    This workes for both cold and warm eddies, which are set to values 2 and 1 in dataset
    """

    # Transform eddy centerpoints from coords to index position from eta array
    eddy_center = [np.argwhere(eta.Y.values == eddy_center[0])[0][0],np.argwhere(eta.X.values == eddy_center[1])[0][0]]

    eddies = eddiesDataset
    Eta = eta
    eta = eta.values


    def pos_X_search(eddy_location=list(),warm=False,cold=False):
        condXmax = 0
        condXmin = 0
        if warm:
            for i in range(0,100): # 100*2km radius
                # Test is extent is found and stop for loop
                if -1-i+eddy_location[1] < 0 or 1+i+eddy_location[1] > len(Eta.X)-1:
                    if -1-i+eddy_location[1] < 0:
                        condXmin = True
                    if 1+i+eddy_location[1] > len(Eta.X)-1:
                        condXmax = True
                if condXmax and condXmin:
                    continue
                
                try:
                    min_X = eta[eddy_location[0]][eddy_location[1]-i]
                except:
                    pass
                try:
                    max_X = eta[eddy_location[0]][eddy_location[1]+i]
                except:
                    pass

                # Check the change in SSH level from each point outwards from center
                try:
                    change_min = min_X - eta[eddy_location[0]][eddy_location[1]-i-1]
                except:
                    pass
                try:
                    change_max = max_X - eta[eddy_location[0]][eddy_location[1]+i+1]
                except:
                    pass

                # Test conditions if extent is reached
                if change_max < 0 and condXmax == False:
                    condXmax = eddy_location[1] + i
                if change_min < 0 and condXmin == False:
                    condXmin = eddy_location[1] - i
                
                
        
            if condXmax==False or condXmin==False:
                pass
                print('Error in X domain: No change in eta detected. To low extent etc')
            X_axis = [condXmin,condXmax]
                        
            return X_axis
        
        elif cold:
            for i in range(0,100): # 100*2km radius
                # Test is extent is found and stop for loop
                if -1-i+eddy_location[1] < 0 or 1+i+eddy_location[1] > len(Eta.X)-1:
                    if -1-i+eddy_location[1] < 0:
                        condXmin = True
                    if 1+i+eddy_location[1] > len(Eta.X)-1:
                        condXmax = True
                if condXmax and condXmin:
                    continue
                
                try:
                    min_X = eta[eddy_location[0]][eddy_location[1]-i]
                except:
                    pass
                try:
                    max_X = eta[eddy_location[0]][eddy_location[1]+i]
                except:
                    pass

                # Check the change in SSH level from each point outwards from center
                try:
                    change_min = min_X - eta[eddy_location[0]][eddy_location[1]-i-1]
                except:
                    pass
                try:
                    change_max = max_X - eta[eddy_location[0]][eddy_location[1]+i+1]
                except:
                    pass

                # Test conditions if extent is reached
                if change_max > 0 and condXmax == False:
                    condXmax = eddy_location[1] + i
                if change_min > 0 and condXmin == False:
                    condXmin = eddy_location[1] - i
                
        
            if condXmax==False or condXmin==False:
                pass
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
                if -1-i+eddy_location[1] < 0 or 1+i+eddy_location[1] > len(Eta.X)-1:
                    if -1-i+eddy_location[1] < 0:
                        condXmin = True
                    if 1+i+eddy_location[1] > len(Eta.X)-1:
                        condXmax = True
                if -1-i+eddy_location[0] < 0 or 1+i+eddy_location[0] > len(Eta.Y)-1:
                    if -1-i+eddy_location[0] < 0:
                        condYmin = True
                    if 1+i+eddy_location[0] > len(Eta.Y)-1:
                        condYmax = True

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
                    condXmax = [eddy_location[1]+i,eddy_location[0]+i]
                if change_minX < 0 and condXmin == False:
                    condXmin = [eddy_location[1]-i,eddy_location[0]-i]
                if change_minY < 0 and condYmin == False:
                    condYmin = [eddy_location[1]-i,eddy_location[0]+i]
                if change_maxY < 0 and condYmax == False:
                    condYmax = [eddy_location[1]+i,eddy_location[0]-i]
                
                
        
            if condXmax==False or condXmin==False or condYmin==False or condYmax==False:
                pass
                print('Error in XY domain: No change in eta detected. To low extent etc')
            Axis = [condXmin,condXmax,condYmin,condYmax]
                        
            return Axis
        
        elif cold:
            for i in range(0,47): # 100*2km radius
                # Test is extent is found and stop for loop
                # Test is extent is found and stop for loop
                if -1-i+eddy_location[1] < 0 or 1+i+eddy_location[1] > len(Eta.X)-1:
                    if -1-i+eddy_location[1] < 0:
                        condXmin = True
                    if 1+i+eddy_location[1] > len(Eta.X)-1:
                        condXmax = True
                if -1-i+eddy_location[0] < 0 or 1+i+eddy_location[0] > len(Eta.Y)-1:
                    if -1-i+eddy_location[0] < 0:
                        condYmin = True
                    if 1+i+eddy_location[0] > len(Eta.Y)-1:
                        condYmax = True

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
                if change_maxX > 0 and condXmax == False:
                    condXmax = [eddy_location[1]+i,eddy_location[0]+i]
                if change_minX > 0 and condXmin == False:
                    condXmin = [eddy_location[1]-i,eddy_location[0]-i]
                if change_minY > 0 and condYmin == False:
                    condYmin = [eddy_location[1]-i,eddy_location[0]+i]
                if change_maxY > 0 and condYmax == False:
                    condYmax = [eddy_location[1]+i,eddy_location[0]-i]
                
                
        
            if condXmax==False or condXmin==False or condYmin==False or condYmax==False:
                pass
                print('Error in XY domain: No change in eta detected. To low extent etc')
            Axis = [condXmin,condXmax,condYmin,condYmax]
                        
            return Axis


    def pos_Y_search(eddy_location=list(),warm=False,cold=False):
        condYmax = 0
        condYmin = 0
        if warm:
            for i in range(0,47): # 47*2km radius
                # Test is extent is found and stop for loop
                if -1-i+eddy_location[0] < 0 or 1+i+eddy_location[0] > len(Eta.Y)-1:
                    if -1-i+eddy_location[0] < 0:
                        condYmin = True
                    if 1+i+eddy_location[0] > len(Eta.Y)-1:
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
                    condYmax = eddy_location[0] + i
                if change_min < 0 and condYmin == False:
                    condYmin = eddy_location[0] - i
                
            if condYmax==False or condYmin==False:
                pass
                print('Error in Y domain: No change in eta detected. To low extent etc')
            X_axis = [condYmin,condYmax]
                        
            return X_axis
        
        elif cold:
            for i in range(0,47): # 47*2km radius
                if -1-i+eddy_location[0] < 0 or 1+i+eddy_location[0] > 46:
                    if -1-i+eddy_location[0] < 0:
                        condYmin = True
                    if 1+i+eddy_location[0] > 46:
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
                    condYmax = eddy_location[0] + i
                elif change_min > 0 and condYmin == False:
                    condYmin = eddy_location[0] - i
        
            if condYmax==False or condYmin==False:
                pass
                print('Error in Y domain: No change in eta detected. To low extent etc')
            X_axis = [condYmin,condYmax]
                        
            return X_axis


    def area_of_inner_eddy(threshold=float(),domainX=list(),domainY=list(),warm=False,cold=False,eddies=eddies):
        if warm:
            # Search domain error fix, sets to max extent of grid, if not, uses found domain
            if domainX == [0,0]:
                Xrange = np.arange(len(Eta.X))
            else:
                Xrange = np.arange(domainX[0],domainX[1])
            if domainY == [0,0]:
                Yrange = np.arange(len(Eta.Y))
            else:
                Yrange = np.arange(domainY[0],domainY[1])

            # Changing values in eddies dataset for inner region
            for j in Yrange:
                for i in Xrange:
                    if eta[j][i] < threshold or eddies[j][i] == 2: # value 1 indicates area of inner warm eddy, add more for other layers
                        continue
                    eddies[j][i] = 1
            
            # Check for outliers
            test = eddies
            for j in Yrange:
                for i in Xrange:
                    if eddies[j][i] != 1: # value 1 indicates area of inner warm eddy, add more for other layers
                        continue
                    if eddies[j][i-1] !=1 and eddies[j][i+1] !=1 and eddies[j][i-2] !=1 and eddies[j][i+2] !=1:
                        test[j][i] = 0
                    if eddies[j-1][i] !=1 and eddies[j+1][i] !=1 and eddies[j-2][i] !=1 and eddies[j+2][i] !=1:
                        test[j][i] = 0
            eddies = test

        elif cold:
            # Search domain error fix, sets to max extent of grid, if not, uses found domain
            if domainX == [0,0]:
                Xrange = np.arange(len(eta.X))
            else:
                Xrange = np.arange(domainX[0],domainX[1])
            if domainY == [0,0]:
                Yrange = np.arange(len(Eta.Y))
            else:
                Yrange = np.arange(domainY[0],domainY[1])

            # Changing values in eddies dataset for inner region
            for j in Yrange:
                for i in Xrange:
                    if eta[j][i] > threshold or eddies[j][i] == 1: # value 2 indicates area of inner eddy, add more for other layers
                        continue
                    eddies[j][i] = 2
            
            # Check for outliers
            test = eddies
            for j in Yrange:
                for i in Xrange:
                    if eddies[j][i] != 2: # value 2 indicates area of inner cold eddy, add more for other layers
                        continue
                    if eddies[j][i-1] !=2 and eddies[j][i+1] !=2 and eddies[j][i-2] !=2 and eddies[j][i+2] !=2:
                        test[j][i] = 0
                    if eddies[j-1][i] !=2 and eddies[j+1][i] !=2 and eddies[j-2][i] !=2 and eddies[j+2][i] !=2:
                        test[j][i] = 0
            eddies = test
        
        return eddies


    if warm:
        # Computes domain of eddy
        domainX = pos_X_search(eddy_center,warm=True)
        domainY = pos_Y_search(eddy_center,warm=True)
        domainXY = pos_XY_search(eddy_center,warm=True)
        # print('Domain of eddy: ',[domainX,domainY])
        # print('Domain of eddy XY: ', domainXY)

        totDomain = [[],[]]
        if domainX[0] != 0 and not isinstance(domainX[0], bool):
            totDomain[0].append(domainX[0])
        if domainX[1] != 0 and not isinstance(domainX[1], bool):
            totDomain[0].append(domainX[1])
        if domainY[0] != 0 and not isinstance(domainY[0], bool):
            totDomain[1].append(domainY[0])
        if domainY[1] != 0 and not isinstance(domainY[1], bool):
            totDomain.append(domainY[1])
        for i in domainXY:
            try:
                if i[0] != 0 and not isinstance(i[0], bool):
                    totDomain[0].append(i[0])
            except:
                continue
            if i[1] != 0 and not isinstance(i[1], bool):
                totDomain[1].append(i[1])
        
        # Find the maximum eta value all boundaries to use as threshold
        max_eta_boundary_coord = []
        max_eta_boundary_value = []
        for j in totDomain[1]:
            for i in totDomain[0]:
                max_eta_boundary_coord.append([i,j])
        if domainX[0] != 0 and not isinstance(domainX[0], bool):
            max_eta_boundary_coord.append([domainX[0],eddy_center[0]])
        if domainX[1] != 0 and not isinstance(domainX[1], bool):
            max_eta_boundary_coord.append([domainX[1],eddy_center[0]])
        if domainY[0] != 0 and not isinstance(domainY[0], bool):    
            max_eta_boundary_coord.append([eddy_center[1],domainY[0]])
        if domainY[1] != 0 and not isinstance(domainY[1], bool):
            max_eta_boundary_coord.append([eddy_center[1],domainY[1]])
        
        for i in max_eta_boundary_coord:
            try:
                max_eta_boundary_value.append(eta[i[1]][i[0]])
            except:
                continue
        try:
            threshold = np.max(max_eta_boundary_value)

            try:
                totDomain = [np.min(totDomain[0]),np.max(totDomain[0])],[np.min(totDomain[1]),np.max(totDomain[1])]
                dataset = area_of_inner_eddy(threshold=threshold,domainX=[totDomain[0][0],totDomain[0][1]],domainY=[totDomain[1][0],totDomain[1][1]],warm=True,eddies=eddies)
            except:
                dataset = area_of_inner_eddy(threshold=threshold,domainX=[0,0],domainY=[0,0],warm=True,eddies=eddies)
        except:
            print('Error in eddy: ',eddy_center,' Skiped')

        
    elif cold:
        domainX = pos_X_search(eddy_center,cold=True)
        domainY = pos_Y_search(eddy_center,cold=True)
        domainXY = pos_XY_search(eddy_center,cold=True)
        # print('Domain of eddy: ',[domainX,domainY])
        # print('Domain of eddy XY: ', domainXY)

        totDomain = [[],[]]
        if domainX[0] != 0 and not isinstance(domainX[0], bool):
            totDomain[0].append(domainX[0])
        if domainX[1] != 0 and not isinstance(domainX[1], bool):
            totDomain[0].append(domainX[1])
        if domainY[0] != 0 and not isinstance(domainY[0], bool):
            totDomain[1].append(domainY[0])
        if domainY[1] != 0 and not isinstance(domainY[1], bool):
            totDomain.append(domainY[1])
        for i in domainXY:
            try:
                if i[0] != 0 and not isinstance(i[0], bool):
                    totDomain[0].append(i[0])
            except:
                continue
            if i[1] != 0 and not isinstance(i[1], bool):
                totDomain[1].append(i[1])
        
        # Find the minimum eta value all boundaries to use as threshold
        min_eta_boundary_coord = []
        min_eta_boundary_value = []
        for j in totDomain[1]:
            for i in totDomain[0]:
                min_eta_boundary_coord.append([i,j])
        if domainX[0] != 0 and not isinstance(domainX[0], bool):
            min_eta_boundary_coord.append([domainX[0],eddy_center[0]])
        if domainX[1] != 0 and not isinstance(domainX[1], bool):
            min_eta_boundary_coord.append([domainX[1],eddy_center[0]])
        if domainY[0] != 0 and not isinstance(domainY[0], bool):    
            min_eta_boundary_coord.append([eddy_center[1],domainY[0]])
        if domainY[1] != 0 and not isinstance(domainY[1], bool):
            min_eta_boundary_coord.append([eddy_center[1],domainY[1]])
        
        for i in min_eta_boundary_coord:
            try:
                min_eta_boundary_value.append(eta[i[1]][i[0]])
            except:
                continue
        try:
            threshold = np.min(min_eta_boundary_value)
            try:
                totDomain = [np.min(totDomain[0]),np.max(totDomain[0])],[np.min(totDomain[1]),np.max(totDomain[1])]
                dataset = area_of_inner_eddy(threshold=threshold,domainX=[totDomain[0][0],totDomain[0][1]],domainY=[totDomain[1][0],totDomain[1][1]],cold=True,eddies=eddies)
            except:
                dataset = area_of_inner_eddy(threshold=threshold,domainX=[0,0],domainY=[0,0],cold=True,eddies=eddies)
        except:
            print('Error in eddy: ',eddy_center,' Skiped')
    
    try:
        return dataset
    except:
        return eddies


def outer_eddy_region(hor_vel,eddiesDataset):
    """
    Detection of stream and current area around eddies.
    """
    data = eddiesDataset.copy()

    # Thresholds
    current = 0.5 # m/s and greater
    T = len(eddiesDataset.Y)
    pbar = tqdm(total=T, desc="Running outer region algorythm")

    for j in range(len(eddiesDataset.Y)):
        pbar.update(1)
        for i in range(len(eddiesDataset.X)):
            if data[j][i] == 0:
                if hor_vel[j][i].values >= current:
                    data[j,i] = 3
                elif i>=10 and i<=len(eddiesDataset.X)-10 and j>=10 and j<=len(eddiesDataset.Y)-10 and hor_vel[j][i].values > 0.2 and hor_vel[j][i] < current and eddiesDataset[j-10:j+10,i-10:i+10].max()>0:
                    data[j,i] = 4

    pbar.close()
    # print('Test of max value: ',data.max())
    return data


def newEddyMethod(eddy_center=list,OW=xr.DataArray,hor_vel=xr.DataArray,eddiesDataset=xr.DataArray,warm=False,cold=False):
    """
    Using the eddy search algorythm based on Matsuoka et al. 2016
    """
    # Transform eddy centerpoints from coords to index position from eta array
    eddy_center = [np.argwhere(OW.Y.values == eddy_center[0])[0][0],np.argwhere(OW.X.values == eddy_center[1])[0][0]]
    eddies = eddiesDataset


    def pos_X_search(eddy_location=list()):
        condXmax = 0
        condXmin = 0

        for i in range(0,100): # 100*2km radius
            # Test is extent is found and stop for loop
            if -1-i+eddy_location[1] < 0 or 1+i+eddy_location[1] > len(OW.X)-1:
                if -1-i+eddy_location[1] < 0:
                    condXmin = True
                if 1+i+eddy_location[1] > len(OW.X)-1:
                    condXmax = True
            if condXmax and condXmin:
                break
            
            max_extent = OW[eddy_location[0],eddy_location[1]+i+1]
            min_extent = OW[eddy_location[0],eddy_location[1]-i-1]

            # Test conditions if extent is reached
            if max_extent >= 0 and condXmax == False:
                condXmax = eddy_location[1] + i
            if min_extent >= 0 and condXmin == False:
                condXmin = eddy_location[1] - i
    
        if condXmax==False or condXmin==False:
            print('Error in X domain: No change in Okubo-Weiss detected. To low extent etc')
        X_axis = [condXmin,condXmax]
                    
        return X_axis


    def pos_XY_search(eddy_location=list()):
        condXmax = 0
        condXmin = 0
        condYmax = 0
        condYmin = 0
        
        for i in range(0,100): # 100*2km radius
            # Test is extent is found and stop for loop
            if -1-i+eddy_location[1] < 0 or 1+i+eddy_location[1] > len(OW.X)-1:
                if -1-i+eddy_location[1] < 0:
                    condXmin = True
                if 1+i+eddy_location[1] > len(OW.X)-1:
                    condXmax = True
            if -1-i+eddy_location[0] < 0 or 1+i+eddy_location[0] > len(OW.Y)-1:
                if -1-i+eddy_location[0] < 0:
                    condYmin = True
                if 1+i+eddy_location[0] > len(OW.Y)-1:
                    condYmax = True

            if condXmax and condXmin and condYmax and condYmin:
                break


            max_X_extent = OW[eddy_location[0]+i+1,eddy_location[1]+i+1]
            min_X_extent = OW[eddy_location[0]-i-1,eddy_location[1]+i+1]
            max_Y_extent = OW[eddy_location[0]+i+1,eddy_location[1]-i-1]
            min_Y_extent = OW[eddy_location[0]-i-1,eddy_location[1]-i-1]


            # Test conditions if extent is reached
            if max_X_extent >= 0 and condXmax == False:
                condXmax = [eddy_location[1]+i,eddy_location[0]+i]
            if min_X_extent >= 0 and condXmin == False:
                condXmin = [eddy_location[1]-i,eddy_location[0]+i]
            if min_Y_extent >= 0 and condYmin == False:
                condYmin = [eddy_location[1]-i,eddy_location[0]-i]
            if max_Y_extent >= 0 and condYmax == False:
                condYmax = [eddy_location[1]+i,eddy_location[0]-i]
            
            
    
        if condXmax==False or condXmin==False or condYmin==False or condYmax==False:
            pass
            print('Error in XY domain: No change in eta detected. To low extent etc')
        Axis = [condXmin,condXmax,condYmin,condYmax]
                    
        return Axis


    def pos_Y_search(eddy_location=list()):
        condYmax = 0
        condYmin = 0
        for i in range(0,100): # 100*2km radius
            # Test is extent is found and stop for loop
            if -1-i+eddy_location[0] < 0 or 1+i+eddy_location[0] > len(OW.Y)-1:
                if -1-i+eddy_location[0] < 0:
                    condYmin = True
                if 1+i+eddy_location[0] > len(OW.Y)-1:
                    condYmax = True
            if condYmax and condYmin:
                continue
                
            max_Y = OW[eddy_location[0]+i+1,eddy_location[1]]
            min_Y = OW[eddy_location[0]-i-1,eddy_location[1]]
            
            # Test conditions if extent is reached
            if max_Y >= 0 and condYmax == False:
                condYmax = eddy_location[0] + i
            if min_Y >= 0 and condYmin == False:
                condYmin = eddy_location[0] - i
            
        if condYmax==False or condYmin==False:
            pass
            print('Error in Y domain: No change in eta detected. To low extent etc')
        X_axis = [condYmin,condYmax]
                    
        return X_axis


    def inner_domain(center=eddy_center,domain=list,warm=False,cold=False,eddies=eddies):
        domain_search = int(np.max([np.max(domain[0])-np.min(domain[0]),np.max(domain[1])-np.min(domain[1])]))
        
        if warm:
            # Find areas with OW < 0 for each step outwards from the center

            # Reached limit conditions
            Xpos = True
            Xneg = True
            Ypos = True
            Yneg = True
            X = center[1]
            Y = center[0]
            eddies[Y,X] = 1
            for i in range(1,domain_search):
                if Xpos==False and Xneg==False and Ypos==False and Yneg==False:
                    break

                # X-Positive
                if Xpos:
                    OW_test = OW[Y-(i-1):Y+(i),X+i]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y-(i-1):Y+(i),X+i] = eddies[Y-(i-1):Y+(i),X+i].where(criteria,other=1)
                    if eddies[Y-(i-1):Y+(i),X+i].max() != 1:
                        Xpos = False
                # X-Negative
                if Xneg:
                    OW_test = OW[Y-(i-1):Y+(i),X-i]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y-(i-1):Y+(i),X-i] = eddies[Y-(i-1):Y+(i),X-i].where(criteria,other=1)
                    if eddies[Y-(i-1):Y+(i),X-i].max() != 1:
                        Xneg = False
                # Y-Positive
                if Ypos:
                    OW_test = OW[Y+i,X-i:X+i+1]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y+i,X-i:X+i+1] = eddies[Y+i,X-i:X+i+1].where(criteria, other=1)
                    if eddies[Y+i,X-i:X+i+1].max() != 1:
                        Ypos = False
                # Y-Negative
                if Yneg:
                    OW_test = OW[Y-i,X-i:X+i+1]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y-i,X-i:X+i+1] = eddies[Y-i,X-i:X+i+1].where(criteria, other=1)
                    if eddies[Y-i,X-i:X+i+1].max() != 1:
                        Yneg = False
            
            # # Filter out outliers
            # data = eddies
            # cleaned = eddies
            # for j in range(domain_search-Y,domain_search+Y+1):
            #     for i in range(domain_search-X,domain_search+X+1):
            #         if data[j,i] == 1:
            #             if data[j,i+1] != 1 and data[j,i-1] != 1 and data[j-1,i] != 1 and data[j+1,i] !=1:
            #                 cleaned[j,i] = 0 
                            
            # eddies = cleaned
            return eddies
        
        elif cold:
            # Find areas with OW < 0 for each step outwards from the center

            # Reached limit conditions
            Xpos = True
            Xneg = True
            Ypos = True
            Yneg = True
            X = center[1]
            Y = center[0]
            eddies[Y,X] = 2
            for i in range(1,domain_search):
                if Xpos==False and Xneg==False and Ypos==False and Yneg==False:
                    break

                # X-Positive
                if Xpos:
                    OW_test = OW[Y-(i-1):Y+(i),X+i]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y-(i-1):Y+(i),X+i] = eddies[Y-(i-1):Y+(i),X+i].where(criteria,other=2)
                    if eddies[Y-(i-1):Y+(i),X+i].max() != 2:
                        Xpos = False
                # X-Negative
                if Xneg:
                    OW_test = OW[Y-(i-1):Y+(i),X-i]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y-(i-1):Y+(i),X-i] = eddies[Y-(i-1):Y+(i),X-i].where(criteria,other=2)
                    if eddies[Y-(i-1):Y+(i),X-i].max() != 2:
                        Xneg = False
                # Y-Positive
                if Ypos:
                    OW_test = OW[Y+i,X-i:X+i+1]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y+i,X-i:X+i+1] = eddies[Y+i,X-i:X+i+1].where(criteria, other=2)
                    if eddies[Y+i,X-i:X+i+1].max() != 2:
                        Ypos = False
                # Y-Negative
                if Yneg:
                    OW_test = OW[Y-i,X-i:X+i+1]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y-i,X-i:X+i+1] = eddies[Y-i,X-i:X+i+1].where(criteria, other=2)
                    if eddies[Y-i,X-i:X+i+1].max() != 2:
                        Yneg = False
            
            # # Filter out outliers
            # data = eddies
            # cleaned = eddies
            # for j in range(domain_search-Y,domain_search+Y+1):
            #     for i in range(domain_search-X,domain_search+X+1):
            #         if data[j,i] == 2:
            #             if data[j,i+1] != 2 and data[j,i-1] != 2 and data[j-1,i] != 2 and data[j+1,i] !=2:
            #                 cleaned[j,i] = 0 
                            
            # eddies = cleaned

            return eddies


    def outer_domain(hor_vel=hor_vel):
        pass


    def currents(hor_vel=hor_vel):
        pass


    domainX = pos_X_search(eddy_center)
    domainY = pos_Y_search(eddy_center)
    domainXY = pos_XY_search(eddy_center)
    # print('Domain of eddy: ',[domainX,domainY])
    # print('Domain of eddy XY: ', domainXY)

    totDomain = [[],[]]
    if domainX[0] != 0 and not isinstance(domainX[0], bool):
        totDomain[0].append(domainX[0])
    if domainX[1] != 0 and not isinstance(domainX[1], bool):
        totDomain[0].append(domainX[1])
    if domainY[0] != 0 and not isinstance(domainY[0], bool):
        totDomain[1].append(domainY[0])
    if domainY[1] != 0 and not isinstance(domainY[1], bool):
        totDomain.append(domainY[1])
    for i in domainXY:
        try:
            if i[0] != 0 and not isinstance(i[0], bool):
                totDomain[0].append(i[0])
        except:
            continue
        if i[1] != 0 and not isinstance(i[1], bool):
            totDomain[1].append(i[1])

    if warm:
        eddies = inner_domain(center=eddy_center,domain=totDomain,warm=True,eddies=eddies)

    elif cold:
        eddies = inner_domain(center=eddy_center,domain=totDomain,cold=True,eddies=eddies)

    return eddies

