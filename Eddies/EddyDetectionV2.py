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

    for j in range(4, mxbndY + 1):
        for i in range(4, mxbndX + 1):
            if Okubo_Weiss_np[j, i] >= std_OW:  # Boundary conditions
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

    return max_coord, min_coord


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

            max_pos_x = np.max(domain[0]) - X
            max_pos_y = np.max(domain[1]) - Y
            min_pos_x = X - np.min(domain[0])
            min_pos_y = Y - np.min(domain[1])

            eddies[Y,X] = 1
            for i in range(1,domain_search):
                if Xpos==False and Xneg==False and Ypos==False and Yneg==False:
                    break

                # X-Positive
                if Xpos and i<=max_pos_x:
                    OW_test = OW[Y-(i-1):Y+(i),X+i]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y-(i-1):Y+(i),X+i] = eddies[Y-(i-1):Y+(i),X+i].where(criteria,other=1)
                    if eddies[Y-(i-1):Y+(i),X+i].max() != 1:
                        Xpos = False
                # X-Negative
                if Xneg and i<=min_pos_x:
                    OW_test = OW[Y-(i-1):Y+(i),X-i]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y-(i-1):Y+(i),X-i] = eddies[Y-(i-1):Y+(i),X-i].where(criteria,other=1)
                    if eddies[Y-(i-1):Y+(i),X-i].max() != 1:
                        Xneg = False
                # Y-Positive
                if Ypos and i<=max_pos_y:
                    OW_test = OW[Y+i,X-i:X+i+1]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y+i,X-i:X+i+1] = eddies[Y+i,X-i:X+i+1].where(criteria, other=1)
                    if eddies[Y+i,X-i:X+i+1].max() != 1:
                        Ypos = False
                # Y-Negative
                if Yneg and i<=min_pos_y:
                    OW_test = OW[Y-i,X-i:X+i+1]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y-i,X-i:X+i+1] = eddies[Y-i,X-i:X+i+1].where(criteria, other=1)
                    if eddies[Y-i,X-i:X+i+1].max() != 1:
                        Yneg = False
        
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

            max_pos_x = np.max(domain[0]) - X
            max_pos_y = np.max(domain[1]) - Y
            min_pos_x = X - np.min(domain[0])
            min_pos_y = Y - np.min(domain[1])

            eddies[Y,X] = 2
            for i in range(1,domain_search):
                if Xpos==False and Xneg==False and Ypos==False and Yneg==False:
                    break

                # X-Positive
                if Xpos and i<=max_pos_x:
                    OW_test = OW[Y-(i-1):Y+(i),X+i]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y-(i-1):Y+(i),X+i] = eddies[Y-(i-1):Y+(i),X+i].where(criteria,other=2)
                    if eddies[Y-(i-1):Y+(i),X+i].max() != 2:
                        Xpos = False
                # X-Negative
                if Xneg and i<=min_pos_x:
                    OW_test = OW[Y-(i-1):Y+(i),X-i]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y-(i-1):Y+(i),X-i] = eddies[Y-(i-1):Y+(i),X-i].where(criteria,other=2)
                    if eddies[Y-(i-1):Y+(i),X-i].max() != 2:
                        Xneg = False
                # Y-Positive
                if Ypos and i<=max_pos_y:
                    OW_test = OW[Y+i,X-i:X+i+1]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y+i,X-i:X+i+1] = eddies[Y+i,X-i:X+i+1].where(criteria, other=2)
                    if eddies[Y+i,X-i:X+i+1].max() != 2:
                        Ypos = False
                # Y-Negative
                if Yneg and i<=min_pos_y:
                    OW_test = OW[Y-i,X-i:X+i+1]
                    criteria = (OW_test >= 0) | (OW_test.isnull())
                    eddies[Y-i,X-i:X+i+1] = eddies[Y-i,X-i:X+i+1].where(criteria, other=2)
                    if eddies[Y-i,X-i:X+i+1].max() != 2:
                        Yneg = False

            return eddies


    def outer_domain(hor_vel=hor_vel):
        pass

        def search_velocity_maximum():
            pass
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
            
            def pos_X_search(eddy_location=list()):
                condXmax = 0
                condXmin = 0

                for i in range(0,100): # 100*2km radius
                    # Test is extent is found and stop for loop
                    if -1-i+eddy_location[1] < 0 or 1+i+eddy_location[1] > len(hor_vel.X)-1:
                        if -1-i+eddy_location[1] < 0:
                            condXmin = True
                        if 1+i+eddy_location[1] > len(OW.X)-1:
                            condXmax = True
                    if condXmax and condXmin:
                        break
                    
                    max_extent = hor_vel[eddy_location[0],eddy_location[1]+i+1] - hor_vel[eddy_location[0],eddy_location[1]]
                    min_extent = hor_vel[eddy_location[0],eddy_location[1]-i-1] - hor_vel[eddy_location[0],eddy_location[1]]

                    # Test conditions if extent is reached
                    if max_extent <= 0.01 and condXmax == False:
                        condXmax = eddy_location[1] + i
                    if min_extent <= 0.01 and condXmin == False:
                        condXmin = eddy_location[1] - i
            
                if condXmax==False or condXmin==False:
                    print('Error in X domain: No change in Okubo-Weiss detected. To low extent etc')
                X_axis = [condXmin,condXmax]
                            
                return X_axis


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


def outer_eddy_region(hor_vel,eddiesDataset,eddycenters):
            """
            Detection of stream and current area around eddies.
            """
            data = eddiesDataset.copy()

            # Thresholds
            current = 0.5 # m/s and greater

            X_max = 0
            X_min = 0
            Y_max = 0
            Y_min = 0
            XY_max = 0
            XY_min = 0

            for pos in eddycenters:
                center_vel = hor_vel[pos[0]][pos[1]].values
                for dist in range(1,100):
                    pass

            for j in range(len(eddiesDataset.Y)):
                for i in range(len(eddiesDataset.X)):
                    if data[j][i].values == 0:
                        if hor_vel[j][i].values >= current:
                            data[j,i] = 3
                        elif i>=10 and i<=len(eddiesDataset.X)-10 and j>=10 and j<=len(eddiesDataset.Y)-10 and hor_vel[j][i].values > 0.2 and hor_vel[j][i].values < current and eddiesDataset[j-15:j+15,i-15:i+15].max().values>0:
                            data[j,i] = 4
            return data