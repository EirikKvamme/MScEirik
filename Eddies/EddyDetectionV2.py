import xarray as xr
import matplotlib.pyplot as plt
import oceanspy as ospy
import numpy as np
from tqdm import tqdm
from matplotlib.path import Path
from scipy.ndimage import laplace

# def function for detection of surface anom creating eddies

def eddyDetection(SSH, Okubo_Weiss, threshold):
    """
    Locates local max/min of SSH based threshold Okubo-Weiss parameter
    """
    
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
            if Okubo_Weiss_np[j, i] >= threshold:  # Boundary conditions
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
            # elif center > np.max([ # Added to detect missing eddies / Does not solve it
            #     SSH[j+2, i],
            #     SSH[j+1, i-1], SSH[j+1, i+1],
            #     SSH[j, i-2], SSH[j, i+2],
            #     SSH[j-1, i-1], SSH[j-1, i+1],
            #     SSH[j-2, i]
            # ]):
            #     max_coord.append([Y[j], X[i]])
            
            elif center < np.min(neighbors):
                min_coord.append([Y[j], X[i]])
            # elif center < np.min([ # Added to detect missing eddies / Does not solve it
            #     SSH[j+2, i],
            #     SSH[j+1, i-1], SSH[j+1, i+1],
            #     SSH[j, i-2], SSH[j, i+2],
            #     SSH[j-1, i-1], SSH[j-1, i+1],
            #     SSH[j-2, i]
            # ]):
            #     min_coord.append([Y[j], X[i]])

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


def outer_eddy_region(hor_vel,eddiesDataset): # eddycenters
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

            # for pos in eddycenters: # Working on
            #     center_vel = hor_vel[pos[0]][pos[1]].values
            #     for dist in range(1,100):
            #         pass

            for j in range(len(eddiesDataset.Y)):
                for i in range(len(eddiesDataset.X)):
                    if data[j][i].values == 0:
                        if hor_vel[j][i].values >= current:
                            data[j,i] = 3
                        elif i>=10 and i<=len(eddiesDataset.X)-10 and j>=10 and j<=len(eddiesDataset.Y)-10 and hor_vel[j][i].values > 0.2 and hor_vel[j][i].values < current and eddiesDataset[j-15:j+15,i-15:i+15].max().values>0:
                            data[j,i] = 4
            return data


def inner_eddy_region_v3(eddyCenterpoints=list,eta=xr.DataArray(),cold=False,warm=False,test_calib=False,eddiesData=xr.DataArray()):
    """
    Inner eddy detection utilizing maximum outer closed contour around each eddy center point.\n
    Warm: Local maximum, Cold: Local minimum, test_calib: Plots contour field with max outer closed contour.\n
    Previously saved eddies dataset can be utilized for further additions.
    """
    data = eddiesData

    if warm:
        for center in eddyCenterpoints:
            eta_around_center = eta.sel(X=slice(center[1]-2.5,center[1]+2.5)).sel(Y=slice(center[0]-2.5,center[0]+2.5))

            # Step 1: define inflexion point for level in contour
            x_indices_center = np.where((eta_around_center.X == center[1]) & (eta_around_center.X == center[1]))[0][0]
            y_indices_center = np.where((eta_around_center.Y == center[0]) & (eta_around_center.Y == center[0]))[0][0]

            max_extent_search = np.min([len(eta_around_center.X.values)-x_indices_center,len(eta_around_center.Y)-y_indices_center])

            eta_levels = []

            for i in range(0,max_extent_search-2): # Second derivative utilizing the forward method
                X_pos = eta_around_center[y_indices_center][x_indices_center+i+2] - 2*eta_around_center[y_indices_center][x_indices_center+i+1] + eta_around_center[y_indices_center][x_indices_center+i]
                if X_pos >= 0:
                    eta_levels.append(eta_around_center[y_indices_center][x_indices_center+i].values)
                    if test_calib:
                        print('Local max i found',i)
                    break

            for i in range(0,max_extent_search): # Second derivative utilizing the forward method
                XY_pos = eta_around_center[y_indices_center+i+2][x_indices_center+i+2] - 2*eta_around_center[y_indices_center+i+1][x_indices_center+i+1] + eta_around_center[y_indices_center+i][x_indices_center+i]
                if XY_pos >= 0:
                    eta_levels.append(eta_around_center[y_indices_center+i][x_indices_center+i].values)
                    if test_calib:
                        print('Local max i found',i)
                    break

            for i in range(0,max_extent_search): # Second derivative utilizing the forward method
                Y_pos = eta_around_center[y_indices_center+i+2][x_indices_center] - 2*eta_around_center[y_indices_center+i+1][x_indices_center] + eta_around_center[y_indices_center+i][x_indices_center]
                if Y_pos >= 0:
                    eta_levels.append(eta_around_center[y_indices_center+i][x_indices_center].values)
                    if test_calib:
                        print('Local max i found',i)
                    break

            for i in range(0,max_extent_search): # Second derivative utilizing the forward method
                X_neg_Y_pos = eta_around_center[y_indices_center+i+2][x_indices_center-i-2] - 2*eta_around_center[y_indices_center+i+1][x_indices_center-i-1] + eta_around_center[y_indices_center+i][x_indices_center-i]
                if X_neg_Y_pos >= 0:
                    eta_levels.append(eta_around_center[y_indices_center+i][x_indices_center-i].values)
                    if test_calib:
                        print('Local max i found',i)
                    break

            for i in range(0,max_extent_search): # Second derivative utilizing the forward method
                X_neg = eta_around_center[y_indices_center][x_indices_center-i-2] - 2*eta_around_center[y_indices_center][x_indices_center-i-1] + eta_around_center[y_indices_center][x_indices_center-i]
                if X_neg >= 0:
                    eta_levels.append(eta_around_center[y_indices_center][x_indices_center-i])
                    if test_calib:
                        print('Local max i found',i)
                    break

            for i in range(0,max_extent_search): # Second derivative utilizing the forward method
                XY_neg = eta_around_center[y_indices_center-i-2][x_indices_center-i-2] - 2*eta_around_center[y_indices_center-i-1][x_indices_center-i-1] + eta_around_center[y_indices_center-i][x_indices_center-i]
                if XY_neg >= 0:
                    eta_levels.append(eta_around_center[y_indices_center-i][x_indices_center-i].values)
                    if test_calib:
                        print('Local max i found',i)
                    break

            for i in range(0,max_extent_search): # Second derivative utilizing the forward method
                Y_neg = eta_around_center[y_indices_center-i-2][x_indices_center] - 2*eta_around_center[y_indices_center-i-1][x_indices_center] + eta_around_center[y_indices_center-i][x_indices_center]
                if Y_neg >= 0:
                    eta_levels.append(eta_around_center[y_indices_center-i][x_indices_center])
                    if test_calib:
                        print('Local max i found',i)
                    break

            for i in range(0,max_extent_search): # Second derivative utilizing the forward method
                X_pos_Y_neg = eta_around_center[y_indices_center-i-2][x_indices_center+i+2] - 2*eta_around_center[y_indices_center-i-1][x_indices_center+i+1] + eta_around_center[y_indices_center-i][x_indices_center+i]
                if X_pos_Y_neg >= 0:
                    eta_levels.append(eta_around_center[y_indices_center-i][x_indices_center+i].values)
                    if test_calib:
                        print('Local max i found',i)
                    break

            max_eta = np.argmax(eta_levels)

            levels = [
                eta_levels[max_eta]
            ]
            levels_array = np.array(levels)

            # Remove duplicates and sort the array
            levels = np.unique(levels_array)

            # Step 2: Generate Contours

            contours = plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels)
            plt.close()
            # Step 3: Collect all X and Y points of the paths of all contours
            all_contour_points = []

            for collection in contours.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            # Step 4: Identify the Center Point
            center_point = (center[1],center[0])

            # Step 5: Process Contour Paths to Handle Jumps
            def process_contour_path(vertices, jump_threshold=0.1):
                segments = []
                current_segment = [vertices[0]]
                
                for i in range(1, len(vertices)):
                    if np.linalg.norm(vertices[i] - vertices[i-1]) > jump_threshold:
                        segments.append(np.array(current_segment))
                        current_segment = [vertices[i]]
                    else:
                        current_segment.append(vertices[i])
                
                if current_segment:
                    segments.append(np.array(current_segment))
                
                return segments

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)

            # Step 6: Find the Outermost Closed Contour Segment
            outermost_contour = None
            max_area = 0

            def is_closed_contour(vertices, tol=1e-1):
                distance = np.linalg.norm(vertices[0] - vertices[-1])
                return distance < tol

            for vertices in processed_contour_segments:
                # Create a Path object from the vertices
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-2):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    if path_obj.contains_point(center_point):
                        # Calculate the area of the contour segment using the shoelace formula
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                                    # Update the outermost contour if this one has a larger area
                        if area > max_area:
                            max_area = area
                            outermost_contour = vertices
            
            if outermost_contour is not None:
                outermost_path = Path(outermost_contour)
                mask = xr.zeros_like(eta_around_center, dtype=bool)

                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                mask_flattened = outermost_path.contains_points(points)
                mask.values = mask_flattened.reshape(mask.shape)

                # Update eddies
                x_indices = np.where((data.X >= center[1] - 2.5) & (data.X <= center[1] + 2.5))[0]
                y_indices = np.where((data.Y >= center[0] - 2.5) & (data.Y <= center[0] + 2.5))[0]

                current_subset = data.isel(X=x_indices, Y=y_indices)
                cond = (~mask) | (current_subset != 0)
                updated_subset = current_subset.where(cond, other=1)

                data.values[np.ix_(y_indices, x_indices)] = updated_subset.values

            
            if test_calib:
                # Step 7: Plot the Data and the Outermost Closed Contour Segment
                plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels)
                if outermost_contour is not None:
                    plt.plot(outermost_contour[:, 0], outermost_contour[:, 1], 'r-', linewidth=2)
                    plt.contourf(mask.X, mask.Y, mask, colors=['white', 'blue'])
                    plt.scatter(*center_point, color='red')  # Mark the center point
                    plt.title("Topographic Data LOCAL MAX with Outermost Closed Contour Segment")
                    plt.show()

                else:
                    print('Local MAX not found')
                    plt.pcolormesh(eta_around_center.X,eta_around_center.Y,eta_around_center)
                    plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels)
                    plt.title("Topographic Data LOCAL MAX with Outermost Closed Contour Segment")
                    plt.show()

                # Print the bounding box of the outermost closed contour segment
                if outermost_contour is not None:
                    min_x, min_y = np.min(outermost_contour, axis=0)
                    max_x, max_y = np.max(outermost_contour, axis=0)
                    print(f"Outermost closed contour segment bounding box: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
                else:
                    print("No closed contour segments found containing the center point.")

    if cold:
        for center in eddyCenterpoints:
            eta_around_center = eta.sel(X=slice(center[1]-2.5,center[1]+2.5)).sel(Y=slice(center[0]-2.5,center[0]+2.5))
            
            # Step 1: define inflexion point for level in contour
            x_indices_center = np.where((eta_around_center.X == center[1]) & (eta_around_center.X == center[1]))[0][0]
            y_indices_center = np.where((eta_around_center.Y == center[0]) & (eta_around_center.Y == center[0]))[0][0]

            max_extent_search = np.min([len(eta_around_center.X.values)-x_indices_center,len(eta_around_center.Y)-y_indices_center])

            eta_levels = []

            for i in range(0,max_extent_search-2): # Second derivative utilizing the forward method
                X_pos = eta_around_center[y_indices_center][x_indices_center+i+2] - 2*eta_around_center[y_indices_center][x_indices_center+i+1] + eta_around_center[y_indices_center][x_indices_center+i]
                if X_pos <= 0:
                    eta_levels.append(eta_around_center[y_indices_center][x_indices_center+i].values)
                    if test_calib:
                        print('Local max i found',i)
                    break

            for i in range(0,max_extent_search): # Second derivative utilizing the forward method
                XY_pos = eta_around_center[y_indices_center+i+2][x_indices_center+i+2] - 2*eta_around_center[y_indices_center+i+1][x_indices_center+i+1] + eta_around_center[y_indices_center+i][x_indices_center+i]
                if XY_pos <= 0:
                    eta_levels.append(eta_around_center[y_indices_center+i][x_indices_center+i].values)
                    if test_calib:
                        print('Local max i found',i)
                    break

            for i in range(0,max_extent_search): # Second derivative utilizing the forward method
                Y_pos = eta_around_center[y_indices_center+i+2][x_indices_center] - 2*eta_around_center[y_indices_center+i+1][x_indices_center] + eta_around_center[y_indices_center+i][x_indices_center]
                if Y_pos <= 0:
                    eta_levels.append(eta_around_center[y_indices_center+i][x_indices_center].values)
                    if test_calib:
                        print('Local max i found',i)
                    break

            for i in range(0,max_extent_search): # Second derivative utilizing the forward method
                X_neg_Y_pos = eta_around_center[y_indices_center+i+2][x_indices_center-i-2] - 2*eta_around_center[y_indices_center+i+1][x_indices_center-i-1] + eta_around_center[y_indices_center+i][x_indices_center-i]
                if X_neg_Y_pos <= 0:
                    eta_levels.append(eta_around_center[y_indices_center+i][x_indices_center-i].values)
                    if test_calib:
                        print('Local max i found',i)
                    break

            for i in range(0,max_extent_search): # Second derivative utilizing the forward method
                X_neg = eta_around_center[y_indices_center][x_indices_center-i-2] - 2*eta_around_center[y_indices_center][x_indices_center-i-1] + eta_around_center[y_indices_center][x_indices_center-i]
                if X_neg <= 0:
                    eta_levels.append(eta_around_center[y_indices_center][x_indices_center-i])
                    if test_calib:
                        print('Local max i found',i)
                    break

            for i in range(0,max_extent_search): # Second derivative utilizing the forward method
                XY_neg = eta_around_center[y_indices_center-i-2][x_indices_center-i-2] - 2*eta_around_center[y_indices_center-i-1][x_indices_center-i-1] + eta_around_center[y_indices_center-i][x_indices_center-i]
                if XY_neg <= 0:
                    eta_levels.append(eta_around_center[y_indices_center-i][x_indices_center-i].values)
                    if test_calib:
                        print('Local max i found',i)
                    break

            for i in range(0,max_extent_search): # Second derivative utilizing the forward method
                Y_neg = eta_around_center[y_indices_center-i-2][x_indices_center] - 2*eta_around_center[y_indices_center-i-1][x_indices_center] + eta_around_center[y_indices_center-i][x_indices_center]
                if Y_neg <= 0:
                    eta_levels.append(eta_around_center[y_indices_center-i][x_indices_center])
                    if test_calib:
                        print('Local max i found',i)
                    break

            for i in range(0,max_extent_search): # Second derivative utilizing the forward method
                X_pos_Y_neg = eta_around_center[y_indices_center-i-2][x_indices_center+i+2] - 2*eta_around_center[y_indices_center-i-1][x_indices_center+i+1] + eta_around_center[y_indices_center-i][x_indices_center+i]
                if X_pos_Y_neg <= 0:
                    eta_levels.append(eta_around_center[y_indices_center-i][x_indices_center+i].values)
                    if test_calib:
                        print('Local max i found',i)
                    break

            min_eta = np.argmin(eta_levels)

            levels = [
                eta_levels[min_eta]
            ]
            levels_array = np.array(levels)

            # Remove duplicates and sort the array
            levels = np.unique(levels_array)

            # Step 2: Generate Contours

            contours = plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels)
            plt.close()
            # Step 3: Collect all X and Y points of the paths of all contours
            all_contour_points = []

            for collection in contours.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            # Step 4: Identify the Center Point
            center_point = (center[1],center[0])

            # Step 5: Process Contour Paths to Handle Jumps
            def process_contour_path(vertices, jump_threshold=0.1):
                segments = []
                current_segment = [vertices[0]]
                
                for i in range(1, len(vertices)):
                    if np.linalg.norm(vertices[i] - vertices[i-1]) > jump_threshold:
                        segments.append(np.array(current_segment))
                        current_segment = [vertices[i]]
                    else:
                        current_segment.append(vertices[i])
                
                if current_segment:
                    segments.append(np.array(current_segment))
                
                return segments

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)

            # Step 6: Find the Outermost Closed Contour Segment
            outermost_contour = None
            max_area = 0

            def is_closed_contour(vertices, tol=1e-1):
                distance = np.linalg.norm(vertices[0] - vertices[-1])
                return distance < tol

            for vertices in processed_contour_segments:
                # Create a Path object from the vertices
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-2):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    if path_obj.contains_point(center_point):
                        # Calculate the area of the contour segment using the shoelace formula
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                                    # Update the outermost contour if this one has a larger area
                        if area > max_area:
                            max_area = area
                            outermost_contour = vertices
            
            if outermost_contour is not None:
                outermost_path = Path(outermost_contour)
                mask = xr.zeros_like(eta_around_center, dtype=bool)

                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                mask_flattened = outermost_path.contains_points(points)
                mask.values = mask_flattened.reshape(mask.shape)

                # Update eddies
                x_indices = np.where((data.X >= center[1] - 2.5) & (data.X <= center[1] + 2.5))[0]
                y_indices = np.where((data.Y >= center[0] - 2.5) & (data.Y <= center[0] + 2.5))[0]

                current_subset = data.isel(X=x_indices, Y=y_indices)
                cond = (~mask) | (current_subset != 0)
                updated_subset = current_subset.where(cond, other=2)

                data.values[np.ix_(y_indices, x_indices)] = updated_subset.values

            if test_calib:
                # Step 7: Plot the Data and the Outermost Closed Contour Segment
                plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels)
                if outermost_contour is not None:
                    plt.plot(outermost_contour[:, 0], outermost_contour[:, 1], 'r-', linewidth=2)
                    plt.scatter(*center_point, color='red')  # Mark the center point
                    plt.title("Topographic Data LOCAL MIN with Outermost Closed Contour Segment")
                    plt.show()
                else:
                    print('Local MIN not found')
                    plt.pcolormesh(eta_around_center.X,eta_around_center.Y,eta_around_center)
                    plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels)
                    plt.title("Topographic Data LOCAL MIN with Outermost Closed Contour Segment")
                    plt.show()

                # Print the bounding box of the outermost closed contour segment
                if outermost_contour is not None:
                    min_x, min_y = np.min(outermost_contour, axis=0)
                    max_x, max_y = np.max(outermost_contour, axis=0)
                    print(f"Outermost closed contour segment bounding box: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
                else:
                    print("No closed contour segments found containing the center point.")
    
    return data


def inner_eddy_region_v4(eddyCenterpoints=list,eta=xr.DataArray(),cold=False,warm=False,test_calib=False,eddiesData=xr.DataArray()):
    """
    Based on Matsuoka et al. 2016.\n
    Inner eddy detection utilizing maximum outer closed contour around each eddy center point.\n
    Warm: Local maximum, Cold: Local minimum, test_calib: Plots contour field with max outer closed contour.\n
    Previously saved eddies dataset can be utilized for further additions.
    """
    data = eddiesData

    if warm:
        for center in eddyCenterpoints:
            eta_around_center = eta.sel(X=slice(center[1]-1,center[1]+1)).sel(Y=slice(center[0]-1,center[0]+1))

            # Step 1: define inflexion point for level in contour
            x_indices_center = np.where((eta_around_center.X == center[1]) & (eta_around_center.X == center[1]))[0][0]
            y_indices_center = np.where((eta_around_center.Y == center[0]) & (eta_around_center.Y == center[0]))[0][0]

            # Computes second derivative
            laplacian_eta = laplace(eta_around_center)
            laplacian = xr.zeros_like(eta_around_center)
            laplacian.values = laplacian_eta
            laplacian_eta = laplacian

            # Define outermost contour enclosing the eddy to mask out outlying values
            max_eta = eta_around_center.max().values
            min_eta = eta_around_center.min().values

            step_size = 0.001
            num_elements = int((max_eta-min_eta) / step_size) + 1
            levels_array = np.linspace(min_eta,max_eta,num_elements)

            contours = plt.contour(eta_around_center.X,eta_around_center.Y,eta_around_center,levels=levels_array)
            plt.close()

            # Collect all X and Y points of the paths of all contours
            all_contour_points = []

            for collection in contours.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            # Identify the Center Point
            center_point = (center[1], center[0])

            # Process Contour Paths to Handle Jumps
            def process_contour_path(vertices, jump_threshold=0.1):
                segments = []
                current_segment = [vertices[0]]
                
                for i in range(1, len(vertices)):
                    if np.linalg.norm(vertices[i] - vertices[i-1]) > jump_threshold:
                        segments.append(np.array(current_segment))
                        current_segment = [vertices[i]]
                    else:
                        current_segment.append(vertices[i])
                
                if current_segment:
                    segments.append(np.array(current_segment))
                
                return segments

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)

            # Find the Outermost Closed Contour Segment
            outermost_contour = None
            max_area = 0

            def is_closed_contour(vertices, tol=1e-1):
                distance = np.linalg.norm(vertices[0] - vertices[-1])
                return distance < tol

            def calculate_area(vertices):
                x = vertices[:, 0]
                y = vertices[:, 1]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            def contains_other_contours(outer_contour, all_contours, center_point):
                outer_path = Path(outer_contour)
                for contour in all_contours:
                    if np.array_equal(contour, outer_contour):
                        continue
                    contour_path = Path(contour)
                    if all(outer_path.contains_point(point) for point in contour) and not contour_path.contains_point(center_point):
                        return True
                return False

            for vertices in processed_contour_segments:
                # Create a Path object from the vertices
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-2):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    if path_obj.contains_point(center_point):
                        # Calculate the area of the contour segment using the shoelace formula
                        area = calculate_area(vertices)
                        
                        # Check if this contour contains any other contour that does not contain the center point
                        if not contains_other_contours(vertices, processed_contour_segments, center_point):
                            # Update the outermost contour if this one has a larger area
                            if area > max_area:
                                max_area = area
                                outermost_contour = vertices

            # Finds inflextion points
            cond = laplacian_eta >= 0

            inflection_ssh_values = eta_around_center.where(cond)

            # Mask the area of inflextion point relevant for this eddy
            if outermost_contour is not None:
                outermost_path = Path(outermost_contour)
                mask = xr.zeros_like(eta_around_center, dtype=bool)

                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                mask_flattened = outermost_path.contains_points(points)
                mask.values = mask_flattened.reshape(mask.shape)
                inflection_ssh_values = inflection_ssh_values.where(mask)
            
            # Finds maximum SSH from inflection points
            inflection_ssh_values = inflection_ssh_values.where(inflection_ssh_values < eta_around_center[y_indices_center][x_indices_center].values).values
            # Remove NaN values from inflection_ssh_values
            inflection_ssh_values = inflection_ssh_values[~np.isnan(inflection_ssh_values)]
            only_centerpoint = False
            try:
                max_ssh_index = np.argmax(inflection_ssh_values)
            except:
                only_centerpoint = True # If no maximum is found

            # if inflection_ssh_values.size == 0:
            #     print("inflection_ssh_values is empty")
            #     continue  # Skip this iteration or handle the error appropriately
            
            if only_centerpoint:
                levels = [
                    eta_around_center[y_indices_center][x_indices_center].values
                ]
            else:
                levels = [
                    inflection_ssh_values[max_ssh_index],
                    inflection_ssh_values[max_ssh_index]+0.05
                ]


            levels_array = np.array(levels)

            # Remove duplicates and sort the array
            levels = np.unique(levels_array)

            # Step 2: Generate Contours

            contours = plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels)
            plt.close()
            # Step 3: Collect all X and Y points of the paths of all contours
            all_contour_points = []

            for collection in contours.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            # Step 4: Identify the Center Point
            center_point = (center[1],center[0])

            # Step 5: Process Contour Paths to Handle Jumps
            def process_contour_path(vertices, jump_threshold=0.1):
                segments = []
                current_segment = [vertices[0]]
                
                for i in range(1, len(vertices)):
                    if np.linalg.norm(vertices[i] - vertices[i-1]) > jump_threshold:
                        segments.append(np.array(current_segment))
                        current_segment = [vertices[i]]
                    else:
                        current_segment.append(vertices[i])
                
                if current_segment:
                    segments.append(np.array(current_segment))
                
                return segments

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)

            # Step 6: Find the Outermost Closed Contour Segment
            outermost_contour = None
            max_area = 0

            def is_closed_contour(vertices, tol=1e-1):
                distance = np.linalg.norm(vertices[0] - vertices[-1])
                return distance < tol

            for vertices in processed_contour_segments:
                # Create a Path object from the vertices
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-2):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    if path_obj.contains_point(center_point):
                        # Calculate the area of the contour segment using the shoelace formula
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                            
                        # Check if this contour contains any other contour that does not contain the center point
                        if not contains_other_contours(vertices, processed_contour_segments, center_point):
                            # Update the outermost contour if this one has a larger area
                            if area > max_area:
                                max_area = area
                                outermost_contour = vertices

            
            if outermost_contour is not None:
                outermost_path = Path(outermost_contour)
                mask = xr.zeros_like(eta_around_center, dtype=bool)

                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                mask_flattened = outermost_path.contains_points(points)
                mask.values = mask_flattened.reshape(mask.shape)

                # Update eddies
                x_indices = np.where((data.X >= center[1] - 1) & (data.X <= center[1] + 1))[0]
                y_indices = np.where((data.Y >= center[0] - 1) & (data.Y <= center[0] + 1))[0]

                current_subset = data.isel(X=x_indices, Y=y_indices)
                cond = (~mask) | (current_subset != 0)
                updated_subset = current_subset.where(cond, other=1)

                data.values[np.ix_(y_indices, x_indices)] = updated_subset.values

            
            if test_calib:
                # Step 7: Plot the Data and the Outermost Closed Contour Segment
                plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels)
                if outermost_contour is not None:
                    plt.plot(outermost_contour[:, 0], outermost_contour[:, 1], 'r-', linewidth=2)
                    plt.contourf(mask.X, mask.Y, mask, colors=['white', 'blue'])
                    plt.scatter(*center_point, color='red')  # Mark the center point
                    plt.title("Topographic Data LOCAL MAX with Outermost Closed Contour Segment")
                    plt.show()

                else:
                    print('Local MAX not found')
                    plt.pcolormesh(eta_around_center.X,eta_around_center.Y,eta_around_center)
                    plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels)
                    plt.title("Topographic Data LOCAL MAX with Outermost Closed Contour Segment")
                    plt.show()

                # Print the bounding box of the outermost closed contour segment
                if outermost_contour is not None:
                    min_x, min_y = np.min(outermost_contour, axis=0)
                    max_x, max_y = np.max(outermost_contour, axis=0)
                    print(f"Outermost closed contour segment bounding box: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
                else:
                    print("No closed contour segments found containing the center point.")

    if cold:
        for center in eddyCenterpoints:
            eta_around_center = eta.sel(X=slice(center[1]-1,center[1]+1)).sel(Y=slice(center[0]-1,center[0]+1))

            # Step 1: define inflexion point for level in contour
            x_indices_center = np.where((eta_around_center.X == center[1]) & (eta_around_center.X == center[1]))[0][0]
            y_indices_center = np.where((eta_around_center.Y == center[0]) & (eta_around_center.Y == center[0]))[0][0]

            # Computes second derivative
            laplacian_eta = laplace(eta_around_center)
            laplacian = xr.zeros_like(eta_around_center)
            laplacian.values = laplacian_eta
            laplacian_eta = laplacian

            # Define outermost contour enclosing the eddy to mask out outlying values
            max_eta = eta_around_center.max().values
            min_eta = eta_around_center.min().values

            step_size = 0.001
            num_elements = int((max_eta-min_eta) / step_size) + 1
            levels_array = np.linspace(min_eta,max_eta,num_elements)

            contours = plt.contour(eta_around_center.X,eta_around_center.Y,eta_around_center,levels=levels_array)
            plt.close()

            # Collect all X and Y points of the paths of all contours
            all_contour_points = []

            for collection in contours.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            # Identify the Center Point
            center_point = (center[1], center[0])

            # Process Contour Paths to Handle Jumps
            def process_contour_path(vertices, jump_threshold=0.1):
                segments = []
                current_segment = [vertices[0]]
                
                for i in range(1, len(vertices)):
                    if np.linalg.norm(vertices[i] - vertices[i-1]) > jump_threshold:
                        segments.append(np.array(current_segment))
                        current_segment = [vertices[i]]
                    else:
                        current_segment.append(vertices[i])
                
                if current_segment:
                    segments.append(np.array(current_segment))
                
                return segments

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)

            # Find the Outermost Closed Contour Segment
            outermost_contour = None
            max_area = 0

            def is_closed_contour(vertices, tol=1e-1):
                distance = np.linalg.norm(vertices[0] - vertices[-1])
                return distance < tol

            def calculate_area(vertices):
                x = vertices[:, 0]
                y = vertices[:, 1]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            def contains_other_contours(outer_contour, all_contours, center_point):
                outer_path = Path(outer_contour)
                for contour in all_contours:
                    if np.array_equal(contour, outer_contour):
                        continue
                    contour_path = Path(contour)
                    if all(outer_path.contains_point(point) for point in contour) and not contour_path.contains_point(center_point):
                        return True
                return False

            for vertices in processed_contour_segments:
                # Create a Path object from the vertices
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-2):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    if path_obj.contains_point(center_point):
                        # Calculate the area of the contour segment using the shoelace formula
                        area = calculate_area(vertices)
                        
                        # Check if this contour contains any other contour that does not contain the center point
                        if not contains_other_contours(vertices, processed_contour_segments, center_point):
                            # Update the outermost contour if this one has a larger area
                            if area > max_area:
                                max_area = area
                                outermost_contour = vertices

            # Finds inflextion points
            cond = laplacian_eta <= 0

            inflection_ssh_values = eta_around_center.where(cond)

            # Mask the area of inflextion point relevant for this eddy
            if outermost_contour is not None:
                outermost_path = Path(outermost_contour)
                mask = xr.zeros_like(eta_around_center, dtype=bool)

                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                mask_flattened = outermost_path.contains_points(points)
                mask.values = mask_flattened.reshape(mask.shape)
                inflection_ssh_values = inflection_ssh_values.where(mask)

            # Finds maximum SSH from inflection points
            inflection_ssh_values = inflection_ssh_values.where(inflection_ssh_values > eta_around_center[y_indices_center][x_indices_center].values).values
            # Remove NaN values from inflection_ssh_values
            inflection_ssh_values = inflection_ssh_values[~np.isnan(inflection_ssh_values)]
            only_centerpoint = False
            try:
                min_ssh_index = np.argmin(inflection_ssh_values)
            except:
                only_centerpoint = True # If no minimum is found

            # if inflection_ssh_values.size == 0:
            #     print("inflection_ssh_values is empty")
            #     continue  # Skip this iteration or handle the error appropriately
            if only_centerpoint:
                levels = [
                    eta_around_center[y_indices_center][x_indices_center].values
                ]
            else:
                levels = [
                    inflection_ssh_values[min_ssh_index]-0.05,
                    inflection_ssh_values[min_ssh_index]
                ]
            levels_array = np.array(levels)

            # Remove duplicates and sort the array
            levels = np.unique(levels_array)

            # Step 2: Generate Contours

            contours = plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels)
            plt.close()
            # Step 3: Collect all X and Y points of the paths of all contours
            all_contour_points = []

            for collection in contours.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            # Step 4: Identify the Center Point
            center_point = (center[1],center[0])

            # Step 5: Process Contour Paths to Handle Jumps
            def process_contour_path(vertices, jump_threshold=0.1):
                segments = []
                current_segment = [vertices[0]]
                
                for i in range(1, len(vertices)):
                    if np.linalg.norm(vertices[i] - vertices[i-1]) > jump_threshold:
                        segments.append(np.array(current_segment))
                        current_segment = [vertices[i]]
                    else:
                        current_segment.append(vertices[i])
                
                if current_segment:
                    segments.append(np.array(current_segment))
                
                return segments

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)

            # Step 6: Find the Outermost Closed Contour Segment
            outermost_contour = None
            max_area = 0

            def is_closed_contour(vertices, tol=1e-1):
                distance = np.linalg.norm(vertices[0] - vertices[-1])
                return distance < tol

            for vertices in processed_contour_segments:
                # Create a Path object from the vertices
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-2):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    if path_obj.contains_point(center_point):
                        # Calculate the area of the contour segment using the shoelace formula
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        
                        # Check if this contour contains any other contour that does not contain the center point
                        if not contains_other_contours(vertices, processed_contour_segments, center_point):
                            # Update the outermost contour if this one has a larger area
                            if area > max_area:
                                max_area = area
                                outermost_contour = vertices

            
            if outermost_contour is not None:
                outermost_path = Path(outermost_contour)
                mask = xr.zeros_like(eta_around_center, dtype=bool)

                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                mask_flattened = outermost_path.contains_points(points)
                mask.values = mask_flattened.reshape(mask.shape)

                # Update eddies
                x_indices = np.where((data.X >= center[1] - 1) & (data.X <= center[1] + 1))[0]
                y_indices = np.where((data.Y >= center[0] - 1) & (data.Y <= center[0] + 1))[0]

                current_subset = data.isel(X=x_indices, Y=y_indices)
                cond = (~mask) | (current_subset != 0)
                updated_subset = current_subset.where(cond, other=2)

                data.values[np.ix_(y_indices, x_indices)] = updated_subset.values

            if test_calib:
                # Step 7: Plot the Data and the Outermost Closed Contour Segment
                plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels)
                if outermost_contour is not None:
                    plt.plot(outermost_contour[:, 0], outermost_contour[:, 1], 'r-', linewidth=2)
                    plt.scatter(*center_point, color='red')  # Mark the center point
                    plt.title("Topographic Data LOCAL MIN with Outermost Closed Contour Segment")
                    plt.show()
                else:
                    print('Local MIN not found')
                    plt.pcolormesh(eta_around_center.X,eta_around_center.Y,eta_around_center)
                    plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels)
                    plt.title("Topographic Data LOCAL MIN with Outermost Closed Contour Segment")
                    plt.show()

                # Print the bounding box of the outermost closed contour segment
                if outermost_contour is not None:
                    min_x, min_y = np.min(outermost_contour, axis=0)
                    max_x, max_y = np.max(outermost_contour, axis=0)
                    print(f"Outermost closed contour segment bounding box: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
                else:
                    print("No closed contour segments found containing the center point.")
    
    return data


def outer_eddy_region(eddyCenterpoints=list,hor_vel=xr.DataArray(),test_calib=False,eddiesData=xr.DataArray()):
    data = eddiesData
    for center in eddyCenterpoints:
            hor_vel_around_center = hor_vel.sel(X=slice(center[1]-1,center[1]+1)).sel(Y=slice(center[0]-1,center[0]+1))

            # Step 1: define inflexion point for level in contour

            # Computes second derivative
            laplacian_eta = laplace(hor_vel_around_center.values)

            # Finds inflextion points
            inflection_points = np.where(np.diff(np.sign(laplacian_eta), axis=0) != 0)

            # Ensure inflection points are within bounds
            inflection_points = (inflection_points[0][inflection_points[0] < hor_vel_around_center.shape[0]],
                                 inflection_points[1][inflection_points[1] < hor_vel_around_center.shape[1]])

            if len(inflection_points[0]) == 0 or len(inflection_points[1]) == 0:
                continue  # Skip if no valid inflection points

           # Finds maximum SSH from inflection points
            inflection_speed_values = hor_vel_around_center.values[inflection_points]
            max_speed_index = np.argmax(inflection_speed_values)

            levels = [
                inflection_speed_values[max_speed_index]
            ]
            levels_array = np.array(levels)

            # Remove duplicates and sort the array
            levels = np.unique(levels_array)

            # Step 2: Generate Contours

            contours = plt.contour(hor_vel_around_center.X, hor_vel_around_center.Y, hor_vel_around_center,levels)
            plt.close()
            # Step 3: Collect all X and Y points of the paths of all contours
            all_contour_points = []

            for collection in contours.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            # Step 4: Identify the Center Point
            center_point = (center[1],center[0])

            # Step 5: Process Contour Paths to Handle Jumps
            def process_contour_path(vertices, jump_threshold=0.1):
                segments = []
                current_segment = [vertices[0]]
                
                for i in range(1, len(vertices)):
                    if np.linalg.norm(vertices[i] - vertices[i-1]) > jump_threshold:
                        segments.append(np.array(current_segment))
                        current_segment = [vertices[i]]
                    else:
                        current_segment.append(vertices[i])
                
                if current_segment:
                    segments.append(np.array(current_segment))
                
                return segments

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)

            # Step 6: Find the Outermost Closed Contour Segment
            outermost_contour = None
            max_area = 0

            def is_closed_contour(vertices, tol=1e-1):
                distance = np.linalg.norm(vertices[0] - vertices[-1])
                return distance < tol

            for vertices in processed_contour_segments:
                # Create a Path object from the vertices
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-2):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    if path_obj.contains_point(center_point):
                        # Calculate the area of the contour segment using the shoelace formula
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                                    # Update the outermost contour if this one has a larger area
                        if area > max_area:
                            max_area = area
                            outermost_contour = vertices
            
            if outermost_contour is not None:
                outermost_path = Path(outermost_contour)
                mask = xr.zeros_like(hor_vel_around_center, dtype=bool)

                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                mask_flattened = outermost_path.contains_points(points)
                mask.values = mask_flattened.reshape(mask.shape)

                # Update eddies
                x_indices = np.where((data.X >= center[1] - 1) & (data.X <= center[1] + 1))[0]
                y_indices = np.where((data.Y >= center[0] - 1) & (data.Y <= center[0] + 1))[0]

                current_subset = data.isel(X=x_indices, Y=y_indices)
                cond = (~mask) | (current_subset != 0)
                updated_subset = current_subset.where(cond, other=1)

                data.values[np.ix_(y_indices, x_indices)] = updated_subset.values

            
            if test_calib:
                # Step 7: Plot the Data and the Outermost Closed Contour Segment
                plt.contour(hor_vel_around_center.X, hor_vel_around_center.Y, hor_vel_around_center,levels)
                if outermost_contour is not None:
                    plt.plot(outermost_contour[:, 0], outermost_contour[:, 1], 'r-', linewidth=2)
                    plt.contourf(mask.X, mask.Y, mask, colors=['white', 'blue'])
                    plt.scatter(*center_point, color='red')  # Mark the center point
                    plt.title("Topographic Data LOCAL MAX with Outermost Closed Contour Segment")
                    plt.show()

                else:
                    print('Local MAX not found')
                    plt.pcolormesh(hor_vel_around_center.X,hor_vel_around_center.Y,hor_vel_around_center)
                    plt.contour(hor_vel_around_center.X, hor_vel_around_center.Y, hor_vel_around_center,levels)
                    plt.title("Topographic Data LOCAL MAX with Outermost Closed Contour Segment")
                    plt.show()

                # Print the bounding box of the outermost closed contour segment
                if outermost_contour is not None:
                    min_x, min_y = np.min(outermost_contour, axis=0)
                    max_x, max_y = np.max(outermost_contour, axis=0)
                    print(f"Outermost closed contour segment bounding box: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
                else:
                    print("No closed contour segments found containing the center point.")


def inner_eddy_region_v5(eddyCenterpoints=list,eta=xr.DataArray(),cold=False,warm=False,test_calib=False,eddiesData=xr.DataArray()):
    """
    Based on Matsuoka et al. 2016.\n
    Inner eddy detection utilizing maximum outer closed contour around each eddy center point.\n
    Warm: Local maximum, Cold: Local minimum, test_calib: Plots contour field with max outer closed contour.\n
    Previously saved eddies dataset can be utilized for further additions.\n

    This version utilizes only SSH, where the outermost closed contour without any other local max/min occurs.
    """
    data = eddiesData

    if warm:
        for center in eddyCenterpoints:
            eta_around_center = eta.sel(X=slice(center[1]-1,center[1]+1)).sel(Y=slice(center[0]-1,center[0]+1))
            # Define outermost contour enclosing the eddy to mask out outlying values
            max_eta = eta_around_center.max().values
            min_eta = eta_around_center.min().values

            step_size = 0.001
            num_elements = int((max_eta-min_eta) / step_size) + 1
            levels_array = np.linspace(min_eta,max_eta,num_elements)

            contours = plt.contour(eta_around_center.X,eta_around_center.Y,eta_around_center,levels=levels_array)
            plt.close()

            # Collect all X and Y points of the paths of all contours
            all_contour_points = []

            for collection in contours.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            # Identify the Center Point
            center_point = (center[1], center[0])

            # Process Contour Paths to Handle Jumps
            def process_contour_path(vertices, jump_threshold=0.05):
                segments = []
                current_segment = [vertices[0]]
                
                for i in range(1, len(vertices)):
                    if np.linalg.norm(vertices[i] - vertices[i-1]) > jump_threshold:
                        segments.append(np.array(current_segment))
                        current_segment = [vertices[i]]
                    else:
                        current_segment.append(vertices[i])
                
                if current_segment:
                    segments.append(np.array(current_segment))
                
                return segments

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)

            # Find the Outermost Closed Contour Segment
            outermost_contour = None
            max_area = 0

            def is_closed_contour(vertices, tol=1e-5):
                distance = np.linalg.norm(vertices[0] - vertices[-1])
                return distance < tol

            def calculate_area(vertices):
                x = vertices[:, 0]
                y = vertices[:, 1]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            def contains_other_contours(outer_contour, all_contours, center_point):
                outer_path = Path(outer_contour)
                for contour in all_contours:
                    if np.array_equal(contour, outer_contour):
                        continue
                    contour_path = Path(contour)
                    if all(outer_path.contains_point(point) for point in contour) and not contour_path.contains_point(center_point):
                        return True
                return False

            for vertices in processed_contour_segments:
                # Create a Path object from the vertices
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-5):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    if path_obj.contains_point(center_point):
                        # Calculate the area of the contour segment using the shoelace formula
                        area = calculate_area(vertices)
                        
                        # Check if this contour contains any other contour that does not contain the center point
                        if not contains_other_contours(vertices, processed_contour_segments, center_point):
                            # Update the outermost contour if this one has a larger area
                            if area > max_area:
                                max_area = area
                                outermost_contour = vertices

            # Mask the area of inner eddy eddy
            if outermost_contour is not None:
                outermost_path = Path(outermost_contour)
                mask = xr.zeros_like(eta_around_center, dtype=bool)

                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                mask_flattened = outermost_path.contains_points(points)
                mask.values = mask_flattened.reshape(mask.shape)

                # Update eddies
                x_indices = np.where((data.X >= center[1] - 1) & (data.X <= center[1] + 1))[0]
                y_indices = np.where((data.Y >= center[0] - 1) & (data.Y <= center[0] + 1))[0]

                current_subset = data.isel(X=x_indices, Y=y_indices)
                cond = (~mask) | (current_subset != 0)
                updated_subset = current_subset.where(cond, other=1)

                data.values[np.ix_(y_indices, x_indices)] = updated_subset.values

            if test_calib:
                # Step 7: Plot the Data and the Outermost Closed Contour Segment
                plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels_array)
                if outermost_contour is not None:
                    plt.plot(outermost_contour[:, 0], outermost_contour[:, 1], 'r-', linewidth=2)
                    plt.contourf(mask.X, mask.Y, mask, colors=['white', 'blue'])
                    plt.scatter(*center_point, color='red')  # Mark the center point
                    plt.title("Topographic Data LOCAL MAX with Outermost Closed Contour Segment")
                    plt.show()

                else:
                    print('Local MAX not found')
                    plt.pcolormesh(eta_around_center.X,eta_around_center.Y,eta_around_center)
                    plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels_array)
                    plt.title("Topographic Data LOCAL MAX with Outermost Closed Contour Segment")
                    plt.show()

                # Print the bounding box of the outermost closed contour segment
                if outermost_contour is not None:
                    min_x, min_y = np.min(outermost_contour, axis=0)
                    max_x, max_y = np.max(outermost_contour, axis=0)
                    print(f"Outermost closed contour segment bounding box: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
                else:
                    print("No closed contour segments found containing the center point.")

    if cold:
        for center in eddyCenterpoints:
            eta_around_center = eta.sel(X=slice(center[1]-1,center[1]+1)).sel(Y=slice(center[0]-1,center[0]+1))

            # Define outermost contour enclosing the eddy to mask out outlying values
            max_eta = eta_around_center.max().values
            min_eta = eta_around_center.min().values

            step_size = 0.001
            num_elements = int((max_eta-min_eta) / step_size) + 1
            levels_array = np.linspace(min_eta,max_eta,num_elements)

            contours = plt.contour(eta_around_center.X,eta_around_center.Y,eta_around_center,levels=levels_array)
            plt.close()

            # Collect all X and Y points of the paths of all contours
            all_contour_points = []

            for collection in contours.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            # Identify the Center Point
            center_point = (center[1], center[0])

            # Process Contour Paths to Handle Jumps
            def process_contour_path(vertices, jump_threshold=0.05):
                segments = []
                current_segment = [vertices[0]]
                
                for i in range(1, len(vertices)):
                    if np.linalg.norm(vertices[i] - vertices[i-1]) > jump_threshold:
                        segments.append(np.array(current_segment))
                        current_segment = [vertices[i]]
                    else:
                        current_segment.append(vertices[i])
                
                if current_segment:
                    segments.append(np.array(current_segment))
                
                return segments

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)

            # Find the Outermost Closed Contour Segment
            outermost_contour = None
            max_area = 0

            def is_closed_contour(vertices, tol=1e-5):
                distance = np.linalg.norm(vertices[0] - vertices[-1])
                return distance < tol

            def calculate_area(vertices):
                x = vertices[:, 0]
                y = vertices[:, 1]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            def contains_other_contours(outer_contour, all_contours, center_point):
                outer_path = Path(outer_contour)
                for contour in all_contours:
                    if np.array_equal(contour, outer_contour):
                        continue
                    contour_path = Path(contour)
                    if all(outer_path.contains_point(point) for point in contour) and not contour_path.contains_point(center_point):
                        return True
                return False

            for vertices in processed_contour_segments:
                # Create a Path object from the vertices
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-5):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    if path_obj.contains_point(center_point):
                        # Calculate the area of the contour segment using the shoelace formula
                        area = calculate_area(vertices)
                        
                        # Check if this contour contains any other contour that does not contain the center point
                        if not contains_other_contours(vertices, processed_contour_segments, center_point):
                            # Update the outermost contour if this one has a larger area
                            if area > max_area:
                                max_area = area
                                outermost_contour = vertices

            # Mask the area of inflextion point relevant for this eddy
            if outermost_contour is not None:
                outermost_path = Path(outermost_contour)
                mask = xr.zeros_like(eta_around_center, dtype=bool)

                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                mask_flattened = outermost_path.contains_points(points)
                mask.values = mask_flattened.reshape(mask.shape)

                # Update eddies
                x_indices = np.where((data.X >= center[1] - 1) & (data.X <= center[1] + 1))[0]
                y_indices = np.where((data.Y >= center[0] - 1) & (data.Y <= center[0] + 1))[0]

                current_subset = data.isel(X=x_indices, Y=y_indices)
                cond = (~mask) | (current_subset != 0)
                updated_subset = current_subset.where(cond, other=2)

                data.values[np.ix_(y_indices, x_indices)] = updated_subset.values
                

            if test_calib:
                # Step 7: Plot the Data and the Outermost Closed Contour Segment
                plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels_array,zorder=0)
                if outermost_contour is not None:
                    plt.plot(outermost_contour[:, 0], outermost_contour[:, 1], 'r-', linewidth=2,zorder=1)
                    plt.scatter(center_point[0],center_point[1], color='red',edgecolors='black',zorder=2)  # Mark the center point
                    plt.title("Topographic Data LOCAL MIN with Outermost Closed Contour Segment")
                    plt.show()
                else:
                    pass
                    print('Local MIN not found')
                    plt.pcolormesh(eta_around_center.X,eta_around_center.Y,eta_around_center)
                    plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels_array)
                    plt.title("Topographic Data LOCAL MIN with Outermost Closed Contour Segment")
                    plt.show()

                # Print the bounding box of the outermost closed contour segment
                if outermost_contour is not None:
                    min_x, min_y = np.min(outermost_contour, axis=0)
                    max_x, max_y = np.max(outermost_contour, axis=0)
                    print(f"Outermost closed contour segment bounding box: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
                else:
                    print("No closed contour segments found containing the center point.")
    
    return data


def Full_eddy_region_v1(eddyCenterpoints=list,eta=xr.DataArray(),hor_vel=xr.DataArray(),cold=False,warm=False,test_calib=False,eddiesData=xr.DataArray(),
                        min_vel_diff=0.05,
                        percentage_diff_vel=20):
    """
    Inner eddy detection utilizing maximum outer closed contour around each eddy center point.\n
    Warm: Local maximum, Cold: Local minimum, test_calib: Plots contour field with max outer closed contour.\n
    Previously saved eddies dataset can be utilized for further additions.\n
    \n
    min_vel_diff:[m/s] The mimimum velocity difference between the eddy centre point and the maximum velocity of the inner eddy region.\n
    percentage_diff_vel:[%] Percentage of the difference between the maximum velocity and the eddy center point velocity, and is used to subtract the maximum velocity to include more of the outer eddy region.\n
    \n
    This version utilizes both SSH and velocity maximum of the inner eddy region, where the outermost closed contour without any other local max/min occurs.
    """
    data = eddiesData

    if warm:
        for center in eddyCenterpoints:
            eta_around_center = eta.sel(X=slice(center[1]-1.5,center[1]+1.5)).sel(Y=slice(center[0]-1,center[0]+1))
            # Define outermost contour enclosing the eddy to mask out outlying values
            max_eta = eta_around_center.max().values
            min_eta = eta_around_center.min().values

            step_size = 0.001
            num_elements = int((max_eta-min_eta) / step_size) + 1
            levels_array = np.linspace(min_eta,max_eta,num_elements)

            contours = plt.contour(eta_around_center.X,eta_around_center.Y,eta_around_center,levels=levels_array)
            plt.close()

            # Collect all X and Y points of the paths of all contours
            all_contour_points = []

            for collection in contours.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            # Identify the Center Point
            center_point = (center[1], center[0])

            # Process Contour Paths to Handle Jumps
            def process_contour_path(vertices, jump_threshold=0.05):
                segments = []
                current_segment = [vertices[0]]
                
                for i in range(1, len(vertices)):
                    if np.linalg.norm(vertices[i] - vertices[i-1]) > jump_threshold:
                        segments.append(np.array(current_segment))
                        current_segment = [vertices[i]]
                    else:
                        current_segment.append(vertices[i])
                
                if current_segment:
                    segments.append(np.array(current_segment))
                
                return segments

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)

            # Find the Outermost Closed Contour Segment
            outermost_contour = None
            max_area = 0

            def is_closed_contour(vertices, tol=1e-5):
                distance = np.linalg.norm(vertices[0] - vertices[-1])
                return distance < tol

            def calculate_area(vertices):
                x = vertices[:, 0]
                y = vertices[:, 1]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            def contains_other_contours(outer_contour, all_contours, center_point):
                outer_path = Path(outer_contour)
                for contour in all_contours:
                    if np.array_equal(contour, outer_contour):
                        continue
                    contour_path = Path(contour)
                    if all(outer_path.contains_point(point) for point in contour) and not contour_path.contains_point(center_point):
                        return True
                return False

            for vertices in processed_contour_segments:
                # Create a Path object from the vertices
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-5):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    if path_obj.contains_point(center_point):
                        # Calculate the area of the contour segment using the shoelace formula
                        area = calculate_area(vertices)
                        
                        # Check if this contour contains any other contour that does not contain the center point
                        if not contains_other_contours(vertices, processed_contour_segments, center_point):
                            # Update the outermost contour if this one has a larger area
                            if area > max_area:
                                max_area = area
                                outermost_contour = vertices

            # Mask the area of inner eddy eddy
            if outermost_contour is not None:
                outermost_path = Path(outermost_contour)
                mask = xr.zeros_like(eta_around_center, dtype=bool)

                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                mask_flattened = outermost_path.contains_points(points)
                mask.values = mask_flattened.reshape(mask.shape)

                # Update eddies
                x_indices = np.where((data.X >= center[1] - 1.5) & (data.X <= center[1] + 1.5))[0]
                y_indices = np.where((data.Y >= center[0] - 1) & (data.Y <= center[0] + 1))[0]

                current_subset = data.isel(X=x_indices, Y=y_indices)
                cond = (~mask) | (current_subset != 0) & (current_subset != 4)
                updated_subset = current_subset.where(cond, other=1)

                data.values[np.ix_(y_indices, x_indices)] = updated_subset.values
            outermost_contour_vel = []
            # Mask the outer eddy regions
            if outermost_contour is not None:
                hor_vel_around_center = hor_vel.sel(X=slice(center[1]-1.5,center[1]+1.5)).sel(Y=slice(center[0]-1,center[0]+1))
                hor_vel_in_center = hor_vel_around_center.where(mask)
                hor_vel_max = hor_vel_in_center.max().values
                hor_vel_center = hor_vel_in_center.sel(X=center[1],Y=center[0]).values
                diff_hor_vel = hor_vel_max - hor_vel_center
                

                if diff_hor_vel >= min_vel_diff: # The difference between the eddy center and the maximum velocity of the inner region must be significant
                    contour_vel_num = hor_vel_max - (diff_hor_vel*(percentage_diff_vel/100))
                    contour_vel = plt.contour(hor_vel_around_center.X,hor_vel_around_center.Y,hor_vel_around_center,levels=[contour_vel_num])
                    plt.close()

                    all_contour_points = []

                    for collection in contour_vel.collections:
                        for path in collection.get_paths():
                            # Collect the vertices of the path
                            vertices = path.vertices
                            all_contour_points.append(vertices)

                    processed_contour_segments = []
                    for vertices in all_contour_points:
                        segments = process_contour_path(vertices)
                        processed_contour_segments.extend(segments)

                    max_area = 0
                    indices = np.column_stack(np.where(mask))

                    x_coords = hor_vel_in_center.X.values
                    y_coords = hor_vel_in_center.Y.values

                    # Convert indices to actual coordinates
                    area_points = np.array([[x_coords[j], y_coords[i]] for i, j in indices])
                    
                    for vertices in processed_contour_segments:
                        # Create a Path object from the vertices
                        path_obj = Path(vertices)
                        
                        # Check if the contour segment is closed
                        if is_closed_contour(vertices, tol=1e-5):  # Adjust the tolerance value as needed
                            # Check if the center point is inside the contour segment
                            if any(path_obj.contains_point(point) for point in area_points):
                                    outermost_contour_vel.append(vertices)
                        
                        if len(outermost_contour_vel) != 0:
                            for vertices in outermost_contour_vel:
                                outermost_path = Path(vertices)
                                mask = xr.zeros_like(eta_around_center, dtype=bool)

                                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                                mask_flattened = outermost_path.contains_points(points)
                                mask.values = mask_flattened.reshape(mask.shape)

                                # Update eddies
                                x_indices = np.where((data.X >= center[1] - 1.5) & (data.X <= center[1] + 1.5))[0]
                                y_indices = np.where((data.Y >= center[0] - 1) & (data.Y <= center[0] + 1))[0]

                                current_subset = data.isel(X=x_indices, Y=y_indices)
                                cond = (~mask) | (current_subset != 0)
                                updated_subset = current_subset.where(cond, other=4)

                                data.values[np.ix_(y_indices, x_indices)] = updated_subset.values




            if test_calib:
                # Step 7: Plot the Data and the Outermost Closed Contour Segment
                plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels_array)
                if outermost_contour is not None:
                    plt.plot(outermost_contour[:, 0], outermost_contour[:, 1], 'r-', linewidth=2)
                    plt.contourf(mask.X, mask.Y, mask, colors=['white', 'blue'])
                    plt.scatter(*center_point, color='red')  # Mark the center point
                    plt.title("Topographic Data LOCAL MAX with Outermost Closed Contour Segment")
                    plt.show()

                else:
                    print('Local MAX not found')
                    plt.pcolormesh(eta_around_center.X,eta_around_center.Y,eta_around_center)
                    plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels_array)
                    plt.title("Topographic Data LOCAL MAX with Outermost Closed Contour Segment")
                    plt.show()

                # Outer eddy region test configuration plot
                if outermost_contour_vel is not None:
                    fig, ax = plt.subplots()
                    ax.pcolormesh(eta_around_center.X, eta_around_center.Y, eta_around_center)
                    ax.contour(hor_vel_around_center.X, hor_vel_around_center.Y, hor_vel_around_center,cmap='binary')
                    ax.plot(outermost_contour_vel[:, 0], outermost_contour_vel[:, 1], 'r-', linewidth=2)
                    # ax.contourf(mask.X, mask.Y, mask, colors=['white', 'blue'])
                    ax.scatter(*center_point, color='red')  # Mark the center point
                    ax.set_title("Topographic Data LOCAL MAX with velocity maximum segment")
                    fig.show()
                    
                else:
                    print("No Outer eddy region found.")

    if cold:
        for center in eddyCenterpoints:
            eta_around_center = eta.sel(X=slice(center[1]-1.5,center[1]+1.5)).sel(Y=slice(center[0]-1,center[0]+1))

            # Define outermost contour enclosing the eddy to mask out outlying values
            max_eta = eta_around_center.max().values
            min_eta = eta_around_center.min().values

            step_size = 0.001
            num_elements = int((max_eta-min_eta) / step_size) + 1
            levels_array = np.linspace(min_eta,max_eta,num_elements)

            contours = plt.contour(eta_around_center.X,eta_around_center.Y,eta_around_center,levels=levels_array)
            plt.close()

            # Collect all X and Y points of the paths of all contours
            all_contour_points = []

            for collection in contours.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            # Identify the Center Point
            center_point = (center[1], center[0])

            # Process Contour Paths to Handle Jumps
            def process_contour_path(vertices, jump_threshold=0.05):
                segments = []
                current_segment = [vertices[0]]
                
                for i in range(1, len(vertices)):
                    if np.linalg.norm(vertices[i] - vertices[i-1]) > jump_threshold:
                        segments.append(np.array(current_segment))
                        current_segment = [vertices[i]]
                    else:
                        current_segment.append(vertices[i])
                
                if current_segment:
                    segments.append(np.array(current_segment))
                
                return segments

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)

            # Find the Outermost Closed Contour Segment
            outermost_contour = None
            max_area = 0

            def is_closed_contour(vertices, tol=1e-5):
                distance = np.linalg.norm(vertices[0] - vertices[-1])
                return distance < tol

            def calculate_area(vertices):
                x = vertices[:, 0]
                y = vertices[:, 1]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            def contains_other_contours(outer_contour, all_contours, center_point):
                outer_path = Path(outer_contour)
                for contour in all_contours:
                    if np.array_equal(contour, outer_contour):
                        continue
                    contour_path = Path(contour)
                    if all(outer_path.contains_point(point) for point in contour) and not contour_path.contains_point(center_point):
                        return True
                return False

            for vertices in processed_contour_segments:
                # Create a Path object from the vertices
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-5):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    if path_obj.contains_point(center_point):
                        # Calculate the area of the contour segment using the shoelace formula
                        area = calculate_area(vertices)
                        
                        # Check if this contour contains any other contour that does not contain the center point
                        if not contains_other_contours(vertices, processed_contour_segments, center_point):
                            # Update the outermost contour if this one has a larger area
                            if area > max_area:
                                max_area = area
                                outermost_contour = vertices

            # Mask the area of inflextion point relevant for this eddy
            if outermost_contour is not None:
                outermost_path = Path(outermost_contour)
                mask = xr.zeros_like(eta_around_center, dtype=bool)

                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                mask_flattened = outermost_path.contains_points(points)
                mask.values = mask_flattened.reshape(mask.shape)

                # Update eddies
                x_indices = np.where((data.X >= center[1] - 1.5) & (data.X <= center[1] + 1.5))[0]
                y_indices = np.where((data.Y >= center[0] - 1) & (data.Y <= center[0] + 1))[0]

                current_subset = data.isel(X=x_indices, Y=y_indices)
                cond = (~mask) | (current_subset != 0) & (current_subset != 4)
                updated_subset = current_subset.where(cond, other=2)

                data.values[np.ix_(y_indices, x_indices)] = updated_subset.values

            # Mask the outer eddy regions
            if outermost_contour is not None:
                hor_vel_around_center = hor_vel.sel(X=slice(center[1]-1.5,center[1]+1.5)).sel(Y=slice(center[0]-1,center[0]+1))
                hor_vel_in_center = hor_vel_around_center.where(mask)
                hor_vel_max = hor_vel_in_center.max().values
                hor_vel_center = hor_vel_in_center.sel(X=center[1],Y=center[0]).values
                diff_hor_vel = hor_vel_max - hor_vel_center
                

                if diff_hor_vel >= min_vel_diff: # The difference between the eddy center and the maximum velocity of the inner region must be significant
                    contour_vel_num = hor_vel_max - (diff_hor_vel*(percentage_diff_vel/100))
                    contour_vel = plt.contour(hor_vel_around_center.X,hor_vel_around_center.Y,hor_vel_around_center,levels=[contour_vel_num])
                    plt.close()

                    all_contour_points = []

                    for collection in contour_vel.collections:
                        for path in collection.get_paths():
                            # Collect the vertices of the path
                            vertices = path.vertices
                            all_contour_points.append(vertices)

                    processed_contour_segments = []
                    for vertices in all_contour_points:
                        segments = process_contour_path(vertices)
                        processed_contour_segments.extend(segments)

                    outermost_contour_vel = []
                    max_area = 0
                    indices = np.column_stack(np.where(mask))

                    x_coords = hor_vel_in_center.X.values
                    y_coords = hor_vel_in_center.Y.values

                    # Convert indices to actual coordinates
                    area_points = np.array([[x_coords[j], y_coords[i]] for i, j in indices])
                    
                    for vertices in processed_contour_segments:
                        # Create a Path object from the vertices
                        path_obj = Path(vertices)
                        
                        # Check if the contour segment is closed
                        if is_closed_contour(vertices, tol=1e-5):  # Adjust the tolerance value as needed
                            # Check if the center point is inside the contour segment
                            if any(path_obj.contains_point(point) for point in area_points):
                                    outermost_contour_vel.append(vertices)
                        
                        if len(outermost_contour_vel) != 0:
                            for vertices in outermost_contour_vel:
                                outermost_path = Path(vertices)
                                mask = xr.zeros_like(eta_around_center, dtype=bool)

                                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                                mask_flattened = outermost_path.contains_points(points)
                                mask.values = mask_flattened.reshape(mask.shape)

                                # Update eddies
                                x_indices = np.where((data.X >= center[1] - 1.5) & (data.X <= center[1] + 1.5))[0]
                                y_indices = np.where((data.Y >= center[0] - 1) & (data.Y <= center[0] + 1))[0]

                                current_subset = data.isel(X=x_indices, Y=y_indices)
                                cond = (~mask) | (current_subset != 0)
                                updated_subset = current_subset.where(cond, other=4)

                                data.values[np.ix_(y_indices, x_indices)] = updated_subset.values
                

            if test_calib:
                # Step 7: Plot the Data and the Outermost Closed Contour Segment
                plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels_array,zorder=0)
                if outermost_contour is not None:
                    plt.plot(outermost_contour[:, 0], outermost_contour[:, 1], 'r-', linewidth=2,zorder=1)
                    plt.scatter(center_point[0],center_point[1], color='red',edgecolors='black',zorder=2)  # Mark the center point
                    plt.title("Topographic Data LOCAL MIN with Outermost Closed Contour Segment")
                    plt.show()
                else:
                    pass
                    print('Local MIN not found')
                    plt.pcolormesh(eta_around_center.X,eta_around_center.Y,eta_around_center)
                    plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels_array)
                    plt.title("Topographic Data LOCAL MIN with Outermost Closed Contour Segment")
                    plt.show()

                # Print the bounding box of the outermost closed contour segment
                if outermost_contour is not None:
                    min_x, min_y = np.min(outermost_contour, axis=0)
                    max_x, max_y = np.max(outermost_contour, axis=0)
                    print(f"Outermost closed contour segment bounding box: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
                else:
                    print("No closed contour segments found containing the center point.")
    
    return data


def Full_eddy_region_v2(eddyCenterpoints=list,eta=xr.DataArray(),cold=False,warm=False,test_calib=False,eddiesData=xr.DataArray()):
    """
    Eddy detection utilizing maximum outer closed contour around each eddy center point.\n
    Utilizes a inner eddy region by determining the outermost closed contour without any other local maxima and a outer region of the outermost closed contour containing the eddy center point.\n 
    Warm: Local maximum, Cold: Local minimum, test_calib: Plots contour field with max outer closed contour.\n
    Previously saved eddies dataset can be utilized for further additions.\n
    \n
    This version utilizes only SSH and defined eddy center points.
    """
    data = eddiesData

    if warm:
        for center in eddyCenterpoints:
            eta_around_center = eta.sel(X=slice(center[1]-1.5,center[1]+1.5)).sel(Y=slice(center[0]-1,center[0]+1))
            # Define outermost contour enclosing the eddy to mask out outlying values
            max_eta = eta_around_center.max().values
            min_eta = eta_around_center.min().values

            step_size = 0.001
            num_elements = int((max_eta-min_eta) / step_size) + 1
            levels_array = np.linspace(min_eta,max_eta,num_elements)

            contours = plt.contour(eta_around_center.X,eta_around_center.Y,eta_around_center,levels=levels_array)
            plt.close()

            # Collect all X and Y points of the paths of all contours
            all_contour_points = []

            for collection in contours.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            # Identify the Center Point
            center_point = (center[1], center[0])

            # Process Contour Paths to Handle Jumps
            def process_contour_path(vertices, jump_threshold=0.05):
                segments = []
                current_segment = [vertices[0]]
                
                for i in range(1, len(vertices)):
                    if np.linalg.norm(vertices[i] - vertices[i-1]) > jump_threshold:
                        segments.append(np.array(current_segment))
                        current_segment = [vertices[i]]
                    else:
                        current_segment.append(vertices[i])
                
                if current_segment:
                    segments.append(np.array(current_segment))
                
                return segments

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)

            # Find the Outermost Closed Contour Segment
            outermost_contour = None
            outer_outermost_contour = None
            max_area = 0
            max_area_outer = 0

            def is_closed_contour(vertices, tol=1e-5):
                distance = np.linalg.norm(vertices[0] - vertices[-1])
                return distance < tol

            def calculate_area(vertices):
                x = vertices[:, 0]
                y = vertices[:, 1]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            def contains_other_contours(outer_contour, all_contours, center_point):
                outer_path = Path(outer_contour)
                for contour in all_contours:
                    if np.array_equal(contour, outer_contour):
                        continue
                    contour_path = Path(contour)
                    if all(outer_path.contains_point(point) for point in contour) and not contour_path.contains_point(center_point):
                        return True
                return False

            for vertices in processed_contour_segments:
                # Create a Path object from the vertices
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-5):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    if path_obj.contains_point(center_point):
                        # Calculate the area of the contour segment using the shoelace formula
                        area = calculate_area(vertices)
                        
                        # Check if this contour contains any other contour that does not contain the center point
                        if not contains_other_contours(vertices, processed_contour_segments, center_point):
                            # Update the outermost contour if this one has a larger area
                            if area > max_area:
                                max_area = area
                                outermost_contour = vertices
                        else:
                            if area > max_area_outer:
                                max_area_outer = area
                                outer_outermost_contour = vertices

            # Mask the area of inner eddy eddy
            if outermost_contour is not None:
                outermost_path = Path(outermost_contour)
                mask = xr.zeros_like(eta_around_center, dtype=bool)

                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                mask_flattened = outermost_path.contains_points(points)
                mask.values = mask_flattened.reshape(mask.shape)

                # Update eddies
                x_indices = np.where((data.X >= center[1] - 1.5) & (data.X <= center[1] + 1.5))[0]
                y_indices = np.where((data.Y >= center[0] - 1) & (data.Y <= center[0] + 1))[0]

                current_subset = data.isel(X=x_indices, Y=y_indices)
                cond = (~mask) | (current_subset != 0) & (current_subset != 4)
                updated_subset = current_subset.where(cond, other=1)

                data.values[np.ix_(y_indices, x_indices)] = updated_subset.values

            # Mask the area of outer region of the eddy
            if outer_outermost_contour is not None:
                outermost_path = Path(outer_outermost_contour)
                mask = xr.zeros_like(eta_around_center, dtype=bool)

                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                mask_flattened = outermost_path.contains_points(points)
                mask.values = mask_flattened.reshape(mask.shape)

                # Update eddies
                x_indices = np.where((data.X >= center[1] - 1.5) & (data.X <= center[1] + 1.5))[0]
                y_indices = np.where((data.Y >= center[0] - 1) & (data.Y <= center[0] + 1))[0]

                current_subset = data.isel(X=x_indices, Y=y_indices)
                cond = (~mask) | (current_subset != 0)
                updated_subset = current_subset.where(cond, other=4)

                data.values[np.ix_(y_indices, x_indices)] = updated_subset.values
            



            if test_calib:
                # Step 7: Plot the Data and the Outermost Closed Contour Segment
                plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels_array)
                if outermost_contour is not None:
                    plt.plot(outermost_contour[:, 0], outermost_contour[:, 1], 'r-', linewidth=2)
                    plt.contourf(mask.X, mask.Y, mask, colors=['white', 'blue'])
                    plt.scatter(*center_point, color='red')  # Mark the center point
                    plt.title("Topographic Data LOCAL MAX with Outermost Closed Contour Segment")
                    plt.show()

                else:
                    print('Local MAX not found')
                    plt.pcolormesh(eta_around_center.X,eta_around_center.Y,eta_around_center)
                    plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels_array)
                    plt.title("Topographic Data LOCAL MAX with Outermost Closed Contour Segment")
                    plt.show()
                    

    if cold:
        for center in eddyCenterpoints:
            eta_around_center = eta.sel(X=slice(center[1]-1.5,center[1]+1.5)).sel(Y=slice(center[0]-1,center[0]+1))

            # Define outermost contour enclosing the eddy to mask out outlying values
            max_eta = eta_around_center.max().values
            min_eta = eta_around_center.min().values

            step_size = 0.001
            num_elements = int((max_eta-min_eta) / step_size) + 1
            levels_array = np.linspace(min_eta,max_eta,num_elements)

            contours = plt.contour(eta_around_center.X,eta_around_center.Y,eta_around_center,levels=levels_array)
            plt.close()

            # Collect all X and Y points of the paths of all contours
            all_contour_points = []

            for collection in contours.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            # Identify the Center Point
            center_point = (center[1], center[0])

            # Process Contour Paths to Handle Jumps
            def process_contour_path(vertices, jump_threshold=0.05):
                segments = []
                current_segment = [vertices[0]]
                
                for i in range(1, len(vertices)):
                    if np.linalg.norm(vertices[i] - vertices[i-1]) > jump_threshold:
                        segments.append(np.array(current_segment))
                        current_segment = [vertices[i]]
                    else:
                        current_segment.append(vertices[i])
                
                if current_segment:
                    segments.append(np.array(current_segment))
                
                return segments

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)

            # Find the Outermost Closed Contour Segment
            outermost_contour = None
            outer_outermost_contour = None
            max_area = 0
            max_area_outer = 0

            def is_closed_contour(vertices, tol=1e-5):
                distance = np.linalg.norm(vertices[0] - vertices[-1])
                return distance < tol

            def calculate_area(vertices):
                x = vertices[:, 0]
                y = vertices[:, 1]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            def contains_other_contours(outer_contour, all_contours, center_point):
                outer_path = Path(outer_contour)
                for contour in all_contours:
                    if np.array_equal(contour, outer_contour):
                        continue
                    contour_path = Path(contour)
                    if all(outer_path.contains_point(point) for point in contour) and not contour_path.contains_point(center_point):
                        return True
                return False

            for vertices in processed_contour_segments:
                # Create a Path object from the vertices
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-5):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    if path_obj.contains_point(center_point):
                        # Calculate the area of the contour segment using the shoelace formula
                        area = calculate_area(vertices)
                        
                        # Check if this contour contains any other contour that does not contain the center point
                        if not contains_other_contours(vertices, processed_contour_segments, center_point):
                            # Update the outermost contour if this one has a larger area
                            if area > max_area:
                                max_area = area
                                outermost_contour = vertices
                        else:
                            if area > max_area_outer:
                                max_area_outer = area
                                outer_outermost_contour = vertices

            # Mask the area of inflextion point relevant for this eddy
            if outermost_contour is not None:
                outermost_path = Path(outermost_contour)
                mask = xr.zeros_like(eta_around_center, dtype=bool)

                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                mask_flattened = outermost_path.contains_points(points)
                mask.values = mask_flattened.reshape(mask.shape)

                # Update eddies
                x_indices = np.where((data.X >= center[1] - 1.5) & (data.X <= center[1] + 1.5))[0]
                y_indices = np.where((data.Y >= center[0] - 1) & (data.Y <= center[0] + 1))[0]

                current_subset = data.isel(X=x_indices, Y=y_indices)
                cond = (~mask) | (current_subset != 0) & (current_subset != 4)
                updated_subset = current_subset.where(cond, other=2)

                data.values[np.ix_(y_indices, x_indices)] = updated_subset.values

            # Mask the area of outer region of the eddy
            if outer_outermost_contour is not None:
                outermost_path = Path(outer_outermost_contour)
                mask = xr.zeros_like(eta_around_center, dtype=bool)

                X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                mask_flattened = outermost_path.contains_points(points)
                mask.values = mask_flattened.reshape(mask.shape)

                # Update eddies
                x_indices = np.where((data.X >= center[1] - 1.5) & (data.X <= center[1] + 1.5))[0]
                y_indices = np.where((data.Y >= center[0] - 1) & (data.Y <= center[0] + 1))[0]

                current_subset = data.isel(X=x_indices, Y=y_indices)
                cond = (~mask) | (current_subset != 0)
                updated_subset = current_subset.where(cond, other=4)

                data.values[np.ix_(y_indices, x_indices)] = updated_subset.values
                

            if test_calib:
                # Step 7: Plot the Data and the Outermost Closed Contour Segment
                plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels_array,zorder=0)
                if outermost_contour is not None:
                    plt.plot(outermost_contour[:, 0], outermost_contour[:, 1], 'r-', linewidth=2,zorder=1)
                    plt.scatter(center_point[0],center_point[1], color='red',edgecolors='black',zorder=2)  # Mark the center point
                    plt.title("Topographic Data LOCAL MIN with Outermost Closed Contour Segment")
                    plt.show()
                else:
                    pass
                    print('Local MIN not found')
                    plt.pcolormesh(eta_around_center.X,eta_around_center.Y,eta_around_center)
                    plt.contour(eta_around_center.X, eta_around_center.Y, eta_around_center,levels_array)
                    plt.title("Topographic Data LOCAL MIN with Outermost Closed Contour Segment")
                    plt.show()

                # Print the bounding box of the outermost closed contour segment
                if outermost_contour is not None:
                    min_x, min_y = np.min(outermost_contour, axis=0)
                    max_x, max_y = np.max(outermost_contour, axis=0)
                    print(f"Outermost closed contour segment bounding box: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
                else:
                    print("No closed contour segments found containing the center point.")
    
    return data


def EddyTracking(eddyDataset=xr.DataArray(),center_points=list,EddiesTracked=None):
    if EddiesTracked is not None:
        data = EddiesTracked
    
    for center in center_points:
        pass