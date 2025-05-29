import xarray as xr
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pandas as pd
from tqdm import tqdm

import warnings
import matplotlib

# Suppress specific MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

with open("eddyCenterpoints_fullYear.txt",'r') as f:
        data = f.read()
        eddyLocation = ast.literal_eval(data)

#eddyLocation = eddyLocation[175:186] # test


eddy_df = xr.open_dataset('/nird/projects/NS9608K/MSc_EK/Data/Eddies_fullYear_final.nc')['EddyDetection']
eddy_area = eddy_df.where(eddy_df != 4, other=np.nan)
#eddy_area = eddy_area[175:186] # Test

# Test few timesteps
# eddy_area = eddy_area[0:2]
# eddyLocation = eddyLocation[0:2]

#eddy_time = xr.zeros_like(eddy_area).rename('EddyID')
eddy_time = xr.DataArray(np.full_like(eddy_area, '0',dtype=object), dims=eddy_area.dims, coords=eddy_area.coords).rename('EddyID')
ID_locMax = []

ID_num = 1

def is_closed_contour(vertices, tol=1e-5):
                distance = np.linalg.norm(vertices[0] - vertices[-1])
                return distance < tol


# Process Contour Paths to Handle Jumps
def process_contour_path(vertices, jump_threshold=0.2):
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


def convert_to_list(input_string):
    try:
        return ast.literal_eval(input_string)
    except (ValueError, SyntaxError):
        print('Error return: ',[input_string])
        return [input_string]



time_ = np.arange(len(eddyLocation))
# time_ = np.arange(0,11) # Test run
for time in tqdm(time_,desc='Running warm eddy ID'):
    eddy_max = []
    for eddy in eddyLocation[time][0]:
        if time == 0:
            contour = plt.contourf(eddy_area.X,eddy_area.Y,eddy_area[time],[0,1])
            plt.close()

            all_contour_points = []

            for collection in contour.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)
            
            for vertices in processed_contour_segments:
                # Create a Path object from the vertices       
                region = Path(vertices)
                if region.contains_point((eddy[1],eddy[0])):
                    if eddy_time[time].sel(X=eddy[1],Y=eddy[0],method='nearest').values != '0':
                        continue
                    mask = xr.zeros_like(eddy_time[0], dtype=bool)

                    X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                    points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                    mask_flattened = region.contains_points(points)
                    mask.values = mask_flattened.reshape(mask.shape)

                    cond = (~mask) | (eddy_time[time] != '0')
                    update = eddy_time[time].where(cond,other=str([str(ID_num)]))
                    eddy_time[time] = update
                    eddy_max.append([[str(ID_num)],eddy])
                    ID_num += 1
                    break
        
        else:
            contour = plt.contourf(eddy_area.X,eddy_area.Y,eddy_area[time],[0,1])
            plt.close()

            all_contour_points = []

            for collection in contour.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)
            
            for vertices in processed_contour_segments:
                # Check if the center point is inside the contour segment
                region = Path(vertices)
                if region.contains_point((eddy[1],eddy[0])):
                    if eddy_time[time].sel(X=eddy[1],Y=eddy[0],method='nearest').values != '0':
                        break
                    mask = xr.zeros_like(eddy_time[0], dtype=bool)

                    X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                    points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                    mask_flattened = region.contains_points(points)
                    mask.values = mask_flattened.reshape(mask.shape)

                    filtered_grid = eddy_time[time-1].where((mask) & (eddy_time[time-1] != '0'))
                    filtered_grid = filtered_grid.stack(points=("X","Y")).values
                    ID_in_region = [ids for ids in filtered_grid if not (isinstance(ids, float) and np.isnan(ids))] # Finds all IDs without nan
                    ID_in_region = set(ID_in_region) # Give unique IDs
                    
                    # Ensure IDs are lists of lists
                    ID_in_region = [convert_to_list(ids) if isinstance(ids, str) else [str(int(ids))] for ids in ID_in_region]
                    pre_IDs = [X[0] for X in ID_locMax[-1]]
                    
                    # Flatten the lists
                    flattened_ID_in_region = [item for sublist in ID_in_region for item in sublist]
                    flattened_pre_IDs = [item for sublist in pre_IDs for item in sublist]
                    # Find intersection
                    intersection = list(set(flattened_ID_in_region).intersection(flattened_pre_IDs))

                    if len(intersection) != 0: # If the previous timestep contains the IDs in local maximum
                        ID = intersection
                        cond = (~mask) | (eddy_time[time] != '0')
                        update = eddy_time[time].where(cond,other=str(ID))
                        eddy_time[time] = update
                        eddy_max.append([ID,eddy])
                        break
                    else:
                        try:
                            filtered_grid = eddy_time[time-2].where((mask) & (eddy_time[time-2] != '0'))
                            filtered_grid = filtered_grid.stack(points=("X","Y")).values
                            ID_in_region = [ids for ids in filtered_grid if not (isinstance(ids, float) and np.isnan(ids))] # Finds all IDs without nan
                            ID_in_region = set(ID_in_region) # Give unique IDs
                            # Ensure IDs are lists of lists
                            ID_in_region = [convert_to_list(ids) if isinstance(ids, str) else [str(int(ids))] for ids in ID_in_region]
                            pre_IDs = [X[0] for X in ID_locMax[-2]]

                            # Flatten the lists
                            flattened_ID_in_region = [item for sublist in ID_in_region for item in sublist]
                            flattened_pre_IDs = [item for sublist in pre_IDs for item in sublist]
                            # Find unique IDs
                            inRegion_set = set(flattened_ID_in_region)
                            preID_set = set(flattened_pre_IDs)
                            
                            intersection = list(inRegion_set.intersection(preID_set))
                            if len(intersection) != 0:
                                ID = intersection
                                cond = (~mask) | (eddy_time[time] != '0')
                                update = eddy_time[time].where(cond,other=str(ID))
                                eddy_time[time] = update
                                eddy_max.append([ID,eddy])
                                break
                            else:
                                cond = (~mask) | (eddy_time[time] != '0')
                                update = eddy_time[time].where(cond,other=str([str(ID_num)]))
                                eddy_time[time] = update
                                eddy_max.append([[str(ID_num)],eddy])
                                ID_num += 1
                                break
                        except:
                            cond = (~mask) | (eddy_time[time] != '0')
                            update = eddy_time[time].where(cond,other=str([str(ID_num)]))
                            eddy_time[time] = update
                            eddy_max.append([[str(ID_num)],eddy])
                            ID_num += 1
                            break

    # Test if Bifuraction has occured, and add letter to the same ID in time
    letter_ID = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','æ','ø','å']
    new_eddy_max = []
    ID_eddy_max = [X[0] for X in eddy_max]
    unique_eddy_max = [tuple(sublist) for sublist in ID_eddy_max]
    unique_eddy_max = list(set(unique_eddy_max))
    if len(ID_eddy_max) == len(unique_eddy_max):
        ID_locMax.append(eddy_max)
    else:
        print('Bifurcation occurs!')
        SameID = [[] for X in unique_eddy_max] # Empty list of list for each unique ID
        for j,unique in enumerate(unique_eddy_max):
            for i,ID in enumerate(eddy_max):
                if unique == tuple(ID[0]):
                    SameID[j].append(ID) # Add each ID to list of unique IDs
        for j,data in enumerate(SameID):
            if len(data) != 1:
                for i,subdata in enumerate(data):
                    ID_multi = []
                    for multiIDs in subdata[0]:
                        ID_multi.append(multiIDs+letter_ID[i]) # Adds same letter to ID of multiple of unique (Also if multi ID is within)
                    new_eddy_max.append([ID_multi,subdata[1]])
            else:
                new_eddy_max.append(SameID[j][0])
        
        contour = plt.contourf(eddy_area.X,eddy_area.Y,eddy_area[time],[0,1])
        plt.close()

        all_contour_points = []

        for collection in contour.collections:
            for path in collection.get_paths():
                # Collect the vertices of the path
                vertices = path.vertices
                all_contour_points.append(vertices)

        processed_contour_segments = []
        for vertices in all_contour_points:
            segments = process_contour_path(vertices)
            processed_contour_segments.extend(segments)

        for eddy in new_eddy_max:
            for vertices in processed_contour_segments:
                    # Check if the center point is inside the contour segment
                    region = Path(vertices)
                    if region.contains_point((eddy[1][1],eddy[1][0])):
                        mask = xr.zeros_like(eddy_time[0], dtype=bool)

                        X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                        points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                        mask_flattened = region.contains_points(points)
                        mask.values = mask_flattened.reshape(mask.shape)
                        cond = (~mask)
                        update = eddy_time[time].where(cond,other=str(eddy[0]))
                        eddy_time[time] = update
        ID_locMax.append(new_eddy_max)


ID_locMin = []
for time in tqdm(time_,desc='Running cold eddy ID'):
    eddy_min = []
    for eddy in eddyLocation[time][1]:
        if time == 0:
            contour = plt.contourf(eddy_area.X,eddy_area.Y,eddy_area[time],[1.5,2])
            plt.close()

            all_contour_points = []

            for collection in contour.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)
            
            for vertices in processed_contour_segments:
                # Check if the center point is inside the contour segment
                region = Path(vertices)
                if region.contains_point((eddy[1],eddy[0])):
                    if eddy_time[time].sel(X=eddy[1],Y=eddy[0],method='nearest').values != '0':
                        continue
                    mask = xr.zeros_like(eddy_time[0], dtype=bool)

                    X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                    points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                    mask_flattened = region.contains_points(points)
                    mask.values = mask_flattened.reshape(mask.shape)

                    cond = (~mask) | (eddy_time[time] != '0')
                    update = eddy_time[time].where(cond,other=str([str(ID_num)]))
                    eddy_time[time] = update
                    eddy_min.append([[str(ID_num)],eddy])
                    ID_num += 1
                    break
        
        else:
            contour = plt.contourf(eddy_area.X,eddy_area.Y,eddy_area[time],[1.5,2])
            plt.close()

            all_contour_points = []

            for collection in contour.collections:
                for path in collection.get_paths():
                    # Collect the vertices of the path
                    vertices = path.vertices
                    all_contour_points.append(vertices)

            processed_contour_segments = []
            for vertices in all_contour_points:
                segments = process_contour_path(vertices)
                processed_contour_segments.extend(segments)
            
            for vertices in processed_contour_segments:              
                # Check if the center point is inside the contour segment
                region = Path(vertices)
                if region.contains_point((eddy[1],eddy[0])):
                    if eddy_time[time].sel(X=eddy[1],Y=eddy[0],method='nearest').values != '0':
                        continue
                    mask = xr.zeros_like(eddy_time[0], dtype=bool)

                    X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                    points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                    mask_flattened = region.contains_points(points)
                    mask.values = mask_flattened.reshape(mask.shape)

                    filtered_grid = eddy_time[time-1].where((mask) & (eddy_time[time-1] != '0'))
                    filtered_grid = filtered_grid.stack(points=("X","Y")).values
                    ID_in_region = [ids for ids in filtered_grid if not (isinstance(ids, float) and np.isnan(ids))] # Finds all IDs without nan
                    ID_in_region = set(ID_in_region) # Give unique IDs
                    
                    # Ensure IDs are lists of lists
                    ID_in_region = [convert_to_list(ids) if isinstance(ids, str) else [str(int(ids))] for ids in ID_in_region]
                    pre_IDs = [X[0] for X in ID_locMin[-1]]

                    # Flatten the lists
                    flattened_ID_in_region = [item for sublist in ID_in_region for item in sublist]
                    flattened_pre_IDs = [item for sublist in pre_IDs for item in sublist]
                    # Find unique IDs
                    inRegion_set = set(flattened_ID_in_region)
                    preID_set = set(flattened_pre_IDs)
                    
                    intersection = inRegion_set.intersection(preID_set)

                    if len(intersection) != 0: # If the previous timestep contains the IDs in local maximum
                        ID = list(intersection)
                        cond = (~mask) | (eddy_time[time] != '0')
                        update = eddy_time[time].where(cond,other=str(ID))
                        eddy_time[time] = update
                        eddy_min.append([ID,eddy])
                        break
                    else:
                        try:
                            filtered_grid = eddy_time[time-2].where((mask) & (eddy_time[time-2] != '0'))
                            filtered_grid = filtered_grid.stack(points=("X","Y")).values
                            ID_in_region = [ids for ids in filtered_grid if not (isinstance(ids, float) and np.isnan(ids))] # Finds all IDs without nan
                            ID_in_region = set(ID_in_region) # Give unique IDs
                            # Ensure IDs are lists of lists
                            ID_in_region = [convert_to_list(ids) if isinstance(ids, str) else [str(int(ids))] for ids in ID_in_region]
                            pre_IDs = [X[0] for X in ID_locMin[-2]]

                            # Flatten the lists
                            flattened_ID_in_region = [item for sublist in ID_in_region for item in sublist]
                            flattened_pre_IDs = [item for sublist in pre_IDs for item in sublist]
                            # Find unique IDs
                            inRegion_set = set(flattened_ID_in_region)
                            preID_set = set(flattened_pre_IDs)
                            
                            intersection = inRegion_set.intersection(preID_set)
                            
                            if len(intersection) != 0:
                                ID = list(intersection)
                                cond = (~mask) | (eddy_time[time] != '0')
                                update = eddy_time[time].where(cond,other=str(ID))
                                eddy_time[time] = update
                                eddy_min.append([ID,eddy])
                                break
                            else:
                                cond = (~mask) | (eddy_time[time] != '0')
                                update = eddy_time[time].where(cond,other=str([str(ID_num)]))
                                eddy_time[time] = update
                                eddy_min.append([[str(ID_num)],eddy])
                                ID_num += 1
                                break
                        except:
                            cond = (~mask) | (eddy_time[time] != '0')
                            update = eddy_time[time].where(cond,other=str([str(ID_num)]))
                            eddy_time[time] = update
                            eddy_min.append([[str(ID_num)],eddy])
                            ID_num += 1
                            break
            
    # Test if Bifuraction has occured, and add letter to the same ID in time
    letter_ID = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','æ','ø','å']
    new_eddy_min = []
    ID_eddy_min = [X[0] for X in eddy_min]
    unique_eddy_min = [tuple(sublist) for sublist in ID_eddy_min]
    unique_eddy_min = list(set(unique_eddy_min))
    if len(ID_eddy_min) == len(unique_eddy_min):
        ID_locMin.append(eddy_min) # No Bifurcation events
    else: # Bifurcation events
        SameID = [[] for X in unique_eddy_min] # Empty list of list for each unique ID
        for j,unique in enumerate(unique_eddy_min):
            for i,ID in enumerate(eddy_min):
                if unique == tuple(ID[0]):
                    SameID[j].append(ID) # Add each ID to list of unique IDs
        for j,data in enumerate(SameID):
            if len(data) != 1: # Test if ID has Bifuraction
                for i,subdata in enumerate(data):
                    ID_multi = []
                    for multiIDs in subdata[0]:
                        ID_multi.append(multiIDs+letter_ID[i]) # Adds same letter to ID of multiple of unique (Also if multi ID is within)
                    new_eddy_min.append([ID_multi,subdata[1]])
            else:
                new_eddy_min.append(SameID[j][0])

        contour = plt.contourf(eddy_area.X,eddy_area.Y,eddy_area[time],[1.5,2])
        plt.close()

        all_contour_points = []

        for collection in contour.collections:
            for path in collection.get_paths():
                # Collect the vertices of the path
                vertices = path.vertices
                all_contour_points.append(vertices)

        processed_contour_segments = []
        for vertices in all_contour_points:
            segments = process_contour_path(vertices)
            processed_contour_segments.extend(segments)

        for eddy in new_eddy_min:
            for vertices in processed_contour_segments:
                    # Check if the center point is inside the contour segment
                    region = Path(vertices)
                    if region.contains_point((eddy[1][1],eddy[1][0])):
                        mask = xr.zeros_like(eddy_time[0], dtype=bool)

                        X_vals, Y_vals = np.meshgrid(mask.X, mask.Y)
                        points = np.vstack((X_vals.flatten(), Y_vals.flatten())).T

                        mask_flattened = region.contains_points(points)
                        mask.values = mask_flattened.reshape(mask.shape)
                        cond = (~mask)
                        update = eddy_time[time].where(cond,other=str(eddy[0]))
                        eddy_time[time] = update
        ID_locMin.append(new_eddy_min)


# Flatten the data
flattened_data1 = []
for time_index, time_data in enumerate(ID_locMax):
    for eddy_index, eddy_data in enumerate(time_data):
        flattened_data1.append([time_index, eddy_data[0], eddy_data[1][0], eddy_data[1][1]])

# Create a DataFrame
df = pd.DataFrame(flattened_data1, columns=['Time', 'ID', 'Latitude', 'Longitude'])

# Write to Excel
df.to_csv('/nird/projects/NS9608K/MSc_EK/Data/EddyResults/Tracking/locMAX_final.csv', index=False)

# Flatten the data
flattened_data2 = []
for time_index, time_data in enumerate(ID_locMin):
    for eddy_index, eddy_data in enumerate(time_data):
        flattened_data2.append([time_index, eddy_data[0], eddy_data[1][0], eddy_data[1][1]])

# Create a DataFrame
df2 = pd.DataFrame(flattened_data2, columns=['Time', 'ID', 'Latitude', 'Longitude'])

# Write to Excel
df2.to_csv('/nird/projects/NS9608K/MSc_EK/Data/EddyResults/Tracking/locMin_final.csv', index=False)

eddy_time.to_netcdf('/nird/projects/NS9608K/MSc_EK/Data/EddyResults/Tracking/EddyAreaID_final.nc')