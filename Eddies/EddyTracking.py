import xarray as xr
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pandas as pd
from tqdm import tqdm


with open("eddyCenterpoints_fullYear.txt",'r') as f:
        data = f.read()
        eddyLocation = ast.literal_eval(data)


eddy_df = xr.open_dataset('/nird/projects/NS9608K/MSc_EK/Data/Eddies_fullYear.nc')['EddyDetection']
eddy_area = eddy_df.where(eddy_df != 4, other=np.nan)

eddy_time = xr.zeros_like(eddy_area).rename('EddyID')
ID_locMax = []

ID_num = 0

def is_closed_contour(vertices, tol=1e-5):
                distance = np.linalg.norm(vertices[0] - vertices[-1])
                return distance < tol


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


time_ = np.arange(len(eddyLocation))
for time in tqdm(time_,desc='Running warm eddy ID'):
    eddy_max = []
    for eddy in eddyLocation[time][0]:
        if time == 0:
            contour = plt.contourf(eddy_area.X,eddy_area.Y,eddy_area[time])
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
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-5):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    region = Path(vertices)
                    if region.contains_point((eddy[1],eddy[0])):
                        eddy_max.append([ID_num,eddy])
                        ID_num += 1
                        break
        
        else:
            contour = plt.contourf(eddy_area.X,eddy_area.Y,eddy_area[time])
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
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-5):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    region = Path(vertices)
                    if region.contains_point((eddy[1],eddy[0])):
                        points = [(X[1][1],X[1][0]) for X in ID_locMax[-1]]
                        if any(region.contains_point(point) for point in points):
                             for find_ID in ID_locMax[-1]:
                                  if region.contains_point((find_ID[1][1],find_ID[1][0])):
                                    ID = find_ID[0]
                                    eddy_max.append([ID,eddy])
                                    break
                            
                        else:
                            try:
                                points = [(X[1][1],X[1][0]) for X in ID_locMax[-2]]
                                if any(region.contains_point(point) for point in points):
                                    for find_ID in ID_locMax[-2]:
                                        if region.contains_point((find_ID[1][1],find_ID[1][0])):
                                            ID = find_ID[0]
                                            eddy_max.append([ID,eddy])
                                            break
                                else:
                                    try:
                                        points = [(X[1][1],X[1][0]) for X in ID_locMax[-3]]
                                        if any(region.contains_point(point) for point in points):
                                            for find_ID in ID_locMax[-3]:
                                                if region.contains_point((find_ID[1][1],find_ID[1][0])):
                                                    ID = find_ID[0]
                                                    eddy_max.append([ID,eddy])
                                                    break
                                        else:
                                            eddy_max.append([ID_num,eddy])
                                            ID_num += 1
                                    except:
                                        eddy_max.append([ID_num,eddy])
                                        ID_num += 1
                            except:
                                eddy_max.append([ID_num,eddy])
                                ID_num += 1
            
    ID_locMax.append(eddy_max)

ID_num = ID_locMax[-1][-1][0] + 1
ID_locMin = []
time_ = np.arange(len(eddyLocation))
for time in tqdm(time_,desc='Running cold eddy ID'):
    eddy_min = []
    for eddy in eddyLocation[time][1]:
        if time == 0:
            contour = plt.contourf(eddy_area.X,eddy_area.Y,eddy_area[time])
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
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-5):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    region = Path(vertices)
                    if region.contains_point((eddy[1],eddy[0])):
                        eddy_min.append([ID_num,eddy])
                        ID_num += 1
                        break
        
        else:
            contour = plt.contourf(eddy_area.X,eddy_area.Y,eddy_area[time])
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
                path_obj = Path(vertices)
                
                # Check if the contour segment is closed
                if is_closed_contour(vertices, tol=1e-5):  # Adjust the tolerance value as needed
                    # Check if the center point is inside the contour segment
                    region = Path(vertices)
                    if region.contains_point((eddy[1],eddy[0])):
                        points = [(X[1][1],X[1][0]) for X in ID_locMin[-1]]
                        if any(region.contains_point(point) for point in points):
                             for find_ID in ID_locMin[-1]:
                                  if region.contains_point((find_ID[1][1],find_ID[1][0])):
                                    ID = find_ID[0]
                                    eddy_min.append([ID,eddy])
                                    break
                            
                        else:
                            try:
                                points = [(X[1][1],X[1][0]) for X in ID_locMin[-2]]
                                if any(region.contains_point(point) for point in points):
                                    for find_ID in ID_locMin[-2]:
                                        if region.contains_point((find_ID[1][1],find_ID[1][0])):
                                            ID = find_ID[0]
                                            eddy_min.append([ID,eddy])
                                            break
                                else:
                                    try:
                                        points = [(X[1][1],X[1][0]) for X in ID_locMin[-3]]
                                        if any(region.contains_point(point) for point in points):
                                            for find_ID in ID_locMin[-3]:
                                                if region.contains_point((find_ID[1][1],find_ID[1][0])):
                                                    ID = find_ID[0]
                                                    eddy_min.append([ID,eddy])
                                                    break
                                        else:
                                            eddy_min.append([ID_num,eddy])
                                            ID_num += 1
                                    except:
                                        eddy_min.append([ID_num,eddy])
                                        ID_num += 1
                            except:
                                eddy_min.append([ID_num,eddy])
                                ID_num += 1
            
    ID_locMin.append(eddy_min)


# Flatten the data
flattened_data1 = []
for time_index, time_data in enumerate(ID_locMax):
    for eddy_index, eddy_data in enumerate(time_data):
        flattened_data1.append([time_index, eddy_data[0], eddy_data[1][0], eddy_data[1][1]])

# Create a DataFrame
df = pd.DataFrame(flattened_data1, columns=['Time', 'ID', 'Latitude', 'Longitude'])

# Write to Excel
df.to_csv('/nird/projects/NS9608K/MSc_EK/Data/locMAX.csv', index=False)

# Flatten the data
flattened_data2 = []
for time_index, time_data in enumerate(ID_locMin):
    for eddy_index, eddy_data in enumerate(time_data):
        flattened_data2.append([time_index, eddy_data[0], eddy_data[1][0], eddy_data[1][1]])

# Create a DataFrame
df2 = pd.DataFrame(flattened_data2, columns=['Time', 'ID', 'Latitude', 'Longitude'])

# Write to Excel
df2.to_csv('/nird/projects/NS9608K/MSc_EK/Data/locMin.csv', index=False)