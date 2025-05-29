import EddyDetectionV2 as eddy
import oceanspy as ospy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import SymLogNorm
from tqdm import tqdm
import ast
from matplotlib.animation import FuncAnimation, FFMpegFileWriter


# Enable/disable computation of different domains ##################
config_parent_domain = True
config_child_domain = True

# Computes full year run eddy detection

time = ['2017-09-01T00:00:00.000000000','2018-08-31T00:00:00.000000000']
####################################################################


# Import data #######################################################################################################################
depth = xr.open_dataset(r'/nird/projects/NS9608K/MSc_EK/Data/Depth_res.nc')['Depth'].sel(Y=slice(70,75)).sel(X=slice(-22,2))
depth_no_nan = depth
depth = depth.where(depth > 0, np.nan)

df = xr.open_dataset('/nird/projects/NS9608K/MSc_EK/Data/Background_var_eddies.nc')

df_OW = df['Okubo_Weiss']
df_OW = df_OW*(1/(10**(-9)))
df_eta = df['Eta']


# Delete no longer used variables which uses memory
del df
#####################################################################################################################################
# Child domain
domain_center = [[-20,0],[71,74]]
OW_center = df_OW.sel(X=slice(domain_center[0][0],domain_center[0][1])).sel(Y=slice(domain_center[1][0],domain_center[1][1])).sel(Z=-1)
eta_center = df_eta.sel(X=slice(domain_center[0][0],domain_center[0][1])).sel(Y=slice(domain_center[1][0],domain_center[1][1]))

# Parent domain
eta = df_eta.sel(X=slice(domain_center[0][0]-2,domain_center[0][1]+2)).sel(Y=slice(domain_center[1][0]-1,domain_center[1][1]+1))
OW = df_OW.sel(X=slice(domain_center[0][0]-2,domain_center[0][1]+2)).sel(Y=slice(domain_center[1][0]-1,domain_center[1][1]+1)).sel(Z=-1)

# Resample data
eta = eta.resample(time='D').mean(dim='time')
eta_center = eta_center.resample(time='D').mean(dim='time')
OW = OW.resample(time='D').mean(dim='time')
OW_center = OW_center.resample(time='D').mean(dim='time')

# Define time extent
eta = eta.sel(time=slice(time[0],time[1]))
eta_center = eta_center.sel(time=slice(time[0],time[1]))
OW = OW.sel(time=slice(time[0],time[1]))
OW_center = OW_center.sel(time=slice(time[0],time[1]))


eta = eta.where(depth > 0, np.nan)
OW = OW.where(depth > 0, np.nan)

# Computes or loads eddy centerpoints #####################################################################
run = config_child_domain

if run:
    eddyLocation = []
    OW_th = -0.1*OW_center.std().values
    T = len(eta)
    pbar = tqdm(total=T, desc="Running eddy centerpoint algorythm")
    for i in range(len(eta)):
        eddyLocation.append(eddy.eddyDetection(eta_center[i],OW_center[i],OW_th))
        pbar.update(1)
    pbar.close()
    with open("eddyCenterpoints_fullYear.txt",'w') as f:
        for time in range(len(eddyLocation)):
            if time != len(eddyLocation)-1:
                f.write(str(eddyLocation[time])+',')
            else:
                f.write(str(eddyLocation[time]))

else:
    print('###Loading previously saved eddy centerpoints###')
    with open("eddyCenterpoints_fullYear.txt",'r') as f:
        data = f.read()
        eddyLocation = ast.literal_eval(data)
############################################################################################################

# Run region detection ##################################################################################
run = config_parent_domain
if run:
    eddies = xr.full_like(eta,fill_value=0)
    eddies = eddies.rename("EddyDetection")
    T = len(eddyLocation)
    pbar = tqdm(total=T, desc="Running algorythm")
    for time in range(len(eddyLocation)):
        eddies[time] = eddy.inner_eddy_region_v5(eddyLocation[time][0],eta=eta[time],warm=True,cold=False,test_calib=False,eddiesData=eddies[time])
        eddies[time] = eddy.inner_eddy_region_v5(eddyLocation[time][1],eta=eta[time],warm=False,cold=True,test_calib=False,eddiesData=eddies[time])
        pbar.update(1)
    ###############################################################
    eddies = eddies.where(eddies != 0, np.nan)
    eddies.to_netcdf('/nird/projects/NS9608K/MSc_EK/Data/Eddies_fullYear_final.nc')
    pbar.close()
else:
    eddies = xr.open_dataset('/nird/projects/NS9608K/MSc_EK/Data/Eddies_fullYear_final.nc')
    eddies = eddies['EddyDetection']
##########################################################################################################
print('### Complete ###')