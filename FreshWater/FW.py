import oceanspy as ospy
import numpy as np
import xarray as xr


def FWT(mooringDataset=xr.Dataset,ref_salinity=int):
    # Open dataset as OceanDataset
    od = ospy.OceanDataset(mooringDataset)
    # Subsample into monthly mean values
    od = od.subsample.cutout(timeFreq='ME',sampMethod='mean')
    # Compute total transport through the mooring array
    od = od.compute.mooring_volume_transport()

    # Defining salinity integration depth
    transport = od.dataset['transport']
    salinity = od.dataset['S']#.squeeze()
    FW = transport.where(salinity<ref_salinity) * ((ref_salinity-salinity)/ref_salinity)
    
    return FW

def Transport(mooringDataset=xr.Datset):
    pass
