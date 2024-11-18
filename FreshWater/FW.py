import oceanspy as ospy
import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt


def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def FWT(mooringDataset=xr.Dataset,ref_salinity=int):
    # Open dataset as OceanDataset
    od = ospy.OceanDataset(mooringDataset)

    # Subsample into monthly mean values
    od = od.subsample.cutout(timeFreq='D',sampMethod='mean')

    # Compute total transport through the mooring array
    od = od.compute.mooring_volume_transport()

    # # Low-pass 5 day filter
    # cutoff_frequency = 1 / (5 * 24 * 3600)  # 5 days in seconds
    # sampling_frequency = 1 / (np.diff(od.dataset['time'].values).mean() / np.timedelta64(1, 's'))  # Sampling frequency in Hz

    # # Apply the filter to the transport variable
    # od.dataset['transport'] = xr.apply_ufunc(
    #     lowpass_filter,
    #     od.dataset['transport'],
    #     cutoff_frequency,
    #     sampling_frequency,
    #     input_core_dims=[['time']],
    #     vectorize=True
    # )

    # Defining salinity integration depth
    transport = od.dataset['transport']
    salinity = od.dataset['S']#.squeeze()
    FW = transport.where(salinity<ref_salinity) * ((ref_salinity-salinity)/ref_salinity)
    
    return FW


def Transport(mooringDataset=xr.Dataset):
    # Open dataset as OceanDataset
    od = ospy.OceanDataset(mooringDataset)
    # Subsample into monthly mean values
    od = od.subsample.cutout(timeFreq='D',sampMethod='mean')
    # Compute total transport through the mooring array
    od = od.compute.mooring_volume_transport()

    # Defining salinity integration depth
    transport = od.dataset['transport']
    
    return transport


def FWC(obs=False,model=False,data=list,ref_salinity=float):
    return_data = []
    if obs:
        for subdata in data:
            sec = []
            for i in range(len(subdata.time)):
                sec.append(np.cumsum((ref_salinity-subdata.SA[i])/ref_salinity))
            return_data.append(sec)

    elif model:
        for subdata in data:
            sec = []
            for subsubdata in subdata:
                sec.append(np.cumsum((ref_salinity-subsubdata.S)/ref_salinity))
            return_data.append(sec)

    return return_data