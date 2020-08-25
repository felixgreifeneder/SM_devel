import pandas as pd
import xarray as xr
from pathlib import Path
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

# load datasets
airpals_path = '/home/fgreifeneder@eurac.edu/Documents/sm_paper/smapvex16/air_pals/SV16M_PLTBSM_PALS_VSM_MBhi_M500_v033_v064_20160719_both.txt'
s1_path = '/home/fgreifeneder@eurac.edu/Documents/sm_paper/smapvex16/s1sm/original_dates/SMCS1_20160719_00_063_A.tif'

airpals_data = pd.read_csv(airpals_path, skipinitialspace=True)
s1_data = xr.open_rasterio(s1_path)

# extract values
smlist = list()

for irow in range(airpals_data.shape[0]):
    smlist.append(s1_data.interp(x=airpals_data['Lon'].iloc[irow], y=airpals_data['Lat'].iloc[irow],
                                 method='linear').values[0])

airpals_data['s1sm'] = smlist
print('done')
