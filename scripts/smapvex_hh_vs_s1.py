import pandas as pd
import xarray as xr
from pathlib import Path
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from xrspatial import convolution

# load datasets
hh_path = '/home/fgreifeneder@eurac.edu/Documents/sm_paper/smapvex16/insitu_handheld/SV16M_PSM_SoilMoistureHandheld_Vers3_w_coords.csv'
s1_path = '/home/fgreifeneder@eurac.edu/Documents/sm_paper/smapvex16/s1sm/SMCS1_20160719_001513_063_A.tif'

hh_data = pd.read_csv(hh_path, index_col=[1, 2], parse_dates=[1])
# hh_data['SITE_ID'] = [x.split("-")[0] for x in hh_data['SITE_ID']]
# hh_data = hh_data.groupby('SITE_ID').mean()
#hh_data = hh_data.groupby(level=0).mean()
hh_data = hh_data.xs('Top', level='LOCATION')
hh_data = hh_data.loc[hh_data.index.date == dt.date(year=2016, month=7, day=19)]
s1_data = xr.open_rasterio(s1_path)
s1_data = convolution.convolve_2d(s1_data, np.full((3, 3), 1/3))

# extract values
smlist = list()

for irow in range(hh_data.shape[0]):
    try:
        tmp = s1_data.interp(x=hh_data['Lon'].iloc[irow], y=hh_data['Lat'].iloc[irow],
                                     method='linear').values[0]
        # if tmp > 40:
        #     hh_data['SOIL_MOISTURE'].iloc[irow] = hh_data['SOIL_MOISTURE'].iloc[irow] + 0.1
        smlist.append(tmp)
    except:
        smlist.append(np.nan)

hh_data['s1sm'] = smlist
print('done')
