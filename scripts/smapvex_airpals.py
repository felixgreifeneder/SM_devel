import pandas as pd
import xarray as xr
from pathlib import Path
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

def inside_image(lat, lon, image):
    if (lat < image.y.min()) | (lat > image.y.max()) | (lon < image.x.min()) | (lon > image.x.max()):
        return False
    else:
        return True

# load datasets
airpals_path = '/home/fgreifeneder@eurac.edu/Documents/sm_paper/smapvex16/air_pals/iowa/SV16I_PLTBSM_PALS_VSM_SFhi_M500_v033_v064_20160531_both.txt'
s1_path = '/home/fgreifeneder@eurac.edu/Downloads/SMCS1_20160601_001314_063_A.tif'

airpals_data = pd.read_csv(airpals_path, index_col=False, skipinitialspace=True, sep=",")
s1_data = xr.open_rasterio(s1_path).coarsen(x=10, y=10, boundary='trim').mean()

# extract values
airpals_data['S1SM'] = np.full(airpals_data.shape[0], np.nan)

for irow in range(airpals_data.shape[0]):
    try:
        if inside_image(airpals_data['Lat'].iloc[irow], airpals_data['Lon'].iloc[irow], s1_data):
            tmp = s1_data.interp(x=airpals_data['Lon'].iloc[irow], y=airpals_data['Lat'].iloc[irow],
                                         method='linear').values[0]
            airpals_data['S1SM'].iat[irow] = tmp
    except:
        pass

fig, axs = plt.subplots(2, 4, figsize=(15,8))
for i, x in zip(airpals_data['LC'].unique(), axs.flatten()):
    airpals_data.where(airpals_data['LC'] == i).plot.scatter('VSM', 'S1SM', ax=x)
    x.set_title(str(i))
plt.tight_layout()
plt.savefig('/home/fgreifeneder@eurac.edu/Documents/sm_paper/smapvex16/airpals_vs_s1.png', dpi=600)

