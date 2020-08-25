import pandas as pd
import xarray as xr
from pathlib import Path
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

# create list of available s1 files
filelist = list()
for path in Path('/home/fgreifeneder@eurac.edu/Documents/sm_paper/smapvex16/s1sm/').rglob('*.tif'):
    filelist.append(path)

sv16sm = pd.read_csv('/home/fgreifeneder@eurac.edu/Documents/sm_paper/smapvex16/sv16_sm_measurements_locs.csv')
sv16sm = sv16sm.loc[:, ['SITE_ID', 'DATE', 'VOL_SOIL_M', 'sv16_wit11', 'sv16_wit12']]
sv16sm.columns = ['SITE_ID', 'DATE', 'VOL_SOIL_M', 'x', 'y']

for ismpath in filelist:
    # read s1 raster
    s1img = xr.open_rasterio(ismpath)

    # extract date
    idate = dt.datetime.strptime(ismpath.name[6:14], "%Y%m%d")
    #idate = dt.datetime.strptime("20160524", "%Y%m%d")
    sv16sm_subset = sv16sm.loc[sv16sm['DATE'] == idate.strftime('%Y-%m-%d'), :]
    if sv16sm_subset.size == 0:
        continue

    # extract sm values from raster
    smlist = list()
    for irow in range(sv16sm_subset.shape[0]):
        if np.isfinite(sv16sm_subset['x'].iloc[irow]):
            smlist.append(s1img.interp(x=sv16sm_subset['x'].iloc[irow], y=sv16sm_subset['y'].iloc[irow],
                                       method='linear').values[0])
        else:
            smlist.append(0.0)

    sv16sm.loc[sv16sm['DATE'] == idate.strftime('%Y-%m-%d'), 's1sm'] = smlist
print('something')


