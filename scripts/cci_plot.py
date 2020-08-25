from netCDF4 import Dataset
import datetime
import numpy as np
import pandas as pd
from pytesmo.time_series.plotting import plot_clim_anom
import matplotlib.pyplot as plt
from pytesmo.time_series.anomaly import calc_climatology
from pytesmo.time_series.anomaly import calc_anomaly

# load cci
#cci = Dataset('/mnt/SAT/Workspaces/GrF/01_Data/CCI/extracted_sm_2.nc', 'r', format="NETCDF4")
cci = Dataset('/mnt/SAT/Workspaces/GrF/01_Data/CCI/kassel.nc', 'r', format='NETCDF4')
#sm = np.array(cci['sm'])
sm = np.array(cci['RootMoist']).squeeze()
sm[np.where(sm == np.nan)] = np.nan
sm[sm == -9999] = np.nan

sm_time = cci['time']
#sm_time = np.array([datetime.datetime.fromtimestamp(i*24*60*60) for i in sm_time])
timediff = datetime.datetime(year=1979, month=1, day=1, hour=0, minute=0, second=0)-datetime.datetime.fromtimestamp(0)
sm_time = np.array([datetime.datetime.fromtimestamp(i*24*60*60)+timediff for i in sm_time])
sm = np.nanmean(sm, axis=(1,2))

sm_pd = pd.Series(sm, index=sm_time)
sm_pd = sm_pd.dropna()
sm_df = pd.DataFrame({'Soil Moisture': sm_pd.values}, index=sm_pd.index)

sm_climatology = calc_climatology(sm_pd)
sm_climatology = pd.DataFrame({'Soil Moisture': sm_climatology.values}, index=sm_climatology.index)

fig, axs = plt.subplots(figsize=(25,10))

plot_clim_anom(sm_df, clim=sm_climatology, axes=[axs], clim_color='r', clim_linewidth='0.9')
#sm_anomalies_monthly.plot(ax=axs, linestyle='-', linewidth=0.5)
plt.savefig('/mnt/SAT/Workspaces/GrF/model_kassel_sm_1980_2015.png')
plt.close()
cci.close()











