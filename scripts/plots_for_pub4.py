from SMC.TigerMap import getTimeSeries
import pytesmo.io.ismn.interface as ismn_interface
from pytesmo.time_series.anomaly import calc_climatology
import pandas as pd
import matplotlib.pyplot as plt

# initialise available ISMN data
# ismn = ismn_interface.ISMN_Interface('/mnt/SAT4/DATA/S1_EODC/ISMN/')
# # load data
# ismn_station = ismn.get_station('KLEE', 'COSMOS')
# station_depths = ismn_station.get_depths('soil moisture')
# ismn_sensors = ismn_station.get_sensors('soil moisture', depth_from=0, depth_to=0.1)
# ismn_series = ismn_station.read_variable('soil moisture', depth_from=0, depth_to=0.1, sensor=ismn_sensors[0]).data
#
# s1_ts, s1_clim = getTimeSeries(ismn_station.longitude,
#                                ismn_station.latitude,
#                                '/mnt/SAT/Workspaces/GrF/02_Documents/1_THESIS/pub4/plots/',
#                                100,
#                                57,
#                                '/mnt/SAT/Workspaces/GrF/Processing/ESA_TIGER/GEE/SVR_Model_Python_S1.p',
#                                calc_anomalies=True,
#                                name='KLEE')
#
# s1_clim_dates = [s1_clim['S1'].iloc[x.dayofyear-1] for x in ismn_series.index]
# s1_clim_dates = pd.Series(s1_clim_dates, index=ismn_series.index)
#
# ismn_clim = calc_climatology(ismn_series['soil moisture'])
# ismn_clim_dates = [ismn_clim.values[x.dayofyear-1] for x in ismn_series.index]
# ismn_clim_dates = pd.Series(ismn_clim_dates, index=ismn_series.index)
#
# # scale time-series based on climatology
# s1_clim_s1_dates = [s1_clim['S1'].iloc[x.dayofyear-1] for x in s1_ts.index]
# s1_clim_s1_dates = pd.Series(s1_clim_s1_dates, index=s1_ts.index)
# s1_clim_range = s1_clim['S1'].max() - s1_clim['S1'].min()
# s1_ts_scaled = (s1_ts - s1_clim_s1_dates) / s1_clim_range
#
# ismn_clim_range = ismn_clim.max() - ismn_clim.min()
# ismn_series_scaled = (ismn_series['soil moisture'] - ismn_clim_dates) / ismn_clim_range
#
# fig, ax = plt.subplots(figsize=(6.5,2.7))
# line1, = ax.plot(s1_ts.index, s1_ts_scaled, color='b', linestyle='-', marker='+', label='S1', linewidth=0.2)
# #line2, = ax.plot(s1_clim_dates.index, s1_clim_dates, color='b', linestyle='--', label='S1 climatology', linewidth=0.2)
# line3, = ax.plot(ismn_series.index, ismn_series_scaled, color='r', label='In-situ', linewidth=0.2)
# #line4, = ax.plot(ismn_clim_dates.index, ismn_clim_dates*100, color='r', linestyle='--', label='In-situ climatology', linewidth=0.2)
# x0 = [ismn_series.index[0], s1_ts.index[-1]]
# y0 = [0, 0]
# line4, = ax.plot(x0, y0, color='k', linestyle='--', linewidth=0.2)
#
# plt.setp(ax.get_xticklabels(), fontsize=6)
#
# ax.set_ylabel('Soil Moisture Anomaly')
# plt.legend(handles=[line1,line3])
#
# plt.show()
# plt.savefig('/mnt/SAT/Workspaces/GrF/02_Documents/1_THESIS/pub4/plots/KLEE_anomalies_and_insitu.png', dpi=300)
# plt.close()
#
# print(s1_ts)

trmm = pd.Series.from_csv('/mnt/SAT/Workspaces/GrF/02_Documents/1_THESIS/pub4/plots/giovanni/trmm.txt', header=0,infer_datetime_format=True)
s1_ug = pd.Series.from_csv('/mnt/SAT/Workspaces/GrF/02_Documents/1_THESIS/pub4/plots/giovanni/smc_uasingishu.txt', header=0, infer_datetime_format=True)

trmm = trmm.sort_index()
s1_ug = s1_ug.sort_index()

fig, ax = plt.subplots(figsize=(6.5,2.7))
line1, = ax.plot(s1_ug.index, s1_ug, color='b', linestyle='-', marker='+', label='S1 Soil Moisture', linewidth=0.2)
ax.set_ylim(0,30)

ax2 = ax.twinx()
line2, = ax2.plot(trmm.index, trmm, color='r', label='TRMM Precipitation', linewidth=0.2)

ax.set_ylabel('Surface Soil Moisture [%-Vol.]')
ax2.set_ylabel('Precipitation [mm/h]')
plt.legend(handles=[line1, line2])

#fig.tight_layout()
plt.show()
plt.savefig('/mnt/SAT/Workspaces/GrF/02_Documents/1_THESIS/pub4/plots/giovanni/trmm_vs_s1.png', dpi=300)
plt.close()
