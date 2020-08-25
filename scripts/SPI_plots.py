import geopandas as gpd
import pandas as pd
import os
import json
import datetime as dt
import numpy as np
from standard_precip.spi import SPI
import matplotlib.pyplot as plt

outpath = '/mnt/SAT/Workspaces/GrF/01_Data/InSitu/Province/'
inpath = '/mnt/SAO/DROUGHT/Data/in_situ/ST/monthly/'
inpath2 = '/mnt/SAO/DROUGHT/Data/in_situ/ST/daily/'

# get the station coordinates
stations = gpd.read_file('/mnt/SAT/Workspaces/GrF/01_Data/InSitu/Province/stations.geojson')

n_stations, _ = stations.shape

spi_3_list = list()
spi_6_list = list()
spi_12_list = list()

for st_i in range(n_stations):

    st_code = stations.iloc[st_i]['SCODE']
    st_name = stations.iloc[st_i]['NAME_D']


    if os.path.exists(inpath2 + st_name + '_LT_N_daily.csv'):
        print(st_name)
        #N_series = pd.concat(N)
        #N_series.sort_index(inplace=True)
        N_series = pd.Series(pd.DataFrame.from_csv(inpath2 + st_name + '_LT_N_daily.csv')['mm'])

        # calculate SPI
        tmp = N_series.dropna()
        N_series_monthly = tmp.resample('MS').sum()
        N_series_monthly.dropna(inplace=True)

        if len(N_series_monthly) > 120:

            spi3 = SPI()
            spi3.set_rolling_window_params(span=3, window_type='boxcar', center=False)
            spi3.set_distribution_params(dist_type='gamma')
            spi3_data = spi3.calculate(np.array(N_series_monthly), starting_month=N_series_monthly.index[0].month)
            n_spi3_df = pd.DataFrame({st_name.lower().replace(' ', '_'): spi3_data.squeeze()}, index=N_series_monthly.index)
            spi_3_list.append(n_spi3_df.copy())

            spi6 = SPI()
            spi6.set_rolling_window_params(span=6, window_type='boxcar', center=False)
            spi6.set_distribution_params(dist_type='gamma')
            spi6_data = spi6.calculate(np.array(N_series_monthly), starting_month=N_series_monthly.index[0].month)
            n_spi6_df = pd.DataFrame({st_name.lower().replace(' ', '_'): spi6_data.squeeze()}, index=N_series_monthly.index)
            spi_6_list.append(n_spi6_df.copy())

            spi12 = SPI()
            spi12.set_rolling_window_params(span=12, window_type='boxcar', center=False)
            spi12.set_distribution_params(dist_type='gamma')
            spi12_data = spi12.calculate(np.array(N_series_monthly), starting_month=N_series_monthly.index[0].month)
            n_spi12_df = pd.DataFrame({st_name.lower().replace(' ', '_'): spi12_data.squeeze()},
                                     index=N_series_monthly.index)
            spi_12_list.append(n_spi12_df.copy())

            n_out_df = pd.DataFrame({'N': N_series_monthly,
                                     'SPI3': spi3_data.squeeze(),
                                     'SPI6': spi6_data.squeeze(),
                                     'SPI12': spi12_data.squeeze()}, index=N_series_monthly.index)
            n_out_df.to_csv(outpath + 'monthly/' + st_name + 'N_monthly_SPI.csv')

            if np.any(n_out_df['2015-07-01':'2016-07-01'].SPI3 < -2):
                stations.loc[st_i, 'SPI3drought1516'] = "YES"
            else:
                stations.loc[st_i, 'SPI3drought1516'] = "NO"
            if np.any(n_out_df['2015-07-01':'2016-07-01'].SPI6 < -2):
                stations.loc[st_i, 'SPI6drought1516'] = "YES"
            else:
                stations.loc[st_i, 'SPI6drought1516'] = "NO"
        else:
            stations.loc[st_i, 'SPI3drought1516'] = np.nan
            stations.loc[st_i, 'SPI6drought1516'] = np.nan
    else:
        stations.loc[st_i, 'SPI3drought1516'] = np.nan
        stations.loc[st_i, 'SPI6drought1516'] = np.nan

spi3_df = pd.concat(spi_3_list, axis=1)
spi3_df.to_csv(outpath + 'SPI3.csv', encoding='utf-8')
spi6_df = pd.concat(spi_6_list, axis=1)
spi6_df.to_csv(outpath+ 'SPI6.csv', encoding='utf-8')
spi12_df = pd.concat(spi_12_list, axis=1)
spi12_df.to_csv(outpath+ 'SPI12.csv', encoding='utf-8')

# plot
plt.figure(figsize=(30, 10))
spi3_df.plot()#['2014-01-01':'2017-12-31'].plot()
plt.show()
plt.savefig(outpath + 'SPI3full.png')
plt.close()

plt.figure(figsize=(30, 10))
spi3_df.plot()#['2014-01-01':'2017-12-31'].plot(legend=False)
plt.show()
plt.savefig(outpath + 'SPI3fullnolegend.png')
plt.close()

plt.figure(figsize=(30, 10))
spi6_df.plot(legend=False)#['2014-01-01':'2017-12-31'].plot(legend=False)
plt.show()
plt.savefig(outpath + 'SPI6fullnolegend.png')
plt.close()

plt.figure(figsize=(30, 10))
spi6_df.plot()#['2014-01-01':'2017-12-31'].plot()
plt.show()
plt.savefig(outpath + 'SPI6full.png')
plt.close()

plt.figure(figsize=(30, 10))
spi12_df.plot(legend=False)#['2014-01-01':'2017-12-31'].plot(legend=False)
plt.show()
plt.savefig(outpath + 'SPI12fullnolegend.png')
plt.close()

plt.figure(figsize=(30, 10))
spi12_df.plot()#['2014-01-01':'2017-12-31'].plot()
plt.show()
plt.savefig(outpath + 'SPI12full.png')
plt.close()

stations.to_file(outpath+'drought1516')





