import geopandas as gpd
import pandas as pd
import os
import json
import datetime as dt
import numpy as np
#from standard_precip.spi import SPI

if __name__ == '__main__':

    outpath = 'Q:/SAO/DROUGHT/Data/in_situ/ST/raw/'
    inpath = '/mnt/SAO/case_study_drought/Data/in_situ/ST/'
    inpath2 = '/mnt/SAO/case_study_drought/Data/in_situ/ST/daily/'

    # get the station coordinates
    stations = gpd.read_file("Q:/SAO/DROUGHT/Data/in_situ/ST/2_stations.geojson")

    n_stations, _ = stations.shape



    for st_i in range(n_stations):

        st_code = stations.iloc[st_i]['SCODE']
        st_name = stations.iloc[st_i]['NAME_D']

        if st_name != 'Naturns' and \
            st_name != 'Schlanders' and \
            st_name != 'Laars - Eyrs' and \
            st_name != 'ETSCH BEI SPONDINIG' and \
            st_name != 'RAMBACH BEI LAATSCH' and \
            st_name != 'Marienberg':
            continue

        LT = list()
        N = list()

        for date_i in range(2000,2019):

            exec_string_LT = 'curl "http://daten.buergernetz.bz.it/services/meteo/v1/timeseries?station_code=' + st_code + \
                   '&sensor_code=LT&date_from=' + str(date_i) + '0101&date_to=' + str(date_i) + '1231" > ' + outpath + 'tmpLT' + str(date_i) + '.json'
            exec_string_N = 'curl "http://daten.buergernetz.bz.it/services/meteo/v1/timeseries?station_code=' + st_code + \
                   '&sensor_code=N&date_from=' + str(date_i) + '0101&date_to=' + str(date_i) + '1231" > ' + outpath + 'tmpN' + str(date_i) + '.json'

            os.system(exec_string_LT)
            os.system(exec_string_N)

            tmp_LT_file = json.load(open(outpath + 'tmpLT' + str(date_i) + '.json'))
            tmp_N_file = json.load(open(outpath + 'tmpN' + str(date_i) + '.json'))

            if len(tmp_LT_file) != 0:
                tmp_dates_LT = [dt.datetime.strptime(x['DATE'][0:19], '%Y-%m-%dT%H:%M:%S') for x in tmp_LT_file]
                tmp_values_LT = [x['VALUE'] for x in tmp_LT_file]

                tmp_LT_series = pd.Series(tmp_values_LT, index=tmp_dates_LT)

                LT.append(tmp_LT_series.copy())

                tmp_LT_file = None

            if len(tmp_N_file) != 0:
                tmp_dates_N = [dt.datetime.strptime(x['DATE'][0:19], '%Y-%m-%dT%H:%M:%S') for x in tmp_N_file]
                tmp_values_N = [x['VALUE'] for x in tmp_N_file]

                tmp_N_series = pd.Series(tmp_values_N, index=tmp_dates_N)
                N.append(tmp_N_series.copy())
                tmp_N_file = None

            os.system('rm ' + outpath + 'tmpLT' + str(date_i) + '.json')
            os.system('rm ' + outpath + 'tmpN' + str(date_i) + '.json')



        if len(LT) != 0:
        #if os.path.exists(inpath + st_name + '_LT.csv'):
            LT_series = pd.concat(LT)
            LT_series.sort_index(inplace=True)
            #LT_series = pd.Series.from_csv(inpath + st_name + '_LT.csv')
            outname_LT = outpath + st_name + '_LT.csv'
            LT_series.to_csv(outname_LT)
            #stations.loc[st_i, 'TEMP_MEAN'] = LT_series.mean()
            #stations.loc[st_i, 'TEMP_STD'] = LT_series.std()
        else:
            stations.loc[st_i, 'TEMP_MEAN'] = np.nan
            stations.loc[st_i, 'TEMP_STD'] = np.nan
    #
        if len(N) != 0:
    #     if os.path.exists(inpath2 + st_name + '_LT_N_daily.csv'):
    #        print(st_name)
            N_series = pd.concat(N)
            N_series.sort_index(inplace=True)
    #         N_series = pd.Series(pd.DataFrame.from_csv(inpath2 + st_name + '_LT_N_daily.csv')['mm'])
    #
    #         # calculate SPI
    #         tmp = N_series.dropna()
    #         N_series_monthly = tmp.resample('BMS').sum()
    #         N_series_monthly.dropna(inplace=True)
    #
    #         if len(N_series_monthly) > 120:
    #
    #             spi = SPI()
    #             spi.set_rolling_window_params(span=3, window_type='boxcar')
    #             spi.set_distribution_params(dist_type='gamma')
    #             spi_data = spi.calculate(np.array(N_series_monthly), starting_month=N_series_monthly.index[0].month)
    #             n_droughts = len(np.where(spi_data < -2)[0])
    #             stations.loc[st_i, 'N_DROUGHTS'] = n_droughts
    #
    #             n_spi_df = pd.DataFrame({'N': N_series_monthly, 'SPI': spi_data.squeeze()}, index=N_series_monthly.index)
    #             n_spi_df.to_csv(outpath + 'monthly/' + st_name + 'N_monthly_SPI.csv')
    #         else:
    #             stations.loc[st_i, 'N_DROUGHTS'] = np.nan
    #
            outname_N = outpath + st_name + '_N.csv'
            N_series.to_csv(outname_N)
    #         stations.loc[st_i, 'PRECIP_SUM'] = N_series.sum() / len(np.unique(N_series.index.date))
    #     else:
    #         stations.loc[st_i, 'PRECIP_SUM'] = np.nan
    #         stations.loc[st_i, 'N_DROUGHTS'] = np.nan
    #
    # stations.to_file(outpath+'stations_w_stats')
    #stations.to_csv(outpath+'stations_w_stats.csv')




