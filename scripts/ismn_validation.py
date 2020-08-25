import pytesmo.io.ismn.interface as ismn_interface
from sgrt_devels.derive_smc import extract_time_series_gee
from sgrt_devels.extr_TS import extr_SIG0_LIA_ts_GEE
from sgrt_devels.extr_TS import extr_MODIS_MOD13Q1_ts_GEE
from sgrt_devels.extr_TS import extr_GLDAS_ts_GEE
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import os


def ismn_validation_run(bsize=500, name='500m', fvect1=None, fvect2=None, fvect1desc=None, fvect2desc=None):
    s1path=117

    #bsize=250

    basepath = '//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_' + name + '_reprocess_lt_05/'

    # outpath
    outpath = basepath + 'w_GLDAS_asc_desc/'

    # Calculate prediction standar deviations
    calcstd = False

    # use descending orbits
    desc = False

    # calculate anomalies?
    calc_anomalies=False

    # initialise available ISMN data
    ismn = ismn_interface.ISMN_Interface('T:/ECOPOTENTIAL/reference_data/ISMN/')

    # get list of networks
    networks = ismn.list_networks()

    # initialise S1 SM retrieval
    # mlmodel = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ASCAT/gee/mlmodel0.p', 'rb'))
    mlmodel_avg = "//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_1km_reprocess_lt_05/no_GLDAS_RFmlmodelNoneSVR_2step.p"
    mlmodel = basepath + "w_GLDAS_RFmlmodelNoneSVR_2step.p"
    mlmodel_desc = "//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_50m_reprocess_lt_05_descending/w_GLDAS_RFmlmodelNoneSVR_2step.p"

    # initialse text report
    txtrep = open(outpath + '2_report.txt', 'w')
    txtrep.write('Accuracy report for Soil Moisture validation based on ISMN stations\n\n')
    txtrep.write('Model used: ' + mlmodel + '\n')
    txtrep.write('------------------------------------------------------------------------\n\n')
    txtrep.write('Name, R, RMSE\n')

    xyplot = pd.DataFrame()
    cntr = 1

    #used_stations = np.load('X:/Workspaces/GrF/Processing/S1ALPS/ISMN/gee_global_all_no_deep/ValidStaions.npy')
    used_stations = pickle.load(open(basepath + "testset_meta.p", 'rb'))
    #invalid_col = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ISMN/gee_global005_highdbtollerance/invalid_col.p', 'rb'))
    #used_stations = [invalid_col['ntwkname'][i] + ', ' + invalid_col['stname'][i] for i in range(len(invalid_col['ntwkname']))]

    #for ntwk in networks:
    #for vstation in used_stations:
    s1_ts_list = list()
    station_ts_list = list()
    station_name_list = list()
    gldas_ts_list = list()
    s1_failed_list = list()
    s1_ub_list = list()
    for st_i in range(len(used_stations[0])):
    #for st_i in [0]:

        #ntwk, st_name = [x.strip() for x in vstation.split(',')]
        ntwk = used_stations[1][st_i]
        st_name = used_stations[0][st_i]
        # if st_name != 'ROSASCYN':# and st_name != 'Salem-10-W':
        #     continue

        # get list of available stations
        available_stations = ismn.list_stations(ntwk)

        # iterate through all available ISMN stations
        #for st_name in available_stations:
        try:

            station = ismn.get_station(st_name, ntwk)
            station_vars = station.get_variables()
            # if st_name != '2.10':
            #     continue

            # if (st_name != 'Boulder-14-W'):
            #    continue
            #if st_name not in ['ANTIMONYFL', 'CALIVALLEY', 'Bethlehem']:
            # if st_name not in ['CALIVALLEY']:
            #     continue

            if 'soil moisture' not in station_vars:
                continue

            station_depths = station.get_depths('soil moisture')

            # if 0.0 in station_depths[0]:
            #     sm_sensors = station.get_sensors('soil moisture', depth_from=0, depth_to=0.24)
            #     station_ts = station.read_variable('soil moisture', depth_from=0, depth_to=0.24, sensor=sm_sensors[0])
            # else:
            #     continue

            if 0.0 in station_depths[0]:
                did = np.where(station_depths[0] == 0.0)
                dto = station_depths[1][did]
                sm_sensors = station.get_sensors('soil moisture', depth_from=0, depth_to=dto[0])
                print(sm_sensors[0])
                station_ts = station.read_variable('soil moisture', depth_from=0, depth_to=dto[0], sensor=sm_sensors[0])
            elif 0.05 in station_depths[0]:
                sm_sensors = station.get_sensors('soil moisture', depth_from=0.05, depth_to=0.05)
                station_ts = station.read_variable('soil moisture', depth_from=0.05, depth_to=0.05, sensor=sm_sensors[0])
            else:
                continue

            print(st_name)

            plotpath = outpath + st_name + '.png'

            # if os.path.exists(plotpath):
            #     continue

            # get station ts
            station_ts = station_ts.data['soil moisture']

            # get S1 time series
            s1_ts, s1_ts_std, outliers, s1_failed = extract_time_series_gee(mlmodel,
                                                                 mlmodel_avg,
                                                                 '/mnt/SAT4/DATA/S1_EODC/',
                                                                  outpath,
                                                                  station.latitude,
                                                                  station.longitude,
                                                                  name=st_name,
                                                                  footprint=bsize,
                                                                  calcstd=calcstd,
                                                                  desc=desc,
                                                                  target=station_ts,
                                                                  feature_vect1=fvect1,
                                                                  feature_vect2=fvect2)#,
                                        #s1path=s1path)

            s1_ts2, s1_ts_std2, outliers2, s1_failed2 = extract_time_series_gee(mlmodel_desc,
                                                                            mlmodel_avg,
                                                                            '/mnt/SAT4/DATA/S1_EODC/',
                                                                            outpath,
                                                                            station.latitude,
                                                                            station.longitude,
                                                                            name=st_name,
                                                                            footprint=bsize,
                                                                            calcstd=calcstd,
                                                                            desc=True,
                                                                            target=station_ts,
                                                                            feature_vect1=fvect1desc,
                                                                            feature_vect2=fvect2desc)  # ,

            if (s1_ts is not None) and (s1_ts2 is not None):
                # correct the mean offset between time-series from ascending and descending orbits
                meandiff = s1_ts.mean() - s1_ts2.mean()
                s1_ts2 = s1_ts2 + meandiff
                meandiff_failed = s1_failed.median() - s1_failed2.median()
                s1_failed2 = s1_failed2 + meandiff_failed
                s1_ts = pd.concat([s1_ts, s1_ts2])
                s1_failed = pd.concat([s1_failed, s1_failed2])
            elif (s1_ts is None) and (s1_ts2 is not None):
                s1_ts = s1_ts2
                s1_failed = s1_failed2

            s1_ts.sort_index(inplace=True)
            s1_failed.sort_index(inplace=True)

            if s1_ts is None:
                continue

            if len(s1_ts) < 5:
                continue

            #evi_ts = extr_MODIS_MOD13Q1_ts_GEE(station.longitude, station.latitude, bufferSize=150)
            #evi_ts = pd.Series(evi_ts[1]['EVI'], index=evi_ts[0])

            gldas_ts = extr_GLDAS_ts_GEE(station.longitude, station.latitude, bufferSize=150, yearlist=[2014,2015,2016,2017,2018,2019])
            gldas_ts = gldas_ts / 100.

            start = np.array([s1_ts.index[0], station_ts.index[0]]).max()
            end = np.array([s1_ts.index[-1], station_ts.index[-1]]).min()
            if start > end:
                continue
            station_ts = station_ts[start:end]
            s1_ts = s1_ts[start:end]
            s1_failed = s1_failed[start:end]
            #outliers = outliers[start:end]
            #evi_ts = evi_ts[start:end]
            gldas_ts = gldas_ts[start:end]
            if calcstd == True:
                s1_ts_std = s1_ts_std[start:end]
            if len(s1_ts) < 1:
                continue
            #station_ts = station_ts.iloc[np.where(station_ts > 0.1)]

            #s1_ts = s1_ts[np.where(error_ts == error_ts.min())[0]]

            s1_ts_res = s1_ts.resample('D').mean().rename('s1')
            station_ts_res = station_ts.resample('D').mean().rename('ismn')
            gldas_ts_res = gldas_ts.resample('D').mean().rename('gldas')

            if calc_anomalies == True:
                from pytesmo.time_series import anomaly as pyan
                s1_clim = pyan.calc_climatology(s1_ts_res.interpolate())
                station_clim = pyan.calc_climatology(station_ts_res.interpolate())

                s1_ts = pyan.calc_anomaly(s1_ts, climatology=s1_clim)
                s1_ts_res = pyan.calc_anomaly(s1_ts_res, climatology=s1_clim)
                station_ts = pyan.calc_anomaly(station_ts, climatology=station_clim)
                station_ts_res = pyan.calc_anomaly(station_ts_res, climatology=station_clim)


            # calculate error metrics
            ts_bias = s1_ts_res.subtract(station_ts_res).mean()

            # cdf matching
            tobemerged = [s1_ts_res.dropna(), gldas_ts_res.dropna(), station_ts_res.dropna()]
            s1_and_station = pd.concat(tobemerged, axis=1, join='inner')
            statmask = (s1_and_station['ismn'] > 0) & (s1_and_station['ismn'] < 1) & (s1_and_station['s1'] > 0) & (s1_and_station['s1'] < 1)
            p2 = s1_and_station['ismn'][statmask].std() / s1_and_station['s1'][statmask].std()
            p1 = s1_and_station['ismn'][statmask].mean() - (p2 * s1_and_station['s1'][statmask].mean())
            s1_ts_ub = p1 + (p2 * s1_ts)
            s1_and_station['s1ub'] = p1 + (p2 * s1_and_station['s1'])
            ts_bias = s1_and_station['s1ub'].subtract(s1_and_station['ismn']).median()
            s1_failed = p1 + (p2 * s1_failed)

            xytmp = pd.concat({'y': s1_and_station['s1ub'], 'x': s1_and_station['ismn']}, join='inner', axis=1)
            if cntr == 1:
                xyplot = xytmp
            else:
                xyplot = pd.concat([xyplot, xytmp], axis=0)

            cntr = cntr + 1


            ts_cor = s1_and_station['s1ub'].corr(s1_and_station['ismn'])
            ts_rmse = np.sqrt(np.nanmean(np.square(s1_and_station['s1ub'].subtract(s1_and_station['ismn']))))
            #ts_ubrmse = np.sqrt(np.sum(np.square(s1_ts_res.subtract(s1_ts_res.mean()).subtract(station_ts_res.subtract(station_ts_res.mean())))))
            ts_ubrmse = np.sqrt(np.sum(np.square((s1_and_station['s1ub'] - s1_and_station['s1ub'].mean()) - (s1_and_station['ismn'] - s1_and_station['ismn'].mean()))) / len(s1_and_station['s1ub']))
            print('R: ' + str(ts_cor))
            print('RMSE: ' + str(ts_rmse))
            print('Bias: ' + str(ts_bias))
            txtrep.write(st_name + ', ' + str(ts_cor) + ', ' + str(ts_rmse) + '\n')

            s1_ts_list.append(s1_ts)
            station_ts_list.append(station_ts)
            station_name_list.append(st_name)
            gldas_ts_list.append(gldas_ts)
            s1_failed_list.append(s1_failed)
            s1_ub_list.append(s1_ts_ub)
            # plot
            # plt.figure(figsize=(18, 6))
            fig, ax1 = plt.subplots(figsize=(7.16,1.4), dpi=300)
            line1, = ax1.plot(s1_ts_ub.index, s1_ts_ub, color='b', linestyle='', marker='+', label='Sentinel-1', linewidth=0.2)
            line8, = ax1.plot(s1_failed.index, s1_failed, color='r', linestyle='', marker='+', label='fail', linewidth=0.2)
            line2, = ax1.plot(station_ts.index, station_ts, label='In-Situ', linewidth=0.4)
            if np.any(outliers) and outliers is not None:
                line6, = ax1.plot(s1_ts.index[outliers], s1_ts.iloc[outliers], color='r', linestyle='', marker='o')
            #line3, = ax1.plot(outliers.index, outliers, color='r', linestyle='', marker='*', label='Outliers')
            if calcstd == True:
                line4, = ax1.plot(s1_ts.index, s1_ts - np.sqrt(s1_ts_std), color='k', linestyle='--', linewidth=0.2)
                line5, = ax1.plot(s1_ts.index, s1_ts + np.sqrt(s1_ts_std), color='k', linestyle='--', linewidth=0.2)

            #ax3 = ax1.twinx()
            line6, = ax1.plot(gldas_ts.index, gldas_ts, color='g', linestyle='--', label='GLDAS', linewidth=0.2)
            #line3, = ax3.plot(evi_ts.index, evi_ts, linewidth=0.4, color='r', linestyle='--', label='MOD13Q1:EVI')
            # ax3.axes.tick_params(axis='y', direction='in', labelcolor='r', right='off', labelright='off')

            ax1.set_ylabel('Soil Moisture [m3m-3]', size=8)
            smc_max = np.max([s1_ts.max(), station_ts.max()])
            if smc_max <= 0.5:
                smc_max = 0.5
            ax1.set_ylim((0,smc_max))
            ax1.text(0.85, 0.4, 'R=' + '{:03.2f}'.format(ts_cor) +
                     #'\nRMSE=' + '{:03.2f}'.format(ts_rmse) +
                     '\nBias=' + '{:03.2f}'.format(ts_bias) +
                     '\nRMSE=' + '{:03.2f}'.format(ts_rmse), transform=ax1.transAxes, fontsize=8)
            # ax2.set_ylabel('In-Situ [m3/m3]')
            #ax3.set_ylabel('EVI')

            #fig.tight_layout()
            #plt.legend(handles=[line1, line2], loc='upper left', fontsize=8)#, line3, line6])
            plt.title(st_name, fontsize=8)
            #plt.show()
            plt.tight_layout()
            plt.savefig(plotpath, dpi=300)
            plt.close()
        except:
            print('No data for: ' + st_name)

    pickle.dump((s1_ts_list, s1_ub_list, s1_failed, station_ts_list, gldas_ts_list, station_name_list),
                open('C:/Users/FGreifeneder/OneDrive - Scientific Network South Tyrol/1_THESIS/pub3/images_submission2/w_GLDAS_validation_tss_' + name + '.p', 'wb'))
    scatter_valid = np.where(((xyplot['x'] > 0) & (xyplot['x'] < 1)) & ((xyplot['y'] > 0) & (xyplot['y'] < 1)))
    xyplot = xyplot.iloc[scatter_valid]
    urmse_scatter = np.sqrt(np.sum(np.square((xyplot['y'] - xyplot['y'].mean()) - (xyplot['x'] - xyplot['x'].mean()))) / len(xyplot['y']))
    rmse_scatter = np.sqrt(np.nanmean(np.square(xyplot['x'].subtract(xyplot['y']))))
    r_scatter = xyplot['x'].corr(xyplot['y'])
    #plt.figure(figsize=(3.5, 3), dpi=600)
    xyplot.plot.scatter(x='x', y='y', color='k', xlim=(0, 1), ylim=(0, 1), figsize=(3.5,3), s=1, marker='*')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=8)
    plt.ylabel("$SMC^*_{Tot}$ [m$^3$m$^{-3}$]", size=8)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.text(0.1, 0.5, 'R=' + '{:03.2f}'.format(r_scatter) +
             '\nRMSE=' + '{:03.2f}'.format(rmse_scatter), fontsize=8)# +
             #'\nRMSE=' + '{:03.2f}'.format(rmse_scatter), fontsize=8)
    plt.tick_params(labelsize=8)
    plt.title('True vs. estimated SMC', size=8)
    plt.axes().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(outpath + '1_scatterplot.png', dpi=600)
    plt.close()

    txtrep.write('------------------------------------------------------------------------\n\n')
    txtrep.write('Overall performance:\n')
    txtrep.write('R = ' + str(xyplot['x'].corr(xyplot['y'])) + '\n')
    txtrep.write('RMSE = ' + str(np.sqrt(np.nanmean(np.square(xyplot['x'].subtract(xyplot['y']))))))
    txtrep.write('ubRMSE = ' + str(np.sqrt(np.sum(np.square((xyplot['y'] - xyplot['y'].mean()) - (xyplot['x'] - xyplot['x'].mean()))) / len(xyplot['y']))))
    txtrep.close()


def mazia_vaildation_run(bsize=500, name='500m', fvect1=None, fvect2=None):

    # outpath
    outpath = '//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_' + name + '_reprocess_lt_05/w_GLDAS_station_validation/'

    # Calculate prediction standar deviations
    calcstd = False

    # use descending orbits
    desc = False

    calc_anomalies = False

    # initialise S1 SM retrieval
    # mlmodel = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ASCAT/gee/mlmodel0.p', 'rb'))
    mlmodel = "//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_" + name + "_reprocess_lt_05/w_GLDAS_RFmlmodelNoneSVR_2step.p"

    # initialse text report
    txtrep = open(outpath + '2_Mazia_report.txt', 'w')
    txtrep.write('Accuracy report for Soil Moisture validation based on ISMN stations\n\n')
    txtrep.write('Model used: ' + mlmodel + '\n')
    txtrep.write('------------------------------------------------------------------------\n\n')
    txtrep.write('Name, R, RMSE\n')

    xyplot = pd.DataFrame()
    cntr = 1

    # define mazia station locations
    m_stations = {'I1': [10.57978, 46.68706],
                  'I3': [10.58359, 46.68197],
                  'P1': [10.58295, 46.68586],
                  'P2': [10.58525, 46.68433],
                  'P3': [10.58562, 46.68511]}

    m_station_paths = '//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/01_Data/InSitu/MaziaValley_SWC_2015_16/'

    s1_ts_list = list()
    station_ts_list = list()
    station_name_list = list()
    gldas_ts_list = list()

    for vstation in m_stations:

        st_name = vstation
        st_coordinates = m_stations[vstation]

        try:

            # get in situ data
            full_path2015 = m_station_paths + st_name + '_YEAR_2015.csv'

            insitu2015 = pd.read_csv(
                full_path2015,
                header=0,
                skiprows=[0, 2, 3],
                index_col=0,
                parse_dates=True,
                sep=',')

            full_path2016 = m_station_paths + st_name + '_YEAR_2016.csv'

            insitu2016 = pd.read_csv(
                full_path2016,
                header=0,
                skiprows=[0, 2, 3],
                index_col=0,
                parse_dates=True,
                sep=',')

            insitu = insitu2015.append(insitu2016)

            # get station ts
            station_ts = pd.Series(insitu[['SWC_02_A_Avg', 'SWC_02_B_Avg', 'SWC_02_C_Avg']].mean(axis=1))
            plotpath = outpath + st_name + '.png'

            s1_ts, s1_ts_std, outliers = extract_time_series_gee(mlmodel,
                                                                 mlmodel,
                                                                 '/mnt/SAT4/DATA/S1_EODC/',
                                                                 outpath,
                                                                 st_coordinates[1],
                                                                 st_coordinates[0],
                                                                 name=st_name,
                                                                 footprint=bsize,
                                                                 calcstd=calcstd,
                                                                 desc=desc,
                                                                 target=station_ts,
                                                                 feature_vect1=fvect1,
                                                                 feature_vect2=fvect2)  # ,

            if s1_ts is None:
                continue

            if len(s1_ts) < 5:
                continue

            gldas_ts = extr_GLDAS_ts_GEE(st_coordinates[1], st_coordinates[0], bufferSize=150,
                                         yearlist=[2015, 2016])
            gldas_ts = gldas_ts / 100.

            start = np.array([s1_ts.index[0], station_ts.index[0]]).max()
            end = np.array([s1_ts.index[-1], station_ts.index[-1]]).min()
            if start > end:
                continue
            station_ts = station_ts[start:end]
            s1_ts = s1_ts[start:end]
            gldas_ts = gldas_ts[start:end]
            if calcstd == True:
                s1_ts_std = s1_ts_std[start:end]
            if len(s1_ts) < 1:
                continue

            s1_ts_res = s1_ts.resample('D').mean()
            station_ts_res = station_ts.resample('D').mean()

            if calc_anomalies == True:
                from pytesmo.time_series import anomaly as pyan

                s1_clim = pyan.calc_climatology(s1_ts_res.interpolate())
                station_clim = pyan.calc_climatology(station_ts_res.interpolate())

                s1_ts = pyan.calc_anomaly(s1_ts, climatology=s1_clim)
                s1_ts_res = pyan.calc_anomaly(s1_ts_res, climatology=s1_clim)
                station_ts = pyan.calc_anomaly(station_ts, climatology=station_clim)
                station_ts_res = pyan.calc_anomaly(station_ts_res, climatology=station_clim)

            # calculate error metrics
            ts_bias = s1_ts_res.subtract(station_ts_res).mean()

            tobemerged = [s1_ts_res.dropna(), station_ts_res.dropna()]
            s1_and_station = pd.concat(tobemerged, axis=1, join='inner')
            ts_bias = s1_and_station[0].subtract(s1_and_station[1]).median()

            xytmp = pd.concat({'y': s1_and_station[0] - ts_bias, 'x': s1_and_station[1]}, join='inner',
                              axis=1)
            if cntr == 1:
                xyplot = xytmp
            else:
                xyplot = pd.concat([xyplot, xytmp], axis=0)

            cntr = cntr + 1

            ts_cor = s1_and_station[0].corr(s1_and_station[1])
            ts_rmse = np.sqrt(np.nanmean(np.square(s1_and_station[0].subtract(s1_and_station[1]))))
            ts_ubrmse = np.sqrt(np.sum(np.square((s1_and_station[0] - s1_and_station[0].mean()) - (
                    s1_and_station[1] - s1_and_station[1].mean()))) / len(
                s1_and_station[0]))
            print('R: ' + str(ts_cor))
            print('RMSE: ' + str(ts_rmse))
            print('Bias: ' + str(ts_bias))
            txtrep.write(st_name + ', ' + str(ts_cor) + ', ' + str(ts_rmse) + '\n')

            s1_ts_list.append(s1_ts)
            station_ts_list.append(station_ts)
            station_name_list.append(st_name)
            gldas_ts_list.append(gldas_ts)
            # plot
            fig, ax1 = plt.subplots(figsize=(7.16, 1.4), dpi=300)
            line1, = ax1.plot(s1_ts.index, s1_ts, color='b', linestyle='-', marker='+', label='Sentinel-1',
                              linewidth=0.2)
            line2, = ax1.plot(station_ts.index, station_ts, label='In-Situ', linewidth=0.4)
            if np.any(outliers) and outliers is not None:
                line6, = ax1.plot(s1_ts.index[outliers], s1_ts.iloc[outliers], color='r', linestyle='', marker='o')
            if calcstd == True:
                line4, = ax1.plot(s1_ts.index, s1_ts - np.sqrt(s1_ts_std), color='k', linestyle='--', linewidth=0.2)
                line5, = ax1.plot(s1_ts.index, s1_ts + np.sqrt(s1_ts_std), color='k', linestyle='--', linewidth=0.2)
            line6, = ax1.plot(gldas_ts.index, gldas_ts, color='g', linestyle='--', label='GLDAS', linewidth=0.2)

            ax1.set_ylabel('Soil Moisture [m3m-3]', size=8)
            smc_max = np.max([s1_ts.max(), station_ts.max()])
            if smc_max <= 0.5:
                smc_max = 0.5
            ax1.set_ylim((0, smc_max))
            ax1.text(0.85, 0.4, 'R=' + '{:03.2f}'.format(ts_cor) +
                     # '\nRMSE=' + '{:03.2f}'.format(ts_rmse) +
                     '\nBias=' + '{:03.2f}'.format(ts_bias) +
                     '\nubRMSE=' + '{:03.2f}'.format(ts_ubrmse), transform=ax1.transAxes, fontsize=8)
            plt.title(st_name, fontsize=8)
            plt.tight_layout()
            plt.savefig(plotpath, dpi=300)
            plt.close()
        except:
            print('No data for: ' + st_name)

    pickle.dump((s1_ts_list, station_ts_list, gldas_ts_list, station_name_list),
                open(
                    'C:/Users/FGreifeneder/OneDrive - Scientific Network South Tyrol/1_THESIS/pub3/images_submission2/w_GLDAS_validation_tss_mazia' + name + '.p',
                    'wb'))
    urmse_scatter = np.sqrt(
        np.sum(np.square((xyplot['y'] - xyplot['y'].mean()) - (xyplot['x'] - xyplot['x'].mean()))) / len(xyplot['y']))
    rmse_scatter = np.sqrt(np.nanmean(np.square(xyplot['x'].subtract(xyplot['y']))))
    r_scatter = xyplot['x'].corr(xyplot['y'])
    # plt.figure(figsize=(3.5, 3), dpi=600)
    xyplot.plot.scatter(x='x', y='y', color='k', xlim=(0, 1), ylim=(0, 1), figsize=(3.5, 3), s=1, marker='.')
    plt.xlim(0, 0.7)
    plt.ylim(0, 0.7)
    plt.xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=8)
    plt.ylabel("$SMC^*_{Tot}$ [m$^3$m$^{-3}$]", size=8)
    plt.plot([0, 0.7], [0, 0.7], 'k--')
    plt.text(0.1, 0.5, 'R=' + '{:03.2f}'.format(r_scatter) +
             '\nRMSE=' + '{:03.2f}'.format(rmse_scatter), fontsize=8)  # +
    # '\nRMSE=' + '{:03.2f}'.format(rmse_scatter), fontsize=8)
    plt.tick_params(labelsize=8)
    plt.title('True vs. estimated SMC', size=8)
    plt.axes().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(outpath + '1_Mazia_scatterplot.png', dpi=600)
    plt.close()

    txtrep.write('------------------------------------------------------------------------\n\n')
    txtrep.write('Overall performance:\n')
    txtrep.write('R = ' + str(xyplot['x'].corr(xyplot['y'])) + '\n')
    txtrep.write('RMSE = ' + str(np.sqrt(np.nanmean(np.square(xyplot['x'].subtract(xyplot['y']))))))
    txtrep.write('ubRMSE = ' + str(np.sqrt(
        np.sum(np.square((xyplot['y'] - xyplot['y'].mean()) - (xyplot['x'] - xyplot['x'].mean()))) / len(xyplot['y']))))

    txtrep.close()


if __name__ == '__main__':

    i_name = ['5km','1km', '500m', '250m', '100m', '50m','10m']
    i_res = [5000,1000, 500, 250, 100, 50,10]
    # -- BEST FEATURES NO GLDAS --
    i_feat1 = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        [3, 10, 11, 16],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    ]
    i_feat2 = [
        [0, 1, 2, 3, 4, 5, 6, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33,
         34],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
         33, 34],
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33,
         34],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 34],
        [2, 3, 4, 5, 9, 13, 16, 17, 20, 22, 23, 24, 25, 26, 27, 32, 33, 34],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
         30, 31, 32, 33, 34,35],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
         30, 31, 32, 33, 34]]

    i_feat1_desc = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        [3, 10, 11, 16],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    ]
    i_feat2_desc = [
        [0, 1, 2, 3, 4, 5, 6, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33,
         34],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
         33, 34],
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33,
         34],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 34],
        [2, 3, 4, 5, 9, 13, 16, 17, 20, 22, 23, 24, 25, 26, 27, 32, 33, 34],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
         30, 31, 32, 33, 34,35],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
         30, 31, 32, 33, 34]]
    # -- BEST FEATURES WITH GLDAS --
    # i_feat1 = [
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    #     [2, 3, 7, 8, 10, 11, 12, 13, 14, 16, 17],
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # ]
    # i_feat2 = [[0, 2, 3, 4, 5, 7, 9, 10, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 29, 32, 33, 34, 35],
    #            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    #             28, 29, 31, 32, 33, 34, 35],
    #            [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35],
    #            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 33, 34, 35],
    #            [2, 4, 5, 6, 18, 20, 22, 23, 24, 25, 26, 31, 32, 34, 35],
    #            [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
    #            [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 16, 17, 20, 21, 22, 24, 25, 26, 27, 28, 34, 35]]

    for i in range(5,6):
        print(i_name[i])
        ismn_validation_run(bsize=i_res[i], name=i_name[i], fvect1=i_feat1[i],
                            fvect2=i_feat2[i], fvect1desc = i_feat1_desc[i], fvect2desc = i_feat2_desc[i])

    # for i in range(5,6):
    #     mazia_vaildation_run(bsize=i_res[i], name=i_name[i], fvect1=i_feat1[i],
    #                          fvect2=i_feat2[i])





