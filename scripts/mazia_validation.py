import pytesmo.io.ismn.interface as ismn_interface
from sgrt_devels.derive_smc import extract_time_series_gee
from sgrt_devels.extr_TS import extr_SIG0_LIA_ts_GEE
from sgrt_devels.extr_TS import extr_MODIS_MOD13Q1_ts_GEE
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

if __name__ == '__main__':

    s1path=117

    bsize=500
    name = '500m'

    # outpath
    outpath = '//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_' + name + '_reprocess/station_validation/'

    # Calculate prediction standar deviations
    calcstd = False

    # use descending orbits
    desc = False


    calc_anomalies = False

    # initialise S1 SM retrieval
    # mlmodel = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ASCAT/gee/mlmodel0.p', 'rb'))
    mlmodel = "//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_500m_reprocess/RFmlmodelNoneSVR_2step.p"

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

    for vstation in m_stations:

        st_name = vstation
        st_coordinates = m_stations[vstation]

        try:

            # get in situ data
            full_path2015 = m_station_paths + st_name + '_YEAR_2015.csv'

            insitu2015 = pd.read_csv(
                full_path2015,
                header=0,
                skiprows=[0,2,3],
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
                                                                 target=station_ts)  # ,

            if s1_ts is None:
                continue

            if len(s1_ts) < 5:
                continue

            start = np.array([s1_ts.index[0], station_ts.index[0]]).max()
            end = np.array([s1_ts.index[-1], station_ts.index[-1]]).min()
            if start > end:
                continue
            station_ts = station_ts[start:end]
            s1_ts = s1_ts[start:end]
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
            # plot
            fig, ax1 = plt.subplots(figsize=(7.16, 1.4), dpi=300)
            line1, = ax1.plot(s1_ts.index, s1_ts, color='b', linestyle='-', marker='+', label='Sentinel-1', linewidth=0.2)
            line2, = ax1.plot(station_ts.index, station_ts, label='In-Situ', linewidth=0.4)
            if np.any(outliers) and outliers is not None:
                line6, = ax1.plot(s1_ts.index[outliers], s1_ts.iloc[outliers], color='r', linestyle='', marker='o')
            if calcstd == True:
                line4, = ax1.plot(s1_ts.index, s1_ts - np.sqrt(s1_ts_std), color='k', linestyle='--', linewidth=0.2)
                line5, = ax1.plot(s1_ts.index, s1_ts + np.sqrt(s1_ts_std), color='k', linestyle='--', linewidth=0.2)

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

    pickle.dump((s1_ts_list, station_ts_list, station_name_list),
                    open('C:/Users/FGreifeneder/OneDrive - Scientific Network South Tyrol/1_THESIS/pub3/images_submission2/validation_tss_mazia' + name + '.p', 'wb'))
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
                 '\nubRMSE=' + '{:03.2f}'.format(urmse_scatter), fontsize=8)  # +
        # '\nRMSE=' + '{:03.2f}'.format(rmse_scatter), fontsize=8)
    plt.tick_params(labelsize=8)
    plt.tight_layout()
    plt.title('True vs. estimated SMC', size=8)
    plt.savefig(outpath + '1_Mazia_scatterplot.png', dpi=600)
    plt.close()

    plt.close()

    txtrep.write('------------------------------------------------------------------------\n\n')
    txtrep.write('Overall performance:\n')
    txtrep.write('R = ' + str(xyplot['x'].corr(xyplot['y'])) + '\n')
    txtrep.write('RMSE = ' + str(np.sqrt(np.nanmean(np.square(xyplot['x'].subtract(xyplot['y']))))))
    txtrep.write('ubRMSE = ' + str(np.sqrt(
        np.sum(np.square((xyplot['y'] - xyplot['y'].mean()) - (xyplot['x'] - xyplot['x'].mean()))) / len(xyplot['y']))))

    txtrep.close()






