import pytesmo.io.ismn.interface as ismn_interface
from sgrt_devels.extr_TS import extr_SIG0_LIA_ts_GEE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle

# outpath
outpath = '/mnt/SAT/Workspaces/GrF/02_Documents/1_THESIS/pub3/plots/valid_stations/'

# use descending orbits
desc = False

# initialise available ISMN data
ismn = ismn_interface.ISMN_Interface('/mnt/SAT4/DATA/S1_EODC/ISMN/')

# get the stations the qualified for training
#used_stations = np.load('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ISMN/gee_global_ASC_DESC/ValidStaions.npy')
invalid_col = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ISMN/tmp/invalid_col.p', 'rb'))

# iterate through all stations
#for vstation in used_stations:
for i in range(len(invalid_col['label'])):

    #ntwk, st_name = [x.strip() for x in vstation.split(',')]
    ntwk = invalid_col['ntwkname'][i]
    st_name = invalid_col['stname'][i]

    if os.path.exists(outpath + st_name + '.png'):
        continue

    station = ismn.get_station(st_name, ntwk)
    station_vars = station.get_variables()

    if 'soil moisture' not in station_vars:
        continue

    station_depths = station.get_depths('soil moisture')

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

    # get station ts
    station_ts = station_ts.data['soil moisture']

    # add station ts to data frame
    all_tts = pd.DataFrame({'insitu': station_ts}, index=station_ts.index)

    try:
        # get the s1 time series
        # tts = extr_SIG0_LIA_ts_GEE(station.longitude, station.latitude, maskwinter=False, lcmask=False, tempfilter=False, masksnow=False,
        #                            dual_pol=False)

        # get the temporally filtered time series
        tts_filtered = extr_SIG0_LIA_ts_GEE(station.longitude, station.latitude, maskwinter=False, lcmask=False, tempfilter=True, masksnow=False,
                                   dual_pol=False, desc=False, bufferSize=20)
    except:
        print('Time out')
        continue

    # for ckey in tts.keys():
    #     series_name = ckey
    #     tmpdf = pd.DataFrame({series_name: tts[ckey][1]['sig0']}, index=tts[ckey][0])
    #     # add to complete data frame
    #     all_tts = all_tts.join(tmpdf, how='outer')

    start = 0
    end = 0

    for ckey in tts_filtered.keys():
        series_name = ckey
        tmpdf = pd.DataFrame({series_name: tts_filtered[ckey][1]['sig0']}, index=tts_filtered[ckey][0])
        tmpstart = np.min(tts_filtered[ckey][0])
        tmpend = np.max(tts_filtered[ckey][0])
        # add to complete data frame
        all_tts = all_tts.join(tmpdf, how='outer')

        if start == 0:
            start = tmpstart
        else:
            if tmpstart < start:
                start = tmpstart
        if end == 0:
            end = tmpend
        else:
            if tmpend > end:
                end = tmpend



    # interpolate nans
    all_tts.interpolate(inplace=True)

    # plt.figure(figsize=(4, 2))
    all_tts[start:end].plot(secondary_y=['insitu'], figsize=(5,3), linewidth=0.5)
    plt.savefig(outpath + st_name + '.png')
    plt.close()
