__author__ = 'usergre'

from sgrt_devels.derive_smc import extract_time_series_gee
import pickle
import pandas as pd
import matplotlib.pyplot as plt

stations = {'Vipas2000': [4831141.21, 1516685.53],
            'Vimes2000': [4830444.22, 1517335.96],
            'Vimef2000': [4830996.06, 1516222.11],
            'Vimef1500': [4836014.08, 1512440.54],
            'Domes1500': [4874224.56, 1469179.81],
            'Domef1500': [4874881.06, 1469495.06],
            #'Dopas2000': [4873472.41, 1463915.39],
            'Nemef1500': [4935219.71, 1531329.21],
            'Nemes1500': [4934388.83, 1531173.69],
            'Domef2000': [4889902.12, 1484534.78],
            'Domes2000': [4892088.35, 1481584.32],
            'Nepas2000': [4944063.47, 1503903.05],
            'Vimes1500': [4814092.66, 1512626.88],
            'Nemef2000': [4944025.70, 1503497.29],
            'Nemes2000': [4943880.04, 1503699.99]}

mlmodel = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ASCAT/gee_117/mlmodel117.p', 'rb'))

for i in stations:
    print(i)
    # extract_time_series_gee(mlmodel,
    #                     '/mnt/SAT4/DATA/S1_EODC/',
    #                     '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/monalisa/gee_allalps_pwise/',
    #                     stations[i][1],
    #                     stations[i][0],
    #                     grid='Equi7',
    #                     name=i)

    # get S1 time series
    s1_ts = extract_time_series_gee(mlmodel,
                                    '/mnt/SAT4/DATA/S1_EODC/',
                                    '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ASCAT/gee_117/Monalisa/',
                                    stations[i][1],
                                    stations[i][0],
                                    grid='Equi7',
                                    name=i,
                                    footprint=300)

    if s1_ts is None:
        continue

    # get in-situ data
    insitu_path = '/mnt/SAT/Workspaces/GrF/02_Documents/monalisa/insitu/' + i.lower() + '/csv-zip-content.csv'
    insitu = pd.read_csv(
        insitu_path,
        header=0,
        index_col=0,
        usecols=[3, 4],
        names=['Date', 'insitu'],
        squeeze=True,
        parse_dates=True,
        na_values=-9999,
        sep=';')

    # merge
    mgd = pd.concat([s1_ts.resample('D').mean(), insitu.resample('D').mean()], axis=1, join='inner').fillna(method='ffill')

    # plot
    plotpath = '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ASCAT/gee_117/Monalisa/' + i + '.png'
    plt.figure(figsize=(18, 6))
    mgd.plot(secondary_y=['insitu'])
    plt.savefig(plotpath)
    plt.close()