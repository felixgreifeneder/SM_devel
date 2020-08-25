import ee
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pytesmo.time_series.anomaly import calc_climatology

ee.Initialize()

def getEVIanom(lon,lat):

    # load datasetes
    modis = ee.ImageCollection("MODIS/006/MOD13Q1")
    roi = ee.Geometry.Polygon([[[11.51092529296875, 46.626963598343174],
                                     [11.429901123046875, 46.47207397310214],
                                     [11.605682373046875, 46.41529659311316],
                                     [11.734771728515625, 46.442746381846845],
                                     [11.786956787109375, 46.55996197966659]]])

    poi = ee.Geometry.Point([lon, lat]).buffer(250)

    # mask modis
    modis = modis.map(lambda image: image.updateMask(image.select('SummaryQA').lt(2))) \
                 .filterBounds(roi)

    # get time-series
    def getSeries(image):

        out_feature = image.reduceRegion(ee.Reducer.mean(), poi, 250)

        return ee.Feature(None, {'result': out_feature})

    # get the EVI time-series
    evi_series = modis.map(getSeries).getInfo()

    EVI = np.array([x['properties']['result']['EVI'] for x in evi_series['features']], dtype=np.float)

    ge_dates = np.array([datetime.strptime(x['id'], '%Y_%m_%d') for x in evi_series['features']])

    valid = np.where(np.isfinite(EVI))

    # cut out invalid values
    EVI = EVI[valid] / 10000
    ge_dates = ge_dates[valid]

    # insert into pandas series
    EVI = pd.Series(EVI, index=ge_dates)

    # calculate climatology
    EVI_filled = EVI.reindex(index=pd.date_range(start=datetime(year=2000,month=1,day=1),end=datetime(year=2017,month=12,day=31)), copy=True) \
                    .interpolate()
    evi_clim = calc_climatology(EVI_filled, moving_avg_orig=30, moving_avg_clim=0)

    evi_clim_dates = [evi_clim.values[x.dayofyear-1] for x in EVI.index]
    evi_clim_dates = pd.Series(evi_clim_dates, index=EVI.index)
    evi_clim_dates_f = [evi_clim.values[x.dayofyear - 1] for x in EVI_filled.index]
    evi_clim_dates_f = pd.Series(evi_clim_dates_f, index=EVI_filled.index)

    evi_anomalies = evi_clim_dates - EVI

    return (EVI, evi_clim_dates, evi_anomalies, evi_clim_dates_f, evi_clim)


coords_table = ee.FeatureCollection("users/felixgreifeneder/coordinates_yield_laimburg")
# transform and filter table with points
coords_table = coords_table.map(lambda feature: feature.transform())
coords_table = coords_table.toList(100).getInfo()

coord_list = [x['geometry']['coordinates'] for x in coords_table]
names_list = [x['properties']['Dataset']+str(x['properties']['Ort_Code']) for x in coords_table]

for i in range(len(coord_list)):

    if names_list[i] != 'FQ32':
        continue

    EVI, evi_clim_dates, evi_anomalies, evi_clim_dates_f, clim = getEVIanom(coord_list[i][0],coord_list[i][1])

    # subset
    #evi_anomalies = evi_anomalies['2017-01-01':'2017-12-31']

    # create plot
    fig, ax = plt.subplots(figsize=(6.5,2.7))
    #line1, = ax.plot(evi_anomalies.index, evi_anomalies, color='b', linestyle='-', marker='+', label=names_list[i], linewidth=0.2)
    line1, = ax.plot(EVI.index, EVI, color='b', linestyle='-', label=names_list[i], linewidth=0.5)
    line2, = ax.plot(evi_clim_dates_f, color='r', linestyle='--', linewidth=0.5, label=names_list[i] + ' Climatology')
    #x0 = [evi_anomalies.index[0], evi_anomalies.index[-1]]
    #y0 = [0, 0]
    #line4, = ax.plot(x0, y0, color='k', linestyle='--', linewidth=0.2)

    plt.setp(ax.get_xticklabels(), fontsize=6)

    ax.set_ylabel('MODIS EVI')
    plt.legend(handles=[line1, line2])

    plt.show()
    plt.savefig('/mnt/SAT/Workspaces/GrF/02_Documents/Drought_Insurance/'+names_list[i]+'.png', dpi=300)
    plt.close()

