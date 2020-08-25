import numpy as np
import ee
from sgrt_devels.extr_TS import multitemporalDespeckle
from sgrt_devels.extr_TS import GEtodisk
import datetime as dt

ee.Initialize()

def clipCollection(image):
    return(ee.Image(image).clip(aoi))

aoi = ee.Geometry.Polygon([[[7.06,45.68],[7.07,45.4],[7.58,45.4],[7.57,45.69],[7.06,45.68]]])

s1_collection = ee.ImageCollection("COPERNICUS/S1_GRD")
srtm = ee.Image("USGS/SRTMGL1_003")

s1_fltd = s1_collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
                        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                        .filter(ee.Filter.eq('relativeOrbitNumber_start',88.0)) \
                        .filterBounds(aoi)

s1_fltd = s1_fltd.map(lambda image: ee.Image(image).clip(aoi))

## calculate reference image
s1_fltd_lin = s1_fltd.map(lambda image: ee.Image(10).pow(ee.Image(image).divide(10)))
snowref = ee.Image(10).multiply(s1_fltd_lin.reduce(ee.Reducer.intervalMean(5,95)).log10())

## temporal filtering
radius = 7
units = 'pixels'
s1_denoised_vh = multitemporalDespeckle(s1_fltd.select('VH'), radius, units, {'before': -12, 'after': 12, 'units': 'month'})
#s1_denoised_vh_list = s1_denoised_vh.toList(1000)
#s1_ll = s1_denoised_vh_list.length().getInfo()

# create a list of availalbel dates
tmp = s1_fltd.getInfo()
tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
dates = np.unique(np.array([dt.date(year=int(x[17:21]), month=int(x[21:23]), day=int(x[23:25])) for x in tmp_ids]))

## iterate through all images
for i in dates:
    try:
#for i in range(s1_ll):
    #current_img = ee.Image(s1_denoised_vh_list.get(i))
        current_img = s1_denoised_vh.filterDate(i.strftime('%Y-%m-%d'),(i+dt.timedelta(days=1)).strftime('%Y-%m-%d')).mosaic()

        #scene_name = ee.Image(current_img).get('system:index').getInfo()
        scene_name = 'S1_SNOW_' + i.strftime('%Y%m%d')
        diff_img = current_img.subtract(snowref.select('VH_mean'))
        wetsnowmap = diff_img.lte(-2.6).focal_mode(100, 'square', 'meters', 2)
        # wetsnowmap_out = wetsnowmap.updateMask(wetsnowmap.eq(1))
        wetsnowmap_out = wetsnowmap

        GEtodisk(wetsnowmap_out, scene_name, '/mnt/SAT/Workspaces/GrF/Processing/ECOPOTENTIAL/', 10, aoi)
    except:
        print('ERROR when exporting')


