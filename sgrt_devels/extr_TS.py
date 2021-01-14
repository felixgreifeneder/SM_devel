__author__ = 'felix'

import numpy as np
# import sgrt.common.grids.Equi7Grid as Equi7
# import sgrt.common.utils.SgrtTile as SgrtTile
from osgeo import gdal
from osgeo.gdalconst import *
from netCDF4 import Dataset, num2date
from datetime import datetime
import ee
import datetime as dt
import time
from sgrt_tools.access_google_drive import gdrive
import pandas as pd
import math

def multitemporalDespeckle(images, radius, units, opt_timeWindow=None):
    def mapMeanSpace(i):
        reducer = ee.Reducer.mean()
        kernel = ee.Kernel.square(radius, units)
        mean = i.reduceNeighborhood(reducer, kernel).rename(bandNamesMean)
        ratio = i.divide(mean).rename(bandNamesRatio)
        return (i.addBands(mean).addBands(ratio))

    if opt_timeWindow == None:
        timeWindow = dict(before=-3, after=3, units='month')
    else:
        timeWindow = opt_timeWindow

    bandNames = ee.Image(images.first()).bandNames()
    bandNamesMean = bandNames.map(lambda b: ee.String(b).cat('_mean'))
    bandNamesRatio = bandNames.map(lambda b: ee.String(b).cat('_ratio'))

    # compute spatial average for all images
    meanSpace = images.map(mapMeanSpace)

    # computes a multi-temporal despeckle function for a single image

    def multitemporalDespeckleSingle(image):
        t = image.date()
        fro = t.advance(ee.Number(timeWindow['before']), timeWindow['units'])
        to = t.advance(ee.Number(timeWindow['after']), timeWindow['units'])

        meanSpace2 = ee.ImageCollection(meanSpace).select(bandNamesRatio).filterDate(fro, to) \
            .filter(ee.Filter.eq('relativeOrbitNumber_start', image.get('relativeOrbitNumber_start')))

        b = image.select(bandNamesMean)

        return (b.multiply(meanSpace2.sum()).divide(meanSpace2.count()).rename(bandNames)).set('system:time_start',
                                                                                               image.get(
                                                                                                   'system:time_start'))

    return meanSpace.map(multitemporalDespeckleSingle)


def slope_correction(collection, elevation, model, buffer=0):
    '''This function applies the slope correction on a collection of Sentinel-1 data

       :param collection: ee.Collection of Sentinel-1
       :param elevation: ee.Image of DEM
       :param model: model to be applied (volume/surface)
       :param buffer: buffer in meters for layover/shadow amsk

        :returns: ee.Image
    '''

    def _volumetric_model_SCF(theta_iRad, alpha_rRad):
        '''Code for calculation of volumetric model SCF

        :param theta_iRad: ee.Image of incidence angle in radians
        :param alpha_rRad: ee.Image of slope steepness in range

        :returns: ee.Image
        '''

        # create a 90 degree image in radians
        ninetyRad = ee.Image.constant(90).multiply(np.pi / 180)

        # model
        nominator = (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).tan()
        denominator = (ninetyRad.subtract(theta_iRad)).tan()
        return nominator.divide(denominator)

    def _surface_model_SCF(theta_iRad, alpha_rRad, alpha_azRad):
        '''Code for calculation of direct model SCF

        :param theta_iRad: ee.Image of incidence angle in radians
        :param alpha_rRad: ee.Image of slope steepness in range
        :param alpha_azRad: ee.Image of slope steepness in azimuth

        :returns: ee.Image
        '''

        # create a 90 degree image in radians
        ninetyRad = ee.Image.constant(90).multiply(np.pi / 180)

        # model
        nominator = (ninetyRad.subtract(theta_iRad)).cos()
        denominator = (alpha_azRad.cos()
                       .multiply((ninetyRad.subtract(theta_iRad).add(alpha_rRad)).cos()))

        return nominator.divide(denominator)

    def _erode(image, distance):
        '''Buffer function for raster

        :param image: ee.Image that shoudl be buffered
        :param distance: distance of buffer in meters

        :returns: ee.Image
        '''

        d = (image.Not().unmask(1)
             .fastDistanceTransform(30).sqrt()
             .multiply(ee.Image.pixelArea().sqrt()))

        return image.updateMask(d.gt(distance))

    def _masking(alpha_rRad, theta_iRad, buffer):
        '''Masking of layover and shadow


        :param alpha_rRad: ee.Image of slope steepness in range
        :param theta_iRad: ee.Image of incidence angle in radians
        :param buffer: buffer in meters

        :returns: ee.Image
        '''
        # layover, where slope > radar viewing angle
        layover = alpha_rRad.lt(theta_iRad).rename('layover')

        # shadow
        ninetyRad = ee.Image.constant(90).multiply(np.pi / 180)
        shadow = alpha_rRad.gt(ee.Image.constant(-1).multiply(ninetyRad.subtract(theta_iRad))).rename('shadow')

        # add buffer to layover and shadow
        if buffer > 0:
            layover = _erode(layover, buffer)
            shadow = _erode(shadow, buffer)

            # combine layover and shadow
        no_data_mask = layover.And(shadow).rename('no_data_mask')

        return layover.addBands(shadow).addBands(no_data_mask)

    def _correct(image):
        '''This function applies the slope correction and adds layover and shadow masks

        '''

        # get the image geometry and projection
        geom = image.geometry()
        proj = image.select(1).projection()

        # calculate the look direction
        heading = (ee.Terrain.aspect(image.select('angle'))
                   .reduceRegion(ee.Reducer.mean(), geom, 1000)
                   .get('aspect'))

        # Sigma0 to Power of input image
        sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0))

        # the numbering follows the article chapters
        # 2.1.1 Radar geometry
        theta_iRad = image.select('angle').multiply(np.pi / 180)
        phi_iRad = ee.Image.constant(heading).multiply(np.pi / 180)

        # 2.1.2 Terrain geometry
        alpha_sRad = ee.Terrain.slope(elevation).select('slope').multiply(np.pi / 180).setDefaultProjection(proj).clip(
            geom)
        phi_sRad = ee.Terrain.aspect(elevation).select('aspect').multiply(np.pi / 180).setDefaultProjection(proj).clip(
            geom)

        # we get the height, for export
        height = elevation.setDefaultProjection(proj).clip(geom)

        # 2.1.3 Model geometry
        # reduce to 3 angle
        phi_rRad = phi_iRad.subtract(phi_sRad)

        # slope steepness in range (eq. 2)
        alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

        # slope steepness in azimuth (eq 3)
        alpha_azRad = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

        # local incidence angle (eq. 4)
        theta_liaRad = (alpha_azRad.cos().multiply((theta_iRad.subtract(alpha_rRad)).cos())).acos()
        theta_liaDeg = theta_liaRad.multiply(180 / np.pi)

        # 2.2
        # Gamma_nought
        gamma0 = sigma0Pow.divide(theta_iRad.cos())
        gamma0dB = ee.Image.constant(10).multiply(gamma0.log10()).select(['VV', 'VH'], ['VV_gamma0', 'VH_gamma0'])
        ratio_gamma = (gamma0dB.select('VV_gamma0')
                       .subtract(gamma0dB.select('VH_gamma0'))
                       .rename('ratio_gamma0'))

        if model == 'volume':
            scf = _volumetric_model_SCF(theta_iRad, alpha_rRad)

        if model == 'surface':
            scf = _surface_model_SCF(theta_iRad, alpha_rRad, alpha_azRad)

        # apply model for Gamm0_f
        gamma0_flat = gamma0.divide(scf)
        gamma0_flatDB = (ee.Image.constant(10)
                         .multiply(gamma0_flat.log10())
                         .select(['VV', 'VH'], ['VV_gamma0flat', 'VH_gamma0flat'])
                         )

        masks = _masking(alpha_rRad, theta_iRad, buffer)

        # calculate the ratio for RGB vis
        ratio_flat = (gamma0_flatDB.select('VV_gamma0flat')
                      .subtract(gamma0_flatDB.select('VH_gamma0flat'))
                      .rename('ratio_gamma0flat')
                      )

        if model == 'surface':
            gamma0_flatDB = gamma0_flatDB.rename(['VV_gamma0surf', 'VH_gamma0surf'])

        if model == 'volume':
            gamma0_flatDB = gamma0_flatDB.rename(['VV_gamma0vol', 'VH_gamma0vol'])

        return (image.rename(['VV_sigma0', 'VH_sigma0', 'incAngle'])
                .addBands(gamma0dB)
                .addBands(ratio_gamma)
                .addBands(gamma0_flatDB)
                .addBands(ratio_flat)
                .addBands(alpha_rRad.rename('alpha_rRad'))
                .addBands(alpha_azRad.rename('alpha_azRad'))
                .addBands(phi_sRad.rename('aspect'))
                .addBands(alpha_sRad.rename('slope'))
                .addBands(theta_iRad.rename('theta_iRad'))
                .addBands(theta_liaRad.rename('theta_liaRad'))
                .addBands(masks)
                .addBands(height.rename('elevation'))
                )

        # run and return correction

    return collection.map(_correct, opt_dropNulls=True)


def _s1_track_ts(lon,
                 lat,
                 bufferSize,
                 filtered_collection,
                 track_nr,
                 dual_pol,
                 varmask,
                 returnLIA,
                 masksnow,
                 tempfilter,
                 datesonly=False,
                 radcor=False):
    def getAzimuth(f):
        coords = ee.Array(f.geometry().coordinates().get(0)).transpose()
        crdLons = ee.List(coords.toList().get(0))
        crdLats = ee.List(coords.toList().get(1))
        minLon = crdLons.sort().get(0)
        maxLon = crdLons.sort().get(-1)
        minLat = crdLats.sort().get(0)
        maxLat = crdLats.sort().get(-1)
        azimuth = ee.Number(crdLons.get(crdLats.indexOf(minLat))).subtract(minLon).atan2(ee.Number(crdLats
                                                                                                   .get(
            crdLons.indexOf(minLon))).subtract(minLat)).multiply(180.0 / math.pi).add(180.0)
        return ee.Feature(ee.Geometry.LineString([crdLons.get(crdLats.indexOf(maxLat)), maxLat,
                                                  minLon, crdLats.get(crdLons.indexOf(minLon))]),
                          {'azimuth': azimuth}).copyProperties(f)

    def getLIA(imCollection):
        # function to calculate the local incidence angle, based on azimuth angle and srtm
        srtm = ee.Image("USGS/SRTMGL1_003")
        srtm_slope = ee.Terrain.slope(srtm)
        srtm_aspect = ee.Terrain.aspect(srtm)

        # tmpImg = ee.Image(imCollection.first())
        tmpImg = imCollection

        inc = tmpImg.select('angle')
        azimuth = getAzimuth(tmpImg).get('azimuth')
        srtm_slope_proj = srtm_slope.multiply(
            ee.Image.constant(azimuth).subtract(9.0).subtract(srtm_aspect).multiply(math.pi / 180).cos())
        lia = inc.subtract(ee.Image.constant(90).subtract(ee.Image.constant(90).subtract(srtm_slope_proj))).abs()

        return tmpImg.addBands(ee.Image(lia.select(['angle'], ['lia'])))
        # s = srtm_slope.multiply(ee.Image.constant(277).subtract(srtm_aspect).multiply(math.pi / 180).cos())
        # lia = inc.subtract(ee.Image.constant(90).subtract(ee.Image.constant(90).subtract(s))).abs()

        # return ee.Image(lia.select(['angle'], ['lia']).reproject(srtm.projection()))

    def miscMask(image):
        # masking of low and high db values as well as areas affected by geometry distortion
        tmp = ee.Image(image)

        # mask pixels
        vv = tmp.select('VV')
        if dual_pol == True:
            vh = tmp.select('VH')
            maskvh = vh.gte(-25).bitwiseAnd(vh.lt(0))  # was -25 and 0
        lia = tmp.select('lia')
        maskvv = vv.gte(-25).bitwiseAnd(vv.lt(0))
        masklia1 = lia.gt(20)  # angle 10
        masklia2 = lia.lt(45)  # angle 50
        masklia = masklia1.bitwiseAnd(masklia2)

        if dual_pol == True:
            mask = maskvv.bitwiseAnd(maskvh)
        else:
            mask = maskvv
        mask = mask.bitwiseAnd(masklia)
        # mask = mask.bitwiseAnd(maskslope)
        tmp = tmp.updateMask(mask)

        return (tmp)

    def toln(image):
        tmp = ee.Image(image)

        # Convert to linear
        vv = ee.Image(10).pow(tmp.select('VV').divide(10))
        if dual_pol == True:
            vh = ee.Image(10).pow(tmp.select('VH').divide(10))

        # Convert to ln
        out = vv.log()
        if dual_pol == True:
            out = out.addBands(vh.log())
            out = out.select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            out = out.select(['constant'], ['VV'])

        return out.set('system:time_start', tmp.get('system:time_start'))

    def applyvarmask(image):
        tmp = ee.Image(image)
        tmp = tmp.updateMask(varmask)

        return (tmp)

    def tolin(image):
        tmp = ee.Image(image)

        # Convert to linear
        vh = ee.Image(10).pow(tmp.select('VH').divide(10))

        # Output
        out = vh.select(['constant'], ['VH'])

        return out.set('system:time_start', tmp.get('system:time_start'))

    def tolin_dual(image):
        tmp = ee.Image(image)
        if dual_pol == True:
            lin = ee.Image(10).pow(tmp.divide(10))  # .select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            lin = ee.Image(10).pow(tmp.divide(10))  # .select(['constant'], ['VV'])

        return lin.set('system:time_start', tmp.get('system:time_start'))

    def todb(image):
        tmp = ee.Image(image)

        return ee.Image(10).multiply(tmp.log10()).set('system:time_start', tmp.get('system:time_start'))

    def applysnowmask(image):
        tmp = ee.Image(image)
        sdiff = tmp.select('VH').subtract(snowref)
        wetsnowmap = sdiff.lte(-2.6)  # .focal_mode(100, 'square', 'meters', 3)

        return (tmp.updateMask(wetsnowmap.eq(0)))

    def createAvg(image):
        # average pixels within the time-series foot print
        gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
        # tmp = ee.Image(image).resample()
        tmp = ee.Image(image)

        # Conver to linear before averaging
        tmp = tmp.addBands(ee.Image(10).pow(tmp.select('VV_sigma0').divide(10)))
        tmp = tmp.addBands(ee.Image(10).pow(tmp.select('VV_gamma0').divide(10)))
        tmp = tmp.addBands(ee.Image(10).pow(tmp.select('VV_gamma0vol').divide(10)))
        tmp = tmp.addBands(ee.Image(10).pow(tmp.select('VV_gamma0surf').divide(10)))
        if dual_pol == True:
            tmp = tmp.addBands(ee.Image(10).pow(tmp.select('VH_sigma0').divide(10)))
            tmp = tmp.addBands(ee.Image(10).pow(tmp.select('VH_gamma0').divide(10)))
            tmp = tmp.addBands(ee.Image(10).pow(tmp.select('VH_gamma0vol').divide(10)))
            tmp = tmp.addBands(ee.Image(10).pow(tmp.select('VH_gamma0surf').divide(10)))
            tmp = tmp.select(['constant', 'constant_1', 'constant_2', 'constant_3', 'constant_4',
                              'constant_5', 'constant_6', 'constant_7', 'incAngle', 'theta_liaRad'],
                             ['VV_sigma0', 'VV_gamma0', 'VV_gamma0vol', 'VV_gamma0surf',
                              'VH_sigma0', 'VH_gamma0', 'VH_gamma0vol', 'VH_gamma0surf', 'angle', 'lia'])
        else:
            tmp = tmp.select(['constant', 'constant_1', 'constant_2', 'constant_3',
                              'incAngle', 'theta_liaRad'],
                             ['VV_sigma0', 'VV_gamma0', 'VV_gamma0vol', 'VV_gamma0surf',
                              'angle', 'lia'])

        reduced_img_data = tmp.reduceRegion(ee.Reducer.mean(), gee_roi, 10)
        totcount = ee.Image(1).reduceRegion(ee.Reducer.count(), gee_roi, 10)
        pcount = tmp.reduceRegion(ee.Reducer.count(), gee_roi, 10)
        return ee.Feature(None, {'result': reduced_img_data, 'count': pcount, 'totcount': totcount})

    def cliptoroi(image):
        # average pixels within the time-series foot print
        gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
        return image.clip(gee_roi)

    def apply_no_data_mask(image):
        return image.updateMask(image.select('no_data_mask'))


    #  filter for track
    gee_s1_track_fltd = filtered_collection.filterMetadata('relativeOrbitNumber_start', 'equals', int(track_nr))

    if datesonly == True:
        return gee_s1_track_fltd.size().getInfo()


     #gee_s1_track_fltd = gee_s1_track_fltd.map(cliptoroi)
    # paths to dem
    dem = 'USGS/SRTMGL1_003'

    # list of models
    model = 'volume'
    gee_s1_track_fltd_vol = slope_correction(gee_s1_track_fltd,
                                         ee.Image(dem),
                                         model)

    model = 'surface'
    gee_s1_track_fltd_surf = slope_correction(gee_s1_track_fltd,
                                              ee.Image(dem),
                                              model)

    # combine results based on volume and surface backscattering
    gee_s1_track_fltd = gee_s1_track_fltd_surf.combine(gee_s1_track_fltd_vol, overwrite=True)

    gee_s1_track_fltd = gee_s1_track_fltd.map(apply_no_data_mask)

    if varmask == True:
        # mask pixels with low temporal variability
        # compute temporal statistics
        gee_s1_ln = gee_s1_track_fltd.map(toln)
        k2vv = ee.Image(gee_s1_ln.select('VV').reduce(ee.Reducer.stdDev()))
        if dual_pol == True:
            k2vh = ee.Image(gee_s1_ln.select('VH').reduce(ee.Reducer.stdDev()))
            varmask = k2vv.gt(0.4).And(k2vh.gt(0.4))
        else:
            varmask = k2vv.gt(0.4)
        gee_s1_track_fltd = gee_s1_track_fltd.map(applyvarmask)

    if tempfilter == True:
        # apply a temporal speckle filter
        radius = 3
        units = 'pixels'
        gee_s1_linear = gee_s1_track_fltd.map(tolin_dual)
        gee_s1_dspckld_vv = multitemporalDespeckle(gee_s1_linear.select('VV'), radius, units,
                                                   {'before': -12, 'after': 12, 'units': 'month'})
        gee_s1_dspckld_vv = gee_s1_dspckld_vv.map(todb).select(['constant'], ['VV'])
        if dual_pol == True:
            gee_s1_dspckld_vh = multitemporalDespeckle(gee_s1_linear.select('VH'), radius, units,
                                                       {'before': -12, 'after': 12, 'units': 'month'})
            gee_s1_dspckld_vh = gee_s1_dspckld_vh.map(todb).select(['constant'], ['VH'])
        if dual_pol == False:
            gee_s1_track_fltd = gee_s1_dspckld_vv.combine(gee_s1_track_fltd.select('angle')) \
                .combine(gee_s1_track_fltd.select('lia'))
        else:
            gee_s1_track_fltd = gee_s1_dspckld_vv.combine(gee_s1_dspckld_vh) \
                .combine(gee_s1_track_fltd.select('angle')) \
                .combine(gee_s1_track_fltd.select('lia'))

    # if masksnow == True:
    #     # apply wet snow mask
    #     gee_s1_lin = gee_s1_track_fltd.select('VH').map(tolin)
    #     snowref = ee.Image(10).multiply(gee_s1_lin.reduce(ee.Reducer.intervalMean(5, 100)).log10())
    #     gee_s1_track_fltd = gee_s1_track_fltd.map(applysnowmask)

    # create average for buffer area - i.e. compute time-series
    gee_s1_mapped = gee_s1_track_fltd.map(createAvg)
    tmp = gee_s1_mapped.getInfo()
    # get vv
    vv_sig0 = 10 * np.log10(
        np.array([x['properties']['result']['VV_sigma0'] for x in tmp['features']], dtype=np.float))
    vv_g0 = 10 * np.log10(
        np.array([x['properties']['result']['VV_gamma0'] for x in tmp['features']], dtype=np.float)
    )
    vv_g0vol = 10 * np.log10(
        np.array([x['properties']['result']['VV_gamma0vol'] for x in tmp['features']], dtype=np.float)
    )
    vv_g0surf = 10 * np.log10(
        np.array([x['properties']['result']['VV_gamma0surf'] for x in tmp['features']], dtype=np.float)
    )

    ge_dates = np.array([dt.datetime.strptime(x['id'][17:32], '%Y%m%dT%H%M%S') for x in tmp['features']])

    if datesonly == True:
        return ge_dates

    if dual_pol == True:
        # get vh
        vh_sig0 = 10 * np.log10(
            np.array([x['properties']['result']['VH_sigma0'] for x in tmp['features']], dtype=np.float))
        vh_g0 = 10 * np.log10(
            np.array([x['properties']['result']['VH_gamma0'] for x in tmp['features']], dtype=np.float)
        )
        vh_g0vol = 10 * np.log10(
            np.array([x['properties']['result']['VH_gamma0vol'] for x in tmp['features']], dtype=np.float)
        )
        vh_g0surf = 10 * np.log10(
            np.array([x['properties']['result']['VH_gamma0surf'] for x in tmp['features']], dtype=np.float)
        )
        if masksnow == True:
            vh_sig0_lin = np.array([x['properties']['result']['VH_sigma0'] for x in tmp['features']], dtype=np.float)
            snowref = 10 * np.log10(np.mean(vh_sig0_lin[vh_sig0_lin > np.percentile(vh_sig0_lin, 5)]))
            snowmask = np.where(vh_sig0 - snowref > -2.6)
            vh_sig0 = vh_sig0[snowmask]
            vv_sig0 = vv_sig0[snowmask]
            ge_dates = ge_dates[snowmask]

    if returnLIA == True:
        # get lia
        lia = np.array([x['properties']['result']['lia'] for x in tmp['features']], dtype=np.float)
    else:
        # get angle
        lia = np.array([x['properties']['result']['angle'] for x in tmp['features']], dtype=np.float)

    # get val_count - i.e. compute the fraction of valid pixels within the footprint
    val_count = np.array(
        [np.float(x['properties']['count']['VV_sigma0']) / np.float(x['properties']['totcount']['constant']) for x in
         tmp['features']], dtype=np.float)

    if masksnow == True:
        val_count = val_count[snowmask]

    if bufferSize <= 100:
        valid = np.where(val_count > 0.01)
    else:
        valid = np.where(val_count > 0.1)
    vv_sig0 = vv_sig0[valid]
    vv_g0 = vv_g0[valid]
    vv_g0vol = vv_g0vol[valid]
    vv_g0surf = vv_g0surf[valid]
    if dual_pol == True:
        vh_sig0 = vh_sig0[valid]
        vh_g0 = vh_g0[valid]
        vh_g0vol = vh_g0vol[valid]
        vh_g0surf = vh_g0surf[valid]
    lia = lia[valid]
    ge_dates = ge_dates[valid]

    if dual_pol == True:
        return (pd.DataFrame({'vv_sig0': vv_sig0, 'vh_sig0': vh_sig0, 'lia': lia,
                              'vv_g0': vv_g0, 'vv_g0vol': vv_g0vol, 'vv_g0surf': vv_g0surf,
                              'vh_g0': vh_g0, 'vh_g0vol': vh_g0vol, 'vh_g0surf': vh_g0surf},
                             index=ge_dates))
    else:
        return (pd.DataFrame({'vv_sig0': vv_sig0, 'lia': lia,
                              'vv_g0': vv_g0, 'vv_g0vol': vv_g0vol, 'vv_g0surf': vv_g0surf},
                             index=ge_dates))


def extr_USGS_LC(lon, lat, bufferSize=20):
    ee.Initialize()

    # construc feature collection
    pnt_list = list()
    for i in range(len(lon)):
        pnt_list.append(ee.Geometry.Point(lon[i], lat[i]).buffer(bufferSize))
    pnt_collection = ee.FeatureCollection(pnt_list)

    lc_image = ee.Image(ee.ImageCollection("USGS/NLCD").toList(100).get(-1))
    # roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
    # lc = lc_image.reduceRegion(ee.Reducer.mode(), roi).getInfo()
    lc = lc_image.reduceRegions(pnt_collection, ee.Reducer.mode(), scale=10).getInfo()
    lc_array = np.array([x['properties']['landcover'] for x in lc['features']])
    return (lc_array)


def extr_s2_avgs(lon, lat, bufferSize=20):
    ee.Initialize()

    # construc feature collection
    pnt_list = list()
    for i in range(len(lon)):
        pnt_list.append(ee.Geometry.Point(lon[i], lat[i]).buffer(bufferSize))
    pnt_collection = ee.FeatureCollection(pnt_list)

    s2_collection = ee.ImageCollection("COPERNICUS/S2").map(lambda image: image.updateMask(image.select('QA60').eq(0)))
    s2_reduced = s2_collection.reduce(ee.Reducer.mean())

    s2_sampled = s2_reduced.reduceRegions(pnt_collection, ee.Reducer.mean(), scale=10).getInfo()
    return (s2_sampled)


def extr_gldas_avrgs(lon, lat, varname, bufferSize=100):
    ee.Initialize()

    # construc feature collection
    pnt_list = list()
    for i in range(len(lon)):
        pnt_list.append(ee.Geometry.Point(lon[i], lat[i]).buffer(bufferSize))
    pnt_collection = ee.FeatureCollection(pnt_list)

    # for iyear in range(2000,2018):
    # print(iyear)
    gldas_collection = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select(varname).filterDate('2014-01-01',
                                                                                                  '2017-12-31')
    gldas_reduced = gldas_collection.reduce(ee.Reducer.mean())

    gldas_sampled = gldas_reduced.reduceRegions(pnt_collection,
                                                ee.Reducer.mean(),
                                                scale=50,
                                                tileScale=2).getInfo()

    if 'gldas_out' not in locals():
        gldas_out = np.array([x['properties']['mean'] for x in gldas_sampled['features']])
    else:
        gldas_out_tmp = np.array([x['properties']['mean'] for x in gldas_sampled['features']])
        gldas_out = (gldas_out + gldas_out_tmp) / 2
    return (gldas_out)


def extr_MODIS_MOD13Q1_ts_GEE(lon, lat, bufferSize=100, datefilter=None):
    ee.Reset()
    ee.Initialize()

    def createAvg(image):
        # mask image
        immask = image.select('SummaryQA').eq(ee.Image(0))
        image = image.updateMask(immask)

        reduced_img_data = image.reduceRegion(ee.Reducer.median(),
                                              ee.Geometry.Point(lon, lat).buffer(bufferSize),
                                              50)
        return ee.Feature(None, {'result': reduced_img_data})

    # load collection
    gee_l8_collection = ee.ImageCollection('MODIS/006/MOD13Q1')

    # filter collection
    gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
    gee_l8_fltd = gee_l8_collection.filterBounds(gee_roi)

    if datefilter is not None:
        gee_l8_fltd = gee_l8_fltd.filterDate(datefilter[0], datefilter[1])

    # extract time series
    gee_l8_mpd = gee_l8_fltd.map(createAvg)
    tmp = gee_l8_mpd.getInfo()

    EVI = np.array([x['properties']['result']['EVI'] for x in tmp['features']], dtype=np.float)

    ge_dates = np.array([datetime.strptime(x['id'], '%Y_%m_%d') for x in tmp['features']])

    valid = np.where(np.isfinite(EVI))
    if len(valid[0]) == 0:
        success = 0
        return (None, success)

    # cut out invalid values
    EVI = EVI[valid]
    ge_dates = ge_dates[valid]
    success = 1

    return (pd.Series(EVI, index=ge_dates, name='EVI'), success)


def extr_GLDAS_ts_GEE(lon, lat, varname='SoilMoi0_10cm_inst', bufferSize=20, yearlist=None):
    ee.Reset()
    ee.Initialize()

    def createAvg(image):
        gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)

        reduced_img_data = image.reduceRegion(ee.Reducer.median(), gee_roi, 30)
        return ee.Feature(None, {'result': reduced_img_data})

    if yearlist == None:
        # yearlist = range(1987,2018)
        yearlist = range(2011, 2018)

    SM_list = list()

    for iyear in yearlist:
        # ee.Reset()
        # ee.Initialize()
        print(iyear)
        # load collection
        if iyear < 2000:
            GLDAS_collection = ee.ImageCollection('NASA/GLDAS/V20/NOAH/G025/T3H').select(varname)
        else:
            GLDAS_collection = ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H').select(varname)
        GLDAS_collection = GLDAS_collection.filterDate(str(iyear) + '-01-01', str(iyear) + '-12-31')
        GLDAS_collection = GLDAS_collection.filter(ee.Filter.calendarRange(16, 18, 'hour'))

        # clip
        roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
        GLDAS_collection = GLDAS_collection.map(lambda image: image.clip(roi))

        # extract time series
        time_series = GLDAS_collection.map(createAvg)
        tmp = time_series.getInfo()

        SM = np.array([x['properties']['result'][varname] for x in tmp['features']], dtype=np.float)

        ge_dates = np.array([datetime.strptime(x['id'], 'A%Y%m%d_%H%M') for x in tmp['features']])

        valid = np.where(np.isfinite(SM))

        # cut out invalid values
        SM = SM[valid]
        ge_dates = ge_dates[valid]

        SM_series = pd.Series(SM, index=ge_dates, copy=True, name='GLDAS')

        SM_list.append(SM_series)

    return (pd.concat(SM_list))


def extr_L8_ts_GEE(lon, lat, bufferSize=20):
    ee.Reset()
    ee.Initialize()

    def createAvg(image):
        gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)

        # aerosols
        image = image.updateMask(image.select('sr_aerosol').eq(2).bitwiseOr(
            image.select('sr_aerosol').eq(32).bitwiseOr(
                image.select('sr_aerosol').eq(96).bitwiseOr(
                    image.select('sr_aerosol').eq(160).bitwiseOr(
                        image.select('sr_aerosol').eq(66).bitwiseOr(
                            image.select('sr_aerosol').eq(130)
                        ))))))

        # clouds
        def getQABits(image, start, end, newName):
            pattern = 0
            for i in range(start, end + 1):
                pattern = pattern + int(math.pow(2, i))

            return image.select([0], [newName]).bitwiseAnd(pattern).rightShift(start)

        def cloud_shadows(image):
            QA = image.select('pixel_qa')
            return getQABits(QA, 3, 3, 'Cloud_shadows').eq(0)

        def clouds(image):
            QA = image.select('pixel_qa')
            return getQABits(QA, 5, 5, 'Cloud').eq(0)

        image = image.updateMask(cloud_shadows(image))
        image = image.updateMask(clouds(image))

        # # radiometric saturation
        # image = image.updateMask(image.select('radsat_qa').eq(2))

        reduced_img_data = image.reduceRegion(ee.Reducer.mean(), gee_roi, 30)
        return ee.Feature(None, {'result': reduced_img_data})

    def setresample(image):
        image = image.resample()
        return (image)

    def mask_tree_cover(image):
        tree_cover_image = ee.ImageCollection("GLCF/GLS_TCC").filterBounds(gee_roi).filter(
            ee.Filter.eq('year', 2010)).mosaic()
        treemask = tree_cover_image.select('tree_canopy_cover').clip(gee_roi).lte(20)

        # load lc
        glbcvr = ee.Image("ESA/GLOBCOVER_L4_200901_200912_V2_3").select('landcover')
        # mask water
        watermask = glbcvr.neq(210)
        return image.updateMask(treemask.And(watermask))

    # load collection
    gee_l8_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')  # .map(setresample)

    # filter collection
    gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
    gee_l8_fltd = gee_l8_collection.filterBounds(gee_roi)

    # extract time series
    #gee_l8_fltd = gee_l8_fltd.map(mask_tree_cover)
    gee_l8_mpd = gee_l8_fltd.map(createAvg)
    tmp = gee_l8_mpd.getInfo()

    b1 = np.array([x['properties']['result']['B1'] for x in tmp['features']], dtype=np.float)
    b2 = np.array([x['properties']['result']['B2'] for x in tmp['features']], dtype=np.float)
    b3 = np.array([x['properties']['result']['B3'] for x in tmp['features']], dtype=np.float)
    b4 = np.array([x['properties']['result']['B4'] for x in tmp['features']], dtype=np.float)
    b5 = np.array([x['properties']['result']['B5'] for x in tmp['features']], dtype=np.float)
    b6 = np.array([x['properties']['result']['B6'] for x in tmp['features']], dtype=np.float)
    b7 = np.array([x['properties']['result']['B7'] for x in tmp['features']], dtype=np.float)
    b10 = np.array([x['properties']['result']['B7'] for x in tmp['features']], dtype=np.float)
    b11 = np.array([x['properties']['result']['B7'] for x in tmp['features']], dtype=np.float)

    ge_dates = np.array([datetime.strptime(x['id'][12::], '%Y%m%d') for x in tmp['features']])

    valid = np.where(np.isfinite(b2))

    # cut out invalid values
    b1 = b1[valid]
    b2 = b2[valid]
    b3 = b3[valid]
    b4 = b4[valid]
    b5 = b5[valid]
    b6 = b6[valid]
    b7 = b7[valid]
    b10 = b10[valid]
    b11 = b11[valid]
    ge_dates = ge_dates[valid]

    if b1.size == 0:
        return None
    else:
        return pd.DataFrame({'B1': b1,
                             'B2': b2,
                             'B3': b3,
                             'B4': b4,
                             'B5': b5,
                             'B6': b6,
                             'B7': b7,
                             'B10': b10,
                             'B11': b11}, index=ge_dates)


def extr_L8_median(lon, lat, bufferSize=20, startDate='2014-10-01', endDate='2019-01-01'):
    ee.Initialize()

    def mask_bands(image):
        # aerosols
        image = image.updateMask(image.select('sr_aerosol').eq(2).bitwiseOr(
            image.select('sr_aerosol').eq(32).bitwiseOr(
                image.select('sr_aerosol').eq(96).bitwiseOr(
                    image.select('sr_aerosol').eq(160).bitwiseOr(
                        image.select('sr_aerosol').eq(66).bitwiseOr(
                            image.select('sr_aerosol').eq(130)
                        ))))))

        # clouds
        def getQABits(image, start, end, newName):
            pattern = 0
            for i in range(start, end + 1):
                pattern = pattern + int(math.pow(2, i))

            return image.select([0], [newName]).bitwiseAnd(pattern).rightShift(start)

        def cloud_shadows(image):
            QA = image.select('pixel_qa')
            return getQABits(QA, 3, 3, 'Cloud_shadows').eq(0)

        def clouds(image):
            QA = image.select('pixel_qa')
            return getQABits(QA, 5, 5, 'Cloud').eq(0)

        image = image.updateMask(cloud_shadows(image))
        image = image.updateMask(clouds(image))

        # # radiometric saturation
        # image = image.updateMask(image.select('radsat_qa').eq(2))

        return image

    poi = ee.Geometry.Point(lon, lat).buffer(bufferSize)

    l8_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')

    # filter dates
    l8_collection = l8_collection.filterDate(startDate, endDate)

    # filter bounds
    l8_collection = l8_collection.filterBounds(poi)

    # mask images
    l8_collection = l8_collection.map(mask_bands)

    # reduce
    l8_median = l8_collection.reduce(ee.Reducer.median())

    # sample
    avg_value = l8_median.reduceRegion(ee.Reducer.median(), poi, 30, tileScale=4).getInfo()

    return avg_value


def extr_SIG0_LIA_ts_GEE(lon, lat,
                         bufferSize=20,
                         maskwinter=False,
                         lcmask=False,
                         globcover_mask=False,
                         trackflt=None,
                         masksnow=False,
                         varmask=False,
                         ssmcor=None,
                         dual_pol=True,
                         desc=False,
                         tempfilter=False,
                         returnLIA=False,
                         datesonly=False,
                         datefilter=None,
                         S1B=False,
                         treemask=False,
                         radcor=False):

    ee.Initialize()

    def mask_lc(image):
        tmp = ee.Image(image)

        # load land cover info
        corine = ee.Image('users/felixgreifeneder/corine')

        # create lc mask
        valLClist = [10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29]

        lcmask = corine.eq(valLClist[0]).bitwiseOr(corine.eq(valLClist[1])) \
            .bitwiseOr(corine.eq(valLClist[2])) \
            .bitwiseOr(corine.eq(valLClist[3])) \
            .bitwiseOr(corine.eq(valLClist[4])) \
            .bitwiseOr(corine.eq(valLClist[5])) \
            .bitwiseOr(corine.eq(valLClist[6])) \
            .bitwiseOr(corine.eq(valLClist[7])) \
            .bitwiseOr(corine.eq(valLClist[8])) \
            .bitwiseOr(corine.eq(valLClist[9])) \
            .bitwiseOr(corine.eq(valLClist[10])) \
            .bitwiseOr(corine.eq(valLClist[11]))

        tmp = tmp.updateMask(lcmask)

        return tmp

    def mask_lc_globcover(image):
        tmp = ee.Image(image)

        # load lc
        glbcvr = ee.Image("ESA/GLOBCOVER_L4_200901_200912_V2_3").select('landcover')

        valLClist = [11, 14, 20, 30, 120, 140, 150]

        lcmask = glbcvr.eq(valLClist[0]) \
            .bitwiseOr(glbcvr.eq(valLClist[1])) \
            .bitwiseOr(glbcvr.eq(valLClist[2])) \
            .bitwiseOr(glbcvr.eq(valLClist[3])) \
            .bitwiseOr(glbcvr.eq(valLClist[4])) \
            .bitwiseOr(glbcvr.eq(valLClist[5])) \
            .bitwiseOr(glbcvr.eq(valLClist[6]))

        tmp = tmp.updateMask(lcmask)

        return tmp

    def mask_tree_cover(image):

        copernicus_collection = ee.ImageCollection('COPERNICUS/Landcover/100m/Proba-V/Global')
        copernicus_image = ee.Image(copernicus_collection.toList(1000).get(0))
        treemask = copernicus_image.select('tree-coverfraction').clip(gee_roi).lte(20)

        # load lc
        glbcvr = ee.Image("ESA/GLOBCOVER_L4_200901_200912_V2_3").select('landcover')
        # mask water
        watermask = copernicus_image.select('discrete_classification').neq(80)
        return image.updateMask(treemask.And(watermask))

    def addRefSM(image):
        tmp = ee.Image(image)
        img_date = ee.Date(tmp.get('system:time_start'))
        RefSMtmp = RefSMcollection.filterDate(img_date.format('Y-M-d'))
        current_ssm = ee.ImageCollection(RefSMtmp).toList(10).get(0)

        out_image = tmp.addBands(ee.Image(current_ssm))

        return (out_image)

    def s1_simplyfy_date(image):
        return (image.set('system:time_start', ee.Date(ee.Date(image.get('system:time_start')).format('Y-M-d'))))

    def applyCorrelationMask(image):
        mask = ssm_vv_cor.select('correlation').gt(0.1)
        return (image.updateMask(mask))

    import timeit

    tic = timeit.default_timer()

    # load S1 data
    gee_s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD')  # .map(setresample)

    # filter collection
    gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)

    gee_s1_filtered = gee_s1_collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filterBounds(gee_roi) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation',
                                       'VV'))

    if S1B == False:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.eq('platform_number', 'A'))

    if datefilter is not None:
        gee_s1_filtered = gee_s1_filtered.filterDate(datefilter[0], datefilter[1])

    if desc == False:
        # Select only acquisition from ascending tracks
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    # else:
    #    gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

    if dual_pol == True:
        # select only acquisitions with VV AND VH
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

    if maskwinter == True:
        # Mask winter based on the DOY
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.dayOfYear(121, 304))

    if trackflt is not None:
        # Select only data from a specific S1 track
        if isinstance(trackflt, list):
            gee_s1_filtered = gee_s1_filtered.filter(
                ee.Filter.inList(ee.List(trackflt), 'relativeOrbitNumber_start'))
        else:
            gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.eq('relativeOrbitNumber_start', trackflt))

    if lcmask == True:
        # Mask pixels based on Corine land-cover
        gee_s1_filtered = gee_s1_filtered.map(mask_lc)

    if globcover_mask == True:
        # Mask pixels based on the Globcover land-cover classification
        gee_s1_filtered = gee_s1_filtered.map(mask_lc_globcover)

    if treemask == True:
        gee_s1_filtered = gee_s1_filtered.map(mask_tree_cover)

    if ssmcor is not None:
        # Mask pixels with a low correlation toe coarse resolution soil moisture
        # Mostly relevant if aggregating over a larger area
        RefSMlist = list()
        ssmcor = ssmcor.resample('D').mean().dropna()
        ssmcor = ssmcor.astype(np.float)
        for i in range(len(ssmcor)):
            ssm_img = ee.Image(ssmcor[i]).clip(gee_roi).float()
            ssm_img = ssm_img.set('system:time_start', ssmcor.index[i])
            RefSMlist.append(ssm_img)
        RefSMcollection = ee.ImageCollection(RefSMlist)

        # prepare the join
        s1_joined = gee_s1_filtered.map(s1_simplyfy_date)
        join_filter = ee.Filter.equals(leftField='system:time_start', rightField='system:time_start')
        simple_join = ee.Join.simple()
        s1_joined = simple_join.apply(s1_joined, RefSMcollection, join_filter)

        # create ssm reference SM, image collection
        s1_plus_RefSM = ee.ImageCollection(s1_joined.map(addRefSM, True))
        ssm_vv_cor = s1_plus_RefSM.select(['VV', 'constant']).reduce(ee.Reducer.pearsonsCorrelation())
        gee_s1_filtered = gee_s1_filtered.map(applyCorrelationMask)

    # get the track numbers
    tmp = gee_s1_filtered.getInfo()
    track_series = np.array([x['properties']['relativeOrbitNumber_start'] for x in tmp['features']])
    dir_series = np.array([x['properties']['orbitProperties_pass'] for x in tmp['features']])
    available_tracks, uidx = np.unique(track_series, return_index=True)
    available_directions = dir_series[uidx]
    # create dict with orbit directions
    available_directions = pd.Series(np.where(available_directions == 'ASCENDING', 1, 0), index=available_tracks)

    print('Extracting data from ' + str(len(available_tracks)) + ' Sentinel-1 tracks...')
    print(available_tracks)

    out_dict = {}
    lgths = list()
    for track_nr in available_tracks:
        if datesonly == False:
            out_dict[str(int(track_nr))] = _s1_track_ts(lon, lat, bufferSize,
                                                        gee_s1_filtered,
                                                        track_nr,
                                                        dual_pol,
                                                        varmask,
                                                        returnLIA,
                                                        masksnow,
                                                        tempfilter,
                                                        datesonly,
                                                        radcor)
        else:
            tmp_dates = _s1_track_ts(lon, lat, bufferSize,
                                     gee_s1_filtered,
                                     track_nr,
                                     dual_pol,
                                     varmask,
                                     returnLIA,
                                     masksnow,
                                     tempfilter,
                                     datesonly,
                                     radcor)
            lgths.append(tmp_dates)

    toc = timeit.default_timer()

    if datesonly == True:
        return np.array(lgths)

    print('Time-series extraction finished in ' + "{:10.2f}".format(toc - tic) + 'seconds')

    return out_dict, available_directions


def get_s1_dates(minlon, minlat, maxlon, maxlat, tracknr=None, dualpol=True):
    ee.Initialize()

    # load S1 data
    gee_s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD')

    # construct roi
    roi = ee.Geometry.Polygon([[minlon, minlat], [minlon, maxlat],
                               [maxlon, maxlat], [maxlon, minlat],
                               [minlon, minlat]])
    # roi = get_south_tyrol_roi()

    # ASCENDING acquisitions
    gee_s1_filtered = gee_s1_collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filterBounds(roi) \
        .filter(ee.Filter.eq('platform_number', 'A')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))  # \
    # .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))

    if dualpol == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

    if tracknr is not None:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.eq('relativeOrbitNumber_start', tracknr))

    # create a list of availalbel dates
    tmp = gee_s1_filtered.getInfo()
    tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
    dates = np.array([dt.date(year=int(x[17:21]), month=int(x[21:23]), day=int(x[23:25])) for x in tmp_ids])

    return dates
