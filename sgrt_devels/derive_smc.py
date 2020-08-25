# this script include functions to derive both soil moisture time series and maps

from sgrt_devels.compile_tset import Estimationset
#import sgrt.common.grids.Equi7Grid as Equi7
import pickle

def extract_time_series(model_path, sgrt_root, out_path, lat, lon, grid, name=None):

    mlmodel = pickle.load(open(model_path, 'rb'))

    # initialise grid
    alpGrid = Equi7.Equi7Grid(10)

    # identify tile
    if grid is None:
        Equi7XY = alpGrid.lonlat2equi7xy(lon, lat)
    elif grid == 'Equi7':
        Equi7XY = ['EU', lon, lat]
    TileName = alpGrid.identfy_tile(Equi7XY[0], [Equi7XY[1], Equi7XY[2]])

    # initialise estimation set
    es = Estimationset(sgrt_root, [TileName[7:]],
                       sgrt_root+'Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                       sgrt_root+'Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                       out_path,
                       mlmodel,
                       subgrid='EU',
                       uselc=True)

    #es.ssm_ts_gee_alternative(Equi7XY[1], Equi7XY[2], 3, name=name)
    es.ssm_ts_gee_alternative(lon, lat, 117, name=name)


def extract_time_series_gee(mlmodel, mlmodel_avg, sgrt_root, out_path, lat, lon, grid=None, name=None, footprint=50, s1path=None, calcstd=False, desc=False, target=None,
                            feature_vect1=None, feature_vect2=None):

    #mlmodel = pickle.load(open(model_path, 'rb'))

    # initialise grid
    #alpGrid = Equi7.Equi7Grid(10)

    # identify tile
    if grid is None:
        #Equi7XY = alpGrid.lonlat2equi7xy(lon, lat)
        Equi7XY = ['EU', 0, 0]
        TileName='NoneNoneNoneNoneNoneNone'
    # elif grid == 'Equi7':
    #     Equi7XY = ['EU', lon, lat]
    #     tmp = alpGrid.equi7xy2lonlat('EU', lon, lat)
    #     lon = tmp[0]
    #     lat = tmp[1]
    #     TileName = alpGrid.identfy_tile(Equi7XY[0], [Equi7XY[1], Equi7XY[2]])

    # initialise estimation set
    es = Estimationset(sgrt_root, [TileName[7:]],
                       sgrt_root+'Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                       sgrt_root+'Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                       out_path,
                       mlmodel,
                       mlmodel_avg,
                       subgrid='EU',
                       uselc=True,
                       track=s1path)

    # return (es.ssm_ts_gee(lon, lat, Equi7XY[1], Equi7XY[2], footprint, name=name, plotts=False, calcstd=calcstd,
    #                            desc=desc))
    return(es.ssm_ts_gee2step(lon, lat, Equi7XY[1], Equi7XY[2], footprint, name=name, plotts=False, calcstd=calcstd, desc=desc,
                              feature_vect1=feature_vect1, feature_vect2 = feature_vect2))
    # return (es.ssm_ts_gee(lon, lat, Equi7XY[1], Equi7XY[2], footprint, name=name, plotts=False, calcstd=calcstd,
    #                      desc=desc))
    #return (es.ssm_ts_gee_with_target(lon, lat, Equi7XY[1], Equi7XY[2], footprint, target, name=name, plotts=False, calcstd=calcstd,
    #                      desc=desc))