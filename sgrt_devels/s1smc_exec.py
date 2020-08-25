__author__ = 'usergre'

# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend([ '/home/usergre/winpycharm/Python_SGRT_devel',
#                  '/home/usergre/winpycharm/sgrt'])

if __name__ == '__main__':
    from sgrt_devels.compile_tset import Trainingset
    from sgrt_devels.compile_tset import test2step
    from sgrt_devels.compile_tset import test1step
    import os
    from matplotlib import pyplot as plt
    from sgrt_devels.compile_tset import Estimationset
    import pickle
    from sgrt_devels.derive_smc import extract_time_series_gee

    i_name = ['5km', '1km', '500m', '250m', '100m', '50m', '10m']
    i_res = [5000, 1000, 500, 250, 100, 50, 10]
    i_feat1 = [range(48),
               range(48),
               range(48),
               range(48),
               range(48),
               range(48),
               #[13,15,18,19,22,23,26,33,35,36,38,40,41,43,44,45],
               range(48)
               ]
    i_feat2 = [range(65),
               range(65),
               range(65),
               range(65),
               range(65),
               #range(65),
               [4,5,8,26,27,28,30,31,32,33,35,37,39,41,43,45,47,49,51,53,54],
               #[2,3,4,6,7,27,31,35,37,39,40,41,42,43,45,46,47,52,53,54,55,56,57,58,59,60],
               range(65)
               ]

    i_feat3 = [range(23),
               range(23),
               range(23),
               range(23),
               range(23),
               range(23),
               range(23)]

    for i in range(5, 6):
        print(i)
        projpath = '/mnt/CEPH_PROJECTS/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/newrunFeb2020_descending/'
        if not os.path.exists(projpath):
            os.mkdir(projpath)
        if not os.path.exists(projpath + 'station_ts'):
            os.mkdir(projpath + 'station_ts')
        if not os.path.exists(projpath + 'station_validation'):
            os.mkdir(projpath + 'station_validation')
        results = Trainingset(projpath,
                              uselc=False,
                              subgrid='EU',
                              ssm_target='ISMN',
                              sig0_source='GEE',
                              track=None,
                              desc=False,
                              footprint=i_res[i],
                              feature_vect1=i_feat1[i],
                              feature_vect2=i_feat2[i],
                              prefix='no_GLDAS_',
                              extr_data=False,
                              indtestset=False)

        results.RF_loo_1step()
        # results.RF_loo()
        # results.optimize_rf_1step()
        # results.optimize_SVR_1step()
        # results.optimize_SVR()
        # results.SVR_loo_1step()
        # results.SVR_loo()
        # results.optimize_rf()
        # results.optimize_rf_g0est_1step()
        # results.RF_loo_g0_1step()
