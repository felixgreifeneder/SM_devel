__author__ = 'felix'

if __name__ == '__main__':
    from sgrt_devels.compile_tset import Trainingset
    import os
    import time


    working_path = '/mnt/CEPH_PROJECTS/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/cali_model/'

    global_sm = Trainingset(working_path,
                            uselc=True,
                            track=None,
                            desc=True,
                            footprint=50)

    global_sm.create_trainingset()
    # global_sm.create_learning_curve(
    #     '/mnt/CEPH_PROJECTS/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/newrun_ft_descending_v2/no_GLDAS_GBRmlmodel_1step.p',
    #     feature_vect=[2, 19, 22, 23, 33, 34, 35, 36, 43, 44, 46, 47, 48, 50])
    global_sm.create_test_set()

    # global_sm.parameter_selection_1step(ml='RF', frange=53)
    # global_sm.parameter_selection_1step(ml='GBR', frange=53)

    # asc + desc updated temporal features - fixed reference period
    start_time = time.time()
    # global_sm.create_temp_acc_plot('no_GLDAS', feature_vect=[2, 19, 22, 23, 33, 34, 35, 36, 43, 44, 46, 47, 48, 50])
    # # below - based on GBR feature selection
    # global_sm.train_GBR_LOGO_1step(feature_vect=[2, 19, 22, 23, 33, 34, 35, 36, 43, 44, 46, 47, 48, 50],
    #                              prefix='no_GLDAS')
    # below - based on RF feature selection
    # global_sm.train_GBR_LOGO_1step(feature_vect=[2,  6, 13, 19, 22, 23, 29, 30, 33, 34, 38, 39, 41, 42, 43, 44, 47, 48, 49, 50],
    #                                prefix='no_GLDAS_', ml='ADA')
    # global_sm.train_SVR_LOGO_1step(feature_vect=[2, 19, 22, 23, 33, 34, 35, 36, 43, 44, 46, 47, 48, 50],
    #                              prefix='no_GLDAS_')
    # below: feature selection fo os only application
    #global_sm.train_GBR_LOGO_1step(feature_vect=[2,  4,  5,  6,  7,  9, 11, 19, 22, 23, 27, 29, 31, 32, 33, 34, 35,
    #                                             36, 37, 40, 41, 45, 46, 47, 48, 49, 50], prefix='no_GLDAS')
    global_sm.create_temp_acc_plot('no_GLDAS_', feature_vect=[2,  4,  5,  6,  7,  9, 11, 19, 22, 23, 27, 29, 31, 32, 33, 34, 35,
                                                 36, 37, 40, 41, 45, 46, 47, 48, 49, 50])
    print('--- %s seconds ---' % (time.time() - start_time))

