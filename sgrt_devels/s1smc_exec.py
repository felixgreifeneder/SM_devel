__author__ = 'felix'

if __name__ == '__main__':
    from sgrt_devels.compile_tset import Trainingset
    import os

    working_path = '/mnt/CEPH_PROJECTS/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/newrun_ft_descending/'

    global_sm = Trainingset(working_path,
                            uselc=True,
                            track=None,
                            desc=True,
                            footprint=50)

    global_sm.create_trainingset()
    # global_sm.create_learning_curve(
    #     '/mnt/CEPH_PROJECTS/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/newrun_ft_descending/no_GLDAS_GBRmlmodel_1step.p',
    #     feature_vect=[2, 6, 27, 30, 31, 36, 39, 41, 43, 44, 47, 50, 53, 54, 55, 56, 58])
    global_sm.create_test_set()

    #global_sm.parameter_selection_1step(ml='GBR', frange=60)
    #global_sm.parameter_selection_1step(ml='GBR', frange=64)


    # for asc + desc:
    global_sm.train_GBR_LOGO_1step(feature_vect=[2, 6, 27, 30, 31, 36, 39, 41, 43, 44, 47, 50, 53, 54, 55, 56, 58],
                                   prefix='no_GLDAS_')
    global_sm.train_GBR_LOGO_1step(
        feature_vect=[6, 11, 26, 27, 30, 31, 36, 37, 42, 44, 45, 46, 52, 54, 56, 57, 58, 61, 62, 63],
        prefix='w_GLDAS_')
    # for asc
    # global_sm.train_GBR_LOGO_1step(feature_vect=[0,  2,  6,  9, 11, 13, 26, 27, 28, 30, 31, 36, 40, 41, 42, 43, 44,
    #                                              46, 50, 51, 52, 53, 54, 55, 56, 57, 58],
    #                               prefix='no_GLDAS_')
    # #
    # global_sm.train_GBR_LOGO_1step(feature_vect=[ 2, 19, 26, 27, 30, 31, 37, 46, 48, 54, 56, 57, 60, 61, 62],
    #                                prefix='w_GLDAS_')
    #
    # global_sm.train_GBR_LOGO_2step(feature_vect1=[0, 3, 4, 10, 11, 12, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 26,
    #                                               27, 28, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43],
    #                                feature_vect2=[2, 6, 7, 11, 16, 27, 30, 35, 37, 39, 41, 42, 43, 45, 46, 47, 53,
    #                                               54, 55, 56, 58],
    #                                prefix='no_GLDAS_')

    # global_sm.train_SVR_LOGO_1step(feature_vect=[0, 4, 9, 13, 17, 19, 26, 27,
    #                                              28, 30, 31, 40, 41, 42, 43,
    #                                              44, 46, 51, 52, 53, 54, 55, 56, 57, 58],
    #                                prefix='no_GLDAS_')

    # global_sm.train_SVR_LOGO_2step(feature_vect1=[0, 3, 4, 10, 11, 12, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 26,
    #                                               27, 28, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43],
    #                                feature_vect2=[2, 6, 7, 11, 16, 27, 30, 35, 37, 39, 41, 42, 43, 45, 46, 47, 53,
    #                                               54, 55, 56, 58],
    #                                prefix='no_GLDAS_')
