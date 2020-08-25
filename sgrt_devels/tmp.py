if __name__ == '__main__':
    import pickle
    from sgrt_devels.compile_tset import tree_to_code
    from sgrt_devels.compile_tset import tree_to_code_GEE
    from sgrt_devels.compile_tset import treeToJson
    import sys

    mlmodel_tmp = pickle.load(open(
        "//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_250m/RFmlmodelNoneSVR_2step.p",
        'rb'))
    testtarget1, testfeatures1, testtarget2, testfeatures2 = \
        pickle.load(
            open("//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_250m/testset.p"))

    tree_to_code_GEE(mlmodel_tmp[0],
                 ['vvk1', 'vhk1', 'vvk2', 'vhk2', 'vvk3', 'vhk3', 'vvk4', 'vhk4', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6',
                  'b7', 'b10', 'b11', 'trees'],
                 '//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_250m/')

    # tree_to_code_GEE(mlmodel_tmp[2],
    #              ['dVV', 'dVH',
    #               'vvk1', 'vhk1', 'vvk2', 'vhk2', 'vvk3', 'vhk3', 'vvk4', 'vhk4',
    #               'dGLDAS_SM',
    #               'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'],
    #              '//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_250m/')

    # sys.path.append('//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_250m')
    # from decisiontree import tree as dtree
    #
    # est1 = dtree(testfeatures1[0, 0], testfeatures1[0, 1], testfeatures1[0, 2], testfeatures1[0, 3], testfeatures1[0, 4],
    #       testfeatures1[0, 5], testfeatures1[0, 6], testfeatures1[0, 7], testfeatures1[0, 8], testfeatures1[0, 9],
    #       testfeatures1[0, 10], testfeatures1[0, 11], testfeatures1[0, 12], testfeatures1[0, 13], testfeatures1[0, 14],
    #       testfeatures1[0, 15], testfeatures1[0, 16], testfeatures1[0, 17])
    # est2 = mlmodel_tmp[0].predict(testfeatures1[0, :].reshape(1, -1))
    # print(est1)
    # print(est2)