__author__ = 'usergre'

# import sgrt.common.recursive_filesearch as rsearch
import h5py
import os
import glob
import sklearn.preprocessing
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# from sgrt.common.grids.Equi7Grid import Equi7Tile
# from sgrt.common.grids.Equi7Grid import Equi7Grid
# from sgrt.common.utils.SgrtTile import SgrtTile
from osgeo import gdal, gdalconst
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.tree import DecisionTreeRegressor
from time import time
import math
from scipy.ndimage import median_filter
import datetime as dt
import ee
import scipy.stats
from sklearn import neighbors
import ascat
import pytesmo.io.ismn.interface as ismn_interface
import copy
from scipy.stats import pearsonr
# from sklearn.feature_selection import RFECV
from sklearn.tree import _tree
from scipy.stats import moment
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error


def tree_to_code(mlmodel, feature_names, outpath):
    '''
    Outputs a decision tree model as a Python function

    Parameters:
    -----------
    tree: decision tree model
        The decision tree to represent as a function
    feature_names: list
        The feature names of the dataset used for building the decision tree
    '''

    outfile = open(outpath + 'decisiontree.py', 'w')

    trees = mlmodel.estimators_
    init_value = mlmodel.init_.constant_[0][0]
    scores = mlmodel.train_score_
    learning_rate = mlmodel.learning_rate

    # print "def tree({}):".format(", ".join(feature_names))
    outfile.write("def tree({}):".format(", ".join(feature_names)) + '\n')
    outfile.write('  prediction = ' + str(init_value) + '\n')
    outfile.write('  learning_rate = ' + str(learning_rate) + '\n')

    for j in range(len(trees)):

        outfile.write('  tree_prediction = 0 \n')
        outfile.write('  tree_weight = 1 \n')  # + str(scores[j]) + '\n')

        tree = trees[j][0]

        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                # print "{}if {} <= {}:".format(indent, name, threshold)
                outfile.write("{}if {} <= {}:".format(indent, name, threshold) + '\n')
                recurse(tree_.children_left[node], depth + 1)
                # print "{}else:  # if {} > {}".format(indent, name, threshold)
                outfile.write("{}else:  # if {} > {}".format(indent, name, threshold) + '\n')
                recurse(tree_.children_right[node], depth + 1)
            else:
                # print "{}return {}".format(indent, tree_.value[node])
                outfile.write(
                    "{}tree_prediction = tree_prediction + {}".format(indent, tree_.value[node].ravel()[0]) + '\n')

        recurse(0, 1)

        outfile.write('  prediction = prediction + (learning_rate*tree_prediction) \n')

    outfile.write('  return prediction')
    outfile.close()


def tree_to_code_GEE(mlmodel, feature_names, outpath):
    '''
    Outputs a decision tree model as a Python function

    Parameters:
    -----------
    tree: decision tree model
        The decision tree to represent as a function
    feature_names: list
        The feature names of the dataset used for building the decision tree
    '''

    outfile = open(outpath + 'no_GLDAS_decisiontree_GEE_1step.py', 'w')

    trees = mlmodel.estimators_
    init_value = mlmodel.init_.constant_[0][0]
    scores = mlmodel.train_score_
    learning_rate = mlmodel.learning_rate

    outfile.write('import ee \n\n')

    # print "def tree({}):".format(", ".join(feature_names))
    # outfile.write("def tree({}):".format(", ".join(feature_names)) + '\n')
    outfile.write('def tree(feature_stack): \n\n')
    outfile.write('  prediction = ee.Image(' + str(init_value) + ')\n')
    outfile.write('  learning_rate = ee.Image(' + str(learning_rate) + ')\n')
    # outfile.write('  prediction = feature_stack.expression("' + str(init_value) + '+ "+\n')

    for j in range(len(trees)):

        outfile.write('  tree_prediction = ee.Image(0) \n')
        outfile.write('  tree_weight = ee.Image(1) \n')  # + str(scores[j]) + '\n')

        tree = trees[j][0]

        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        # outfile.write('"(" +\n')
        outfile.write(
            '  tree_prediction = tree_prediction.where(feature_stack.mask().reduce(ee.Reducer.allNonZero()).eq(1), feature_stack.expression(\n')

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                # print "{}if {} <= {}:".format(indent, name, threshold)
                outfile.write('"')
                outfile.write("(b('{}') <= {}) ? ".format(name, threshold))
                outfile.write('" + \n')
                recurse(tree_.children_left[node], depth + 1)
                # print "{}else:  # if {} > {}".format(indent, name, threshold)
                outfile.write('"')
                outfile.write(':  ')
                outfile.write('" + \n')
                recurse(tree_.children_right[node], depth + 1)
            else:
                # print "{}return {}".format(indent, tree_.value[node])
                outfile.write('"')
                outfile.write("{}".format(tree_.value[node].ravel()[0]))
                outfile.write('" + \n')

        recurse(0, 1)

        # outfile.write('") * ' + str(learning_rate) +' +" \n')
        outfile.write('""))\n')
        outfile.write('  prediction = prediction.add(learning_rate.multiply(tree_prediction)) \n')

    # outfile.write('"0"))\n')
    outfile.write('  return prediction')
    outfile.close()


def tree_to_code_rpart(mlmodel, feature_names, outpath):
    '''
    Outputs a decision tree model as a Python function

    Parameters:
    -----------
    tree: decision tree model
        The decision tree to represent as a function
    feature_names: list
        The feature names of the dataset used for building the decision tree
    '''

    outfile = open(outpath + 'decisiontree_GEE_step2.py', 'w')

    trees = mlmodel.estimators_
    init_value = mlmodel.init_.constant_[0][0]
    scores = mlmodel.train_score_
    learning_rate = mlmodel.learning_rate

    outfile.write('import ee \n\n')

    # print "def tree({}):".format(", ".join(feature_names))
    # outfile.write("def tree({}):".format(", ".join(feature_names)) + '\n')
    outfile.write('def tree(feature_stack): \n\n')
    outfile.write('  prediction = ee.Image(' + str(init_value) + ')\n')
    outfile.write('  learning_rate = ee.Image(' + str(learning_rate) + ')\n')
    # outfile.write('  prediction = feature_stack.expression("' + str(init_value) + '+ "+\n')

    for j in range(len(trees)):

        outfile.write('  tree_prediction = ee.Image(0) \n')
        outfile.write('  tree_weight = ee.Image(1) \n')  # + str(scores[j]) + '\n')

        tree = trees[j][0]

        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        # outfile.write('"(" +\n')
        outfile.write(
            '  tree_prediction = tree_prediction.where(feature_stack.mask().reduce(ee.Reducer.allNonZero()).eq(1), feature_stack.expression(\n')

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                # print "{}if {} <= {}:".format(indent, name, threshold)
                outfile.write('"')
                outfile.write("(b('{}') <= {}) ? ".format(name, threshold))
                outfile.write('" + \n')
                recurse(tree_.children_left[node], depth + 1)
                # print "{}else:  # if {} > {}".format(indent, name, threshold)
                outfile.write('"')
                outfile.write(':  ')
                outfile.write('" + \n')
                recurse(tree_.children_right[node], depth + 1)
            else:
                # print "{}return {}".format(indent, tree_.value[node])
                outfile.write('"')
                outfile.write("{}".format(tree_.value[node].ravel()[0]))
                outfile.write('" + \n')

        recurse(0, 1)

        # outfile.write('") * ' + str(learning_rate) +' +" \n')
        outfile.write(')\n')
        outfile.write('  prediction = prediction.add(learning_rate.multiply(tree_prediction)) \n')

    # outfile.write('"0"))\n')
    outfile.write('  return prediction')
    outfile.close()


def treeToJson(decision_tree, feature_names=None):
    from warnings import warn

    js = ""

    def node_to_str(tree, node_id, criterion):
        if not isinstance(criterion, sklearn.tree.tree.six.string_types):
            criterion = "impurity"

        value = tree.value[node_id]
        if tree.n_outputs == 1:
            value = value[0, :]

        jsonValue = ', '.join([str(x) for x in value])

        if tree.children_left[node_id] == sklearn.tree._tree.TREE_LEAF:
            return '"node": "%s", "split": "%s", "impurity": "%s", "n": "%s", "yval": [%s]' \
                   % (node_id,
                      criterion,
                      tree.impurity[node_id],
                      tree.n_node_samples[node_id],
                      jsonValue)
        else:
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = tree.feature[node_id]

            if "=" in feature:
                ruleType = "="
                ruleValue = "false"
            else:
                ruleType = "<="
                ruleValue = "%.4f" % tree.threshold[node_id]

            return '"id": "%s", "rule": "%s %s %s", "%s": "%s", "samples": "%s"' \
                   % (node_id,
                      feature,
                      ruleType,
                      ruleValue,
                      criterion,
                      tree.impurity[node_id],
                      tree.n_node_samples[node_id])

    def recurse(tree, node_id, criterion, parent=None, depth=0):
        tabs = "  " * depth
        js = ""

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        js = js + "\n" + \
             tabs + "{\n" + \
             tabs + "  " + node_to_str(tree, node_id, criterion)

        if left_child != sklearn.tree._tree.TREE_LEAF:
            js = js + ",\n" + \
                 tabs + '  "left": ' + \
                 recurse(tree, \
                         left_child, \
                         criterion=criterion, \
                         parent=node_id, \
                         depth=depth + 1) + ",\n" + \
                 tabs + '  "right": ' + \
                 recurse(tree, \
                         right_child, \
                         criterion=criterion, \
                         parent=node_id,
                         depth=depth + 1)

        js = js + tabs + "\n" + \
             tabs + "}"

        return js

    if isinstance(decision_tree, sklearn.tree.tree.Tree):
        js = js + recurse(decision_tree, 0, criterion="impurity")
    else:
        js = js + recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

    return js


def stackh5(root_path=None, out_path=None):
    # find h5 files to create time series
    filelist = rsearch.search_file(root_path, 'SMAP_L4*.h5')

    # load grid coodrinates
    ftmp = h5py.File(filelist[0])
    EASE_lats = np.array(ftmp['cell_lat'])
    EASE_lons = np.array(ftmp['cell_lon'])
    ftmp.close()

    # get number of files
    nfiles = len(filelist)

    # iterate through all files
    time_sec = np.full(nfiles, -9999, dtype=np.float64)
    sm_stack = np.full((nfiles, EASE_lats.shape[0], EASE_lats.shape[1]), -9999, dtype=np.float32)
    stemp_stack = np.full((nfiles, EASE_lats.shape[0], EASE_lats.shape[1]), -9999, dtype=np.float32)

    for find in range(nfiles):
        # load file
        ftmp = h5py.File(filelist[find], 'r')
        tmpsm = ftmp['Analysis_Data/sm_surface_analysis']
        tmptemp = ftmp['Analysis_Data/soil_temp_layer1_analysis']
        # sm_subset = tmpsm[rowmin:rowmax, colmin:colmax]
        tmptime = ftmp['time'][0]

        time_sec[find] = tmptime
        sm_stack[find, :, :] = np.array(tmpsm)
        stemp_stack[find, :, :] = np.array(tmptemp)
        ftmp.close()

    # convert seconds since 2000-01-01 11:58:55.816 to datetime
    time_dt = [dt.datetime(2000, 1, 1, 11, 58, 55, 816) + dt.timedelta(seconds=x) for x in time_sec]

    # write to .h5 file
    f = h5py.File(out_path + 'SMAPL4_SMC_2015.h5', 'w')
    h5sm = f.create_dataset('SM_array', data=sm_stack)
    h5temp = f.create_dataset('SoilTemp', data=stemp_stack)
    h5lats = f.create_dataset('LATS', data=EASE_lats)
    h5lons = f.create_dataset('LONS', data=EASE_lons)
    h5time = f.create_dataset('time', data=time_sec)
    f.close()


def dem2equi7(out_path=None, dem_path=None):
    # this routine re-projects the DEM to the Equi7 grid and derives slope and aspect

    grid = Equi7Grid(10)
    grid.resample(dem_path, out_path, gdal_path="/usr/local/bin", sgrid_ids=['EU'], e7_folder=False,
                  outshortname="EDUDEM", withtilenameprefix=True, image_nodata=-32767, tile_nodata=-9999,
                  qlook_flag=False, resampling_type='bilinear')

    # get list of resampled DEM tiles
    filelist = [x for x in glob.glob(out_path + '*.tif')]

    # iterate through all files and derive slope and aspect
    for file in filelist:

        if (file.find('aspect') == -1) and (file.find('slope') == -1):
            # aspect
            aspect_path = out_path + os.path.basename(file)[:-4] + '_aspect.tif'
            if not os.path.exists(aspect_path):
                os.system('/usr/local/bin/gdaldem aspect ' + file + ' ' + aspect_path + ' -co "COMPRESS=LZW"')

            # slope
            slope_path = out_path + os.path.basename(file)[:-4] + '_slope.tif'
            if not os.path.exists(slope_path):
                os.system('/usr/local/bin/gdaldem slope ' + file + ' ' + slope_path + ' -co "COMPRESS=LZW"')


def test2step(basepath, prefix='noGLDAS_', model='RFmlmodelNoneSVR_2step.p'):
    # filter nan values

    mlmodel_tmp = pickle.load(open(basepath + prefix + model, 'rb'))
    mlmodel1 = mlmodel_tmp[0]
    mlmodel2 = mlmodel_tmp[2]
    scaler1 = mlmodel_tmp[1]
    scaler2 = mlmodel_tmp[3]
    outldetector1 = mlmodel_tmp[4]
    outldetector2 = mlmodel_tmp[5]

    # mlmodel_avg = pickle.load(open("//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_1km/RFmlmodelNoneSVR_2step.p", "rb"))
    # mlmodel1 = mlmodel_avg[0]
    # scaler1 = mlmodel_avg[1]
    # outldetector1 = mlmodel_avg[4]

    testtarget1, testfeatures1, testtarget2, testfeatures2 = \
        pickle.load(open(basepath + "testset.p", "rb"))

    # valid = np.where((testtarget1+testtarget2 > 0) & (testtarget1+testtarget2 < 0.8))

    # testtarget1, testfeatures1, notused1, notused2 = \
    #     pickle.load(open("//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_1km/testset.p", "rb"))

    if scaler1 is not None:
        features1_scaled = scaler1.transform(testfeatures1)
    else:
        features1_scaled = testfeatures1
    if scaler2 is not None:
        features2_scaled = scaler2.transform(testfeatures2)
    else:
        features2_scaled = testfeatures2

    x_test1 = features1_scaled
    x_test2 = features2_scaled

    # detect outliers
    # x_outliers1 = outldetector1.predict(x_test1)
    # x_outliers2 = outldetector2.predict(x_test2)
    # x_outliers_combo = x_outliers1 | x_outliers2
    # x_outliers_good = np.where(x_outliers_combo == 1)[0]
    # x_outliers_bad = np.where(x_outliers_combo == -1)[0]

    # prediction of the average sm
    y_pred1 = mlmodel1.predict(x_test1)
    y_pred2 = mlmodel2.predict(x_test2)
    # pred1ext = np.array([y_pred1[xi] for xi in self.idlist])
    y_pred_tot = y_pred1 + y_pred2  # y_pred1 +

    # np.random.seed(15)
    # indices2 = np.arange(0, len(y_pred_tot))
    # indices = np.random.choice(indices2, int(len(y_pred_tot)*0.1), replace=False)
    # indices2 = np.delete(indices2, indices)

    y_tot = testtarget1 + testtarget2  # testtarget1 +

    # r = np.corrcoef(y_tot[x_outliers_good], y_pred_tot[x_outliers_good])
    # urmse = np.sqrt(np.sum(np.square((y_pred_tot[x_outliers_good] - np.mean(y_pred_tot[x_outliers_good])) - (y_tot[x_outliers_good] - np.mean(y_tot[x_outliers_good])))) / len(y_tot[x_outliers_good]))
    # bias = np.mean(y_pred_tot[x_outliers_good] - y_tot[x_outliers_good])
    # error = np.sqrt(np.sum(np.square(y_pred_tot[x_outliers_good] - y_tot[x_outliers_good])) / len(y_tot[x_outliers_good]))

    r = np.corrcoef(y_tot, y_pred_tot)
    urmse = np.sqrt(np.sum(np.square((y_pred_tot - np.mean(y_pred_tot)) - (
            y_tot - np.mean(y_tot)))) / len(y_tot))
    bias = np.mean(y_pred_tot - y_tot)
    error = np.sqrt(
        np.sum(np.square(y_pred_tot - y_tot)) / len(y_tot))

    r_avg = np.corrcoef(testtarget1, y_pred1)
    r_rel = np.corrcoef(testtarget2, y_pred2)
    urmse_avg = np.sqrt(np.sum(np.square((y_pred1 - np.mean(y_pred1)) - (
            testtarget1 - np.mean(testtarget1)))) / len(testtarget1))
    urmse_rel = np.sqrt(np.sum(np.square((y_pred2 - np.mean(y_pred2)) - (
            testtarget2 - np.mean(testtarget2)))) / len(testtarget2))
    rmse_avg = np.sqrt(np.sum(np.square(y_pred1 - testtarget1)) / len(testtarget1))
    rmse_rel = np.sqrt(np.sum(np.square(y_pred2 - testtarget2)) / len(testtarget2))
    bias_avg = np.mean(y_pred1 - testtarget1)
    bias_rel = np.mean(y_pred2 - testtarget2)

    # print('Prediction of average soil moisture')
    # print('R: ' + str(r[0, 1]))
    # print('RMSE. ' + str(error))

    print(mlmodel1)
    print(mlmodel2)

    pltlims = 1.0

    # create plots
    # create plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(3.5, 9), dpi=600)
    # plt.figure(figsize=(3.5, 3)))
    ax1.scatter(testtarget1, y_pred1, c='k', label='True vs Est', s=1, marker='*')
    ax1.set_xlim(0, pltlims)
    ax1.set_ylim(0, pltlims)
    ax1.set_xlabel("$SMC_{Avg}$ [m$^3$m$^{-3}$]", size=8)
    ax1.set_ylabel("$SMC^*_{Avg}$ [m$^3$m$^{-3}$]", size=8)
    ax1.set_title('a)')
    ax1.plot([0, pltlims], [0, pltlims], 'k--', linewidth=0.8)
    ax1.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r_avg[0, 1]) +
             '\nRMSE=' + '{:03.2f}'.format(rmse_avg) +
             '\nBias=' + '{:03.2f}'.format(bias_avg), fontsize=8)
    ax1.set_aspect('equal', 'box')
    # ax1.tight_layout()
    # plt.savefig(
    #    basepath + prefix +  'RFtruevsest_independent_2step_average.png', dpi=600)
    # plt.close()

    # plt.figure(figsize=(3.5, 3))
    ax2.scatter(testtarget2, y_pred2, c='k', label='True vs Est', s=1, marker='*')
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel("$SMC_{Rel}$ [m$^3$m$^{-3}$]", size=8)
    ax2.set_ylabel("$SMC^*_{Rel}$ [m$^3$m$^{-3}$]", size=8)
    ax2.set_title('b)')
    ax2.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', linewidth=0.8)
    ax2.text(-0.15, 0.12, 'R=' + '{:03.2f}'.format(r_rel[0, 1]) +
             '\nRMSE=' + '{:03.2f}'.format(rmse_rel) +
             '\nBias=' + '{:03.2f}'.format(bias_rel), fontsize=8)
    ax2.set_aspect('equal', 'box')
    # ax2.tight_layout()
    # plt.savefig(
    #     basepath + prefix + 'RFtruevsest_independent_2step_relative.png', dpi=600)
    # plt.close()

    # plt.figure(figsize=(3.5, 3))
    ax3.scatter(y_tot, y_pred_tot, c='k', label='True vs Est', s=1, marker='*')
    ax3.set_xlim(0, pltlims)
    ax3.set_ylim(0, pltlims)
    ax3.set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=8)
    ax3.set_ylabel("$SMC^*_{Tot}$ [m$^3$m$^{-3}$]", size=8)
    ax3.set_title('c)')
    ax3.plot([0, pltlims], [0, pltlims], 'k--', linewidth=0.8)
    ax3.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r[0, 1]) +
             '\nRMSE=' + '{:03.2f}'.format(error) +
             '\nBias=' + '{:03.2f}'.format(bias), fontsize=8)
    ax3.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(basepath + prefix + 'RFtruevsest_independent_2step_combined.png', dpi=600)
    plt.close()

    # cdf and differernce plots
    cdf_y_tot = cdf(y_tot)
    cdf_y_tot_pred = cdf(y_pred_tot)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.5, 6), dpi=600)
    ax1.plot(cdf_y_tot[1][0:-1], cdf_y_tot[0], linewidth=1, color='k', label='ISMN')
    ax1.plot(cdf_y_tot_pred[1][0:-1], cdf_y_tot_pred[0], linewidth=0.8, color='k', linestyle='--', label='S1')
    ax1.set_ylabel('Cumulative frequency', size=8)
    ax1.set_xlim((0, 0.7))
    ax1.grid(b=True)
    ax1.legend(fontsize=8)
    ax1.tick_params(axis="y", labelsize=8)
    plt.tight_layout()

    ismn_sorted = np.argsort(y_tot)
    y_tot_s = y_tot[ismn_sorted]
    y_pred_tot_s = y_pred_tot[ismn_sorted]
    # ax2.plot(y_tot_s, y_pred_tot_s - y_tot_s, color='k', linestyle='', marker='*')
    ax2.plot(cdf_y_tot[1][0:-1], cdf_y_tot[0] - cdf_y_tot_pred[0], color='k', linewidth=0.8)
    ax2.set_ylabel('Cumulative frequency diff. (true - estimated)', size=8)
    ax2.set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=8)
    ax2.set_xlim((0, 0.7))
    ax2.set_ylim((-0.25, 0.25))
    ax2.grid(b=True)

    plt.tick_params(labelsize=8)
    plt.tight_layout()
    # plt.show()
    plt.savefig(basepath + prefix + 'CDF_DIFF_RF.png', dpi=600)
    plt.close()


def cdf(data):
    data_size = len(data)

    # Set bins edges
    data_set = sorted(set(data))
    bins = np.append(data_set, data_set[-1] + 1)

    # Use the histogram function to bin the data
    # counts, bin_edges = np.histogram(data, bins=bins, density=False)
    counts, bin_edges = np.histogram(data, bins=100, range=(0, 0.65), density=False)

    counts = counts.astype(float) / data_size

    # Find the cdf
    cdf = np.cumsum(counts)

    # Plot the cdf
    return (cdf, bin_edges)


def test1step(basepath, prefix='noGLDAS_', model='RFmlmodelNoneSVR_1step.p'):
    # filter nan values

    mlmodel_tmp = pickle.load(open(basepath + prefix + model, 'rb'))
    mlmodel = mlmodel_tmp[0]
    scaler = mlmodel_tmp[1]
    outl_detector = mlmodel_tmp[2]

    testtarget, testfeatures = \
        pickle.load(open(basepath + "testset_1step.p", 'rb'))

    # scores, names, best_features = pickle.load(open(
    #     "//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/test/feature_importance_1step.p",
    #     'rb'))

    if scaler is not None:
        features_scaled = scaler.transform(testfeatures)
    else:
        features_scaled = testfeatures

    x_test = features_scaled

    # detect outliers
    if outl_detector is not None:
        outliers = outl_detector.predict(x_test)
    else:
        outliers = None

    # prediction of the average sm
    y_pred = mlmodel.predict(x_test)
    r = np.corrcoef(testtarget, y_pred)
    urmse = np.sqrt(
        np.sum(np.square((y_pred - np.mean(y_pred)) -
                         (testtarget - np.mean(testtarget)))) /
        len(testtarget))
    bias = np.mean(y_pred - testtarget)
    error = np.sqrt(
        np.sum(np.square(y_pred - testtarget)) / len(
            testtarget))

    print('Prediction of soil moisture')
    print('R: ' + str(r[0, 1]))
    print('RMSE. ' + str(error))

    # print(mlmodel.best_estimator_)

    pltlims = 1

    # create plots
    # create plots
    fig, (ax1) = plt.subplots(1, 1, sharex=False, figsize=(3.5, 3), dpi=600)
    # plt.figure(figsize=(3.5, 3)))
    ax1.scatter(testtarget, y_pred, c='k', label='True vs Est', s=1, marker='*')
    ax1.set_xlim(0, pltlims)
    ax1.set_ylim(0, pltlims)
    ax1.set_xlabel("$SMC$ [m$^3$m$^{-3}$]", size=8)
    ax1.set_ylabel("$SMC$ [m$^3$m$^{-3}$]", size=8)
    ax1.set_title('a)')
    ax1.plot([0, pltlims], [0, pltlims], 'k--', linewidth=0.8)
    ax1.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r[0, 1]) +
             '\nRMSE=' + '{:03.2f}'.format(error) +
             '\nBias=' + '{:03.2f}'.format(bias), fontsize=8)
    ax1.set_aspect('equal', 'box')
    ax1.tight_layout()
    plt.savefig(
        basepath + prefix + 'RFtruevsest_independent_2step_average.png', dpi=600)
    plt.close()

    # cdf and differernce plots
    cdf_y_tot = cdf(testtarget)
    cdf_y_tot_pred = cdf(y_pred)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.5, 6), dpi=600)
    ax1.plot(cdf_y_tot[1][0:-1], cdf_y_tot[0], linewidth=1, color='k', label='ISMN')
    ax1.plot(cdf_y_tot_pred[1][0:-1], cdf_y_tot_pred[0], linewidth=0.8, color='k', linestyle='--', label='S1')
    ax1.set_ylabel('Cumulative frequency', size=8)
    ax1.set_xlim((0, 0.7))
    ax1.grid(b=True)
    ax1.legend(fontsize=8)
    ax1.tick_params(axis="y", labelsize=8)
    plt.tight_layout()

    ismn_sorted = np.argsort(testtarget)
    y_tot_s = testtarget[ismn_sorted]
    y_pred_tot_s = y_pred[ismn_sorted]
    # ax2.plot(y_tot_s, y_pred_tot_s - y_tot_s, color='k', linestyle='', marker='*')
    ax2.plot(cdf_y_tot[1][0:-1], cdf_y_tot[0] - cdf_y_tot_pred[0], color='k', linewidth=0.8)
    ax2.set_ylabel('Cumulative frequency diff. (true - estimated)', size=8)
    ax2.set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=8)
    ax2.set_xlim((0, 0.7))
    ax2.set_ylim((-0.25, 0.25))
    ax2.grid(b=True)

    plt.tick_params(labelsize=8)
    plt.tight_layout()
    # plt.show()
    plt.savefig(basepath + prefix + 'CDF_DIFF_RF.png', dpi=600)
    plt.close()


def test_fw_bw(basepath):
    # filter nan values

    mlmodel_tmp = pickle.load(open(basepath + "mlmodelNoneRF_fw_bw.p", 'rb'))
    mlmodel_s0vv = mlmodel_tmp[0]
    mlmodel_s0vh = mlmodel_tmp[1]
    mlmodel_ssm = mlmodel_tmp[3]

    testtarget, testfeatures = \
        pickle.load(open(basepath + "testset_1step.p"))

    testtargets0vv, testtargets0vh, testfeaturess0 = \
        pickle.load(open(basepath + 'testset_fw_bw.p'))

    # prediction of sm
    x_s0test_dry = np.copy(testfeaturess0)
    x_s0test_dry[:, 0] = 0

    s0vv_dry = mlmodel_s0vv.predict(x_s0test_dry)
    s0vh_dry = mlmodel_s0vh.predict(x_s0test_dry)

    valid = (testfeatures[:, 0] - s0vv_dry >= 0) & (testfeatures[:, 1] - s0vh_dry >= 0)

    testfeatures = testfeatures[valid, :]
    testfeatures[:, 0] = 10 * np.log10(testfeatures[:, 0] - s0vv_dry[valid])
    testfeatures[:, 1] = 10 * np.log10(testfeatures[:, 1] - s0vh_dry[valid])
    testtarget = testtarget[valid]

    ssm_pred = mlmodel_ssm.predict(testfeatures)

    # np.random.seed(15)
    # indices2 = np.arange(0, len(y_pred_tot))
    # indices = np.random.choice(indices2, int(len(y_pred_tot)*0.1), replace=False)
    # indices2 = np.delete(indices2, indices)

    r = np.corrcoef(testtarget, ssm_pred)
    urmse = np.sqrt(np.sum(np.square((ssm_pred - np.mean(ssm_pred)) -
                                     (testtarget - np.mean(testtarget)))) /
                    len(testtarget))
    bias = np.mean(ssm_pred - testtarget)
    error = np.sqrt(np.sum(np.square(ssm_pred - testtarget)) / len(testtarget))

    print('Prediction of average soil moisture')
    print('R: ' + str(r[0, 1]))
    print('RMSE. ' + str(error))

    # print(mlmodel.best_estimator_)

    pltlims = 0.7

    # create plots
    plt.figure(figsize=(6, 6))
    plt.scatter(testtarget, ssm_pred, c='g', label='True vs Est')
    plt.xlim(0, pltlims)
    plt.ylim(0, pltlims)
    plt.xlabel("True  average SMC [m3m-3]")
    plt.ylabel("Estimated average SMC [m3m-3]")
    plt.plot([-0.2, pltlims], [-0.2, pltlims], 'k--')
    plt.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r[0, 1]) +
             '\nubRMSE=' + '{:03.2f}'.format(urmse) +
             '\nBias=' + '{:03.2f}'.format(bias), weight='bold')
    plt.savefig(basepath + 'FW_BW_truevsest_independent.png')
    plt.close()


def plotSSM_vs_SSMpred_pwise(basepath):
    sig0lia = pickle.load(open(basepath + 'sig0lia_dictNone.p', 'rb'))

    X = np.array(sig0lia['ssm'])
    Y = np.array(sig0lia['ssm_p_pred'])

    r = np.corrcoef(X, Y)
    urmse = np.sqrt(
        np.sum(np.square((Y - np.mean(Y)) - (X - np.mean(X)))) / len(X))
    bias = np.mean(Y - X)
    error = np.sqrt(np.sum(np.square(Y - X)) / len(Y))

    # create plots
    plt.figure(figsize=(6, 6))
    plt.scatter(X, Y, c='g', label='True vs Est')
    plt.xlim(0, 0.7)
    plt.ylim(0, 0.7)
    plt.xlabel("True SMC [m3m-3]")
    plt.ylabel("Estimated p-wise SMC [m3m-3]")
    plt.plot([0, 0.7], [0, 0.7], 'k--')
    plt.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r[0, 1]) +
             '\nRMSE=' + '{:03.2f}'.format(error) +
             '\nubRMSE=' + '{:03.2f}'.format(urmse) +
             '\nBias=' + '{:03.2f}'.format(bias), weight='bold')
    plt.savefig(basepath + 'station_ts/corplot.png')
    plt.close()


class Trainingset(object):

    def __init__(self, outpath, uselc=True, subgrid='EU',
                 months=list([5, 6, 7, 8, 9]), ssm_target='SMAP', sig0_source='SGRT', track=117, desc=False,
                 footprint=None, feature_vect1=None, feature_vect2=None, feature_vect3=None, prefix='noGLDAS_', extr_data=False,
                 indtestset=True):

        self.outpath = outpath
        self.uselc = uselc
        self.subgrid = subgrid
        self.months = months
        self.ssm_target = ssm_target
        self.sig0_source = sig0_source
        self.track = track
        self.desc = desc
        self.footprint = footprint
        self.prefix = prefix
        self.feature_vect1 = feature_vect1
        self.feature_vect2 = feature_vect2
        self.feature_vect3 = feature_vect3

        if extr_data:
            sig0lia = self.create_trainingset()
            return
        else:
            sig0lia, _ = pickle.load(open(self.outpath + 'sig0lia_dict' + str(self.track) + '.p', 'rb'),
                                     encoding='latin-1')  # instaed of self.outpath
            # sig0lia_mazia = pickle.load(open(self.outpath + 'sig0lia_dict_Mazia' + str(self.track) + '.p', 'rb'),
            # encoding='latin-1')

        #
        # # apply additional filters
        # valLClist = [11, 14, 20, 30, 120, 140, 150]
        # COPERNICUS
        valLClist = [30, 40, 60, 90]
        ndvi = (np.array(sig0lia['L8_b5']) - np.array(sig0lia['L8_b4'])) / (
                np.array(sig0lia['L8_b5']) + np.array(sig0lia['L8_b4']))
        sig0lia = sig0lia.where((sig0lia['gldas_swe'] == 0) & (sig0lia['gldas_soilt'] > 275) &
                                ([lci in valLClist for lci in np.array(sig0lia['lc'])]) & (ndvi < 0.7)).dropna()

        # Recompute temporal statistics after masking
        # identify the unique sites
        locations_array = np.array([sig0lia.index.get_level_values(0), sig0lia.index.get_level_values(1)]).transpose()
        unique_locations, indices, unique_counts = np.unique(locations_array, axis=0, return_index=True,
                                                             return_counts=True)
        # create a location index
        tmp_loc_id = np.full(sig0lia.shape[0], fill_value=0)
        unique_locations_tracks, tindices, tunique_c = np.unique(np.array(
            [sig0lia.index.get_level_values(0), sig0lia.index.get_level_values(1),
             sig0lia.index.get_level_values(2)]).transpose(), axis=0, return_index=True,
                                                                 return_counts=True)
        for uniqu_idx in range(len(tindices)):
            tmp_loc_id[tindices[uniqu_idx]:tindices[uniqu_idx] + tunique_c[uniqu_idx]] = uniqu_idx
        self.loc_id = tmp_loc_id
        sig0lia['locid'] = self.loc_id

        # recomupte temporal statistics after masking
        tmp_avg_targets = [
            'ssm_mean', 'gldas_mean', 'usdasm_mean', 'gldas_et_mean', 'gldas_swe_mean', 'gldas_soilt_mean',
            'plant_water_mean', 'gldas_precip_mean', 'gldas_snowmelt_mean', 'ndvi_mean', 'L8_b1_median',
            'L8_b2_median', 'L8_b3_median', 'L8_b4_median', 'L8_b5_median', 'L8_b6_median',
            'L8_b7_median', 'L8_b10_median', 'L8_b11_median'
        ]
        tmp_avg_sources = [
            'ssm', 'gldas', 'usdasm', 'gldas_et', 'gldas_swe', 'gldas_soilt',
            'plant_water', 'gldas_precip', 'gldas_snowmelt', 'ndvi', 'L8_b1',
            'L8_b2', 'L8_b3', 'L8_b4', 'L8_b5', 'L8_b6',
            'L8_b7', 'L8_b10', 'L8_b11'
        ]
        for i_sttn in range(unique_locations.shape[0]):
            sttn_tracks = sig0lia.loc[unique_locations[i_sttn, 0],
                                      unique_locations[i_sttn, 1]].index.get_level_values(0).unique()
            for i_sttn_tracks in sttn_tracks:
                for var_i in range(len(tmp_avg_targets)):
                    sig0lia.loc[(unique_locations[i_sttn, 0],
                                 unique_locations[i_sttn, 1],
                                 i_sttn_tracks), tmp_avg_targets[var_i]] = sig0lia.loc[(unique_locations[i_sttn, 0],
                                                                                        unique_locations[i_sttn, 1],
                                                                                        i_sttn_tracks),
                                                                                       tmp_avg_sources[var_i]].median()
                # sig0
                tmp_vv_lin = np.power(10, sig0lia.loc[(unique_locations[i_sttn, 0],
                                                       unique_locations[i_sttn, 1],
                                                       i_sttn_tracks), 'sig0vv'] / 10)
                tmp_vh_lin = np.power(10, sig0lia.loc[(unique_locations[i_sttn, 0],
                                                       unique_locations[i_sttn, 1],
                                                       i_sttn_tracks), 'sig0vv'] / 10)
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vv_tmean'] = 10 * np.log10(tmp_vv_lin.median())
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vh_tmean'] = 10 * np.log10(tmp_vh_lin.median())
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vv_k1'] = np.mean(np.log(tmp_vv_lin))
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vh_k1'] = np.mean(np.log(tmp_vh_lin))
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vv_k2'] = np.std(np.log(tmp_vv_lin))
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vh_k2'] = np.std(np.log(tmp_vh_lin))
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vv_k3'] = moment(np.log(tmp_vv_lin), moment=3)
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vh_k3'] = moment(np.log(tmp_vh_lin), moment=3)
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vv_k4'] = moment(np.log(tmp_vv_lin), moment=4)
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vh_k4'] = moment(np.log(tmp_vh_lin), moment=4)
                # gamma0
                tmp_gammavv_lin = np.power(10, sig0lia.loc[(unique_locations[i_sttn, 0],
                                                       unique_locations[i_sttn, 1],
                                                       i_sttn_tracks), 'gamma0vv'] / 10)
                tmp_gammavh_lin = np.power(10, sig0lia.loc[(unique_locations[i_sttn, 0],
                                                       unique_locations[i_sttn, 1],
                                                       i_sttn_tracks), 'gamma0vv'] / 10)
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vv_gammatmean'] = 10 * np.log10(tmp_gammavv_lin.median())
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vh_gammatmean'] = 10 * np.log10(tmp_gammavh_lin.median())
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vv_gammak1'] = np.mean(np.log(tmp_gammavv_lin))
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vh_gammak1'] = np.mean(np.log(tmp_gammavh_lin))
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vv_gammak2'] = np.std(np.log(tmp_gammavv_lin))
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vh_gammak2'] = np.std(np.log(tmp_gammavh_lin))
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vv_gammak3'] = moment(np.log(tmp_gammavv_lin), moment=3)
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vh_gammak3'] = moment(np.log(tmp_gammavh_lin), moment=3)
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vv_gammak4'] = moment(np.log(tmp_gammavv_lin), moment=4)
                sig0lia.loc[(unique_locations[i_sttn, 0],
                             unique_locations[i_sttn, 1],
                             i_sttn_tracks), 'vh_gammak4'] = moment(np.log(tmp_gammavh_lin), moment=4)

        #
        # # select 10 random stations in different land cover classes
        if indtestset:
            np.random.seed(78)
            rndm_choice = np.random.choice(unique_locations.shape[0], size=10)
            sig0lia_test = pd.concat([sig0lia.loc[unique_locations[i, 0], unique_locations[i, 1]] for i in rndm_choice])
            test_stations_locations = unique_locations[rndm_choice,]
            # remove test-set from sig0lia
            sig0lia.drop(index=sig0lia_test.index, inplace=True)
            # reset the loc indices
            training_stations_indices = np.delete(indices, rndm_choice)
            training_stations_counts = np.delete(unique_counts, rndm_choice)
            tmp_locid = np.full(sig0lia.shape[0], fill_values=0)
            for uniqu_idx in training_stations_indices:
                tmp_locid[training_stations_indices[uniqu_idx]:training_stations_counts[uniqu_idx]] = uniqu_idx
            self.loc_id = tmp_locid

        # create training and test sets
        self.target1, self.features1, self.target2, self.target2_2, self.features2, self.target3, self.features3 = \
            self.create_training_and_testing_array(sig0lia)

        if indtestset:
            self.testtarget1, self.testfeatures1, self.testtarget2, self.testtarget2_2, self.testfeatures2 = \
                self.create_training_and_testing_array(sig0lia_test)

            # create testset metadata
            test_stations_names, unidxs = np.unique(sig0lia_test['stname'].to_numpy(), return_index=True)
            test_stations_ntwks = sig0lia_test['ntwkname'].iloc[unidxs]

            # save the testset
            pickle.dump(
                (self.testtarget1, self.testfeatures1, self.testtarget2, self.testtarget2_2, self.testfeatures2),
                open(self.outpath + 'testset.p', 'wb'))
            pickle.dump((test_stations_names, test_stations_ntwks, test_stations_locations),
                        open(self.outpath + 'testset_meta.p', 'wb'))

        self.sig0lia = sig0lia
        print('HELLO')

    def create_trainingset(self):

        # # Get the locations of training points
        if self.ssm_target == 'ISMN':
            self.points = self.create_random_points(sgrid='ISMN')
        elif self.ssm_target == 'Mazia':
            self.points = self.create_random_points(sgrid='Mazia')
        # # # # #
        # # # # # # extract parameters
        if self.sig0_source == 'GEE':
            sig0lia = self.extr_sig0_lia_gee()

        if self.ssm_target == 'Mazia':
            pickle.dump(sig0lia, open(self.outpath + 'sig0lia_dict_Mazia' + str(self.track) + '.p', 'wb'))
        else:
            pickle.dump(sig0lia, open(self.outpath + 'sig0lia_dict' + str(self.track) + '.p', 'wb'))

        return sig0lia

    def create_training_and_testing_array(self, df):
        # define training sets
        # step1_subs = df.drop_duplicates([
        #     'ssm_mean', 'vv_k1', 'vh_k1', 'vv_k2', 'vh_k2', 'vv_k3', 'vh_k3', 'vv_k4', 'vh_k4', 'vv_tmean', 'vh_tmean',
        #     'lia', 'lc', 'bare_perc', 'crops_perc', 'forest_type', 'grass_perc', 'moss_perc', 'urban_perc',
        #     'waterp_perc', 'waters_perc', 'sand', 'clay', 'trees', 'bulk', 'L8_b1_median', 'L8_b2_median',
        #     'L8_b3_median', 'L8_b4_median', 'L8_b5_median', 'L8_b6_median', 'L8_b7_median', 'L8_b10_median',
        #     'L8_b11_median', 'lon', 'lat', 'gldas_mean', 'usdasm_mean'
        # ])
        step1_subs = df.drop_duplicates(['locid'])
        self.sub_loc = step1_subs['locid'].values
        target1 = np.array(step1_subs['ssm_mean'], dtype=np.float32)
        features1 = np.vstack((np.array(step1_subs['vv_k1'], dtype=np.float32),  # 0
                               np.array(step1_subs['vh_k1'], dtype=np.float32),  # 1
                               np.array(step1_subs['vv_k2'], dtype=np.float32),  # 2
                               np.array(step1_subs['vh_k2'], dtype=np.float32),  # 3
                               np.array(step1_subs['vv_k3'], dtype=np.float32),  # 4
                               np.array(step1_subs['vh_k3'], dtype=np.float32),  # 5
                               np.array(step1_subs['vv_k4'], dtype=np.float32),  # 6
                               np.array(step1_subs['vh_k4'], dtype=np.float32),  # 7
                               np.array(step1_subs['vv_tmean'], dtype=np.float32),  # 8
                               np.array(step1_subs['vh_tmean'], dtype=np.float32),  # 9
                               np.array(step1_subs['vv_gammak1'], dtype=np.float32),  # 10
                               np.array(step1_subs['vh_gammak1'], dtype=np.float32),  # 11
                               np.array(step1_subs['vv_gammak2'], dtype=np.float32),  # 12
                               np.array(step1_subs['vh_gammak2'], dtype=np.float32),  # 13
                               np.array(step1_subs['vv_gammak3'], dtype=np.float32),  # 14
                               np.array(step1_subs['vh_gammak3'], dtype=np.float32),  # 15
                               np.array(step1_subs['vv_gammak4'], dtype=np.float32),  # 16
                               np.array(step1_subs['vh_gammak4'], dtype=np.float32),  # 17
                               np.array(step1_subs['vv_gammatmean'], dtype=np.float32),  # 18
                               np.array(step1_subs['vh_gammatmean'], dtype=np.float32),  # 19
                               np.array(step1_subs['lia'], dtype=np.float32),  # 20
                               np.array(step1_subs['lc'], dtype=np.float32),  # 21
                               np.array(step1_subs['bare_perc'], dtype=np.float32),  # 22
                               np.array(step1_subs['crops_perc'], dtype=np.float32),  # 23
                               np.array(step1_subs['trees'], dtype=np.float32),  # 24
                               np.array(step1_subs['forest_type'], dtype=np.float32),  # 25
                               np.array(step1_subs['grass_perc'], dtype=np.float32),  # 26
                               np.array(step1_subs['moss_perc'], dtype=np.float32),  # 27
                               np.array(step1_subs['urban_perc'], dtype=np.float32),  # 28
                               np.array(step1_subs['waterp_perc'], dtype=np.float32),  # 29
                               np.array(step1_subs['waters_perc'], dtype=np.float32),  # 30
                               np.array(step1_subs['L8_b1_median'], dtype=np.float32),  # 31
                               np.array(step1_subs['L8_b2_median'], dtype=np.float32),  # 32
                               np.array(step1_subs['L8_b3_median'], dtype=np.float32),  # 33
                               np.array(step1_subs['L8_b4_median'], dtype=np.float32),  # 34
                               np.array(step1_subs['L8_b5_median'], dtype=np.float32),  # 35
                               np.array(step1_subs['L8_b6_median'], dtype=np.float32),  # 36
                               np.array(step1_subs['L8_b7_median'], dtype=np.float32),  # 37
                               np.array(step1_subs['L8_b10_median'], dtype=np.float32),  # 38
                               np.array(step1_subs['L8_b11_median'], dtype=np.float32),  # 39
                               np.array(step1_subs['ndvi_mean'], dtype=np.float32),  # 40
                               np.array(step1_subs['bulk'], dtype=np.float32),  # 41
                               np.array(step1_subs['clay'], dtype=np.float32),  # 42
                               np.array(step1_subs['sand'], dtype=np.float32),  # 43
                               np.array(step1_subs['lon'], dtype=np.float32),  # 44
                               np.array(step1_subs['lat'], dtype=np.float32),  # 45
                               np.array(step1_subs['gldas_mean'], dtype=np.float32),  # 46
                               np.array(step1_subs['usdasm_mean'], dtype=np.float32))).transpose()  # 47

        if self.feature_vect1 is not None:
            features1 = features1[:, self.feature_vect1]

        target2 = np.array(df['ssm'], dtype=np.float32) - np.array(df['ssm_mean'], dtype=np.float32)
        target2_2 = np.array(df['ssm'], dtype=np.float32)
        features2 = np.vstack((np.array(df['sig0vv'], dtype=np.float32),  # 0
                               np.array(df['sig0vh'], dtype=np.float32),  # 1
                               np.array(df['sig0vv'], dtype=np.float32) -
                               np.array(df['vv_tmean'], dtype=np.float32),  # 2
                               np.array(df['sig0vh'], dtype=np.float32) -
                               np.array(df['vh_tmean'], dtype=np.float32),  # 3
                               np.array(df['gamma0vv'], dtype=np.float32),  # 4
                               np.array(df['gamma0vh'], dtype=np.float32),  # 5
                               np.array(df['gamma0vv'], dtype=np.float32) -
                               np.array(df['vv_gammatmean'], dtype=np.float32),  # 6
                               np.array(df['gamma0vh'], dtype=np.float32) -
                               np.array(df['vh_gammatmean'], dtype=np.float32),  # 7
                               np.array(df['lia'], dtype=np.float32),  # 8
                               np.array(df['vv_k1'], dtype=np.float32),  # 9
                               np.array(df['vh_k1'], dtype=np.float32),  # 10
                               np.array(df['vv_k2'], dtype=np.float32),  # 11
                               np.array(df['vh_k2'], dtype=np.float32),  # 12
                               np.array(df['vv_k3'], dtype=np.float32),  # 13
                               np.array(df['vh_k3'], dtype=np.float32),  # 14
                               np.array(df['vv_k4'], dtype=np.float32),  # 15
                               np.array(df['vh_k4'], dtype=np.float32),  # 16
                               np.array(df['vv_gammak1'], dtype=np.float32),  # 17
                               np.array(df['vh_gammak1'], dtype=np.float32),  # 18
                               np.array(df['vv_gammak2'], dtype=np.float32),  # 19
                               np.array(df['vh_gammak2'], dtype=np.float32),  # 20
                               np.array(df['vv_gammak3'], dtype=np.float32),  # 21
                               np.array(df['vh_gammak3'], dtype=np.float32),  # 22
                               np.array(df['vv_gammak4'], dtype=np.float32),  # 23
                               np.array(df['vh_gammak4'], dtype=np.float32),  # 24
                               np.array(df['lc'], dtype=np.float32),  # 25
                               np.array(df['bare_perc'], dtype=np.float32),  # 26
                               np.array(df['crops_perc'], dtype=np.float32),  # 27
                               np.array(df['trees'], dtype=np.float32),  # 28
                               np.array(df['forest_type'], dtype=np.float32),  # 29
                               np.array(df['grass_perc'], dtype=np.float32),  # 30
                               np.array(df['moss_perc'], dtype=np.float32),  # 31
                               np.array(df['urban_perc'], dtype=np.float32),  # 32
                               np.array(df['waterp_perc'], dtype=np.float32),  # 33
                               np.array(df['waters_perc'], dtype=np.float32),  # 34
                               np.array(df['L8_b1'], dtype=np.float32),  # 35
                               np.array(df['L8_b1_median'], dtype=np.float32),  # 36
                               np.array(df['L8_b2'], dtype=np.float32),  # 37
                               np.array(df['L8_b2_median'], dtype=np.float32),  # 38
                               np.array(df['L8_b3'], dtype=np.float32),  # 39
                               np.array(df['L8_b3_median'], dtype=np.float32),  # 40
                               np.array(df['L8_b4'], dtype=np.float32),  # 41
                               np.array(df['L8_b4_median'], dtype=np.float32),  # 42
                               np.array(df['L8_b5'], dtype=np.float32),  # 43
                               np.array(df['L8_b5_median'], dtype=np.float32),  # 44
                               np.array(df['L8_b6'], dtype=np.float32),  # 45
                               np.array(df['L8_b6_median'], dtype=np.float32),  # 46
                               np.array(df['L8_b7'], dtype=np.float32),  # 47
                               np.array(df['L8_b7_median'], dtype=np.float32),  # 48
                               np.array(df['L8_b10'], dtype=np.float32),  # 49
                               np.array(df['L8_b10_median'], dtype=np.float32),  # 50
                               np.array(df['L8_b11'], dtype=np.float32),  # 51
                               np.array(df['L8_b11_median'], dtype=np.float32),  # 52
                               np.array(df['L8_timediff'], dtype=np.float32),  # 53
                               np.array(df['ndvi'], dtype=np.float32),  # 54
                               np.array(df['ndvi_mean'], dtype=np.float32),  # 55
                               np.array(df['lon'], dtype=np.float32),  # 56
                               np.array(df['lat'], dtype=np.float32),  # 57
                               np.array(df['bulk'], dtype=np.float32),  # 58
                               np.array(df['clay'], dtype=np.float32),  # 59
                               np.array(df['sand'], dtype=np.float32),  # 60
                               np.array(df['gldas'], dtype=np.float32),  # 61
                               np.array(df['gldas_mean'], dtype=np.float32),  # 62
                               np.array(df['usdasm'], dtype=np.float32),  # 63
                               np.array(df['usdasm_mean'], dtype=np.float32),  # 64
                               )).transpose()

        if self.feature_vect2 is not None:
            features2 = features2[:, self.feature_vect2]

        target3 = np.array(df['gamma0vv'], dtype=np.float32)

        features3 = np.vstack((np.array(df['ssm'], dtype=np.float32),  # 0
                               np.array(df['lia'], dtype=np.float32),  # 1
                               np.array(df['lc'], dtype=np.float32),  # 2
                               np.array(df['bare_perc'], dtype=np.float32),  # 3
                               np.array(df['crops_perc'], dtype=np.float32),  # 4
                               np.array(df['trees'], dtype=np.float32),  # 5
                               np.array(df['forest_type'], dtype=np.float32),  # 6
                               np.array(df['grass_perc'], dtype=np.float32),  # 7
                               np.array(df['moss_perc'], dtype=np.float32),  # 8
                               np.array(df['urban_perc'], dtype=np.float32),  # 9
                               np.array(df['waterp_perc'], dtype=np.float32),  # 10
                               np.array(df['waters_perc'], dtype=np.float32),  # 11
                               np.array(df['L8_b1'], dtype=np.float32),  # 12
                               np.array(df['L8_b2'], dtype=np.float32),  # 13
                               np.array(df['L8_b3'], dtype=np.float32),  # 14
                               np.array(df['L8_b4'], dtype=np.float32),  # 15
                               np.array(df['L8_b5'], dtype=np.float32),  # 16
                               np.array(df['L8_b6'], dtype=np.float32),  # 17
                               np.array(df['L8_b7'], dtype=np.float32),  # 18
                               np.array(df['L8_b10'], dtype=np.float32),  # 19
                               np.array(df['L8_b11'], dtype=np.float32),  # 20
                               np.array(df['L8_timediff'], dtype=np.float32),  # 21
                               np.array(df['ndvi'], dtype=np.float32))).transpose() # 22

        if self.feature_vect3 is not None:
            features3 = features3[:, self.feature_vect3]

        return target1, features1, target2, target2_2, features2, target3, features3

    def get_track_extent(self):

        ee.Initialize()

        boundaries = ee.FeatureCollection("USDOS/LSIB/2013")

        europe = boundaries.filter(ee.Filter.Or(ee.Filter.eq('cc', 'IT'),
                                                ee.Filter.eq('cc', 'AU'),
                                                ee.Filter.eq('cc', 'SZ'),
                                                ee.Filter.eq('cc', 'GM'),
                                                ee.Filter.eq('cc', 'FR'),
                                                ee.Filter.eq('cc', 'AU'),
                                                ee.Filter.eq('cc', 'PO'),
                                                ee.Filter.eq('cc', 'SP'),
                                                ee.Filter.eq('cc', 'UK'),
                                                ee.Filter.eq('cc', 'PL')))
        europe = europe.geometry().convexHull()

        s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
            .filter(ee.Filter.eq('platform_number', 'A')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .filter(ee.Filter.eq('relativeOrbitNumber_start', self.track)) \
            .filterBounds(europe)

        return (s1.geometry().bounds().getInfo())

    def reduceTrainingSetComplexity(self, sig0lia):

        from sklearn import svm
        from sklearn.ensemble import IsolationForest

        tarray = np.column_stack((sig0lia['ssm'], sig0lia['sig0vv'], sig0lia['sig0vh'], sig0lia['vv_k1'],
                                  sig0lia['vh_k1'], sig0lia['vv_k2'], sig0lia['vh_k2']))
        np.random.shuffle(tarray)

        # selected start set
        startset_ind = np.random.choice(tarray.shape[0], size=10)
        training_set = tarray[startset_ind, :]
        pool = np.delete(tarray, startset_ind, axis=0)

        clf = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=0.1)
        clf.fit(training_set)

        # fit the model
        for i in range(pool.shape[0]):

            candidate = np.array(pool[i, :]).reshape(1, -1)
            cand_pred = clf.predict(candidate)

            if cand_pred[0] == -1:
                # clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
                training_set = np.vstack((training_set, candidate))
                clf.fit(training_set)

        # subsub = np.random.choice(training_set.shape[0], size=999)
        # training_set = training_set[subsub, :]
        print(training_set.shape)

        outset = {'ssm': training_set[:, 0],
                  'sig0vv': training_set[:, 1],
                  'sig0vh': training_set[:, 2],
                  'vv_k1': training_set[:, 3],
                  'vh_k1': training_set[:, 4],
                  'vv_k2': training_set[:, 5],
                  'vh_k2': training_set[:, 6]}
        return (outset)

    def train_model(self):

        import scipy.stats
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.decomposition import PCA
        from sklearn.linear_model import TheilSenRegressor
        from sklearn.neural_network import MLPRegressor

        # models = dict()

        # tracks = self.features[:,-1]
        # untracks = np.unique(tracks)

        # for itrack in untracks:

        # filter bad ssm values
        valid = np.where(self.target > 0)  # & (self.features[:,-1] == itrack))
        track_target = self.target[valid[0]]
        track_features = self.features[valid[0], :]
        weights = self.weights[valid[0]]
        # filter nan values
        valid = ~np.any(np.isinf(track_features), axis=1)
        track_target = track_target[valid]
        track_features = track_features[valid, :]
        weights = weights[valid]
        # filter nan
        valid = ~np.any(np.isnan(track_features), axis=1)
        track_target = track_target[valid]
        track_features = track_features[valid, :]
        weights = weights[valid]

        # scaling
        scaler = sklearn.preprocessing.RobustScaler().fit(track_features)
        features = scaler.transform(track_features)

        # perform outlier detection
        from sklearn.ensemble import IsolationForest
        # from sklearn import svm
        isof = IsolationForest(behaviour='new', contamination='auto', random_state=42)
        x_outliers = isof.fit(features).predict(features)
        features = features[np.where(x_outliers == 1)[0], :]
        track_target = track_target[np.where(x_outliers == 1)[0]]
        weights = weights[np.where(x_outliers == 1)[0]]

        x_train, x_test, y_train, y_test, weights_train, weights_test = train_test_split(features, track_target,
                                                                                         weights,
                                                                                         test_size=0.9,
                                                                                         train_size=0.1,
                                                                                         random_state=70)  # , random_state=42)
        # x_test = x_train
        # y_test = y_train
        # x_train = features
        # y_train = track_target
        # x_test = features
        # y_test = track_target
        # weights_train = weights
        # x_test = x_train
        # y_test = y_train

        # ...----...----...----...----...----...----...----...----...----...
        # SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR
        # ---....---....---....---....---....---....---....---....---....---

        # specify kernel
        svr_rbf = SVR()

        # parameter tuning -> grid search
        start = time()
        print('Model training startet at: ' + str(start))
        #
        # SVR --- SVR --- SVR --- SVR --- SVR --- SVR --- SVR
        #
        init_c = [-2, 0]
        init_g = [-2, -0.5]
        init_e = [-2, -0.5]

        for t_i in range(3):
            # SVR -- SVR -- SVR -- SVR -- SVR -- SVR
            dictCV1 = dict(C=np.logspace(init_c[0], init_c[1], 4),
                           # gamma=np.logspace(-2, -0.5, 5),
                           gamma=np.logspace(init_g[0], init_g[1], 4),
                           # epsilon=np.logspace(-2, -0.5, 5),
                           epsilon=np.logspace(init_e[0], init_e[1], 4),
                           # degree=np.array([1,2,3]),
                           # coef0=[0.01,1,10],
                           kernel=['rbf'])

            mlmodel = GridSearchCV(estimator=SVR(),
                                   param_grid=dictCV1,
                                   n_jobs=-1,
                                   verbose=1,
                                   # pre_dispatch='2*n_jobs',
                                   # pre_dispatch='all',
                                   # cv=ShuffleSplit(n_splits=10, random_state=42),
                                   cv=KFold(n_splits=10, shuffle=True, random_state=42),
                                   # cv=LeaveOneOut(),
                                   scoring='r2')  # ,

            from sklearn.utils import parallel_backend
            with parallel_backend('multiprocessing'):
                mlmodel.fit(x_train, y_train, sample_weight=weights_train)

            winner_c = mlmodel.best_estimator_.get_params()['C']
            winner_g = mlmodel.best_estimator_.get_params()['gamma']
            winner_e = mlmodel.best_estimator_.get_params()['epsilon']

            loc_c = np.where(dictCV1['C'] == winner_c)[0][0]
            if loc_c == 0:
                init_c = [np.log10(dictCV1['C'][0]), np.log10(dictCV1['C'][1])]
            elif loc_c == 3:
                init_c = [np.log10(dictCV1['C'][2]), np.log10(dictCV1['C'][3])]
            else:
                init_c = [np.log10(dictCV1['C'][loc_c - 1]), np.log10(dictCV1['C'][loc_c + 1])]

            loc_g = np.where(dictCV1['gamma'] == winner_g)[0][0]
            if loc_g == 0:
                init_g = [np.log10(dictCV1['gamma'][0]), np.log10(dictCV1['gamma'][1])]
            elif loc_g == 3:
                init_g = [np.log10(dictCV1['gamma'][2]), np.log10(dictCV1['gamma'][3])]
            else:
                init_g = [np.log10(dictCV1['gamma'][loc_g - 1]), np.log10(dictCV1['gamma'][loc_g + 1])]

            loc_e = np.where(dictCV1['epsilon'] == winner_e)[0][0]
            if loc_e == 0:
                init_e = [np.log10(dictCV1['epsilon'][0]), np.log10(dictCV1['epsilon'][1])]
            elif loc_e == 3:
                init_e = [np.log10(dictCV1['epsilon'][2]), np.log10(dictCV1['epsilon'][3])]
            else:
                init_e = [np.log10(dictCV1['epsilon'][loc_e - 1]), np.log10(dictCV1['epsilon'][loc_e + 1])]

        print(mlmodel.best_estimator_)
        # print(gdCV.best_params_)
        # prediction on test set
        y_CV_rbf = mlmodel.predict(x_test)
        # print(gdCV.feature_importances_)

        true = y_test
        est = y_CV_rbf

        # models[str(itrack)] = (copy.deepcopy(gdCV), copy.deepcopy(scaler))

        r = np.corrcoef(true, est)
        error = np.sqrt(np.sum(np.square(true - est)) / len(true))
        urmse = np.sqrt(np.sum(np.square((est - np.mean(est)) - (true - np.mean(true)))) / len(true))

        # print(gdCV.best_params_)
        print('Elapse time for training: ' + str(time() - start))

        print('SVR performance based on test-set')
        print('R: ' + str(r[0, 1]))
        print('RMSE. ' + str(error))

        if (self.ssm_target == 'SMAP') or (self.ssm_target == 'ISMN'):
            pltlims = 0.7
        else:
            pltlims = 100

        # create plots
        plt.figure(figsize=(6, 6))
        plt.scatter(true, est, c='g', label='True vs Est')
        # plt.xlim(0,pltlims)
        # plt.ylim(0,pltlims)
        plt.xlabel("True SMC [m3m-3]")
        plt.ylabel("Estimated SMC [m3m-3]")
        plt.plot([0, pltlims], [0, pltlims], 'k--')
        plt.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r[0, 1]) + '\nRMSE=' + '{:03.2f}'.format(error), weight='bold')
        plt.savefig(self.outpath + 'truevsest1step.png')
        plt.close()

        # self.SVRmodel = gdCV
        # self.scaler = scaler
        pickle.dump((mlmodel.best_estimator_, scaler, isof),
                    open(self.outpath + 'mlmodel' + str(self.track) + 'SVR_1step.p', 'wb'))
        return (mlmodel, scaler, isof)

    def train_model2step(self):

        # filter bad ssm values
        valid = np.where(self.target1 > 0)  # & (self.features[:,-1] == itrack))
        self.target1 = self.target1[valid[0]]
        self.features1 = self.features1[valid[0], :]
        self.weights_1 = self.weights_1[valid[0]]
        # filter nan values
        valid = ~np.any(np.isinf(self.features1), axis=1)
        self.target1 = self.target1[valid]
        self.features1 = self.features1[valid, :]
        self.weights_1 = self.weights_1[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features1), axis=1)
        self.target1 = self.target1[valid]
        self.features1 = self.features1[valid, :]
        self.weights_1 = self.weights_1[valid]

        # filter nan values
        valid = ~np.any(np.isinf(self.features2), axis=1)
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]
        self.weights_2 = self.weights_2[valid]
        self.loc_id = self.loc_id[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features2), axis=1)
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]
        self.weights_2 = self.weights_2[valid]
        self.loc_id = self.loc_id[valid]

        # scaling
        scaler1 = sklearn.preprocessing.RobustScaler().fit(self.features1)
        features1_scaled = scaler1.transform(self.features1)

        scaler2 = sklearn.preprocessing.RobustScaler().fit(self.features2)
        features2_scaled = scaler2.transform(self.features2)

        isof1 = None
        isof2 = None

        x_train1, x_test1, y_train1, y_test1, weights_train1, weights_test1 = train_test_split(features1_scaled,
                                                                                               self.target1,
                                                                                               self.weights_1,
                                                                                               test_size=0.1,
                                                                                               train_size=0.9,
                                                                                               random_state=70)

        x_train2, x_test2, y_train2, y_test2, weights_train2, weights_test2, loc_train, loc_test = train_test_split(
            features2_scaled, self.target2, self.weights_2, self.loc_id,
            test_size=0.1,
            train_size=0.9, random_state=70)

        x_train1 = features1_scaled
        x_test1 = features1_scaled
        y_train1 = self.target1
        y_test1 = self.target1
        weights_train1 = self.weights_1

        x_train2 = features2_scaled
        x_test2 = features2_scaled
        y_train2 = self.target2
        y_test2 = self.target2
        weights_train2 = self.weights_2

        init_c = [-2, 2]
        init_g = [-2, -0.5]
        init_e = [-2, -0.5]
        init_nu = [-2, 0]

        # for t_i in range(3):
        # SVR -- SVR -- SVR -- SVR -- SVR -- SVR
        dictCV1 = dict(C=np.logspace(init_c[0], init_c[1], 4),
                       gamma=np.logspace(init_g[0], init_g[1], 4),
                       epsilon=np.logspace(init_e[0], init_e[1], 4),
                       kernel=['rbf'])

        mlmodel1 = GridSearchCV(estimator=SVR(cache_size=200),
                                param_grid=dictCV1,
                                n_jobs=4,
                                verbose=1,
                                # pre_dispatch='2*n_jobs',
                                # pre_dispatch='all',
                                # cv=ShuffleSplit(n_splits=10, random_state=42),
                                cv=KFold(n_splits=10, random_state=42),
                                # cv=LeaveOneOut(),
                                scoring='r2')  # ,
        from sklearn.utils import parallel_backend
        with parallel_backend('multiprocessing'):
            mlmodel1.fit(x_train1, y_train1)

            # winner_c = mlmodel1.best_estimator_.get_params()['C']
            # winner_g = mlmodel1.best_estimator_.get_params()['gamma']
            # winner_e = mlmodel1.best_estimator_.get_params()['epsilon']
            #
            # loc_c = np.where(dictCV1['C'] == winner_c)[0][0]
            # if loc_c == 0:
            #     init_c = [np.log10(dictCV1['C'][0]), np.log10(dictCV1['C'][1])]
            # elif loc_c == 3:
            #     init_c = [np.log10(dictCV1['C'][2]), np.log10(dictCV1['C'][3])]
            # else:
            #     init_c = [np.log10(dictCV1['C'][loc_c-1]), np.log10(dictCV1['C'][loc_c+1])]
            #
            # loc_g = np.where(dictCV1['gamma'] == winner_g)[0][0]
            # if loc_g == 0:
            #     init_g = [np.log10(dictCV1['gamma'][0]), np.log10(dictCV1['gamma'][1])]
            # elif loc_g == 3:
            #     init_g = [np.log10(dictCV1['gamma'][2]), np.log10(dictCV1['gamma'][3])]
            # else:
            #     init_g = [np.log10(dictCV1['gamma'][loc_g - 1]), np.log10(dictCV1['gamma'][loc_g + 1])]
            #
            # loc_e = np.where(dictCV1['epsilon'] == winner_e)[0][0]
            # if loc_e == 0:
            #     init_e = [np.log10(dictCV1['epsilon'][0]), np.log10(dictCV1['epsilon'][1])]
            # elif loc_e == 3:
            #     init_e = [np.log10(dictCV1['epsilon'][2]), np.log10(dictCV1['epsilon'][3])]
            # else:
            #     init_e = [np.log10(dictCV1['epsilon'][loc_e - 1]), np.log10(dictCV1['epsilon'][loc_e + 1])]

        print(mlmodel1.best_estimator_)

        init_c = [-2, 2]
        init_g = [-2, -0.5]
        init_e = [-2, -0.5]
        init_nu = [-2, 0]

        # for t_i in range(3):
        dictCV2 = dict(C=np.logspace(init_c[0], init_c[1], 4),
                       gamma=np.logspace(init_g[0], init_g[1], 4),
                       epsilon=np.logspace(init_e[0], init_e[1], 4),
                       kernel=['rbf'])

        mlmodel2 = GridSearchCV(estimator=SVR(),
                                param_grid=dictCV2,
                                n_jobs=4,
                                verbose=1,
                                iid=False,
                                # pre_dispatch='2*n_jobs',
                                # pre_dispatch='all',
                                # cv=KFold(n_splits=10, shuffle=True, random_state=42),
                                cv=GroupKFold(n_splits=10))  # ,

        from sklearn.utils import parallel_backend
        with parallel_backend('multiprocessing'):
            mlmodel2.fit(x_train2, y_train2, groups=self.loc_id)

            # winner_c = mlmodel2.best_estimator_.get_params()['C']
            # winner_g = mlmodel2.best_estimator_.get_params()['gamma']
            # winner_e = mlmodel2.best_estimator_.get_params()['epsilon']
            #
            # # loc_nu = np.where(dictCV2['nu'] == winner_nu)[0][0]
            # # if loc_c == 0:
            # #     init_c = [np.log10(dictCV2['nu'][0]), np.log10(dictCV2['nu'][1])]
            # # elif loc_c == 3:
            # #     init_c = [np.log10(dictCV2['nu'][2]), np.log10(dictCV2['nu'][3])]
            # # else:
            # #     init_c = [np.log10(dictCV2['nu'][loc_nu - 1]), np.log10(dictCV2['nu'][loc_nu + 1])]
            #
            # loc_c = np.where(dictCV2['C'] == winner_c)[0][0]
            # if loc_c == 0:
            #     init_c = [np.log10(dictCV2['C'][0]), np.log10(dictCV2['C'][1])]
            # elif loc_c == 3:
            #     init_c = [np.log10(dictCV2['C'][2]), np.log10(dictCV2['C'][3])]
            # else:
            #     init_c = [np.log10(dictCV2['C'][loc_c - 1]), np.log10(dictCV2['C'][loc_c + 1])]
            #
            # loc_g = np.where(dictCV2['gamma'] == winner_g)[0][0]
            # if loc_g == 0:
            #     init_g = [np.log10(dictCV2['gamma'][0]), np.log10(dictCV2['gamma'][1])]
            # elif loc_g == 3:
            #     init_g = [np.log10(dictCV2['gamma'][2]), np.log10(dictCV2['gamma'][3])]
            # else:
            #     init_g = [np.log10(dictCV2['gamma'][loc_g - 1]), np.log10(dictCV2['gamma'][loc_g + 1])]
            #
            # loc_e = np.where(dictCV2['epsilon'] == winner_e)[0][0]
            # if loc_e == 0:
            #     init_e = [np.log10(dictCV2['epsilon'][0]), np.log10(dictCV2['epsilon'][1])]
            # elif loc_e == 3:
            #     init_e = [np.log10(dictCV2['epsilon'][2]), np.log10(dictCV2['epsilon'][3])]
            # else:
            #     init_e = [np.log10(dictCV2['epsilon'][loc_e - 1]), np.log10(dictCV2['epsilon'][loc_e + 1])]

        print(mlmodel2.best_estimator_)

        # prediction of the average sm
        y_pred1 = mlmodel1.predict(x_test1)

        r = np.corrcoef(y_test1, y_pred1)
        urmse = np.sqrt(np.sum(np.square((y_pred1 - np.mean(y_pred1)) - (y_test1 - np.mean(y_test1)))) / len(y_test1))
        bias = np.mean(y_pred1 - y_test1)
        error = np.sqrt(np.sum(np.square(y_pred1 - y_test1)) / len(y_test1))

        print('Prediction of average soil moisture')
        print('R: ' + str(r[0, 1]))
        print('RMSE. ' + str(error))

        # print(mlmodel1.best_estimator_)
        # print(mlmodel2.best_estimator_)

        if (self.ssm_target == 'SMAP') or (self.ssm_target == 'ISMN'):
            pltlims = 0.7
        else:
            pltlims = 100

        # create plots
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test1, y_pred1, c='g', label='True vs Est')
        plt.xlim(0, pltlims)
        plt.ylim(0, pltlims)
        plt.xlabel("True  average SMC [m3m-3]")
        plt.ylabel("Estimated average SMC [m3m-3]")
        plt.plot([0, pltlims], [0, pltlims], 'k--')
        plt.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r[0, 1]) +
                 '\nubRMSE=' + '{:03.2f}'.format(urmse) +
                 '\nBias=' + '{:03.2f}'.format(bias), weight='bold')
        plt.savefig(self.outpath + self.prefix + 'SVR_truevsest_average.png')
        plt.close()

        # prediction of the relative sm
        y_pred2 = mlmodel2.predict(x_test2)

        r = np.corrcoef(y_test2, y_pred2)
        urmse = np.sqrt(np.sum(np.square((y_pred2 - np.mean(y_pred2)) - (y_test2 - np.mean(y_test2)))) / len(y_test2))
        bias = np.mean(y_pred2 - y_test2)
        error = np.sqrt(np.sum(np.square(y_pred2 - y_test2)) / len(y_test2))

        print('Prediction of relative soil moisture')
        print('R: ' + str(r[0, 1]))
        print('RMSE. ' + str(error))

        if (self.ssm_target == 'SMAP') or (self.ssm_target == 'ISMN'):
            pltlims = 0.2
        else:
            pltlims = 100

        # create plots
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test2, y_pred2, c='g', label='True vs Est')
        plt.xlim(-0.2, pltlims)
        plt.ylim(-0.2, pltlims)
        plt.xlabel("True  relative SMC [m3m-3]")
        plt.ylabel("Estimated relative SMC [m3m-3]")
        plt.plot([-0.2, pltlims], [-0.2, pltlims], 'k--')
        plt.text(-0.15, 0.15, 'R=' + '{:03.2f}'.format(r[0, 1]) +
                 '\nubRMSE=' + '{:03.2f}'.format(urmse) +
                 '\nBias=' + '{:03.2f}'.format(bias), weight='bold')
        plt.savefig(self.outpath + self.prefix + 'SVR_truevsest_relative.png')
        plt.close()

        pickle.dump((mlmodel1.best_estimator_, scaler1, mlmodel2.best_estimator_, scaler2, isof1, isof2),
                    open(self.outpath + self.prefix + 'mlmodel' + str(self.track) + 'SVR_2step.p', 'wb'))
        return (mlmodel1, scaler1, mlmodel2, scaler2, isof1, isof2)

    def train_model_rf(self):
        import scipy.stats
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import TheilSenRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import ExtraTreesRegressor

        # filter bad ssm values
        valid = np.where(self.target > 0)  # & (self.features[:,-1] == itrack))
        track_target = self.target[valid[0]]
        track_features = self.features[valid[0], :]
        weights = self.weights[valid[0]]
        # filter nan values
        valid = ~np.any(np.isinf(track_features), axis=1)
        track_target = track_target[valid]
        track_features = track_features[valid, :]
        weights = weights[valid]
        # filter nan
        valid = ~np.any(np.isnan(track_features), axis=1)
        track_target = track_target[valid]
        track_features = track_features[valid, :]
        weights = weights[valid]

        # scaling
        # scaler = sklearn.preprocessing.RobustScaler().fit(track_features)
        # features = scaler.transform(track_features)
        scaler = None
        features = track_features

        # perform outlier detection
        # from sklearn.ensemble import IsolationForest
        # # from sklearn import svm
        # isof = IsolationForest(behaviour='new', contamination='auto', random_state=42)
        # x_outliers = isof.fit(features).predict(features)
        # features = features[np.where(x_outliers == 1)[0], :]
        # track_target = track_target[np.where(x_outliers == 1)[0]]
        # weights = weights[np.where(x_outliers == 1)[0]]
        isof = None

        x_train, x_test, y_train, y_test, weights_train, weights_test = train_test_split(features, track_target,
                                                                                         weights,
                                                                                         test_size=0.2,
                                                                                         train_size=0.8,
                                                                                         random_state=70)  # , random_state=42)
        x_train = features
        y_train = track_target
        x_test = features
        y_test = track_target
        weights_train = weights
        # x_test = x_train
        # y_test = y_train

        # ...----...----...----...----...----...----...----...----...----...
        # SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR
        # ---....---....---....---....---....---....---....---....---....---

        # specify kernel
        svr_rbf = SVR()

        # parameter tuning -> grid search
        start = time()
        print('Model training startet at: ' + str(start))
        #
        # Random Forest Random Forest Random Forest Random Forest
        #
        mlmodel = GradientBoostingRegressor(n_estimators=2000,
                                            # oob_score=True,
                                            verbose=True,
                                            random_state=42)  # ,

        # mlmodel = AdaBoostRegressor(n_estimators=100,
        #                             random_state=42)

        # mlmodel = GradientBoostingRegressor(n_estimators=500,
        #                                     random_state=42,
        #                                     loss='ls',
        #                                     learning_rate=0.01,
        #                                     max_depth=4,
        #                                     min_samples_split=2)

        mlmodel.fit(x_train, y_train)

        # print(mlmodel.best_estimator_)
        # print(gdCV.best_params_)
        # prediction on test set
        y_CV_rbf = mlmodel.predict(x_test)
        # print(gdCV.feature_importances_)

        true = y_test
        est = y_CV_rbf

        # models[str(itrack)] = (copy.deepcopy(gdCV), copy.deepcopy(scaler))

        r = np.corrcoef(true, est)
        error = np.sqrt(np.sum(np.square(true - est)) / len(true))
        urmse = np.sqrt(np.sum(np.square((est - np.mean(est)) - (true - np.mean(true)))) / len(true))

        # print(gdCV.best_params_)
        print('Elapse time for training: ' + str(time() - start))

        print('SVR performance based on test-set')
        print('R: ' + str(r[0, 1]))
        print('RMSE. ' + str(error))

        if (self.ssm_target == 'SMAP') or (self.ssm_target == 'ISMN'):
            pltlims = 0.7
        else:
            pltlims = 100

        # create plots
        plt.figure(figsize=(6, 6))
        plt.scatter(true, est, c='g', label='True vs Est')
        # plt.xlim(0,pltlims)
        # plt.ylim(0,pltlims)
        plt.xlabel("True SMC [m3m-3]")
        plt.ylabel("Estimated SMC [m3m-3]")
        plt.plot([0, pltlims], [0, pltlims], 'k--')
        plt.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r[0, 1]) + '\nRMSE=' + '{:03.2f}'.format(
            error) + '\nubRMSE=' + '{:03.2f}'.format(urmse))
        plt.savefig(self.outpath + self.prefix + 'RFtruevsest1step.png')
        plt.close()

        # self.SVRmodel = gdCV
        # self.scaler = scaler
        pickle.dump((mlmodel, scaler, isof),
                    open(self.outpath + self.prefix + 'RFmlmodel' + str(self.track) + 'SVR_1step.p', 'wb'))
        return (mlmodel, scaler, isof)

    def train_model2step_rf(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.ensemble import ExtraTreesRegressor

        # filter bad ssm values
        valid = np.where((self.target1 > 0) & (self.target1 < 0.9))  # & (self.features[:,-1] == itrack))
        self.target1 = self.target1[valid[0]]
        self.features1 = self.features1[valid[0], :]
        # self.weights_1 = self.weights_1[valid[0]]
        # filter nan values
        valid = ~np.any(np.isinf(self.features1), axis=1)
        self.target1 = self.target1[valid]
        self.features1 = self.features1[valid, :]
        # self.weights_1 = self.weights_1[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features1), axis=1)
        self.target1 = self.target1[valid]
        self.features1 = self.features1[valid, :]
        # self.weights_1 = self.weights_1[valid]

        # filter bad ssm values
        valid = ~np.any(np.isinf(self.features2), axis=1)
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]
        # self.weights_2 = self.weights_2[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features2), axis=1)
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]
        # self.weights_2 = self.weights_2[valid]

        tree_grid = {  # 'bootstrap': [True, False],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [100, 200, 400, 600, 800, 1000]}

        isof1 = None
        isof2 = None

        mlmodel1 = GradientBoostingRegressor(n_estimators=100,
                                             # oob_score=True,
                                             verbose=True,
                                             random_state=42)  # ,
        # bootstrap=True,
        # n_jobs=4)

        mlmodel2 = GradientBoostingRegressor(n_estimators=1000,
                                             # oob_score=True,
                                             verbose=True,
                                             random_state=42)  # ,
        # bootstrap=True,
        # n_jobs=4)

        mlmodel1.fit(self.features1, self.target1)  # , weights_train1)
        mlmodel2.fit(self.features2, self.target2)  # , weights_train2)

        # prediction of the average sm
        y_pred1 = mlmodel1.predict(self.features1)

        r = np.corrcoef(self.target1, y_pred1)
        urmse = np.sqrt(np.sum(np.square((y_pred1 - np.mean(y_pred1)) - (self.target1 - np.mean(self.target1)))) / len(
            self.target1))
        bias = np.mean(y_pred1 - self.target1)
        error = np.sqrt(np.sum(np.square(y_pred1 - self.target1)) / len(self.target1))

        print('Prediction of average soil moisture')
        print('R: ' + str(r[0, 1]))
        print('RMSE. ' + str(error))

        # print(mlmodel1.best_estimator_)
        # print(mlmodel2.best_estimator_)

        if (self.ssm_target == 'SMAP') or (self.ssm_target == 'ISMN'):
            pltlims = 0.7
        else:
            pltlims = 100

        # create plots
        plt.figure(figsize=(6, 6))
        plt.scatter(self.target1, y_pred1, c='g', label='True vs Est')
        plt.xlim(0, pltlims)
        plt.ylim(0, pltlims)
        plt.xlabel("True  average SMC [m3m-3]")
        plt.ylabel("Estimated average SMC [m3m-3]")
        plt.plot([0, pltlims], [0, pltlims], 'k--')
        plt.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r[0, 1]) +
                 '\nubRMSE=' + '{:03.2f}'.format(urmse) +
                 '\nBias=' + '{:03.2f}'.format(bias), weight='bold')
        plt.savefig(self.outpath + self.prefix + 'RFtruevsest_average.png')
        plt.close()

        # prediction of the relative sm
        y_pred2 = mlmodel2.predict(self.features2)

        r = np.corrcoef(self.target2, y_pred2)
        urmse = np.sqrt(np.sum(np.square((y_pred2 - np.mean(y_pred2)) - (self.target2 - np.mean(self.target2)))) / len(
            self.target2))
        bias = np.mean(y_pred2 - self.target2)
        error = np.sqrt(np.sum(np.square(y_pred2 - self.target2)) / len(self.target2))

        print('Prediction of relative soil moisture')
        print('R: ' + str(r[0, 1]))
        print('RMSE. ' + str(error))

        if (self.ssm_target == 'SMAP') or (self.ssm_target == 'ISMN'):
            pltlims = 0.5
        else:
            pltlims = 100

        # create plots
        plt.figure(figsize=(6, 6))
        plt.scatter(self.target2, y_pred2, c='g', label='True vs Est')
        plt.xlim(-pltlims, pltlims)
        plt.ylim(-pltlims, pltlims)
        plt.xlabel("True  relative SMC [m3m-3]")
        plt.ylabel("Estimated relative SMC [m3m-3]")
        plt.plot([-pltlims, pltlims], [-pltlims, pltlims], 'k--')
        plt.text(-0.15, 0.15, 'R=' + '{:03.2f}'.format(r[0, 1]) +
                 '\nubRMSE=' + '{:03.2f}'.format(urmse) +
                 '\nBias=' + '{:03.2f}'.format(bias), weight='bold')
        plt.savefig(self.outpath + self.prefix + 'RFtruevsest_relative.png')
        plt.close()

        pickle.dump((mlmodel1, None, mlmodel2, None, isof1, isof2),
                    open(self.outpath + self.prefix + 'RFmlmodel' + str(self.track) + 'SVR_2step.p', 'wb'))
        return (mlmodel1, None, mlmodel2, None, isof1, isof2)

    def RF_loo(self):
        from sklearn.ensemble import GradientBoostingRegressor

        # filter nan values
        valid = ~np.any(np.isinf(self.features1), axis=1)
        self.target1 = self.target1[valid]
        self.features1 = self.features1[valid, :]
        self.sub_loc = self.sub_loc[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features1), axis=1)
        self.target1 = self.target1[valid]
        self.features1 = self.features1[valid, :]
        self.sub_loc = self.sub_loc[valid]

        # filter bad ssm values
        valid = ~np.any(np.isinf(self.features2), axis=1)
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features2), axis=1)
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]

        tree_grid = {  # 'bootstrap': [True, False],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [100, 200, 400, 600, 800, 1000]}

        true_vect1 = list()
        pred_vect1 = list()
        true_vect2 = list()
        pred_vect2 = list()
        true_vect_tot = list()
        pred_vect_tot = list()
        tree_list = list()
        ndvi_list = list()
        lc_list = list()
        ex_var_list = list()
        r2_list = list()
        rmse_list = list()

        from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.ensemble import HistGradientBoostingRegressor

        mlmodel1 = HistGradientBoostingRegressor(max_iter=500, verbose=False, random_state=42)
        mlmodel2 = HistGradientBoostingRegressor(max_iter=500, verbose=False, random_state=42)

        for iid in np.unique(self.loc_id):
            print(iid)
            tmp_features_train1 = self.features1[self.sub_loc != iid, :]
            tmp_target_train1 = self.target1[self.sub_loc != iid]
            tmp_features_test1 = self.features1[self.sub_loc == iid, :]
            tmp_target_test1 = self.target1[self.sub_loc == iid]

            tmp_features_train2 = self.features2[self.loc_id != iid, :]
            tmp_target_train2 = self.target2[self.loc_id != iid]
            tmp_features_test2 = self.features2[self.loc_id == iid, :]
            tmp_target_test2 = self.target2[self.loc_id == iid]

            mlmodel1.fit(tmp_features_train1, tmp_target_train1)  # , tmp_weights1)
            mlmodel2.fit(tmp_features_train2, tmp_target_train2)  # , tmp_weights2)

            # prediction of the average and relative sm
            tmp_pred1 = mlmodel1.predict(tmp_features_test1)
            pred_vect1 += list(tmp_pred1)
            true_vect1 += list(tmp_target_test1)
            tmp_pred2 = mlmodel2.predict(tmp_features_test2)
            pred_vect2 += list(tmp_pred2)
            true_vect2 += list(tmp_target_test2)
            pred_vect_tot += list(tmp_pred1 + tmp_pred2)
            true_vect_tot += list(tmp_target_test1 + tmp_target_test2)

            # collect ancillary data
            tree_list += list(self.sig0lia['trees'][self.sig0lia['locid'] == iid].values)
            ndvi_list += list(self.sig0lia['ndvi'][self.sig0lia['locid'] == iid].values)
            lc_list += list(self.sig0lia['lc'][self.sig0lia['locid'] == iid].values)

            # collect average scores
            ex_var_list.append(explained_variance_score(tmp_target_test2, tmp_pred2))
            r2_list.append(r2_score(tmp_target_test2, tmp_pred2))
            rmse_list.append(mean_squared_error(tmp_target_test2, tmp_pred2, squared=False))

            print('Explained variance: ' + str(explained_variance_score(tmp_target_test2, tmp_pred2)))
            print('R2: ' + str(r2_score(tmp_target_test2, tmp_pred2)))
            print('RMSE: ' + str(mean_squared_error(tmp_target_test2, tmp_pred2, squared=False)))

        # create the estimation models
        mlmodel1 = GradientBoostingRegressor(n_estimators=500,
                                             verbose=True,
                                             random_state=42)

        mlmodel2 = GradientBoostingRegressor(n_estimators=500,
                                             verbose=True,
                                             random_state=42)
        mlmodel1.fit(self.features1, self.target1)
        mlmodel2.fit(self.features2, self.target2)

        pickle.dump((mlmodel1, None, mlmodel2, None, None, None),
                    open(self.outpath + self.prefix + 'RFmlmodel_2step.p', 'wb'))

        pred_vect1 = np.array(pred_vect1)
        true_vect1 = np.array(true_vect1)
        pred_vect2 = np.array(pred_vect2)
        true_vect2 = np.array(true_vect2)
        pred_vect_tot = np.array(pred_vect_tot)
        true_vect_tot = np.array(true_vect_tot)
        tree_list = np.array(tree_list)
        ndvi_list = np.array(ndvi_list)
        lc_list = np.array(lc_list)
        ex_var_list = np.array(ex_var_list)
        r2_list = np.array(r2_list)
        rmse_list = np.array(rmse_list)
        np.savez(self.outpath + 'loo_tmp.npz', pred_vect1, true_vect1, pred_vect2, true_vect2, pred_vect_tot,
                 true_vect_tot)
        np.savez(self.outpath + 'loo_RF_2step_scores.npz', ex_var_list, r2_list, rmse_list)
        r = np.corrcoef(true_vect_tot, pred_vect_tot)
        urmse = np.sqrt(np.sum(np.square((pred_vect_tot - np.mean(pred_vect_tot)) - (
                true_vect_tot - np.mean(true_vect_tot)))) / len(true_vect_tot))
        bias = np.mean(pred_vect_tot - true_vect_tot)
        error = np.sqrt(
            np.sum(np.square(pred_vect_tot - true_vect_tot)) / len(true_vect_tot))

        r_avg = np.corrcoef(true_vect1, pred_vect1)
        r_rel = np.corrcoef(true_vect2, pred_vect2)
        urmse_avg = np.sqrt(np.sum(np.square((pred_vect1 - np.mean(pred_vect1)) - (
                true_vect1 - np.mean(true_vect1)))) / len(true_vect1))
        urmse_rel = np.sqrt(np.sum(np.square((pred_vect2 - np.mean(pred_vect2)) - (
                true_vect2 - np.mean(true_vect2)))) / len(true_vect2))
        rmse_avg = np.sqrt(np.sum(np.square(pred_vect1 - true_vect1)) / len(true_vect1))
        rmse_rel = np.sqrt(np.sum(np.square(pred_vect2 - true_vect2)) / len(true_vect2))
        bias_avg = np.mean(pred_vect1 - true_vect1)
        bias_rel = np.mean(pred_vect2 - true_vect2)

        pltlims = 1.0

        # create plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(3.5, 9), dpi=600)
        # plt.figure(figsize=(3.5, 3)))
        ax1.scatter(true_vect1, pred_vect1, c='k', label='True vs Est', s=1, marker='*')
        ax1.set_xlim(0, pltlims)
        ax1.set_ylim(0, pltlims)
        ax1.set_xlabel("$SMC_{Avg}$ [m$^3$m$^{-3}$]", size=8)
        ax1.set_ylabel("$SMC^*_{Avg}$ [m$^3$m$^{-3}$]", size=8)
        ax1.set_title('a)')
        ax1.plot([0, pltlims], [0, pltlims], 'k--', linewidth=0.8)
        ax1.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r_avg[0, 1]) +
                 '\nRMSE=' + '{:03.2f}'.format(rmse_avg) +
                 '\nBias=' + '{:03.2f}'.format(bias_avg), fontsize=8)
        ax1.set_aspect('equal', 'box')

        ax2.scatter(true_vect2, pred_vect2, c='k', label='True vs Est', s=1, marker='*')
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xlabel("$SMC_{Rel}$ [m$^3$m$^{-3}$]", size=8)
        ax2.set_ylabel("$SMC^*_{Rel}$ [m$^3$m$^{-3}$]", size=8)
        ax2.set_title('b)')
        ax2.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', linewidth=0.8)
        ax2.text(-0.4, 0.12, 'R=' + '{:03.2f}'.format(r_rel[0, 1]) +
                 '\nRMSE=' + '{:03.2f}'.format(rmse_rel) +
                 '\nBias=' + '{:03.2f}'.format(bias_rel), fontsize=8)
        ax2.set_aspect('equal', 'box')

        ax3.scatter(true_vect_tot, pred_vect_tot, c='k', label='True vs Est', s=1, marker='*')
        ax3.set_xlim(0, pltlims)
        ax3.set_ylim(0, pltlims)
        ax3.set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=8)
        ax3.set_ylabel("$SMC^*_{Tot}$ [m$^3$m$^{-3}$]", size=8)
        ax3.set_title('c)')
        ax3.plot([0, pltlims], [0, pltlims], 'k--', linewidth=0.8)
        ax3.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r[0, 1]) +
                 '\nRMSE=' + '{:03.2f}'.format(error) +
                 '\nBias=' + '{:03.2f}'.format(bias), fontsize=8)
        ax3.set_aspect('equal', 'box')
        plt.tight_layout()
        plt.savefig(self.outpath + self.prefix + 'RFtruevsest_LOO_2step_combined.png', dpi=600)
        plt.close()

        # cdf and differernce plots
        cdf_y_tot = cdf(true_vect_tot)
        cdf_y_tot_pred = cdf(pred_vect_tot)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.5, 6), dpi=600)
        ax1.plot(cdf_y_tot[1][0:-1], cdf_y_tot[0], linewidth=1, color='k', label='ISMN')
        ax1.plot(cdf_y_tot_pred[1][0:-1], cdf_y_tot_pred[0], linewidth=0.8, color='k', linestyle='--', label='S1')
        ax1.set_ylabel('Cumulative frequency', size=8)
        ax1.set_xlim((0, 0.7))
        ax1.grid(b=True)
        ax1.legend(fontsize=8)
        ax1.tick_params(axis="y", labelsize=8)
        plt.tight_layout()

        ismn_sorted = np.argsort(true_vect_tot)
        y_tot_s = true_vect_tot[ismn_sorted]
        y_pred_tot_s = true_vect_tot[ismn_sorted]
        # ax2.plot(y_tot_s, y_pred_tot_s - y_tot_s, color='k', linestyle='', marker='*')
        ax2.plot(cdf_y_tot[1][0:-1], cdf_y_tot[0] - cdf_y_tot_pred[0], color='k', linewidth=0.8)
        ax2.set_ylabel('Cumulative frequency diff. (true - estimated)', size=8)
        ax2.set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=8)
        ax2.set_xlim((0, 0.7))
        ax2.set_ylim((-0.25, 0.25))
        ax2.grid(b=True)

        plt.tick_params(labelsize=8)
        plt.tight_layout()
        plt.savefig(self.outpath + self.prefix + 'CDF_DIFF_RF_LOO.png', dpi=600)
        plt.close()

    def RF_loo_1step(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.ensemble import ExtraTreesRegressor


        # filter bad ssm values
        valid = ~np.any(np.isinf(self.features2), axis=1)
        self.target2_2 = self.target2_2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features2), axis=1)
        self.target2_2 = self.target2_2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]

        tree_grid = {  # 'bootstrap': [True, False],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [100, 200, 400, 600, 800, 1000]}

        true_vect1 = list()
        pred_vect1 = list()
        true_vect2 = list()
        pred_vect2 = list()
        true_vect_tot = list()
        pred_vect_tot = list()
        tree_list = list()
        ndvi_list = list()
        lc_list = list()
        ex_var_list = list()
        r2_list = list()
        rmse_list = list()

        from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.ensemble import HistGradientBoostingRegressor

        # mlmodel2 = GradientBoostingRegressor(n_estimators=1000,
        #                                      verbose=False,
        #                                      max_depth=10,
        #                                      max_features='auto',
        #                                      min_samples_leaf=4,
        #                                      min_samples_split=2,
        #                                      random_state=42)
        mlmodel2 = HistGradientBoostingRegressor(max_iter=500, verbose=False, random_state=42)

        for iid in np.unique(self.loc_id):
            print(iid)

            tmp_features_train2 = self.features2[self.loc_id != iid, :]
            tmp_target_train2 = self.target2_2[self.loc_id != iid]
            tmp_features_test2 = self.features2[self.loc_id == iid, :]
            tmp_target_test2 = self.target2_2[self.loc_id == iid]

            mlmodel2.fit(tmp_features_train2, tmp_target_train2)

            # prediction of the average and relative sm
            tmp_pred2 = mlmodel2.predict(tmp_features_test2)
            pred_vect2 += list(tmp_pred2)
            true_vect2 += list(tmp_target_test2)
            pred_vect_tot += list(tmp_pred2)
            true_vect_tot += list(tmp_target_test2)

            # collect ancillary data
            tree_list += list(self.sig0lia['trees'][self.loc_id == iid])
            ndvi_list += list(self.sig0lia['ndvi'][self.loc_id == iid])
            lc_list += list(self.sig0lia['lc'][self.loc_id == iid])

            # collect average scores
            ex_var_list.append(explained_variance_score(tmp_target_test2, tmp_pred2))
            r2_list.append(r2_score(tmp_target_test2, tmp_pred2))
            rmse_list.append(mean_squared_error(tmp_target_test2, tmp_pred2, squared=False))

            print('Explained variance: ' + str(explained_variance_score(tmp_target_test2, tmp_pred2)))
            print('R2: ' + str(r2_score(tmp_target_test2, tmp_pred2)))
            print('RMSE: ' + str(mean_squared_error(tmp_target_test2, tmp_pred2, squared=False)))

        mlmodel2 = GradientBoostingRegressor(n_estimators=500,
                                             verbose=True,
                                             random_state=42)

        mlmodel2.fit(self.features2, self.target2_2)

        pickle.dump((None, None, mlmodel2, None, None, None),
                    open(self.outpath + self.prefix + 'RFmlmodel_1step.p', 'wb'))

        pred_vect2 = np.array(pred_vect2)
        true_vect2 = np.array(true_vect2)
        pred_vect_tot = np.array(pred_vect_tot)
        true_vect_tot = np.array(true_vect_tot)
        tree_list = np.array(tree_list)
        ndvi_list = np.array(ndvi_list)
        lc_list = np.array(lc_list)
        ex_var_list = np.array(ex_var_list)
        r2_list = np.array(r2_list)
        rmse_list = np.array(rmse_list)
        np.savez(self.outpath + 'loo_RF_1step_scores.npz', ex_var_list, r2_list, rmse_list)
        np.savez(self.outpath + 'loo_1stp_tmp.npz', pred_vect2, true_vect2, pred_vect_tot,
                 true_vect_tot)
        r = np.corrcoef(true_vect_tot, pred_vect_tot)
        urmse = np.sqrt(np.sum(np.square((pred_vect_tot - np.mean(pred_vect_tot)) - (
                true_vect_tot - np.mean(true_vect_tot)))) / len(true_vect_tot))
        bias = np.mean(pred_vect_tot - true_vect_tot)
        error = np.sqrt(
            np.sum(np.square(pred_vect_tot - true_vect_tot)) / len(true_vect_tot))

        pltlims = 1.0

        # create plots
        fig, ax3 = plt.subplots(1, 1, sharex=False, figsize=(3.5, 3), dpi=600)

        ax3.scatter(true_vect_tot, pred_vect_tot, c='k', label='True vs Est', s=1, marker='*')
        ax3.set_xlim(0, pltlims)
        ax3.set_ylim(0, pltlims)
        ax3.set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=8)
        ax3.set_ylabel("$SMC^*_{Tot}$ [m$^3$m$^{-3}$]", size=8)
        ax3.set_title('c)')
        ax3.plot([0, pltlims], [0, pltlims], 'k--', linewidth=0.8)
        ax3.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r[0, 1]) +
                 '\nRMSE=' + '{:03.2f}'.format(error) +
                 '\nBias=' + '{:03.2f}'.format(bias), fontsize=8)
        ax3.set_aspect('equal', 'box')
        plt.tight_layout()
        plt.savefig(self.outpath + self.prefix + 'RFtruevsest_LOO_1step_combined.png', dpi=600)
        plt.close()

        # cdf and differernce plots
        cdf_y_tot = cdf(true_vect_tot)
        cdf_y_tot_pred = cdf(pred_vect_tot)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.5, 6), dpi=600)
        ax1.plot(cdf_y_tot[1][0:-1], cdf_y_tot[0], linewidth=1, color='k', label='ISMN')
        ax1.plot(cdf_y_tot_pred[1][0:-1], cdf_y_tot_pred[0], linewidth=0.8, color='k', linestyle='--', label='S1')
        ax1.set_ylabel('Cumulative frequency', size=8)
        ax1.set_xlim((0, 0.7))
        ax1.grid(b=True)
        ax1.legend(fontsize=8)
        ax1.tick_params(axis="y", labelsize=8)
        plt.tight_layout()

        ismn_sorted = np.argsort(true_vect_tot)
        y_tot_s = true_vect_tot[ismn_sorted]
        y_pred_tot_s = true_vect_tot[ismn_sorted]
        # ax2.plot(y_tot_s, y_pred_tot_s - y_tot_s, color='k', linestyle='', marker='*')
        ax2.plot(cdf_y_tot[1][0:-1], cdf_y_tot[0] - cdf_y_tot_pred[0], color='k', linewidth=0.8)
        ax2.set_ylabel('Cumulative frequency diff. (true - estimated)', size=8)
        ax2.set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=8)
        ax2.set_xlim((0, 0.7))
        ax2.set_ylim((-0.25, 0.25))
        ax2.grid(b=True)

        plt.tick_params(labelsize=8)
        plt.tight_layout()
        # plt.show()
        plt.savefig(self.outpath + self.prefix + 'CDF_DIFF_RF_1step_LOO.png', dpi=600)
        plt.close()

    def RF_loo_g0_1step(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.ensemble import ExtraTreesRegressor

        # filter bad ssm values
        valid = ~np.any(np.isinf(self.features3), axis=1)
        self.target3 = self.target3[valid]
        self.features3 = self.features3[valid, :]
        self.loc_id = self.loc_id[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features3), axis=1)
        self.target3 = self.target3[valid]
        self.features3 = self.features3[valid, :]
        self.loc_id = self.loc_id[valid]

        tree_grid = {  # 'bootstrap': [True, False],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [100, 200, 400, 600, 800, 1000]}

        true_vect_g0 = list()
        pred_vect_g0 = list()
        true_vect_ssm = list()
        pred_vect_ssm = list()
        ex_var_list_g0 = list()
        r2_list_g0 = list()
        rmse_list_g0 = list()
        ex_var_list_ssm = list()
        r2_list_ssm = list()
        rmse_list_ssm = list()

        from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.ensemble import HistGradientBoostingRegressor

        mlmodel = HistGradientBoostingRegressor(max_iter=500, verbose=False, random_state=42)

        for iid in np.unique(self.loc_id):
            print(iid)

            tmp_features_train2 = self.features3[self.loc_id != iid, :]
            tmp_target_train2 = self.target3[self.loc_id != iid]
            tmp_features_test2 = self.features3[self.loc_id == iid, :]
            tmp_target_test2 = self.target3[self.loc_id == iid]

            mlmodel.fit(tmp_features_train2, tmp_target_train2)

            # prediction of the average and relative sm
            tmp_pred2 = mlmodel.predict(tmp_features_test2)
            pred_vect_g0 += list(tmp_pred2)
            true_vect_g0 += list(tmp_target_test2)

            # collect average scores
            ex_var_list_g0.append(explained_variance_score(tmp_target_test2, tmp_pred2))
            r2_list_g0.append(r2_score(tmp_target_test2, tmp_pred2))
            rmse_list_g0.append(mean_squared_error(tmp_target_test2, tmp_pred2, squared=False))

            print('Explained variance: ' + str(explained_variance_score(tmp_target_test2, tmp_pred2)))
            print('R2: ' + str(r2_score(tmp_target_test2, tmp_pred2)))
            print('RMSE: ' + str(mean_squared_error(tmp_target_test2, tmp_pred2, squared=False)))

        # create the estimation models
        mlmodel3 = GradientBoostingRegressor(n_estimators=500,
                                             verbose=True,
                                             random_state=42)
        mlmodel3.fit(self.features3, self.target3)

        # estimate dry g0
        tmpfeatures3 = self.features3.copy()
        tmpfeatures3[:, 0] = 0
        g0dry = mlmodel3.predict(self.features3)
        target4 = self.features3[:, 0]
        features4 = np.vstack((self.target3 - g0dry,
                               self.features3[:, 1])).transpose()

        for iid in np.unique(self.loc_id):
            print(iid)

            tmp_features_train2 = features4[self.loc_id != iid, :]
            tmp_target_train2 = target4[self.loc_id != iid]
            tmp_features_test2 = features4[self.loc_id == iid, :]
            tmp_target_test2 = target4[self.loc_id == iid]

            mlmodel.fit(tmp_features_train2, tmp_target_train2)

            # prediction of the average and relative sm
            tmp_pred2 = mlmodel.predict(tmp_features_test2)
            pred_vect_ssm += list(tmp_pred2)
            true_vect_ssm += list(tmp_target_test2)

            # collect average scores
            ex_var_list_ssm.append(explained_variance_score(tmp_target_test2, tmp_pred2))
            r2_list_ssm.append(r2_score(tmp_target_test2, tmp_pred2))
            rmse_list_ssm.append(mean_squared_error(tmp_target_test2, tmp_pred2, squared=False))

            print('Explained variance: ' + str(explained_variance_score(tmp_target_test2, tmp_pred2)))
            print('R2: ' + str(r2_score(tmp_target_test2, tmp_pred2)))
            print('RMSE: ' + str(mean_squared_error(tmp_target_test2, tmp_pred2, squared=False)))

        # create the estimation models
        mlmodel4 = GradientBoostingRegressor(n_estimators=500,
                                             verbose=True,
                                             random_state=42)
        mlmodel4.fit(features4, target4)

        pickle.dump((None, None, mlmodel3, mlmodel4, None, None),
                    open(self.outpath + self.prefix + 'RFmlmodel_g0_1step.p', 'wb'))

        pred_vect_g0 = np.array(pred_vect_g0)
        true_vect_g0 = np.array(true_vect_g0)
        pred_vect_ssm = np.array(pred_vect_ssm)
        true_vect_ssm = np.array(true_vect_ssm)

        ex_var_list_g0 = np.array(ex_var_list_g0)
        r2_list_g0 = np.array(r2_list_g0)
        rmse_list_g0 = np.array(rmse_list_g0)
        ex_var_list_ssm = np.array(ex_var_list_ssm)
        r2_list_ssm = np.array(r2_list_ssm)
        rmse_list_ssm = np.array(rmse_list_ssm)

        r_g0 = np.corrcoef(true_vect_g0, pred_vect_g0)
        bias_g0 = np.mean(pred_vect_g0 - true_vect_g0)
        error_g0 = np.sqrt(
            np.sum(np.square(pred_vect_g0 - true_vect_g0)) / len(true_vect_g0))
        r_ssm = np.corrcoef(true_vect_ssm, pred_vect_ssm)
        bias_ssm = np.mean(pred_vect_ssm - true_vect_ssm)
        error_ssm = np.sqrt(
            np.sum(np.square(pred_vect_ssm - true_vect_ssm)) / len(true_vect_ssm))

        pltlims = 0

        # create plots
        fig, ax3 = plt.subplots(1, 1, sharex=False, figsize=(3.5, 3), dpi=600)

        ax3.scatter(true_vect_g0, pred_vect_g0, c='k', label='True vs Est', s=1, marker='*')
        ax3.set_xlim(-18, pltlims)
        ax3.set_ylim(-18, pltlims)
        ax3.set_xlabel("SIG0 [dB]", size=8)
        ax3.set_ylabel("SIG0 [dB]", size=8)
        ax3.set_title('c)')
        ax3.plot([-18, pltlims], [-18, pltlims], 'k--', linewidth=0.8)
        ax3.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r_g0[0, 1]) +
                 '\nRMSE=' + '{:03.2f}'.format(error_g0) +
                 '\nBias=' + '{:03.2f}'.format(bias_g0), fontsize=8)
        ax3.set_aspect('equal', 'box')
        plt.tight_layout()
        plt.savefig(self.outpath + self.prefix + 'RFtruevsest_g0_LOO_1step_combined.png', dpi=600)
        plt.close()

        pltlims = 1.0

        # create plots
        fig, ax3 = plt.subplots(1, 1, sharex=False, figsize=(3.5, 3), dpi=600)

        ax3.scatter(true_vect_ssm, pred_vect_ssm, c='k', label='True vs Est', s=1, marker='*')
        ax3.set_xlim(0, pltlims)
        ax3.set_ylim(0, pltlims)
        ax3.set_xlabel("SSM", size=8)
        ax3.set_ylabel("SSM", size=8)
        ax3.set_title('c)')
        ax3.plot([-18, pltlims], [-18, pltlims], 'k--', linewidth=0.8)
        ax3.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r_ssm[0, 1]) +
                 '\nRMSE=' + '{:03.2f}'.format(error_ssm) +
                 '\nBias=' + '{:03.2f}'.format(bias_ssm), fontsize=8)
        ax3.set_aspect('equal', 'box')
        plt.tight_layout()
        plt.savefig(self.outpath + self.prefix + 'RFtruevsest_ssm_LOO_1step_combined.png', dpi=600)
        plt.close()

        # cdf and differernce plots
        cdf_y_tot = cdf(true_vect_ssm)
        cdf_y_tot_pred = cdf(pred_vect_ssm)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.5, 6), dpi=600)
        ax1.plot(cdf_y_tot[1][0:-1], cdf_y_tot[0], linewidth=1, color='k', label='ISMN')
        ax1.plot(cdf_y_tot_pred[1][0:-1], cdf_y_tot_pred[0], linewidth=0.8, color='k', linestyle='--', label='S1')
        ax1.set_ylabel('Cumulative frequency', size=8)
        ax1.set_xlim((0, 0.7))
        ax1.grid(b=True)
        ax1.legend(fontsize=8)
        ax1.tick_params(axis="y", labelsize=8)
        plt.tight_layout()

        ismn_sorted = np.argsort(true_vect_ssm)
        y_tot_s = true_vect_ssm[ismn_sorted]
        y_pred_tot_s = true_vect_ssm[ismn_sorted]
        # ax2.plot(y_tot_s, y_pred_tot_s - y_tot_s, color='k', linestyle='', marker='*')
        ax2.plot(cdf_y_tot[1][0:-1], cdf_y_tot[0] - cdf_y_tot_pred[0], color='k', linewidth=0.8)
        ax2.set_ylabel('Cumulative frequency diff. (true - estimated)', size=8)
        ax2.set_xlabel("SIG0 [dB]", size=8)
        ax2.set_xlim((0, 0.7))
        ax2.set_ylim((-0.25, 0.25))
        ax2.grid(b=True)

        plt.tick_params(labelsize=8)
        plt.tight_layout()
        # plt.show()
        plt.savefig(self.outpath + self.prefix + 'CDF_DIFF_RF_ssm_1step_LOO.png', dpi=600)
        plt.close()

    def SVR_loo_1step(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error

        # filter bad ssm values
        valid = ~np.any(np.isinf(self.features2), axis=1)
        self.target2_2 = self.target2_2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features2), axis=1)
        self.target2_2 = self.target2_2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]

        # scaling
        scaler = sklearn.preprocessing.RobustScaler().fit(self.features2)
        x_train2 = scaler.transform(self.features2)

        y_train2 = self.target2_2

        init_c = [-2, 2]
        init_g = [-2, -0.5]
        init_e = [-2, -0.5]

        # for t_i in range(3):
        # SVR -- SVR -- SVR -- SVR -- SVR -- SVR
        # dictCV1 = dict(C=np.logspace(init_c[0], init_c[1], 3),
        #                gamma=np.logspace(init_g[0], init_g[1], 3),
        #                epsilon=np.logspace(init_e[0], init_e[1], 3),
        #                kernel=['rbf'])
        #
        # mlmodel2 = GridSearchCV(estimator=SVR(),
        #                         param_grid=dictCV1,
        #                         n_jobs=-1,
        #                         verbose=1,
        #                         cv=KFold(5, shuffle=True, random_state=42),
        #                         scoring='r2')
        mlmodel2 = SVR(C=1.0, epsilon=0.01, gamma=0.05623413251903491, kernel='rbf')

        true_vect1 = list()
        pred_vect1 = list()
        true_vect2 = list()
        pred_vect2 = list()
        true_vect_tot = list()
        pred_vect_tot = list()
        tree_list = list()
        ndvi_list = list()
        lc_list = list()
        ex_var_list = list()
        r2_list = list()
        rmse_list = list()

        for iid in np.unique(self.loc_id):
            print(iid)

            tmp_features_train2 = x_train2[self.loc_id != iid, :]
            tmp_target_train2 = self.target2_2[self.loc_id != iid]
            tmp_features_test2 = x_train2[self.loc_id == iid, :]
            tmp_target_test2 = self.target2_2[self.loc_id == iid]

            mlmodel2.fit(tmp_features_train2, tmp_target_train2)

            # prediction of the average and relative sm
            tmp_pred2 = mlmodel2.predict(tmp_features_test2)
            pred_vect2 += list(tmp_pred2)
            true_vect2 += list(tmp_target_test2)
            pred_vect_tot += list(tmp_pred2)
            true_vect_tot += list(tmp_target_test2)

            # collect ancillary data
            tree_list += list(self.sig0lia['trees'][self.loc_id == iid])
            ndvi_list += list(self.sig0lia['ndvi'][self.loc_id == iid])
            lc_list += list(self.sig0lia['lc'][self.loc_id == iid])

            # collect average scores
            ex_var_list.append(explained_variance_score(tmp_target_test2, tmp_pred2))
            r2_list.append(r2_score(tmp_target_test2, tmp_pred2))
            rmse_list.append(mean_squared_error(tmp_target_test2, tmp_pred2, squared=False))

            # print sscore of current iid
            print('Explained variance: ' + str(explained_variance_score(tmp_target_test2, tmp_pred2)))
            print('R2: ' + str(r2_score(tmp_target_test2, tmp_pred2)))
            print('RMSE: ' + str(mean_squared_error(tmp_target_test2, tmp_pred2, squared=False)))

        pred_vect2 = np.array(pred_vect2)
        true_vect2 = np.array(true_vect2)
        pred_vect_tot = np.array(pred_vect_tot)
        true_vect_tot = np.array(true_vect_tot)
        tree_list = np.array(tree_list)
        ndvi_list = np.array(ndvi_list)
        lc_list = np.array(lc_list)
        ex_var_list = np.array(ex_var_list)
        r2_list = np.array(r2_list)
        rmse_list = np.array(rmse_list)
        np.savez(self.outpath + 'loo_SVR_1step_scores.npz', ex_var_list, r2_list, rmse_list)
        np.savez(self.outpath + 'SVRloo_1stp_tmp.npz', pred_vect2, true_vect2, pred_vect_tot,
                 true_vect_tot)
        r = np.corrcoef(true_vect_tot, pred_vect_tot)
        urmse = np.sqrt(np.sum(np.square((pred_vect_tot - np.mean(pred_vect_tot)) - (
                true_vect_tot - np.mean(true_vect_tot)))) / len(true_vect_tot))
        bias = np.mean(pred_vect_tot - true_vect_tot)
        error = np.sqrt(
            np.sum(np.square(pred_vect_tot - true_vect_tot)) / len(true_vect_tot))

        pltlims = 1.0

        # create plots
        fig, ax3 = plt.subplots(1, 1, sharex=False, figsize=(3.5, 3), dpi=600)

        ax3.scatter(true_vect_tot, pred_vect_tot, c='k', label='True vs Est', s=1, marker='*')
        ax3.set_xlim(0, pltlims)
        ax3.set_ylim(0, pltlims)
        ax3.set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=8)
        ax3.set_ylabel("$SMC^*_{Tot}$ [m$^3$m$^{-3}$]", size=8)
        ax3.set_title('c)')
        ax3.plot([0, pltlims], [0, pltlims], 'k--', linewidth=0.8)
        ax3.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r[0, 1]) +
                 '\nRMSE=' + '{:03.2f}'.format(error) +
                 '\nBias=' + '{:03.2f}'.format(bias), fontsize=8)
        ax3.set_aspect('equal', 'box')
        plt.tight_layout()
        plt.savefig(self.outpath + self.prefix + 'SVRtruevsest_LOO_1step_combined.png', dpi=600)
        plt.close()

        # cdf and differernce plots
        cdf_y_tot = cdf(true_vect_tot)
        cdf_y_tot_pred = cdf(pred_vect_tot)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.5, 6), dpi=600)
        ax1.plot(cdf_y_tot[1][0:-1], cdf_y_tot[0], linewidth=1, color='k', label='ISMN')
        ax1.plot(cdf_y_tot_pred[1][0:-1], cdf_y_tot_pred[0], linewidth=0.8, color='k', linestyle='--', label='S1')
        ax1.set_ylabel('Cumulative frequency', size=8)
        ax1.set_xlim((0, 0.7))
        ax1.grid(b=True)
        ax1.legend(fontsize=8)
        ax1.tick_params(axis="y", labelsize=8)
        plt.tight_layout()

        ismn_sorted = np.argsort(true_vect_tot)
        y_tot_s = true_vect_tot[ismn_sorted]
        y_pred_tot_s = true_vect_tot[ismn_sorted]
        # ax2.plot(y_tot_s, y_pred_tot_s - y_tot_s, color='k', linestyle='', marker='*')
        ax2.plot(cdf_y_tot[1][0:-1], cdf_y_tot[0] - cdf_y_tot_pred[0], color='k', linewidth=0.8)
        ax2.set_ylabel('Cumulative frequency diff. (true - estimated)', size=8)
        ax2.set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=8)
        ax2.set_xlim((0, 0.7))
        ax2.set_ylim((-0.25, 0.25))
        ax2.grid(b=True)

        plt.tick_params(labelsize=8)
        plt.tight_layout()
        # plt.show()
        plt.savefig(self.outpath + self.prefix + 'CDF_DIFF_SVR_1step_LOO.png', dpi=600)
        plt.close()

    def SVR_loo(self):
        from sklearn.ensemble import GradientBoostingRegressor

        # filter nan values
        valid = ~np.any(np.isinf(self.features1), axis=1)
        self.target1 = self.target1[valid]
        self.features1 = self.features1[valid, :]
        self.sub_loc = self.sub_loc[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features1), axis=1)
        self.target1 = self.target1[valid]
        self.features1 = self.features1[valid, :]
        self.sub_loc = self.sub_loc[valid]

        # filter bad ssm values
        valid = ~np.any(np.isinf(self.features2), axis=1)
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features2), axis=1)
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]

        # scaling
        scaler1 = sklearn.preprocessing.RobustScaler().fit(self.features1)
        x_train1 = scaler1.transform(self.features1)
        scaler2 = sklearn.preprocessing.RobustScaler().fit(self.features2)
        x_train2 = scaler2.transform(self.features2)

        y_train1 = self.target1
        y_train2 = self.target2

        init_c = [-2, 2]
        init_g = [-2, -0.5]
        init_e = [-2, -0.5]

        # find the best parameters for the two models
        dictCV = dict(C=np.logspace(init_c[0], init_c[1], 3),
                       gamma=np.logspace(init_g[0], init_g[1], 3),
                       epsilon=np.logspace(init_e[0], init_e[1], 3),
                       kernel=['rbf'])

        pmsearch1 = GridSearchCV(estimator=SVR(),
                                param_grid=dictCV,
                                n_jobs=-1,
                                verbose=1,
                                cv=KFold(5, shuffle=True, random_state=42),
                                scoring='r2')
        pmsearch2 = GridSearchCV(estimator=SVR(),
                                 param_grid=dictCV,
                                 n_jobs=-1,
                                 verbose=1,
                                 cv=GroupKFold(),
                                 scoring='r2')
        pmsearch1.fit(x_train1, y_train1)
        pmsearch2.fit(x_train2, y_train2, groups=self.loc_id)

        mlmodel1 = SVR(C=pmsearch1.best_params_['C'],
                       epsilon=pmsearch1.best_params_['epsilon'],
                       gamma=pmsearch1.best_params_['gamma'],
                       kernel='rbf')
        mlmodel2 = SVR(C=pmsearch2.best_params_['C'],
                       epsilon=pmsearch2.best_params_['epsilon'],
                       gamma=pmsearch2.best_params_['gamma'],
                       kernel='rbf')

        true_vect1 = list()
        pred_vect1 = list()
        true_vect2 = list()
        pred_vect2 = list()
        true_vect_tot = list()
        pred_vect_tot = list()
        tree_list = list()
        ndvi_list = list()
        lc_list = list()
        ex_var_list = list()
        r2_list = list()
        rmse_list = list()

        for iid in np.unique(self.loc_id):
            print(iid)
            tmp_features_train1 = self.features1[self.sub_loc != iid, :]
            tmp_target_train1 = self.target1[self.sub_loc != iid]
            tmp_features_test1 = self.features1[self.sub_loc == iid, :]
            tmp_target_test1 = self.target1[self.sub_loc == iid]

            tmp_features_train2 = self.features2[self.loc_id != iid, :]
            tmp_target_train2 = self.target2[self.loc_id != iid]
            tmp_features_test2 = self.features2[self.loc_id == iid, :]
            tmp_target_test2 = self.target2[self.loc_id == iid]

            mlmodel1.fit(tmp_features_train1, tmp_target_train1)  # , tmp_weights1)
            mlmodel2.fit(tmp_features_train2, tmp_target_train2)  # , tmp_weights2)

            # prediction of the average and relative sm
            tmp_pred1 = mlmodel1.predict(tmp_features_test1)
            pred_vect1 += list(tmp_pred1)
            true_vect1 += list(tmp_target_test1)
            tmp_pred2 = mlmodel2.predict(tmp_features_test2)
            pred_vect2 += list(tmp_pred2)
            true_vect2 += list(tmp_target_test2)
            pred_vect_tot += list(tmp_pred1 + tmp_pred2)
            true_vect_tot += list(tmp_target_test1 + tmp_target_test2)

            # collect ancillary data
            tree_list += list(self.sig0lia['trees'][self.sig0lia['locid'] == iid].values)
            ndvi_list += list(self.sig0lia['ndvi'][self.sig0lia['locid'] == iid].values)
            lc_list += list(self.sig0lia['lc'][self.sig0lia['locid'] == iid].values)

            # collect average scores
            ex_var_list.append(explained_variance_score(tmp_target_test2, tmp_pred2))
            r2_list.append(r2_score(tmp_target_test2, tmp_pred2))
            rmse_list.append(mean_squared_error(tmp_target_test2, tmp_pred2, squared=False))

            print('Explained variance: ' + str(explained_variance_score(tmp_target_test2, tmp_pred2)))
            print('R2: ' + str(r2_score(tmp_target_test2, tmp_pred2)))
            print('RMSE: ' + str(mean_squared_error(tmp_target_test2, tmp_pred2, squared=False)))

        pred_vect1 = np.array(pred_vect1)
        true_vect1 = np.array(true_vect1)
        pred_vect2 = np.array(pred_vect2)
        true_vect2 = np.array(true_vect2)
        pred_vect_tot = np.array(pred_vect_tot)
        true_vect_tot = np.array(true_vect_tot)
        tree_list = np.array(tree_list)
        ndvi_list = np.array(ndvi_list)
        lc_list = np.array(lc_list)
        ex_var_list = np.array(ex_var_list)
        r2_list = np.array(r2_list)
        rmse_list = np.array(rmse_list)
        np.savez(self.outpath + 'loo_tmp.npz', pred_vect1, true_vect1, pred_vect2, true_vect2, pred_vect_tot,
                 true_vect_tot)
        np.savez(self.outpath + 'loo_SVR_2step_scores.npz', ex_var_list, r2_list, rmse_list)
        r = np.corrcoef(true_vect_tot, pred_vect_tot)
        urmse = np.sqrt(np.sum(np.square((pred_vect_tot - np.mean(pred_vect_tot)) - (
                true_vect_tot - np.mean(true_vect_tot)))) / len(true_vect_tot))
        bias = np.mean(pred_vect_tot - true_vect_tot)
        error = np.sqrt(
            np.sum(np.square(pred_vect_tot - true_vect_tot)) / len(true_vect_tot))

        r_avg = np.corrcoef(true_vect1, pred_vect1)
        r_rel = np.corrcoef(true_vect2, pred_vect2)
        urmse_avg = np.sqrt(np.sum(np.square((pred_vect1 - np.mean(pred_vect1)) - (
                true_vect1 - np.mean(true_vect1)))) / len(true_vect1))
        urmse_rel = np.sqrt(np.sum(np.square((pred_vect2 - np.mean(pred_vect2)) - (
                true_vect2 - np.mean(true_vect2)))) / len(true_vect2))
        rmse_avg = np.sqrt(np.sum(np.square(pred_vect1 - true_vect1)) / len(true_vect1))
        rmse_rel = np.sqrt(np.sum(np.square(pred_vect2 - true_vect2)) / len(true_vect2))
        bias_avg = np.mean(pred_vect1 - true_vect1)
        bias_rel = np.mean(pred_vect2 - true_vect2)

        pltlims = 1.0

        # create plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(3.5, 9), dpi=600)
        # plt.figure(figsize=(3.5, 3)))
        ax1.scatter(true_vect1, pred_vect1, c='k', label='True vs Est', s=1, marker='*')
        ax1.set_xlim(0, pltlims)
        ax1.set_ylim(0, pltlims)
        ax1.set_xlabel("$SMC_{Avg}$ [m$^3$m$^{-3}$]", size=8)
        ax1.set_ylabel("$SMC^*_{Avg}$ [m$^3$m$^{-3}$]", size=8)
        ax1.set_title('a)')
        ax1.plot([0, pltlims], [0, pltlims], 'k--', linewidth=0.8)
        ax1.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r_avg[0, 1]) +
                 '\nRMSE=' + '{:03.2f}'.format(rmse_avg) +
                 '\nBias=' + '{:03.2f}'.format(bias_avg), fontsize=8)
        ax1.set_aspect('equal', 'box')

        ax2.scatter(true_vect2, pred_vect2, c='k', label='True vs Est', s=1, marker='*')
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xlabel("$SMC_{Rel}$ [m$^3$m$^{-3}$]", size=8)
        ax2.set_ylabel("$SMC^*_{Rel}$ [m$^3$m$^{-3}$]", size=8)
        ax2.set_title('b)')
        ax2.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', linewidth=0.8)
        ax2.text(-0.15, 0.12, 'R=' + '{:03.2f}'.format(r_rel[0, 1]) +
                 '\nRMSE=' + '{:03.2f}'.format(rmse_rel) +
                 '\nBias=' + '{:03.2f}'.format(bias_rel), fontsize=8)
        ax2.set_aspect('equal', 'box')

        ax3.scatter(true_vect_tot, pred_vect_tot, c='k', label='True vs Est', s=1, marker='*')
        ax3.set_xlim(0, pltlims)
        ax3.set_ylim(0, pltlims)
        ax3.set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=8)
        ax3.set_ylabel("$SMC^*_{Tot}$ [m$^3$m$^{-3}$]", size=8)
        ax3.set_title('c)')
        ax3.plot([0, pltlims], [0, pltlims], 'k--', linewidth=0.8)
        ax3.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r[0, 1]) +
                 '\nRMSE=' + '{:03.2f}'.format(error) +
                 '\nBias=' + '{:03.2f}'.format(bias), fontsize=8)
        ax3.set_aspect('equal', 'box')
        plt.tight_layout()
        plt.savefig(self.outpath + self.prefix + 'SVRtruevsest_LOO_2step_combined.png', dpi=600)
        plt.close()

        # cdf and differernce plots
        cdf_y_tot = cdf(true_vect_tot)
        cdf_y_tot_pred = cdf(pred_vect_tot)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.5, 6), dpi=600)
        ax1.plot(cdf_y_tot[1][0:-1], cdf_y_tot[0], linewidth=1, color='k', label='ISMN')
        ax1.plot(cdf_y_tot_pred[1][0:-1], cdf_y_tot_pred[0], linewidth=0.8, color='k', linestyle='--', label='S1')
        ax1.set_ylabel('Cumulative frequency', size=8)
        ax1.set_xlim((0, 0.7))
        ax1.grid(b=True)
        ax1.legend(fontsize=8)
        ax1.tick_params(axis="y", labelsize=8)
        plt.tight_layout()

        ismn_sorted = np.argsort(true_vect_tot)
        y_tot_s = true_vect_tot[ismn_sorted]
        y_pred_tot_s = true_vect_tot[ismn_sorted]
        # ax2.plot(y_tot_s, y_pred_tot_s - y_tot_s, color='k', linestyle='', marker='*')
        ax2.plot(cdf_y_tot[1][0:-1], cdf_y_tot[0] - cdf_y_tot_pred[0], color='k', linewidth=0.8)
        ax2.set_ylabel('Cumulative frequency diff. (true - estimated)', size=8)
        ax2.set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=8)
        ax2.set_xlim((0, 0.7))
        ax2.set_ylim((-0.25, 0.25))
        ax2.grid(b=True)

        plt.tick_params(labelsize=8)
        plt.tight_layout()
        plt.savefig(self.outpath + self.prefix + 'CDF_DIFF_SVR_LOO_2step.png', dpi=600)
        plt.close()

    def optimize_rf(self):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GroupKFold

        # filter bad ssm values
        valid = np.where((self.target1 > 0) & (self.target1 < 0.9))  # & (self.features[:,-1] == itrack))
        self.target1 = self.target1[valid[0]]
        self.features1 = self.features1[valid[0], :]
        # filter nan values
        valid = ~np.any(np.isinf(self.features1), axis=1)
        self.target1 = self.target1[valid]
        self.features1 = self.features1[valid, :]
        # filter nan
        valid = ~np.any(np.isnan(self.features1), axis=1)
        self.target1 = self.target1[valid]
        self.features1 = self.features1[valid, :]

        # filter bad ssm values
        valid = ~np.any(np.isinf(self.features2), axis=1)
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features2), axis=1)
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]

        x_train1 = self.features1
        y_train1 = self.target1

        x_train2 = self.features2
        y_train2 = self.target2

        testtarget1 = y_train1
        testfeatures1 = x_train1
        testtarget2 = y_train2
        testfeatures2 = x_train2

        from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.ensemble import HistGradientBoostingRegressor

        mlmodel1 = GradientBoostingRegressor(n_estimators=500,
                                             verbose=False,
                                             random_state=42)

        mlmodel2 = GradientBoostingRegressor(n_estimators=500,
                                             verbose=False,
                                             random_state=42)

        histmodel = HistGradientBoostingRegressor(max_iter=500, verbose=False, random_state=42)

        r1list = list()
        r2list = list()
        urmse1list = list()
        urmse2list = list()
        rmse1list = list()
        rmse2list = list()

        f = open(self.outpath + self.prefix + 'tuning_report.txt', 'w+')
        f.write('FEATURES2: \n')

        if self.prefix == 'w_GLDAS_':
            nf2 = 65
        else:
            nf2 = 61

        featuresidx2 = list(range(nf2))
        for i in range(nf2):
            mlmodel2.fit(x_train2[:, featuresidx2], y_train2)

            scores = cross_val_score(histmodel, x_train2[:, featuresidx2], y_train2, groups=self.loc_id,
                                     scoring='r2', cv=GroupKFold(), n_jobs=-1, verbose=True)
            print("Relative SMC Cor: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
            r = scores.mean()
            urmse = scores.mean()
            error = scores.mean()

            r2list.append(r)
            urmse2list.append(urmse)
            rmse2list.append(error)

            f.write('ubRMSE: ' + str(urmse))
            f.write('; RMSE: ' + str(error))
            f.write('; R: ' + str(r))
            f.write(';   ' + ','.join(map(str, featuresidx2)) + '\n')

            del featuresidx2[np.argmin(mlmodel2.feature_importances_)]

        f.write('FEATURES1: \n')

        if self.prefix == 'w_GLDAS_':
            nf1 = 48
        else:
            nf1 = 46

        featuresidx1 = list(range(nf1))
        for i in range(nf1):
            mlmodel1.fit(x_train1[:, featuresidx1], y_train1)
            # # prediction of the average sm

            scores = cross_val_score(histmodel, x_train1[:, featuresidx1], y_train1,
                                     scoring='r2', cv=KFold(random_state=42), n_jobs=-1, verbose=True)
            print("Average SMC Cor: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            r = scores.mean()
            urmse = scores.mean()
            error = scores.mean()

            r1list.append(r)
            urmse1list.append(urmse)
            rmse1list.append(error)

            f.write('ubRMSE: ' + str(urmse))
            f.write('; RMSE: ' + str(error))
            f.write('; R: ' + str(r))
            f.write(';   ' + ','.join(map(str, featuresidx1)) + '\n')

            del featuresidx1[np.argmin(mlmodel1.feature_importances_)]

        f.close()
        return mlmodel1, None, mlmodel2, None, None, None

    def optimize_rf_1step(self):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GroupKFold

        # filter bad ssm values
        valid = ~np.any(np.isinf(self.features2), axis=1)
        self.target2_2 = self.target2_2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features2), axis=1)
        self.target2_2 = self.target2_2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]

        x_train2 = self.features2
        y_train2 = self.target2_2

        from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.ensemble import HistGradientBoostingRegressor

        mlmodel2 = GradientBoostingRegressor(n_estimators=500,
                                             verbose=False,
                                             random_state=42)

        histmodel = HistGradientBoostingRegressor(max_iter=500,
                                                  verbose=False,
                                                  random_state=42)

        r1list = list()
        r2list = list()
        urmse1list = list()
        urmse2list = list()
        rmse1list = list()
        rmse2list = list()

        f = open(self.outpath + self.prefix + 'tuning_report_1step.txt', 'w+')
        f.write('FEATURES2: \n')

        if self.prefix == 'w_GLDAS_':
            nf2 = 65
        else:
            nf2 = 61

        featuresidx2 = list(range(nf2))
        for i in range(nf2):
            scores = cross_val_score(histmodel, x_train2[:, featuresidx2], y_train2, groups=self.loc_id,
                                     scoring='r2', cv=GroupKFold(), n_jobs=-1, verbose=True)
            print("Relative SMC Cor: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
            r = scores.mean()
            urmse = scores.mean()
            error = scores.mean()

            r2list.append(r)
            urmse2list.append(urmse)
            rmse2list.append(error)

            f.write('ubRMSE: ' + str(urmse))
            f.write('; RMSE: ' + str(error))
            f.write('; R: ' + str(r))
            f.write(';   ' + ','.join(map(str, featuresidx2)) + '\n')

            mlmodel2.fit(x_train2[:, featuresidx2], y_train2)

            del featuresidx2[np.argmin(mlmodel2.feature_importances_)]

        f.close()

    def optimize_rf_g0est_1step(self):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GroupKFold

        # filter bad ssm values
        valid = ~np.any(np.isinf(self.features3), axis=1)
        self.target3 = self.target3[valid]
        self.features3 = self.features3[valid, :]
        self.loc_id = self.loc_id[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features3), axis=1)
        self.target3 = self.target3[valid]
        self.features3 = self.features3[valid, :]
        self.loc_id = self.loc_id[valid]

        x_train = self.features3
        y_train = self.target3

        from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.ensemble import HistGradientBoostingRegressor

        mlmodel = GradientBoostingRegressor(n_estimators=500,
                                             verbose=False,
                                             random_state=42)

        histmodel = HistGradientBoostingRegressor(max_iter=500,
                                                  verbose=False,
                                                  random_state=42)

        r1list = list()
        r2list = list()
        urmse1list = list()
        urmse2list = list()
        rmse1list = list()
        rmse2list = list()

        f = open(self.outpath + self.prefix + 'tuning_reportg0_1step.txt', 'w+')
        f.write('FEATURES2: \n')

        if self.prefix == 'w_GLDAS_':
            nf2 = 23
        else:
            nf2 = 23

        featuresidx = list(range(nf2))
        for i in range(nf2):
            scores = cross_val_score(histmodel, x_train[:, featuresidx], y_train, groups=self.loc_id,
                                     scoring='r2', cv=GroupKFold(), n_jobs=-1, verbose=True)
            print("Relative SMC Cor: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
            r = scores.mean()
            urmse = scores.mean()
            error = scores.mean()

            r2list.append(r)
            urmse2list.append(urmse)
            rmse2list.append(error)

            f.write('ubRMSE: ' + str(urmse))
            f.write('; RMSE: ' + str(error))
            f.write('; R: ' + str(r))
            f.write(';   ' + ','.join(map(str, featuresidx)) + '\n')

            mlmodel.fit(x_train[:, featuresidx], y_train)

            del featuresidx[np.argmin(mlmodel.feature_importances_)]

        f.close()

    def optimize_SVR(self):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GroupKFold

        # filter bad ssm values
        valid = np.where((self.target1 > 0) & (self.target1 < 0.9))  # & (self.features[:,-1] == itrack))
        self.target1 = self.target1[valid[0]]
        self.features1 = self.features1[valid[0], :]
        # filter nan values
        valid = ~np.any(np.isinf(self.features1), axis=1)
        self.target1 = self.target1[valid]
        self.features1 = self.features1[valid, :]
        # filter nan
        valid = ~np.any(np.isnan(self.features1), axis=1)
        self.target1 = self.target1[valid]
        self.features1 = self.features1[valid, :]

        # filter bad ssm values
        valid = ~np.any(np.isinf(self.features2), axis=1)
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features2), axis=1)
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]

        # scaling
        scaler1 = sklearn.preprocessing.RobustScaler().fit(self.features1)
        scaler2 = sklearn.preprocessing.RobustScaler().fit(self.features2)
        x_train1 = scaler1.transform(self.features1)
        x_train2 = scaler2.transform(self.features2)

        y_train1 = self.target1
        y_train2 = self.target2

        init_c = [-2, 2]
        init_g = [-2, -0.5]
        init_e = [-2, -0.5]
        init_nu = [-2, 0]

        # for t_i in range(3):
        # SVR -- SVR -- SVR -- SVR -- SVR -- SVR
        dictCV = dict(C=np.logspace(init_c[0], init_c[1], 3),
                       gamma=np.logspace(init_g[0], init_g[1], 3),
                       epsilon=np.logspace(init_e[0], init_e[1], 3),
                       kernel=['rbf'])

        mlmodel1 = GridSearchCV(estimator=SVR(),
                                param_grid=dictCV,
                                n_jobs=-1,
                                verbose=1,
                                cv=KFold(),
                                scoring='r2')

        mlmodel2 = GridSearchCV(estimator=SVR(),
                                param_grid=dictCV,
                                n_jobs=-1,
                                verbose=1,
                                cv=GroupKFold(),
                                scoring='r2')  # ,

        r1list = list()
        r2list = list()
        urmse1list = list()
        urmse2list = list()
        rmse1list = list()
        rmse2list = list()

        f = open(self.outpath + self.prefix + 'tuning_report_SVR.txt', 'w+')
        f.write('FEATURES2: \n')

        if self.prefix == 'w_GLDAS_':
            nf2 = 65
        else:
            nf2 = 61

        # calculate mutual info
        from sklearn.feature_selection import mutual_info_regression
        mi2 = list(mutual_info_regression(x_train2[:,range(nf2)], y_train2))

        featuresidx2 = list(range(nf2))
        for i in range(nf2):
            mlmodel2.fit(x_train2[:, featuresidx2], y_train2, groups=self.loc_id)
            scores = mlmodel2.best_score_
            print("Relative SMC Cor: %0.3f " % mlmodel2.best_score_)
            r = mlmodel2.best_score_
            urmse = mlmodel2.best_score_
            error = mlmodel2.best_score_

            r2list.append(r)
            urmse2list.append(urmse)
            rmse2list.append(error)

            f.write('ubRMSE: ' + str(urmse))
            f.write('; RMSE: ' + str(error))
            f.write('; R: ' + str(r))
            f.write(';   ' + ','.join(map(str, featuresidx2)) + '\n')

            del featuresidx2[np.argmin(mi2)]
            del mi2[np.argmin(mi2)]

        f.write('FEATURES1: \n')

        if self.prefix == 'w_GLDAS_':
            nf1 = 48
        else:
            nf1 = 46

        # calculate mutual info
        mi1 = list(mutual_info_regression(x_train1[:,range(nf1)], y_train1))

        featuresidx1 = list(range(nf1))
        for i in range(nf1):
            mlmodel1.fit(x_train1[:, featuresidx1], y_train1)
            scores = mlmodel1.best_score_
            print("Relative SMC Cor: %0.3f " % mlmodel1.best_score_)
            r = mlmodel1.best_score_
            urmse = mlmodel1.best_score_
            error = mlmodel1.best_score_

            r1list.append(r)
            urmse1list.append(urmse)
            rmse1list.append(error)

            f.write('ubRMSE: ' + str(urmse))
            f.write('; RMSE: ' + str(error))
            f.write('; R: ' + str(r))
            f.write(';   ' + ','.join(map(str, featuresidx1)) + '\n')

            del featuresidx1[np.argmin(mi1)]
            del mi1[np.argmin(mi1)]

        f.close()
        return mlmodel1, None, mlmodel2, None, None, None

    def optimize_SVR_1step(self):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.model_selection import cross_val_score

        # filter bad ssm values
        valid = ~np.any(np.isinf(self.features2), axis=1)
        self.target2_2 = self.target2_2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features2), axis=1)
        self.target2_2 = self.target2_2[valid]
        self.features2 = self.features2[valid, :]
        self.loc_id = self.loc_id[valid]

        # scaling
        scaler = sklearn.preprocessing.RobustScaler().fit(self.features2)
        x_train2 = scaler.transform(self.features2)

        #x_train2 = self.features2
        y_train2 = self.target2_2

        init_c = [-2, 2]
        init_g = [-2, -0.5]
        init_e = [-2, -0.5]
        init_nu = [-2, 0]

        # for t_i in range(3):
        # SVR -- SVR -- SVR -- SVR -- SVR -- SVR
        dictCV1 = dict(C=np.logspace(init_c[0], init_c[1], 3),
                       gamma=np.logspace(init_g[0], init_g[1], 3),
                       epsilon=np.logspace(init_e[0], init_e[1], 3),
                       kernel=['rbf'])

        mlmodel2 = GridSearchCV(estimator=SVR(),
                                param_grid=dictCV1,
                                n_jobs=-1,
                                verbose=1,
                                # pre_dispatch='2*n_jobs',
                                # pre_dispatch='all',
                                # cv=ShuffleSplit(n_splits=10, random_state=42),
                                cv=KFold(5, shuffle=True, random_state=42),
                                # cv=LeaveOneOut(),
                                scoring='r2')  # ,

        r1list = list()
        r2list = list()
        urmse1list = list()
        urmse2list = list()
        rmse1list = list()
        rmse2list = list()

        f = open(self.outpath + self.prefix + 'tuning_report_SVR_1step.txt', 'w+')
        f.write('FEATURES2: \n')

        if self.prefix == 'w_GLDAS_':
            nf2 = 65
        else:
            nf2 = 61

        # calculate mutual info
        from sklearn.feature_selection import mutual_info_regression
        mi = list(mutual_info_regression(x_train2[:, range(nf2)], y_train2))

        featuresidx2 = list(range(nf2))
        for i in range(nf2):
            mlmodel2.fit(x_train2[:, featuresidx2], y_train2, groups=self.loc_id)

            #scores = cross_val_score(histmodel, x_train2[:, featuresidx2], y_train2, groups=self.loc_id,
            #                         scoring='neg_mean_squared_error', cv=LeaveOneGroupOut(), n_jobs=8,
            #                         verbose=True)
            scores = mlmodel2.best_score_
            print("Relative SMC Cor: %0.3f " % mlmodel2.best_score_)
            r = mlmodel2.best_score_
            urmse = mlmodel2.best_score_
            error = mlmodel2.best_score_

            r2list.append(r)
            urmse2list.append(urmse)
            rmse2list.append(error)

            f.write('ubRMSE: ' + str(urmse))
            f.write('; RMSE: ' + str(error))
            f.write('; R: ' + str(r))
            f.write(';   ' + ','.join(map(str, featuresidx2)) + '\n')

            del featuresidx2[np.argmin(mi)]
            del mi[np.argmin(mi)]

        f.close()

    def RF_plot_deviance(self):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error

        # model 1
        X1_train, X1_test, y1_train, y1_test = train_test_split(self.features1, self.target1, test_size=0.1,
                                                                random_state=42)
        X2_train, X2_test, y2_train, y2_test = train_test_split(self.features2, self.target2, test_size=0.1,
                                                                random_state=42)

        mlmodel1 = GradientBoostingRegressor(n_estimators=1000,
                                             verbose=False,
                                             random_state=42)

        mlmodel2 = GradientBoostingRegressor(n_estimators=1000,
                                             verbose=False,
                                             random_state=42)

        mlmodel1.fit(X1_train, y1_train)
        mlmodel2.fit(X2_train, y2_train)
        mse1 = mean_squared_error(y1_test, mlmodel1.predict(X1_test))
        mse2 = mean_squared_error(y2_test, mlmodel2.predict(X2_test))

        print("MSE1: %.4f" % mse1)
        print("MSE1: %.4f" % mse2)

        # compute test set deviance
        test_score1 = np.zeros((1000,), dtype=np.float64)
        test_score2 = np.zeros((1000,), dtype=np.float64)

        for i, y1_pred in enumerate(mlmodel1.staged_predict(X1_test)):
            test_score1[i] = mlmodel1.loss_(y1_test, y1_pred)
        for i, y2_pred in enumerate(mlmodel2.staged_predict(X2_test)):
            test_score2[i] = mlmodel1.loss_(y2_test, y2_pred)

        # create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.5, 6), dpi=600)
        ax1.set_title('Deviance $SMC_{Avg}$')
        ax1.plot(np.arange(1000) + 1, mlmodel1.train_score_, 'b-', label='Training Set Deviance')
        ax1.plot(np.arange(1000) + 1, test_score1, 'r-', label='Test Set Deviance')
        ax1.legend(loc='upper right')
        ax1.set_ylabel('Deviance')
        ax2.set_title('Deviance $SMC_{Rel}$')
        ax2.plot(np.arange(1000) + 1, mlmodel2.train_score_, 'b-', label='Training Set Deviance')
        ax2.plot(np.arange(1000) + 1, test_score2, 'r-', label='Test Set Deviance')
        ax2.set_xlabel('Boosting iterations')
        ax2.set_ylabel('Deviance')
        plt.tight_layout()
        plt.savefig(self.outpath + self.prefix + 'boosting.png')

    def train_model_fw_bw_rf(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.ensemble import ExtraTreesRegressor

        # filter bad ssm values
        valid = np.where((self.target > 0) and (self.target < 0.8))  # & (self.features[:,-1] == itrack))
        self.target = self.target[valid[0]]
        self.features = self.features[valid[0], :]
        self.s0vvtarget = self.s0vvtarget[valid[0]]
        self.s0vhtarget = self.s0vhtarget[valid[0]]
        self.s0features = self.s0features[valid[0], :]
        self.target2 = self.target2[valid[0]]
        self.features2 = self.features2[valid[0], :]
        # filter nan values
        valid = ~np.any(np.isinf(self.s0features) | np.isnan(self.s0features), axis=1)
        self.s0vvtarget = self.s0vvtarget[valid]
        self.s0vhtarget = self.s0vhtarget[valid]
        self.s0features = self.s0features[valid, :]
        self.target = self.target[valid]
        self.features = self.features[valid, :]
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]
        # filter nan
        valid = ~np.any(np.isinf(self.features) | np.isnan(self.features), axis=1)
        self.target = self.target[valid]
        self.features = self.features[valid, :]
        self.weights = self.weights[valid]
        self.target = self.target[valid]
        self.features = self.features[valid, :]
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]

        # scaling
        s0features_scaler = sklearn.preprocessing.RobustScaler().fit(self.s0features)
        features_scaler = sklearn.preprocessing.RobustScaler().fit(self.features)

        x_s0train, x_s0test, \
        y_s0vvtrain, y_s0vvtest, \
        y_s0vhtrain, y_s0vhtest, \
        x_train, x_test, \
        y_train, y_test, \
        x_train2, x_test2, \
        y_train2, y_test2 = train_test_split(self.s0features,
                                             self.s0vvtarget,
                                             self.s0vhtarget,
                                             self.features,
                                             self.target,
                                             self.features2,
                                             self.target2,
                                             test_size=0.2,
                                             train_size=0.8,
                                             random_state=70)
        init_c = [-2, 2]
        init_g = [-2, -0.5]
        init_e = [-2, -0.5]
        init_nu = [-2, 0]

        dictCV = dict(C=np.logspace(init_c[0], init_c[1], 4),
                      gamma=np.logspace(init_g[0], init_g[1], 4),
                      epsilon=np.logspace(init_e[0], init_e[1], 4),
                      kernel=['rbf'])
        dictCV_lin = dict(C=np.logspace(init_c[0], init_c[1], 4),
                          epsilon=np.logspace(init_e[0], init_e[1], 4),
                          kernel=['linear'])
        # mlmodel_s0vv = GridSearchCV(estimator=SVR(),
        #                        param_grid=dictCV,
        #                        n_jobs=4,
        #                        verbose=1,
        #                        cv=KFold(n_splits=5, random_state=42))
        # mlmodel_s0vh = GridSearchCV(estimator=SVR(),
        #                             param_grid=dictCV,
        #                             n_jobs=4,
        #                             verbose=1,
        #                             cv=KFold(n_splits=5, random_state=42))
        # mlmodel_ssm = GridSearchCV(estimator=SVR(),
        #                             param_grid=dictCV,
        #                             n_jobs=4,
        #                             verbose=1,
        #                             cv=KFold(n_splits=5, random_state=42),
        #                             scoring='r2')

        mlmodel_s0vv = GradientBoostingRegressor(n_estimators=1000,
                                                 # oob_score=True,
                                                 verbose=True,
                                                 random_state=42)
        mlmodel_s0vh = GradientBoostingRegressor(n_estimators=1000,
                                                 verbose=True,
                                                 random_state=42)
        #
        mlmodel_ssm = GradientBoostingRegressor(n_estimators=1000,
                                                verbose=True,
                                                random_state=42)

        from sklearn.utils import parallel_backend
        with parallel_backend('multiprocessing'):
            mlmodel_s0vv.fit(s0features_scaler.transform(x_s0train), y_s0vvtrain)
            mlmodel_s0vh.fit(s0features_scaler.transform(x_s0train), y_s0vhtrain)

        s0vv_pred = mlmodel_s0vv.predict(s0features_scaler.transform(x_s0test))
        s0vh_pred = mlmodel_s0vh.predict(s0features_scaler.transform(x_s0test))

        x_s0train_avg = np.copy(x_s0train)
        x_s0train_avg[:, 0] = x_train2[:, 0]
        x_s0train_avg[:, 2:11] = x_train
        x_s0train_avg[:, 11] = 0

        s0vv_avg = mlmodel_s0vv.predict(s0features_scaler.transform(x_s0train_avg))
        s0vh_avg = mlmodel_s0vh.predict(s0features_scaler.transform(x_s0train_avg))

        valid = (x_train2[:, 1] - s0vv_avg >= 0) & (x_train2[:, 2] - s0vh_avg >= 0)

        x_train2 = x_train2[valid, 1::]
        x_train2[:, 0] = 10 * np.log10(x_train2[:, 0] - s0vv_avg[valid])
        x_train2[:, 1] = 10 * np.log10(x_train2[:, 1] - s0vh_avg[valid])
        y_train2 = y_train2[valid]

        feature2_scaler = sklearn.preprocessing.RobustScaler().fit(x_train2)
        with parallel_backend('multiprocessing'):
            mlmodel_ssm.fit(feature2_scaler.transform(x_train2), y_train2)

        x_s0test_avg = np.copy(x_s0test)
        x_s0test_avg[:, 0] = x_test2[:, 0]
        x_s0test_avg[:, 2:11] = x_test
        x_s0test_avg[:, 11] = 0

        s0vv_avg = mlmodel_s0vv.predict(s0features_scaler.transform(x_s0test_avg))
        s0vh_avg = mlmodel_s0vh.predict(s0features_scaler.transform(x_s0test_avg))

        valid = (x_test2[:, 1] - s0vv_avg >= 0) & (x_test2[:, 2] - s0vh_avg >= 0)

        x_test2 = x_test2[valid, 1::]
        x_test2[:, 0] = 10 * np.log10(x_test2[:, 0] - s0vv_avg[valid])
        x_test2[:, 1] = 10 * np.log10(x_test2[:, 1] - s0vh_avg[valid])
        y_test2 = y_test2[valid]

        ssm_pred = mlmodel_ssm.predict(feature2_scaler.transform(x_test2))

        # compute performance metrics
        r_vv = np.corrcoef(y_s0vvtest, s0vv_pred)
        urmse_vv = np.sqrt(
            np.sum(np.square((s0vv_pred - np.mean(s0vv_pred)) - (y_s0vvtest - np.mean(y_s0vvtest)))) / len(y_s0vvtest))
        bias_vv = np.mean(s0vv_pred - y_s0vvtest)
        error_vv = np.sqrt(np.sum(np.square(s0vv_pred - y_s0vvtest)) / len(y_s0vvtest))

        r_vh = np.corrcoef(y_s0vhtest, s0vh_pred)
        urmse_vh = np.sqrt(
            np.sum(np.square((s0vh_pred - np.mean(s0vh_pred)) - (y_s0vhtest - np.mean(y_s0vhtest)))) / len(y_s0vhtest))
        bias_vh = np.mean(s0vh_pred - y_s0vhtest)
        error_vh = np.sqrt(np.sum(np.square(s0vh_pred - y_s0vhtest)) / len(y_s0vhtest))

        r_ssm = np.corrcoef(y_test2, ssm_pred)
        urmse_ssm = np.sqrt(
            np.sum(np.square((ssm_pred - np.mean(ssm_pred)) - (y_test2 - np.mean(y_test2)))) / len(y_test2))
        bias_ssm = np.mean(ssm_pred - y_test2)
        error_ssm = np.sqrt(np.sum(np.square(ssm_pred - y_test2)) / len(y_test2))

        pltlims = [0, 0.4]
        # create plots
        plt.figure(figsize=(6, 6))
        plt.scatter(y_s0vvtest, s0vv_pred, c='g', label='True vs Est')
        plt.xlim(pltlims[0], pltlims[1])
        plt.ylim(pltlims[0], pltlims[1])
        plt.xlabel("True  relative SMC [m3m-3]")
        plt.ylabel("Estimated relative SMC [m3m-3]")
        plt.plot([pltlims[0], pltlims[1]], [pltlims[0], pltlims[1]], 'k--')
        plt.text(0.1, 0.3, 'R=' + '{:03.2f}'.format(r_vv[0, 1]) +
                 '\nubRMSE=' + '{:03.2f}'.format(urmse_vv) +
                 '\nBias=' + '{:03.2f}'.format(bias_vv), weight='bold')
        plt.savefig(self.outpath + 'SIG0VV_truevsest_relative.png')
        plt.close()

        pltlims = [0, 0.1]
        plt.figure(figsize=(6, 6))
        plt.scatter(y_s0vhtest, s0vh_pred, c='g', label='True vs Est')
        plt.xlim(pltlims[0], pltlims[1])
        plt.ylim(pltlims[0], pltlims[1])
        plt.xlabel("True  relative SMC [m3m-3]")
        plt.ylabel("Estimated relative SMC [m3m-3]")
        plt.plot([pltlims[0], pltlims[1]], [pltlims[0], pltlims[1]], 'k--')
        plt.text(0.01, 0.08, 'R=' + '{:03.2f}'.format(r_vh[0, 1]) +
                 '\nubRMSE=' + '{:03.2f}'.format(urmse_vh) +
                 '\nBias=' + '{:03.2f}'.format(bias_vh), weight='bold')
        plt.savefig(self.outpath + 'SIG0VH_truevsest_relative.png')
        plt.close()

        pltlims = [-0.5, 0.5]
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test2, ssm_pred, c='g', label='True vs Est')
        plt.xlim(pltlims[0], pltlims[1])
        plt.ylim(pltlims[0], pltlims[1])
        plt.xlabel("True  SMC [m3m-3]")
        plt.ylabel("Estimated SMC [m3m-3]")
        plt.plot([pltlims[0], pltlims[1]], [pltlims[0], pltlims[1]], 'k--')
        plt.text(0.1, 0.4, 'R=' + '{:03.2f}'.format(r_ssm[0, 1]) +
                 '\nubRMSE=' + '{:03.2f}'.format(urmse_ssm) +
                 '\nBias=' + '{:03.2f}'.format(bias_ssm), weight='bold')
        plt.savefig(self.outpath + 'SSM_fw_bw_truevsest_relative.png')
        plt.close()

        pickle.dump((mlmodel_s0vv, mlmodel_s0vh, None, mlmodel_ssm, None),
                    open(self.outpath + 'mlmodel' + str(self.track) + 'RF_fw_bw.p', 'wb'))

    def train_model_linear(self):

        import scipy.stats
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.decomposition import PCA
        from sklearn.linear_model import TheilSenRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.linear_model import HuberRegressor

        # filter bad ssm values
        valid = np.where(self.target > 0)  # & (self.features[:,-1] == itrack))
        track_target = self.target[valid[0]]
        track_features = self.features[valid[0], :]
        weights = self.weights[valid[0]]
        # filter nan values
        valid = ~np.any(np.isinf(track_features), axis=1)
        track_target = track_target[valid]
        track_features = track_features[valid, :]
        weights = weights[valid]
        # filter nan
        valid = ~np.any(np.isnan(track_features), axis=1)
        track_target = track_target[valid]
        track_features = track_features[valid, :]
        weights = weights[valid]

        # scaling
        scaler = sklearn.preprocessing.RobustScaler().fit(track_features)
        features = scaler.transform(track_features)

        # perform outlier detection
        from sklearn.ensemble import IsolationForest
        # from sklearn import svm
        isof = IsolationForest(behaviour='new', contamination='auto', random_state=42)
        x_outliers = isof.fit(features).predict(features)
        features = features[np.where(x_outliers == 1)[0], :]
        track_target = track_target[np.where(x_outliers == 1)[0]]
        weights = weights[np.where(x_outliers == 1)[0]]

        x_train, x_test, y_train, y_test, weights_train, weights_test = train_test_split(features, track_target,
                                                                                         weights,
                                                                                         test_size=0.5,
                                                                                         train_size=0.5,
                                                                                         random_state=70)  # , random_state=42)
        x_train = features
        y_train = track_target
        x_test = features
        y_test = track_target
        weights_train = weights
        # x_test = x_train
        # y_test = y_train

        # ...----...----...----...----...----...----...----...----...----...
        # SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR
        # ---....---....---....---....---....---....---....---....---....---

        # specify kernel
        svr_rbf = SVR()

        # parameter tuning -> grid search
        start = time()
        print('Model training startet at: ' + str(start))
        #
        # SVR --- SVR --- SVR --- SVR --- SVR --- SVR --- SVR
        #
        mlmodel = HuberRegressor()

        mlmodel.fit(x_train, y_train, sample_weight=weights_train)

        # prediction on test set
        y_CV_rbf = mlmodel.predict(x_test)
        # print(gdCV.feature_importances_)

        true = y_test
        est = y_CV_rbf

        r = np.corrcoef(true, est)
        error = np.sqrt(np.sum(np.square(true - est)) / len(true))
        urmse = np.sqrt(np.sum(np.square((est - np.mean(est)) - (true - np.mean(true)))) / len(true))

        # print(gdCV.best_params_)
        print('Elapse time for training: ' + str(time() - start))

        print('SVR performance based on test-set')
        print('R: ' + str(r[0, 1]))
        print('RMSE. ' + str(error))

        if (self.ssm_target == 'SMAP') or (self.ssm_target == 'ISMN'):
            pltlims = 0.7
        else:
            pltlims = 100

        # create plots
        plt.figure(figsize=(6, 6))
        plt.scatter(true, est, c='g', label='True vs Est')
        # plt.xlim(0,pltlims)
        # plt.ylim(0,pltlims)
        plt.xlabel("True SMC [m3m-3]")
        plt.ylabel("Estimated SMC [m3m-3]")
        plt.plot([0, pltlims], [0, pltlims], 'k--')
        plt.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r[0, 1]) + '\nRMSE=' + '{:03.2f}'.format(error), weight='bold')
        plt.savefig(self.outpath + 'truevsest1step_linear.png')
        plt.close()

        # self.SVRmodel = gdCV
        # self.scaler = scaler
        pickle.dump((mlmodel, scaler, isof),
                    open(self.outpath + 'mlmodel' + str(self.track) + 'Linear_1step.p', 'wb'))
        return (mlmodel, scaler, isof)

    def train_model2step_linear(self):

        from sklearn.linear_model import HuberRegressor

        # filter bad ssm values
        valid = np.where(self.target1 > 0)  # & (self.features[:,-1] == itrack))
        self.target1 = self.target1[valid[0]]
        self.features1 = self.features1[valid[0], :]
        self.weights_1 = self.weights_1[valid[0]]
        # filter nan values
        valid = ~np.any(np.isinf(self.features1), axis=1)
        self.target1 = self.target1[valid]
        self.features1 = self.features1[valid, :]
        self.weights_1 = self.weights_1[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features1), axis=1)
        self.target1 = self.target1[valid]
        self.features1 = self.features1[valid, :]
        self.weights_1 = self.weights_1[valid]

        # filter bad ssm values
        valid = ~np.any(np.isinf(self.features2), axis=1)
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]
        self.weights_2 = self.weights_2[valid]
        # filter nan
        valid = ~np.any(np.isnan(self.features2), axis=1)
        self.target2 = self.target2[valid]
        self.features2 = self.features2[valid, :]
        self.weights_2 = self.weights_2[valid]

        # scaling
        scaler1 = sklearn.preprocessing.RobustScaler().fit(self.features1)
        features1_scaled = scaler1.transform(self.features1)

        scaler2 = sklearn.preprocessing.RobustScaler().fit(self.features2)
        features2_scaled = scaler2.transform(self.features2)

        # perform outlier detection
        from sklearn.ensemble import IsolationForest
        # from sklearn import svm
        isof1 = IsolationForest(behaviour='new', contamination='auto', random_state=42)
        x_outliers1 = isof1.fit(features1_scaled).predict(features1_scaled)
        features1_scaled = features1_scaled[np.where(x_outliers1 == 1)[0], :]
        target1_nooutl = self.target1[np.where(x_outliers1 == 1)[0]]
        weights_1 = self.weights_1[np.where(x_outliers1 == 1)[0]]

        isof2 = IsolationForest(behaviour='new', contamination='auto', random_state=42)
        x_outliers2 = isof2.fit(features2_scaled).predict(features2_scaled)
        features2_scaled = features2_scaled[np.where(x_outliers2 == 1)[0], :]
        target2_nooutl = self.target2[np.where(x_outliers2 == 1)[0]]
        weights_2 = self.weights_2[np.where(x_outliers2 == 1)[0]]

        x_train1, x_test1, y_train1, y_test1, weights_train1, weights_test1 = train_test_split(features1_scaled,
                                                                                               target1_nooutl,
                                                                                               weights_1,
                                                                                               test_size=0.1,
                                                                                               train_size=0.9,
                                                                                               random_state=70)

        x_train2, x_test2, y_train2, y_test2, weights_train2, weights_test2 = train_test_split(features2_scaled,
                                                                                               target2_nooutl,
                                                                                               weights_2,
                                                                                               test_size=0.5,
                                                                                               train_size=0.5,
                                                                                               random_state=70)

        x_train1 = features1_scaled
        y_train1 = target1_nooutl
        x_test1 = features1_scaled
        y_test1 = target1_nooutl
        weights_train1 = weights_1

        x_train2 = features2_scaled
        y_train2 = target2_nooutl
        x_test2 = features2_scaled
        y_test2 = target2_nooutl
        weights_train2 = weights_2

        mlmodel1 = HuberRegressor()
        mlmodel1.fit(x_train1, y_train1, sample_weight=weights_train1)

        mlmodel2 = HuberRegressor()

        mlmodel2.fit(x_train2, y_train2, sample_weight=weights_train2)

        # prediction of the average sm
        y_pred1 = mlmodel1.predict(x_test1)

        r = np.corrcoef(y_test1, y_pred1)
        urmse = np.sqrt(np.sum(np.square((y_pred1 - np.mean(y_pred1)) - (y_test1 - np.mean(y_test1)))) / len(y_test1))
        bias = np.mean(y_pred1 - y_test1)
        error = np.sqrt(np.sum(np.square(y_pred1 - y_test1)) / len(y_test1))

        print('Prediction of average soil moisture')
        print('R: ' + str(r[0, 1]))
        print('RMSE. ' + str(error))

        # print(mlmodel1.best_estimator_)
        # print(mlmodel2.best_estimator_)

        if (self.ssm_target == 'SMAP') or (self.ssm_target == 'ISMN'):
            pltlims = 0.7
        else:
            pltlims = 100

        # create plots
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test1, y_pred1, c='g', label='True vs Est')
        plt.xlim(0, pltlims)
        plt.ylim(0, pltlims)
        plt.xlabel("True  average SMC [m3m-3]")
        plt.ylabel("Estimated average SMC [m3m-3]")
        plt.plot([0, pltlims], [0, pltlims], 'k--')
        plt.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r[0, 1]) +
                 '\nubRMSE=' + '{:03.2f}'.format(urmse) +
                 '\nBias=' + '{:03.2f}'.format(bias), weight='bold')
        plt.savefig(self.outpath + 'truevsest_average_linear.png')
        plt.close()

        # prediction of the relative sm
        y_pred2 = mlmodel2.predict(x_test2)

        r = np.corrcoef(y_test2, y_pred2)
        urmse = np.sqrt(np.sum(np.square((y_pred2 - np.mean(y_pred2)) - (y_test2 - np.mean(y_test2)))) / len(y_test2))
        bias = np.mean(y_pred2 - y_test2)
        error = np.sqrt(np.sum(np.square(y_pred2 - y_test2)) / len(y_test2))

        print('Prediction of relative soil moisture')
        print('R: ' + str(r[0, 1]))
        print('RMSE. ' + str(error))

        if (self.ssm_target == 'SMAP') or (self.ssm_target == 'ISMN'):
            pltlims = 0.2
        else:
            pltlims = 100

        # create plots
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test2, y_pred2, c='g', label='True vs Est')
        plt.xlim(-0.2, pltlims)
        plt.ylim(-0.2, pltlims)
        plt.xlabel("True  relative SMC [m3m-3]")
        plt.ylabel("Estimated relative SMC [m3m-3]")
        plt.plot([-0.2, pltlims], [-0.2, pltlims], 'k--')
        plt.text(-0.15, 0.15, 'R=' + '{:03.2f}'.format(r[0, 1]) +
                 '\nubRMSE=' + '{:03.2f}'.format(urmse) +
                 '\nBias=' + '{:03.2f}'.format(bias), weight='bold')
        plt.savefig(self.outpath + 'truevsest_relative_linear.png')
        plt.close()

        pickle.dump((mlmodel1, scaler1, mlmodel2, scaler2, isof1, isof2),
                    open(self.outpath + 'mlmodel' + str(self.track) + 'Linear_2step.p', 'wb'))
        return (mlmodel1, scaler1, mlmodel2, scaler2, isof1, isof2)

    # training the SVR per grid-point
    def train_model_alternative(self):

        from joblib import Parallel, delayed, load, dump
        import sys
        import tempfile
        import shutil
        import os

        # filter bad ssm values
        valid = np.where(self.target > 0)
        self.target = self.target[valid[0]]
        self.features = self.features[valid[0], :]
        # filter nan values
        valid = ~np.any(np.isinf(self.features), axis=1)
        self.target = self.target[valid]
        self.features = self.features[valid, :]
        # filter nan
        valid = ~np.any(np.isnan(self.features), axis=1)
        self.target = self.target[valid]
        self.features = self.features[valid, :]

        model_list = list()
        true = np.array([])
        estimated = np.array([])

        b = np.ascontiguousarray(self.features[:, 0:2]).view(
            np.dtype((np.void, self.features[:, 0:2].dtype.itemsize * self.features[:, 0:2].shape[1])))
        _, idx = np.unique(b, return_index=True)

        # prepare multi processing
        # dump arrays to temporary folder
        temp_folder = tempfile.mkdtemp(dir='/tmp/')
        filename_in = os.path.join(temp_folder, 'joblib_dump1.mmap')
        if os.path.exists(filename_in): os.unlink(filename_in)
        _ = dump((self.features, self.target), filename_in)
        param_memmap = load(filename_in, mmap_mode='r+')

        if not hasattr(sys.stdin, 'close'):
            def dummy_close():
                pass

            sys.stdin.close = dummy_close

        # while (len(self.target) > 1):
        model_list = Parallel(n_jobs=8, verbose=5, max_nbytes=None)(
            delayed(_generate_model)(param_memmap[0], param_memmap[1], i) for i in idx)

        try:
            shutil.rmtree(temp_folder)
        except:
            print("Failed to delete: " + temp_folder)

        # filter model list
        model_list_fltrd = list()
        for tmp_i in range(len(model_list)):
            if model_list[tmp_i]['quality'] == 'good':
                model_list_fltrd.append(model_list[tmp_i])

        model_list = model_list_fltrd

        # generate nn model
        nn_target = np.array([str(x) for x in range(len(model_list))])
        nn_features = np.array([x['model_attr'] for x in model_list])

        clf = neighbors.KNeighborsClassifier()
        clf.fit(nn_features, nn_target)

        pickle.dump((model_list, clf), open(self.outpath + 'mlmodel.p', 'wb'))

        return (model_list, clf)

    def train_model_alternative_linear(self):

        from joblib import Parallel, delayed, load, dump
        import sys
        import tempfile
        import shutil
        import os

        # filter bad ssm values
        valid = np.where(self.target > 0)
        self.target = self.target[valid[0]]
        self.features = self.features[valid[0], :]
        # filter nan values
        valid = ~np.any(np.isinf(self.features), axis=1)
        self.target = self.target[valid]
        self.features = self.features[valid, :]
        # filter nan
        valid = ~np.any(np.isnan(self.features), axis=1)
        self.target = self.target[valid]
        self.features = self.features[valid, :]

        model_list = list()
        true = np.array([])
        estimated = np.array([])

        b = np.ascontiguousarray(self.features[:, 0:2]).view(
            np.dtype((np.void, self.features[:, 0:2].dtype.itemsize * self.features[:, 0:2].shape[1])))
        _, idx = np.unique(b, return_index=True)

        # prepare multi processing
        # dump arrays to temporary folder
        temp_folder = tempfile.mkdtemp(dir='/tmp/')
        filename_in = os.path.join(temp_folder, 'joblib_dump1.mmap')
        if os.path.exists(filename_in): os.unlink(filename_in)
        _ = dump((self.features, self.target), filename_in)
        param_memmap = load(filename_in, mmap_mode='r+')

        if not hasattr(sys.stdin, 'close'):
            def dummy_close():
                pass

            sys.stdin.close = dummy_close

        # while (len(self.target) > 1):
        model_list = Parallel(n_jobs=8, verbose=5, max_nbytes=None)(
            delayed(_generate_model_linear)(param_memmap[0], param_memmap[1], i) for i in idx)
        # model_list = [_generate_model_linear(param_memmap[0], param_memmap[1], i) for i in idx]

        try:
            shutil.rmtree(temp_folder)
        except:
            print("Failed to delete: " + temp_folder)

        # filter model list
        model_list_fltrd = list()
        for tmp_i in range(len(model_list)):
            if model_list[tmp_i]['quality'] == 'good':
                model_list_fltrd.append(model_list[tmp_i])

        model_list = model_list_fltrd
        print(len(model_list))
        # model_list = model_list

        # generate nn model
        nn_target = np.array([str(x) for x in range(len(model_list))])
        nn_features = np.array([x['model_attr'] for x in model_list])

        clf = neighbors.KNeighborsClassifier()
        clf.fit(nn_features, nn_target)

        pickle.dump((model_list, clf), open(self.outpath + 'mlmodel.p', 'wb'))

        return (model_list, clf)

    def train_classifier(self):
        from sklearn.svm import SVC
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report

        # filter nan values
        valid = ~np.any(np.isinf(self.invalid_col_features), axis=1)
        self.invalid_col_label = self.invalid_col_label[valid]
        self.invalid_col_features = self.invalid_col_features[valid, :]
        # filter nan
        valid = ~np.any(np.isnan(self.invalid_col_features), axis=1)
        self.invalid_col_label = self.invalid_col_label[valid]
        self.invalid_col_features = self.invalid_col_features[valid, :]

        x = self.invalid_col_features
        y = self.invalid_col_label

        y_int = list()
        for i in range(len(y)):
            if y[i] == 'valid':
                y_int.append(1)
            elif y[i] == 'invalid':
                y_int.append(0)

        y_int = np.array(y_int)

        # scaling
        scaler = sklearn.preprocessing.RobustScaler().fit(x)
        x_scaled = scaler.transform(x)

        dictCV = dict(C=np.logspace(-2, 2, 10),
                      gamma=np.logspace(-2, -0.5, 10),
                      # epsilon=np.logspace(-2, -0.5, 5),
                      kernel=['rbf'])

        clf = SVC()

        gdCV = GridSearchCV(estimator=clf,
                            param_grid=dictCV,
                            n_jobs=8,
                            verbose=1,
                            pre_dispatch='all',
                            cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=42))

        gdCV.fit(x_scaled, y_int)

        y_pred = gdCV.predict(x_scaled)
        # plot confusion matrix
        plt.figure(figsize=(6, 6))
        confusion_matrix(y_int, y_pred)
        plt.savefig(self.outpath + 'confusion_matrix.png')
        plt.close()

        print(classification_report(y_int, y_pred, target_names=['invalid', 'valid']))

        pickle.dump((gdCV, scaler),
                    open(self.outpath + 'cl_model.p', 'wb'))
        return (gdCV, scaler)

    def get_terrain(self, x, y, dx=1, dy=1):
        # extract elevation, slope and aspect

        # set up grid to determine tile name
        Eq7 = Equi7Grid(10)

        # create ouput array
        topo = np.full((len(self.points), 3), -9999, dtype=np.float32)

        # get tile
        tilename = Eq7.identfy_tile(self.subgrid, (x, y))

        # elevation
        filename = glob.glob(self.dempath + tilename + '*T1.tif')
        elev = gdal.Open(filename[0], gdal.GA_ReadOnly)
        elevBand = elev.GetRasterBand(1)
        elevGeo = elev.GetGeoTransform()
        h = elevBand.ReadAsArray(int((x - elevGeo[0]) / 10), int((elevGeo[3] - y) / 10), dx, dy)
        elev = None

        # aspect
        filename = glob.glob(self.dempath + tilename + '*_aspect.tif')
        asp = gdal.Open(filename[0], gdal.GA_ReadOnly)
        aspBand = asp.GetRasterBand(1)
        aspGeo = asp.GetGeoTransform()
        a = aspBand.ReadAsArray(int((x - aspGeo[0]) / 10), int((aspGeo[3] - y) / 10), dx, dy)
        asp = None

        # slope
        filename = glob.glob(self.dempath + tilename + '*_slope.tif')
        slp = gdal.Open(filename[0], gdal.GA_ReadOnly)
        slpBand = slp.GetRasterBand(1)
        slpGeo = slp.GetGeoTransform()
        s = slpBand.ReadAsArray(int((x - slpGeo[0]) / 10), int((slpGeo[3] - y) / 10), dx, dy)
        slp = None

        if dx == 1:
            return (h[0, 0], a[0, 0], s[0, 0])
        else:
            return (h, a, s)

    # def extr_sig0_lia(self, aoi, hour=None):
    #
    #     import sgrt_devels.extr_TS as exTS
    #     import random
    #     import datetime as dt
    #     import os
    #     from scipy.stats import moment
    #
    #     cntr = 0
    #     if hour == 5:
    #         path = "168"
    #     elif hour == 17:
    #         path = "117"
    #     else:
    #         path = None
    #
    #     valLClist = [10,11,12,13,18,19,20,21,26,27,28,29,32]
    #
    #     # cycle through all points
    #     for px in self.points:
    #
    #         # create 100 random points to sample the 9 km smap pixel
    #         sig0points = set()
    #         cntr2 = 0
    #         broken = 0
    #         while len(sig0points) <= 100:         # TODO: Increase number of subpoints
    #             if cntr2 >= 5000:
    #                 broken = 1
    #                 break
    #
    #             tmp_alfa = random.random()
    #             tmp_c = random.randint(0,4500)
    #             tmp_dx = math.sin(tmp_alfa) * tmp_c
    #             if tmp_alfa > .5:
    #                 tmp_dx = 0 - tmp_dx
    #             tmp_dy = math.cos(tmp_alfa) * tmp_c
    #
    #             tmpx = int(round(px[0] + tmp_dx))
    #             tmpy = int(round(px[1] + tmp_dy))
    #             #tmpx = random.randint(px[0]-4450,px[0]+4450)
    #             #tmpy = random.randint(px[1]-4450,px[1]+4450)
    #
    #             # check land cover
    #             LCpx = self.get_lc(tmpx, tmpy)
    #
    #             # get mean
    #             mean = self.get_sig0mean(tmpx, tmpy)
    #             # mean = np.float32(mean)
    #             # mean[mean != -9999] = mean[mean != -9999] / 100
    #
    #
    #             #if LCpx in valLClist and mean[0] != -9999 and mean[1] != -9999 and \
    #             #                tmpx >= aoi[px[2]][0]+100 and tmpx <= aoi[px[2]][2]-100 and \
    #             #                tmpy >= aoi[px[2]][1]+100 and tmpy <= aoi[px[2]][3]-100:
    #             if LCpx in valLClist and mean[0] != -9999 and mean[1] != -9999 and \
    #                             tmpx >= aoi[px[2]][0] and tmpx <= aoi[px[2]][2] and \
    #                             tmpy >= aoi[px[2]][1] and tmpy <= aoi[px[2]][3]:
    #                 sig0points.add((tmpx, tmpy))
    #
    #             cntr2 = cntr2 + 1
    #
    #         if broken == 1:
    #             continue
    #
    #         # cycle through the create points to retrieve a aerial mean value
    #         # dictionary to hold the time seres
    #         vvdict = {}
    #         vhdict = {}
    #         liadict = {}
    #         # slopelistVV = []
    #         # slopelistVH = []
    #         meanlistVV = []
    #         meanlistVH = []
    #         sdlistVV = []
    #         sdlistVH = []
    #         kdict = {"k1listVV":  [],
    #                  "k1listVH": [],
    #                  "k2listVV": [],
    #                  "k2listVH": [],
    #                  "k3listVV": [],
    #                  "k3listVH": [],
    #                  "k4listVV": [],
    #                  "k4listVH": []}
    #         hlist = []
    #         slist = []
    #         alist = []
    #
    #         # counter
    #         tsnum = 0
    #         for subpx in sig0points:
    #             # get slope
    #             # slope = self.get_slope(subpx[0],subpx[1])
    #             # slope = np.float32(slope)
    #             # slope[slope != -9999] = slope[slope != -9999] / 100
    #             # slopelistVV.append(slope[0])
    #             # slopelistVH.append(slope[1])
    #             # slopelistVV.append(0)
    #             # slopelistVH.append(0)
    #
    #             # get mean
    #             mean = self.get_sig0mean(subpx[0],subpx[1], path)
    #             mean = np.float32(mean)
    #             mean[mean != -9999] = np.power(10,(mean[mean != -9999] / 100)/10)
    #             meanlistVV.append(mean[0])
    #             meanlistVH.append(mean[1])
    #
    #             # get standard deviation
    #             sd = self.get_sig0sd(subpx[0],subpx[1], path)
    #             sd = np.float32(sd)
    #             sd[sd != -9999] = np.power(10,(sd[sd != -9999] / 100)/10)
    #             sdlistVV.append(sd[0])
    #             sdlistVH.append(sd[1])
    #
    #             # get k statistics
    #             #for kn in range(4):
    #             #vvname = "k" + str(kn+1) + "listVV"
    #             #vhname = "k" + str(kn+1) + "listVH"
    #             k = self.get_kN(subpx[0],subpx[1],1,path)
    #             kdict["k1listVV"].append(k[0]/1000.0)
    #             kdict["k1listVH"].append(k[1]/1000.0)
    #
    #             # get height, aspect, and slope
    #             terr = self.get_terrain(subpx[0],subpx[1])
    #             hlist.append(terr[0])
    #             alist.append(terr[1])
    #             slist.append(terr[2])
    #
    #             # get sig0 and lia timeseries
    #             #tmp_series = exTS.read_NORM_SIG0(self.sgrt_root, 'S1AIWGRDH', 'A0112', 'normalized', 10,
    #             #                                 subpx[0], subpx[1], 1, 1, pol_name=['VV','VH'], grid='Equi7')
    #             tmp_series = exTS.extr_SIG0_LIA_ts(self.sgrt_root, 'S1AIWGRDH', 'A0111', 'resampled', 10,
    #                                                subpx[0], subpx[1], 1, 1, pol_name=['VV','VH'], grid='Equi7', sat_pass='A')
    #             # lia_series = exTS.extr_SIG0_LIA_ts(self.sgrt_root, 'S1AIWGRDH', 'A0111', 'resampled', 10, subpx[0], subpx[1], 1, 1, pol_name=['VV','VH'], grid='Equi7')
    #             sig0vv = np.float32(tmp_series[1]['sig0'])
    #             sig0vh = np.float32(tmp_series[1]['sig02'])
    #             lia = np.float32(tmp_series[1]['lia'])
    #             sig0vv[sig0vv != -9999] = sig0vv[sig0vv != -9999]/100
    #             sig0vh[sig0vh != -9999] = sig0vh[sig0vh != -9999]/100
    #             lia[lia != -9999] = lia[lia != -9999]/100
    #
    #             # normalise backscatter
    #             # if slope[0] != 0 and slope[0] != -9999:
    #             #    sig0vv[sig0vv != -9999] = np.power(10,(sig0vv[sig0vv != -9999] - slope[0] * (lia[sig0vv != -9999] - 30))/10)
    #             # else:
    #             sig0vv[sig0vv != -9999] = np.power(10,sig0vv[sig0vv != -9999]/10)
    #             # if slope[1] != 0 and slope[1] != -9999:
    #             #     sig0vh[sig0vh != -9999] = np.power(10,(sig0vh[sig0vh != -9999] - slope[0] * (lia[sig0vh != -9999] - 30))/10)
    #             # else:
    #             sig0vh[sig0vh != -9999] = np.power(10,sig0vh[sig0vh != -9999]/10)
    #
    #             datelist = tmp_series[0]
    #             # datelist = [dt.datetime.fromordinal(x) for x in tmp_series[0]]
    #
    #             # create temporary dataframe
    #             tmp_df_vv = pd.DataFrame(sig0vv.squeeze(), index=datelist)
    #             tmp_df_vh = pd.DataFrame(sig0vh.squeeze(), index=datelist)
    #             tmp_df_lia = pd.DataFrame(lia.squeeze(), index=datelist)
    #
    #             # add to collection
    #             tmp_dict = {str(tsnum): tmp_df_vv[0]}
    #             vvdict.update(tmp_dict)
    #             tmp_dict = {str(tsnum): tmp_df_vh[0]}
    #             vhdict.update(tmp_dict)
    #             tmp_dict = {str(tsnum): tmp_df_lia[0]}
    #             liadict.update(tmp_dict)
    #
    #             tsnum = tsnum + 1
    #
    #             tmp_df_vv = None
    #             tmp_df_vh = None
    #             tmp_df_lia = None
    #             tmp_dict = None
    #
    #         # merge into panda data frame
    #         df_vv = pd.DataFrame(vvdict)
    #         df_vh = pd.DataFrame(vhdict)
    #         df_lia = pd.DataFrame(liadict)
    #
    #         # create mask
    #         arr_vv = np.array(df_vv)
    #         arr_vh = np.array(df_vh)
    #         arr_lia = np.array(df_lia)
    #
    #         mask = (arr_vv == -9999) | (arr_vv < -15.00) | (arr_vh == -9999) | (arr_vh < -15.00) | \
    #                (arr_lia < 10.00) | (arr_lia > 50.00)
    #
    #         df_vv.iloc[mask] = np.nan
    #         df_vh.iloc[mask] = np.nan
    #         df_lia.iloc[mask] = np.nan
    #
    #         # mask months
    #         monthmask = (df_vv.index.map(lambda x: x.month) > 1) & (df_vv.index.map(lambda x: x.month) < 12)
    #
    #         # calculate spatial mean and standard deviation
    #         df_vv_mean = 10*np.log10(df_vv.mean(1)[monthmask])
    #         df_vh_mean = 10*np.log10(df_vh.mean(1)[monthmask])
    #         df_lia_mean = df_lia.mean(1)[monthmask]
    #         df_vv_sstd = 10*np.log10(df_vv.std(1)[monthmask])
    #         df_vh_sstd = 10*np.log10(df_vh.std(1)[monthmask])
    #         df_lia_sstd = df_lia.std(1)[monthmask]
    #
    #         # merge to make sure all fits together
    #         tmp_dict = {'vv': df_vv_mean, 'vh': df_vh_mean, 'lia': df_lia_mean,
    #                     'vv_std': df_vv_sstd, 'vh_std': df_vh_sstd, 'lia_std': df_lia_sstd}
    #         df_bac = pd.DataFrame(tmp_dict)
    #
    #         # Cleanup
    #         tmp_dict = None
    #         df_vv_mean = None
    #         df_vh_mean = None
    #         df_lia_mean = None
    #         df_vv_sstd = None
    #         df_vh_sstd = None
    #         df_lia_sstd = None
    #
    #         # ------------------------------------------
    #         # get ssm
    #         tmp_ssm = self.get_ssm(px[0], px[1])
    #         ssm_series = pd.Series(index=df_bac.index)
    #         ssm_dates = np.array(tmp_ssm[0])
    #
    #         for i in range(len(df_bac.index)):
    #             current_day = df_bac.index[i].date()
    #             id = np.where(ssm_dates == current_day)
    #             if len(id[0]) > 0:
    #                 ssm_series.iloc[i] = tmp_ssm[1][id]
    #
    #         tmp_ssm = None
    #
    #         # convert lists to numpy arrays
    #         meanlistVV = np.array(meanlistVV)
    #         meanlistVH = np.array(meanlistVH)
    #         sdlistVV = np.array(sdlistVV)
    #         sdlistVH = np.array(sdlistVH)
    #         #klistVV = np.array(kdict['k1listVV'])
    #         #klistVH = np.array(kdict['k1listVH'])
    #         klistVV = np.log(meanlistVV)
    #         klistVH = np.log(meanlistVH)
    #
    #         # calculate mean temporal mean and standard deviation and slope
    #         meanMeanVV = 10*np.log10(np.mean(meanlistVV[meanlistVV != -9999]))
    #         meanMeanVH = 10*np.log10(np.mean(meanlistVH[meanlistVH != -9999]))
    #         meanSdVV = 10*np.log10(np.mean(sdlistVV[sdlistVV != -9999]))
    #         meanSdVH = 10*np.log10(np.mean(sdlistVH[sdlistVH != -9999]))
    #         # meanSlopeVV = np.mean(slopelistVV[slopelistVV != -9999])
    #         # meanSlopeVH = np.mean(slopelistVH[slopelistVH != -9999])
    #         # calculate mean of temporal k statistics (upscaling)
    #         meank1VV = np.mean(klistVV[klistVV != -9999])
    #         meank1VH = np.mean(klistVH[klistVH != -9999])
    #         meank2VV = moment(klistVV[klistVV != -9999], moment=2)
    #         meank2VH = moment(klistVH[klistVH != -9999], moment=2)
    #         meank3VV = moment(klistVV[klistVV != -9999], moment=3)
    #         meank3VH = moment(klistVH[klistVH != -9999], moment=3)
    #         meank4VV = moment(klistVV[klistVV != -9999], moment=4)
    #         meank4VH = moment(klistVH[klistVH != -9999], moment=4)
    #         # calculate mean terrain parameters
    #         meanH = np.mean(hlist[hlist != -9999])
    #         meanA = np.mean(alist[alist != -9999])
    #         meanS = np.mean(slist[slist != -9999])
    #
    #         if cntr == 0:
    #             ll = len(list(np.array(df_bac['vv']).squeeze()))
    #             sig0lia_samples = {'ssm': list(np.array(ssm_series).squeeze()),
    #                                'sig0vv': list(np.array(df_bac['vv']).squeeze()),
    #                                'sig0vh': list(np.array(df_bac['vh']).squeeze()),
    #                                'lia': list(np.array(df_bac['lia']).squeeze()),
    #                                'vv_sstd': list(np.array(df_bac['vv_std']).squeeze()),
    #                                'vh_sstd': list(np.array(df_bac['vh_std']).squeeze()),
    #                                'lia_sstd': list(np.array(df_bac['lia_std']).squeeze()),
    #                                'vv_tmean': [meanMeanVV]*ll,
    #                                'vh_tmean': [meanMeanVH]*ll,
    #                                'vv_tstd': [meanSdVV]*ll,
    #                                'vh_tstd': [meanSdVH]*ll,
    #                                # 'vv_slope': [meanSlopeVV]*ll,
    #                                # 'vh_slope': [meanSlopeVH]*ll,
    #                                'vv_k1': [meank1VV]*ll,
    #                                'vh_k1': [meank1VH]*ll,
    #                                'vv_k2': [meank2VV] * ll,
    #                                'vh_k2': [meank2VH] * ll,
    #                                'vv_k3': [meank3VV] * ll,
    #                                'vh_k3': [meank3VH] * ll,
    #                                'vv_k4': [meank4VV] * ll,
    #                                'vh_k4': [meank4VH] * ll,
    #                                'height': [meanH]*ll,
    #                                'aspect': [meanA]*ll,
    #                                'slope': [meanS]*ll}
    #         else:
    #             ll = len(list(np.array(df_bac['vv']).squeeze()))
    #             sig0lia_samples['ssm'].extend(list(np.array(ssm_series).squeeze()))
    #             sig0lia_samples['sig0vv'].extend(list(np.array(df_bac['vv']).squeeze()))
    #             sig0lia_samples['sig0vh'].extend(list(np.array(df_bac['vh']).squeeze()))
    #             sig0lia_samples['lia'].extend(list(np.array(df_bac['lia']).squeeze()))
    #             sig0lia_samples['vv_sstd'].extend(list(np.array(df_bac['vv_std']).squeeze()))
    #             sig0lia_samples['vh_sstd'].extend(list(np.array(df_bac['vh_std']).squeeze()))
    #             sig0lia_samples['lia_sstd'].extend(list(np.array(df_bac['lia_std']).squeeze()))
    #             sig0lia_samples['vv_tmean'].extend([meanMeanVV]*ll)
    #             sig0lia_samples['vh_tmean'].extend([meanMeanVH]*ll)
    #             sig0lia_samples['vv_tstd'].extend([meanSdVV]*ll)
    #             sig0lia_samples['vh_tstd'].extend([meanSdVH]*ll)
    #             # sig0lia_samples['vv_slope'].extend([meanSlopeVV]*ll)
    #             # sig0lia_samples['vh_slope'].extend([meanSlopeVH]*ll)
    #             sig0lia_samples['vv_k1'].extend([meank1VV]*ll)
    #             sig0lia_samples['vh_k1'].extend([meank1VH]*ll)
    #             sig0lia_samples['vv_k2'].extend([meank2VV] * ll)
    #             sig0lia_samples['vh_k2'].extend([meank2VH] * ll)
    #             sig0lia_samples['vv_k3'].extend([meank3VV] * ll)
    #             sig0lia_samples['vh_k3'].extend([meank3VH] * ll)
    #             sig0lia_samples['vv_k4'].extend([meank4VV] * ll)
    #             sig0lia_samples['vh_k4'].extend([meank4VH] * ll)
    #             sig0lia_samples['height'].extend([meanH]*ll)
    #             sig0lia_samples['aspect'].extend([meanA]*ll)
    #             sig0lia_samples['slope'].extend([meanS]*ll)
    #
    #         cntr = cntr + 1
    #         os.system('rm /tmp/*.vrt')
    #
    #     return sig0lia_samples

    # def extr_sig0_lia(self, aoi, hour=None):
    #
    #     import sgrt_devels.extr_TS as exTS
    #     import random
    #     import datetime as dt
    #     import os
    #     from scipy.stats import moment
    #
    #     cntr = 0
    #     if hour == 5:
    #         path = "168"
    #     elif hour == 17:
    #         path = "117"
    #     else:
    #         path = None
    #
    #     valLClist = [10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29]
    #
    #     # cycle through all points
    #     for px in self.points:
    #
    #         print('Grid-Point ' + str(cntr+1) + '/' + str(len(self.points)))
    #
    #         # if cntr < 73:
    #         #     cntr = cntr + 1
    #         #     continue
    #
    #         # define SMAP extent coordinates
    #         px_xmin = px[0] - 4500
    #         px_xmax = px[0] + 4500
    #         px_ymin = px[1] - 4500
    #         px_ymax = px[1] + 4500
    #
    #         # read sig0 stack
    #         tmp_series = exTS.extr_SIG0_LIA_ts(self.sgrt_root, 'S1AIWGRDH', 'A0111', 'resampled', 10,
    #                                                px_xmin, px_ymax, 900, 900, pol_name=['VV', 'VH'], grid='Equi7', subgrid=self.subgrid,
    #                                                sat_pass='A', monthmask=self.months)
    #
    #         vv_stack = np.array(tmp_series[1]['sig0'], dtype=np.float32)
    #         vh_stack = np.array(tmp_series[1]['sig02'], dtype=np.float32)
    #         lia_stack = np.array(tmp_series[1]['lia'], dtype=np.float32)
    #
    #         # get lc
    #         if self.uselc == True:
    #             lc_data = self.get_lc(px_xmin, px_ymax, dx=900, dy=900)
    #
    #         # get terrain
    #         #if self.subgrid == 'EU':
    #             #terr_data = self.get_terrain(px_xmin, px_ymax, dx=900, dy=900)
    #
    #         if (self.uselc == True) and (self.subgrid == 'EU'):
    #             LCmask = np.reshape(np.in1d(lc_data, valLClist), (900,900))
    #
    #         # mask backscatter_stack
    #         n_layers = vv_stack.shape[0]
    #
    #         for li in range(n_layers):
    #
    #             tmp_vv = vv_stack[li,:,:]
    #             tmp_vh = vh_stack[li,:,:]
    #             tmp_lia = lia_stack[li,:,:]
    #
    #             if (self.uselc == True):
    #                 tmp_mask = np.where((tmp_vv < -2500) |
    #                                     (tmp_vv == -9999) |
    #                                     (tmp_vh < -2500) |
    #                                     (tmp_vh == -9999) |
    #                                     (tmp_lia < 1000) |
    #                                     (tmp_lia > 5000) |
    #                                     (LCmask == 0))
    #             else:
    #                 tmp_mask = np.where((tmp_vv < -2500) |
    #                                     (tmp_vv == -9999) |
    #                                     (tmp_vh < -2500) |
    #                                     (tmp_vh == -9999) |
    #                                     (tmp_lia < 1000) |
    #                                     (tmp_lia > 5000))
    #
    #             tmp_vv[tmp_mask] = np.nan
    #             tmp_vh[tmp_mask] = np.nan
    #             tmp_lia[tmp_mask] = np.nan
    #
    #         # db to lin
    #         vv_stack_lin = np.power(10, ((vv_stack/100.0) / 10))
    #         vh_stack_lin = np.power(10, ((vh_stack/100.0) / 10))
    #         lia_stack = lia_stack / 100.0
    #
    #         # create spatial mean
    #         vv_smean = np.full(n_layers, -9999, dtype=np.float32)
    #         vh_smean = np.full(n_layers, -9999, dtype=np.float32)
    #         lia_smean = np.full(n_layers, -9999, dtype=np.float32)
    #
    #         for li in range(n_layers):
    #             vv_smean[li] = 10 * np.log10(np.nanmean(vv_stack_lin[li,:,:]))
    #             vh_smean[li] = 10 * np.log10(np.nanmean(vh_stack_lin[li,:,:]))
    #             lia_smean[li] = np.nanmean(lia_stack[li,:,:])
    #
    #         # create temporal mean and standard deviation
    #         vv_tmean = 10*np.log10(np.nanmean(np.nanmean(vv_stack_lin, axis=0)))
    #         vh_tmean = 10*np.log10(np.nanmean(np.nanmean(vh_stack_lin, axis=0)))
    #         vv_tstd = 10*np.log10(np.nanstd(np.nanstd(vv_stack_lin, axis=0)))
    #         vh_tstd = 10*np.log10(np.nanstd(np.nanstd(vh_stack_lin, axis=0)))
    #
    #         # calculate k-statistics
    #         tmp_vv = np.nanmean(vv_stack/100.0, axis=0)
    #         tmp_vh = np.nanmean(vh_stack/100.0, axis=0)
    #         meank1VV = np.nanmean(tmp_vv)
    #         meank1VH = np.nanmean(tmp_vh)
    #         #meank2VV = moment(tmp_vv.ravel(), moment=2, nan_policy='omit')
    #         #meank2VH = moment(tmp_vh.ravel(), moment=2, nan_policy='omit')
    #         meank2VV = np.nanstd(tmp_vv)
    #         meank2VH = np.nanstd(tmp_vh)
    #         meank3VV = moment(tmp_vv.ravel(), moment=3, nan_policy='omit')
    #         meank3VH = moment(tmp_vh.ravel(), moment=3, nan_policy='omit')
    #         meank4VV = moment(tmp_vv.ravel(), moment=4, nan_policy='omit')
    #         meank4VH = moment(tmp_vh.ravel(), moment=4, nan_policy='omit')
    #
    #         # calculate mean terrain parameters
    #         #H = terr_data[0]
    #         #A = terr_data[1]
    #         #S = terr_data[2]
    #         #meanH = np.mean(H[H != -9999])
    #         #meanA = np.mean(A[A != -9999])
    #         #meanS = np.mean(S[S != -9999])
    #
    #         # ------------------------------------------
    #         # get ssm
    #         if self.ssm_target == 'SMAP':
    #             success, tmp_ssm = self.get_ssm(px[3], px[4])
    #         elif self.ssm_target == 'ASCAT':
    #             ssm_success, tmp_ssm = self.get_ssm(px[3], px[4], source='ASCAT')
    #
    #         ssm_series = pd.Series(index=tmp_series[0])
    #         # ssm_dates = np.array(tmp_ssm[0])
    #
    #         for i in range(len(ssm_series.index)):
    #             current_day = ssm_series.index[i]
    #             tmp_select = tmp_ssm.iloc[np.argmin(np.abs(tmp_ssm.index - current_day))]
    #             ssm_series.iloc[i] = tmp_select
    #             # id = np.where(ssm_dates == current_day)
    #             # if len(id[0]) > 0:
    #             #     ssm_series.iloc[i] = tmp_ssm[1][id]
    #
    #         tmp_ssm = None
    #
    #
    #         if cntr == 0:
    #             ll = len(vv_smean)
    #             sig0lia_samples = {'ssm': list(np.array(ssm_series).squeeze()),
    #                                'sig0vv': list(vv_smean),
    #                                'sig0vh': list(vh_smean),
    #                                'lia': list(lia_smean),
    #                                'vv_tmean': [vv_tmean] * ll,
    #                                'vh_tmean': [vh_tmean] * ll,
    #                                'vv_tstd': [vv_tstd] * ll,
    #                                'vh_tstd': [vh_tstd] * ll,
    #                                'vv_k1': [meank1VV] * ll,
    #                                'vh_k1': [meank1VH] * ll,
    #                                'vv_k2': [meank2VV] * ll,
    #                                'vh_k2': [meank2VH] * ll,
    #                                'vv_k3': [meank3VV] * ll,
    #                                'vh_k3': [meank3VH] * ll,
    #                                'vv_k4': [meank4VV] * ll,
    #                                'vh_k4': [meank4VH] * ll}#,
    #                                #'height': [meanH] * ll,
    #                                #'aspect': [meanA] * ll,
    #                                #'slope': [meanS] * ll}
    #         else:
    #             ll = len(vv_smean)
    #             sig0lia_samples['ssm'].extend(list(np.array(ssm_series).squeeze()))
    #             sig0lia_samples['sig0vv'].extend(list(vv_smean))
    #             sig0lia_samples['sig0vh'].extend(list(vh_smean))
    #             sig0lia_samples['lia'].extend(list(lia_smean))
    #             sig0lia_samples['vv_tmean'].extend([vv_tmean] * ll)
    #             sig0lia_samples['vh_tmean'].extend([vh_tmean] * ll)
    #             sig0lia_samples['vv_tstd'].extend([vv_tstd] * ll)
    #             sig0lia_samples['vh_tstd'].extend([vh_tstd] * ll)
    #             sig0lia_samples['vv_k1'].extend([meank1VV] * ll)
    #             sig0lia_samples['vh_k1'].extend([meank1VH] * ll)
    #             sig0lia_samples['vv_k2'].extend([meank2VV] * ll)
    #             sig0lia_samples['vh_k2'].extend([meank2VH] * ll)
    #             sig0lia_samples['vv_k3'].extend([meank3VV] * ll)
    #             sig0lia_samples['vh_k3'].extend([meank3VH] * ll)
    #             sig0lia_samples['vv_k4'].extend([meank4VV] * ll)
    #             sig0lia_samples['vh_k4'].extend([meank4VH] * ll)
    #             #sig0lia_samples['height'].extend([meanH] * ll)
    #             #sig0lia_samples['aspect'].extend([meanA] * ll)
    #             #sig0lia_samples['slope'].extend([meanS] * ll)
    #
    #         cntr = cntr + 1
    #
    #     os.system('rm /tmp/*.vrt')
    #
    #     return sig0lia_samples

    def svrCor(self, x, y):

        # print('x-shape: ' + str(x.shape) + ', y-shape: ' + str(y.shape))
        x_scaled = sklearn.preprocessing.RobustScaler().fit_transform(x)
        # x_scaled=x

        # dictCV = dict(C=np.logspace(-3, 3, 5),
        #               gamma=np.logspace(-2, -0.5, 5),
        #               epsilon=np.logspace(-2, -0.5, 5),
        #               kernel=['rbf'])
        #
        # mlmodel1 = GridSearchCV(estimator=SVR(),
        #                         param_grid=dictCV,
        #                         n_jobs=1,
        #                         verbose=0,
        #                         pre_dispatch='all',
        #                         cv=LeaveOneOut(),
        #                         scoring='neg_mean_squared_error')

        # mlmodel1 = SVR(kernel='rbf', C=1e3, gamma=0.1)

        mlmodel1 = LinearRegression()

        mlmodel1.fit(x_scaled, y)

        y_pred = mlmodel1.predict(x_scaled)

        pearson_r = pearsonr(y, y_pred)

        error = np.sqrt(np.sum(np.square(y - y_pred)) / len(y))

        return (pearson_r[0], pearson_r[1], error, y_pred)

    def extr_sig0_lia_gee(self):

        import sgrt_devels.extr_TS as exTS
        import datetime as dt
        from scipy.stats import moment
        from scipy.spatial.distance import correlation as dcorr

        cntr = 0
        cntr2 = 0

        # CORINE
        # valLClist = [10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29]
        # GlobCover =
        # valLClist = [11, 14, 20, 30, 120, 140, 150]
        # COPERNICUS
        valLClist = [30, 40, 60, 90]
        sig0lia_samples = dict()
        valStations = list()

        # initialise dict for invalid pixels
        invalid_collection = dict()

        if self.ssm_target == 'ASCAT':
            bsize = 6000
        elif self.ssm_target == 'SMAP':
            bsize = 4000
        elif self.ssm_target == 'ISMN' or self.ssm_target == 'Mazia':
            # bsize = 20
            if self.footprint is not None:
                bsize = self.footprint
            else:
                bsize = 20

        # check if tmp files exist
        if os.path.exists(self.outpath + 'siglia.tmp'):
            sig0lia_samples = pickle.load(open(self.outpath + 'siglia.tmp', 'rb'))
            cntr = sig0lia_samples.shape[0]
            valStations = pickle.load(open(self.outpath + 'stations.tmp', 'rb'))
            cntr2 = len(valStations)


        # cycle through all points
        for px in self.points:

            if (px[6], px[5]) in valStations:
                continue

            print('Grid-Point ' + str(cntr2 + 1) + '/' + str(len(self.points)))
            print(px[6] + ', ' + px[5])

            # extract time series
            tries = 1

            # lon lat
            tmp_lon = px[3]
            tmp_lat = px[4]
            while tries < 4:
                try:
                    if self.ssm_target == 'ISMN' or self.ssm_target == 'Mazia':
                        ssm_success, tmp_ssm = self.get_ssm(px[3], px[4], source=self.ssm_target, stname=px[5],
                                                            network=px[6])
                    else:
                        ssm_success, tmp_ssm = self.get_ssm(px[3], px[4], source=self.ssm_target)
                    if ssm_success == 0:
                        print('Failed to read ISMN data')
                        # cntr2 = cntr2 + 1
                        tries = 4
                        continue

                    # get land-cover
                    tmplc = self.get_lc(tmp_lon, tmp_lat, buffer=bsize, source='Copernicus')
                    tmplc_disc = tmplc['classid'] if tmplc['classid'] is not None else 0
                    tmplc_forestT = tmplc['forestType'] if tmplc['forestType'] is not None else 0
                    tmplc_bare = tmplc['bare'] if tmplc['bare'] is not None else 0
                    tmplc_crops = tmplc['crops'] if tmplc['crops'] is not None else 0
                    tmplc_grass = tmplc['grass'] if tmplc['grass'] is not None else 0
                    tmplc_moss = tmplc['shrub'] if tmplc['shrub'] is not None else 0
                    tmplc_urban = tmplc['urban'] if tmplc['urban'] is not None else 0
                    tmplc_waterp = tmplc['waterp'] if tmplc['waterp'] is not None else 0
                    tmplc_waters = tmplc['waters'] if tmplc['waters'] is not None else 0

                    # get soil info
                    tmp_bulc = self.get_bulk_density(tmp_lon, tmp_lat, buffer=bsize)
                    tmp_clay = self.get_clay_content(tmp_lon, tmp_lat, buffer=bsize)
                    tmp_sand = self.get_sand_content(tmp_lon, tmp_lat, buffer=bsize)

                    tmp_bulc = tmp_bulc['b0']
                    tmp_sand = tmp_sand['b0']
                    tmp_clay = tmp_clay['b0']

                    # get elevation
                    roi = ee.Geometry.Point(px[3], px[4]).buffer(bsize)
                    elev = ee.Image("CGIAR/SRTM90_V4").reduceRegion(ee.Reducer.median(), roi).getInfo()
                    aspe = ee.Terrain.aspect(ee.Image("CGIAR/SRTM90_V4")).reduceRegion(ee.Reducer.median(),
                                                                                       roi).getInfo()
                    slop = ee.Terrain.slope(ee.Image("CGIAR/SRTM90_V4")).reduceRegion(ee.Reducer.median(),
                                                                                      roi).getInfo()
                    elev = elev['elevation']
                    aspe = aspe['aspect']
                    slop = slop['slope']

                    # get tree-cover
                    trees = tmplc['tree']

                    # if the current land cover class is not in the training-set then continue
                    if self.uselc == True:
                        if (tmplc_disc not in valLClist):
                            print('Invalid LC')
                            # cntr2 = cntr2 + 1
                            tries = 4
                            continue

                    # perform a preliminary check of time series length
                    tmp_dates = exTS.extr_SIG0_LIA_ts_GEE(float(px[3]), float(px[4]),
                                                          bufferSize=bsize,
                                                          maskwinter=False,
                                                          trackflt=self.track,
                                                          masksnow=False,
                                                          # ssmcor=tmp_ssm,
                                                          lcmask=False,
                                                          desc=self.desc,
                                                          tempfilter=False,
                                                          returnLIA=False,
                                                          datesonly=True,
                                                          datefilter=[np.min(tmp_ssm.index).strftime('%Y-%m-%d'),
                                                                      np.max(tmp_ssm.index).strftime('%Y-%m-%d')],
                                                          S1B=True,
                                                          treemask=True)

                    if np.max(tmp_dates) < 10:
                        print('S1 time-series < 20')
                        tries = 4
                        # cntr2 = cntr2 + 1
                        continue

                    tmp_series = exTS.extr_SIG0_LIA_ts_GEE(float(px[3]), float(px[4]),
                                                           bufferSize=bsize,
                                                           maskwinter=False,
                                                           trackflt=self.track,
                                                           masksnow=False,
                                                           # ssmcor=tmp_ssm,
                                                           lcmask=False,
                                                           desc=self.desc,
                                                           tempfilter=True,
                                                           returnLIA=True,
                                                           datefilter=[np.min(tmp_ssm.index).strftime('%Y-%m-%d'),
                                                                       np.max(tmp_ssm.index).strftime('%Y-%m-%d')],
                                                           S1B=True,
                                                           treemask=True)
                except:
                    print('Reading from GEE failed - starting retry #' + str(tries))
                    tries = tries + 1
                    continue
                else:
                    break

            if tries > 3:
                print('Failed to read S1 from GEE')
                cntr2 = cntr2 + 1
                continue

            if len(tmp_series.keys()) < 1:
                print('Observations from at least two angles are necessary!')
                cntr2 = cntr2 + 1
                continue

            px_counter = 0

            if cntr != 0:
                out_df = copy.deepcopy(sig0lia_samples)

            for track_key in tmp_series.keys():

                if len(tmp_series[track_key]) < 2:
                    print('S1 time-series <20!')
                    continue

                if cntr2 + 1 == 125:
                    print('Pause')

                vv_series = np.array(tmp_series[track_key]['sig0'], dtype=np.float32)
                vh_series = np.array(tmp_series[track_key]['sig02'], dtype=np.float32)
                lia_series = np.array(tmp_series[track_key]['lia'], dtype=np.float32)
                vv_gamma = np.array(tmp_series[track_key]['gamma0'], dtype=np.float32)
                vh_gamma = np.array(tmp_series[track_key]['gamma02'], dtype=np.float32)

                # db to lin
                vv_series_lin = np.power(10, ((vv_series) / 10))
                vh_series_lin = np.power(10, ((vh_series) / 10))
                vv_gamma_lin = np.power(10, ((vv_gamma) / 10))
                vh_gamma_lin = np.power(10, ((vh_gamma) / 10))

                # ------------------------------------------
                # get ssm
                try:
                    if self.ssm_target == 'ISMN' or self.ssm_target == 'Mazia':
                        ssm_success, tmp_ssm = self.get_ssm(px[3], px[4], source=self.ssm_target, stname=px[5],
                                                            network=px[6])
                    else:
                        ssm_success, tmp_ssm = self.get_ssm(px[3], px[4], source=self.ssm_target)
                except:
                    print('Failed at reading ASCAT time series')
                    ssm_success = 0
                    tmp_ssm = []

                # check if the ssm time series is valid
                if len(tmp_ssm) < 1:
                    print('<1 valid soil moisture value for current grid-point!')
                    continue
                if ssm_success == 0:
                    print('No soil moisture reference available or wetland!')
                    continue
                ssm_series = pd.Series(index=tmp_series[track_key].index)
                # ssm_dates = np.array(tmp_ssm[0])

                # get ndvi
                try:
                    tmpndvi, ndvi_success = exTS.extr_MODIS_MOD13Q1_ts_GEE(px[3], px[4],
                                                                           bufferSize=bsize,
                                                                           datefilter=[
                                                                               np.min(tmp_ssm.index).strftime(
                                                                                   '%Y-%m-%d'),
                                                                               np.max(tmp_ssm.index).strftime(
                                                                                   '%Y-%m-%d')])
                    if ndvi_success == 0:
                        print('No valid NDVI for given location')
                        continue
                except:
                    print('Failed to read NDVI')
                    continue

                try:
                    l8_tmp = exTS.extr_L8_ts_GEE(px[3], px[4], bsize)
                except:
                    print('Landsat extraction failed!')
                    continue

                gldas_series = pd.Series(index=tmp_series[track_key].index)
                gldas_veg_water = pd.Series(index=tmp_series[track_key].index)
                gldas_et = pd.Series(index=tmp_series[track_key].index)
                gldas_swe = pd.Series(index=tmp_series[track_key].index)
                gldas_soilt = pd.Series(index=tmp_series[track_key].index)
                ndvi_series = pd.Series(index=tmp_series[track_key].index)
                gldas_precip = pd.Series(index=tmp_series[track_key].index)
                gldas_snowmelt = pd.Series(index=tmp_series[track_key].index)

                usdasm_series = pd.Series(index=tmp_series[track_key].index)

                l8_series_b1 = pd.Series(index=tmp_series[track_key].index)
                l8_series_b2 = pd.Series(index=tmp_series[track_key].index)
                l8_series_b3 = pd.Series(index=tmp_series[track_key].index)
                l8_series_b4 = pd.Series(index=tmp_series[track_key].index)
                l8_series_b5 = pd.Series(index=tmp_series[track_key].index)
                l8_series_b6 = pd.Series(index=tmp_series[track_key].index)
                l8_series_b7 = pd.Series(index=tmp_series[track_key].index)
                l8_series_b10 = pd.Series(index=tmp_series[track_key].index)
                l8_series_b11 = pd.Series(index=tmp_series[track_key].index)
                l8_series_timediff = pd.Series(index=tmp_series[track_key].index)

                for i in range(len(ssm_series.index)):
                    current_day = ssm_series.index[i]
                    timediff = np.min(np.abs(tmp_ssm.index - current_day))
                    ndvi_timediff = np.min(np.abs(tmpndvi.index - current_day))
                    if (timediff > dt.timedelta(days=1)):
                        continue
                    ssm_series.iloc[i] = tmp_ssm.iloc[np.argmin(np.abs(tmp_ssm.index - current_day))]

                    if (ndvi_timediff > dt.timedelta(days=32)):
                        ndvi_series.iloc[i] = np.nan
                    else:
                        ndvi_series.iloc[i] = tmpndvi.iloc[np.argmin(np.abs(tmpndvi.index - current_day))]

                    l8_timediff = np.min(np.abs(l8_tmp.index - current_day))
                    l8_series_b1.iloc[i] = l8_tmp['B1'].iloc[np.argmin(np.abs(l8_tmp.index - current_day))]
                    l8_series_b2.iloc[i] = l8_tmp['B2'].iloc[np.argmin(np.abs(l8_tmp.index - current_day))]
                    l8_series_b3.iloc[i] = l8_tmp['B3'].iloc[np.argmin(np.abs(l8_tmp.index - current_day))]
                    l8_series_b4.iloc[i] = l8_tmp['B4'].iloc[np.argmin(np.abs(l8_tmp.index - current_day))]
                    l8_series_b5.iloc[i] = l8_tmp['B5'].iloc[np.argmin(np.abs(l8_tmp.index - current_day))]
                    l8_series_b6.iloc[i] = l8_tmp['B6'].iloc[np.argmin(np.abs(l8_tmp.index - current_day))]
                    l8_series_b7.iloc[i] = l8_tmp['B7'].iloc[np.argmin(np.abs(l8_tmp.index - current_day))]
                    l8_series_b10.iloc[i] = l8_tmp['B10'].iloc[np.argmin(np.abs(l8_tmp.index - current_day))]
                    l8_series_b11.iloc[i] = l8_tmp['B11'].iloc[np.argmin(np.abs(l8_tmp.index - current_day))]
                    l8_series_timediff.iloc[i] = l8_timediff.total_seconds()

                    tmp_gldas = self.get_gldas(float(px[3]), float(px[4]),
                                               current_day.strftime('%Y-%m-%d'),
                                               varname=['SoilMoi0_10cm_inst', 'CanopInt_inst', 'Evap_tavg',
                                                        'SWE_inst', 'SoilTMP0_10cm_inst', 'Rainf_f_tavg', 'Qsm_acc'])
                    gldas_series.iloc[i] = tmp_gldas['SoilMoi0_10cm_inst']
                    gldas_veg_water.iloc[i] = tmp_gldas['CanopInt_inst']
                    gldas_et.iloc[i] = tmp_gldas['Evap_tavg']
                    gldas_swe.iloc[i] = tmp_gldas['SWE_inst']
                    gldas_soilt.iloc[i] = tmp_gldas['SoilTMP0_10cm_inst']
                    gldas_precip.iloc[i] = tmp_gldas['Rainf_f_tavg']
                    gldas_snowmelt.iloc[i] = tmp_gldas['Qsm_acc']

                    usdasm_series.iloc[i] = self.get_USDASM(tmp_lon, tmp_lat, current_day.strftime('%Y-%m-%d'))

                # check correlation between s1 and smc
                tmp_vv_pd = pd.Series(vv_series, index=tmp_series[track_key].index)
                tmp_vh_pd = pd.Series(vh_series, index=tmp_series[track_key].index)
                ascat_s1_cor = tmp_vv_pd.corr(ssm_series)
                ascat_s1_cor_vh = tmp_vh_pd.corr(ssm_series)
                vld = np.isfinite(ssm_series) & np.isfinite(tmp_vv_pd) & (gldas_soilt > 275) & (gldas_swe == 0)

                overlap = len(np.where(vld == True)[0])
                print('Valid S1 - ISMN overlap ' + str(overlap))
                if overlap < 2:
                    print('S1 - ISMN overlap too short!')
                    continue

                # calculate k-statistics
                meank1VV = np.mean(np.log(vv_series_lin[vld]))
                meank1VH = np.mean(np.log(vh_series_lin[vld]))
                meank2VV = np.std(np.log(vv_series_lin[vld]))
                meank2VH = np.std(np.log(vh_series_lin[vld]))
                meank3VV = moment(vv_series[vld], moment=3)
                meank3VH = moment(vh_series[vld], moment=3)
                meank4VV = moment(vv_series[vld], moment=4)
                meank4VH = moment(vh_series[vld], moment=4)
                gammak1VV = np.mean(np.log(vv_gamma_lin[vld]))
                gammak1VH = np.mean(np.log(vh_gamma_lin[vld]))
                gammak2VV = np.std(np.log(vv_gamma_lin[vld]))
                gammak2VH = np.std(np.log(vh_gamma_lin[vld]))
                gammak3VV = moment(vv_gamma[vld], moment=3)
                gammak3VH = moment(vh_gamma[vld], moment=3)
                gammak4VV = moment(vv_gamma[vld], moment=4)
                gammak4VH = moment(vh_gamma[vld], moment=4)

                # calculate normal statistics
                vv_tmean = 10 * np.log10(np.mean(vv_series_lin[vld]))
                vh_tmean = 10 * np.log10(np.mean(vh_series_lin[vld]))
                vv_tstd = 10 * np.log10(np.std(vv_series_lin[vld]))
                vh_tstd = 10 * np.log10(np.std(vh_series_lin[vld]))
                vv_gammatmean = 10 * np.log10(np.mean(vv_gamma_lin[vld]))
                vh_gammatmean = 10 * np.log10(np.mean(vh_gamma_lin[vld]))
                vv_gammatstd = 10 * np.log10(np.std(vv_gamma_lin[vld]))
                vh_gammatstd = 10 * np.log10(np.std(vh_gamma_lin[vld]))

                s1_df = np.array([[tmp_vv_pd[vld].values], [tmp_vh_pd[vld].values],
                                  [l8_series_b1[vld].values],
                                  [l8_series_b2[vld].values],
                                  [l8_series_b3[vld].values],
                                  [l8_series_b4[vld].values],
                                  [l8_series_b5[vld].values],
                                  [l8_series_b6[vld].values],
                                  [l8_series_b7[vld].values],
                                  [l8_series_b10[vld].values],
                                  [l8_series_b11[vld].values]])
                s1_df = s1_df.squeeze().transpose()
                p_cor, p_p, p_error, p_pred = self.svrCor(s1_df, ssm_series[vld].values)

                df_for_plot = pd.DataFrame({'S1VV': s1_df[:, 0],
                                            'S1VH': s1_df[:, 1],
                                            'SMC': ssm_series[vld].values,
                                            'SMC_pred': p_pred}, index=ssm_series[vld].index)
                fig, ax = plt.subplots()
                fig.set_size_inches(12, 6)

                df_for_plot.S1VV.plot(title=str(p_cor), ax=ax, style='k-').set_ylabel('SIG0 [dB]')
                df_for_plot.S1VH.plot(ax=ax, style='k--')
                df_for_plot.SMC.plot(ax=ax, secondary_y=True, style='b').set_ylabel('SMC [m^3/m^3]')
                df_for_plot.SMC_pred.plot(ax=ax, secondary_y=True, style='b--')

                ax.legend([ax.get_lines()[0], ax.get_lines()[1],
                           ax.right_ax.get_lines()[0], ax.right_ax.get_lines()[1]],
                          ['S1VV', 'S1VH', 'SMC', 'SMC_pred'])

                plt.savefig(self.outpath + "station_ts/" + px[5] + '_' + track_key + '.png',
                            dpi=600)
                plt.close()

                print('Correlation VV - SIG0 ' + str(ascat_s1_cor))
                print('Correlation VH - SIG0' + str(ascat_s1_cor_vh))
                print(str(p_cor) + ', ' + str(p_p) + ', ' + str(p_error))  # + ', '+ str(distance_r))
                # print(self.svrCor(s1_df[:,0:2], ssm_series[vld].values))

                # if the correlation is below threshold don't add to sm retrieval training set
                mindexlist = [(tmp_lon, tmp_lat, int(track_key), ix) for ix in ssm_series.index[vld]]
                mindex = pd.MultiIndex.from_tuples(mindexlist)
                ll = len(vld[vld == True])
                tmp_dframe = pd.DataFrame({'ssm': list(np.array(ssm_series[vld]).squeeze()),
                                           'ssm_mean': [ssm_series[vld].median()] * ll,
                                           'ssm_p_pred': list(np.array(p_pred)),
                                           'sig0vv': list(vv_series[vld]),
                                           'sig0vh': list(vh_series[vld]),
                                           'gamma0vv': list(vv_gamma[vld]),
                                           'gamma0vh': list(vh_gamma[vld]),
                                           'gldas': list(np.array(gldas_series[vld]).squeeze()),
                                           'gldas_mean': [gldas_series[vld].median()] * ll,
                                           'usdasm': list(np.array(usdasm_series[vld]).squeeze()),
                                           'usdasm_mean': [usdasm_series[vld].median()] * ll,
                                           'plant_water': list(gldas_veg_water[vld]),
                                           'gldas_et': list(gldas_et[vld]),
                                           'gldas_swe': list(gldas_swe[vld]),
                                           'gldas_soilt': list(gldas_soilt[vld]),
                                           'gldas_et_mean': [gldas_et[vld].median()] * ll,
                                           'gldas_swe_mean': [gldas_swe[vld].median()] * ll,
                                           'gldas_soilt_mean': [gldas_soilt[vld].median()] * ll,
                                           'plant_water_mean': [gldas_veg_water[vld].median()] * ll,
                                           'gldas_precip': list(gldas_precip[vld]),
                                           'gldas_precip_mean': [gldas_precip[vld].median()] * ll,
                                           'gldas_snowmelt': list(gldas_snowmelt[vld]),
                                           'gldas_snowmelt_mean': [gldas_snowmelt[vld].median()] * ll,
                                           'ndvi': list(ndvi_series[vld]),
                                           'ndvi_mean': [ndvi_series[vld].median()] * ll,
                                           'lia': list(lia_series[vld]),
                                           'vv_tmean': [vv_tmean] * ll,
                                           'vh_tmean': [vh_tmean] * ll,
                                           'vv_tstd': [vv_tstd] * ll,
                                           'vh_tstd': [vh_tstd] * ll,
                                           'vv_k1': [meank1VV] * ll,
                                           'vh_k1': [meank1VH] * ll,
                                           'vv_k2': [meank2VV] * ll,
                                           'vh_k2': [meank2VH] * ll,
                                           'vv_k3': [meank3VV] * ll,
                                           'vh_k3': [meank3VH] * ll,
                                           'vv_k4': [meank4VV] * ll,
                                           'vh_k4': [meank4VH] * ll,
                                           'vv_gammatmean': [vv_gammatmean] * ll,
                                           'vh_gammatmean': [vh_gammatmean] * ll,
                                           'vv_gammatstd': [vv_tstd] * ll,
                                           'vh_gammatstd': [vh_gammatstd] * ll,
                                           'vv_gammak1': [gammak1VV] * ll,
                                           'vh_gammak1': [gammak1VH] * ll,
                                           'vv_gammak2': [gammak2VV] * ll,
                                           'vh_gammak2': [gammak2VH] * ll,
                                           'vv_gammak3': [gammak3VV] * ll,
                                           'vh_gammak3': [gammak3VH] * ll,
                                           'vv_gammak4': [gammak4VV] * ll,
                                           'vh_gammak4': [gammak4VH] * ll,
                                           'lc': [tmplc_disc] * ll,
                                           'bare_perc': [tmplc_bare] * ll,
                                           'crops_perc': [tmplc_crops] * ll,
                                           'forest_type': [tmplc_forestT] * ll,
                                           'grass_perc': [tmplc_grass] * ll,
                                           'moss_perc': [tmplc_moss] * ll,
                                           'bare_perc': [tmplc_bare] * ll,
                                           'urban_perc': [tmplc_urban] * ll,
                                           'waterp_perc': [tmplc_waterp] * ll,
                                           'waters_perc': [tmplc_waters] * ll,
                                           'sand': [tmp_sand] * ll,
                                           'clay': [tmp_clay] * ll,
                                           'bulk': [tmp_bulc] * ll,
                                           'lon': [px[3]] * ll,
                                           'lat': [px[4]] * ll,
                                           'track': [int(track_key)] * ll,
                                           'trees': [trees] * ll,
                                           'network': [px[6]] * ll,
                                           'station': [px[5]] * ll,
                                           'sensor': [px[7]] * ll,
                                           'L8_b1': list(l8_series_b1[vld]),
                                           'L8_b2': list(l8_series_b2[vld]),
                                           'L8_b3': list(l8_series_b3[vld]),
                                           'L8_b4': list(l8_series_b4[vld]),
                                           'L8_b5': list(l8_series_b5[vld]),
                                           'L8_b6': list(l8_series_b6[vld]),
                                           'L8_b7': list(l8_series_b7[vld]),
                                           'L8_b10': list(l8_series_b10[vld]),
                                           'L8_b11': list(l8_series_b11[vld]),
                                           'L8_timediff': list(l8_series_timediff[vld]),
                                           'L8_b1_median': [l8_series_b1[vld].median()] * ll,
                                           'L8_b2_median': [l8_series_b2[vld].median()] * ll,
                                           'L8_b3_median': [l8_series_b3[vld].median()] * ll,
                                           'L8_b4_median': [l8_series_b4[vld].median()] * ll,
                                           'L8_b5_median': [l8_series_b5[vld].median()] * ll,
                                           'L8_b6_median': [l8_series_b6[vld].median()] * ll,
                                           'L8_b7_median': [l8_series_b7[vld].median()] * ll,
                                           'L8_b10_median': [l8_series_b10[vld].median()] * ll,
                                           'L8_b11_median': [l8_series_b11[vld].median()] * ll,
                                           'overlap': [overlap] * ll}, index=mindex)

                if cntr == 0:
                    out_df = tmp_dframe
                else:
                    out_df = pd.concat([out_df, tmp_dframe], axis=0)

                cntr = cntr + 1
                px_counter = px_counter + 1

            if px_counter >= 1:
                sig0lia_samples = copy.deepcopy(out_df)
                valStations.append((px[6], px[5]))
                sig0lia_samples.to_pickle(self.outpath + 'siglia.tmp')
                pickle.dump(valStations, open(self.outpath + 'stations.tmp', 'wb'))
            else:
                if cntr == px_counter:
                    cntr = 0

            cntr2 = cntr2 + 1
            # if cntr == 10:
            #     valStations = np.unique(np.array(valStations))
            #     np.save(self.outpath + 'ValidStaions.npy', valStations)
            #     return (sig0lia_samples, invalid_collection)

        # os.system('rm /tmp/*.vrt')
        valStations = np.unique(np.array(valStations))
        if self.ssm_target == 'Mazia':
            np.save(self.outpath + 'ValidStaionsMazia.npy', valStations)
        else:
            np.save(self.outpath + 'ValidStaions.npy', valStations)
        return sig0lia_samples, invalid_collection

    def create_random_points(self, sgrid='EASE20'):

        import pygeogrids.netcdf as nc
        if sgrid == 'ISMN':

            # initialise point set
            points = set()

            # initialise available ISMN data
            # ismn = ismn_interface.ISMN_Interface('/mnt/SAT/Workspaces/GrF/01_Data/InSitu/ISMN/')
            ismn = ismn_interface.ISMN_Interface('/mnt/CEPH_PROJECTS/ECOPOTENTIAL/reference_data/ISMN/')
            self.ismn = ismn

            # get list of networks
            networks = ismn.list_networks()

            for ntwk in networks:

                # get list of available stations
                available_stations = ismn.list_stations(ntwk)

                for st_name in available_stations:
                    # load stations
                    station = ismn.get_station(st_name, ntwk)
                    station_vars = station.get_variables()

                    if 'soil moisture' not in station_vars:
                        continue

                    # get available depths measurements
                    station_depths = station.get_depths('soil moisture')

                    if 0.0 in station_depths[0]:

                        did = np.where(station_depths[0] == 0.0)
                        dto = station_depths[1][did]

                        if dto[0] > 0.05:
                            continue

                        sensor = station.get_sensors('soil moisture', 0.0, dto[0])


                    elif 0.05 in station_depths[0]:

                        did = np.where(station_depths[0] == 0.05)
                        dto = station_depths[1][did]

                        if dto[0] > 0.05:
                            continue

                        sensor = station.get_sensors('soil moisture', 0.05, dto[0])

                    else:
                        continue

                    points.add((0, 0, 0, station.longitude, station.latitude, st_name, ntwk, sensor[0]))

        elif sgrid == 'Mazia':

            # initialise point set
            points = set()

            m_stations = {'I1': [10.57978, 46.68706],
                          'I3': [10.58359, 46.68197],
                          'P1': [10.58295, 46.68586],
                          'P2': [10.58525, 46.68433],
                          'P3': [10.58562, 46.68511]}

            for st_name in m_stations.keys():
                points.add((0, 0, 0, m_stations[st_name][0], m_stations[st_name][1],
                            st_name, 'Mazia', 'Hydraprobe'))

        return points

    def get_lc(self, x, y, dx=1, dy=1, buffer=None, source='Corine'):

        # if source == 'Corine':
        #     # set up land cover grid
        #     Eq7LC = Equi7Grid(75)
        #     Eq7Sig0 = Equi7Grid(10)
        #
        #     # get tile name of 75 Equi7 grid to check land-cover
        #     tilename = Eq7LC.identfy_tile(self.subgrid, (x, y))
        #     tilename_res = Eq7Sig0.identfy_tile(self.subgrid, (x,y))
        #
        #     resLCpath = self.outpath + 'LC_' + tilename_res + '.tif'
        #
        #     # check if resampled tile alread exisits
        #     if os.path.exists(resLCpath) == False:
        #
        #         self.tempfiles.append(resLCpath)
        #         # resample lc to sig0
        #         # source
        #         LCtile = SgrtTile(dir_root=self.sgrt_root,
        #                           product_id='S1AIWGRDH',
        #                           soft_id='E0110',
        #                           product_name='CORINE06',
        #                           ftile=tilename,
        #                           src_res=75)
        #
        #         LCfilename = [xs for xs in LCtile._tile_files]
        #         LCfilename = LCtile.dir + '/' + LCfilename[0] + '.tif'
        #
        #         LC = gdal.Open(LCfilename, gdal.GA_ReadOnly)
        #         LC_proj = LC.GetProjection()
        #         LC_geotrans = LC.GetGeoTransform()
        #
        #         #target
        #         Sig0tname = Eq7Sig0.identfy_tile(self.subgrid, (x,y))
        #         S0tile = SgrtTile(dir_root=self.sgrt_root,
        #                           product_id='S1AIWGRDH',
        #                           soft_id='A0111',
        #                           product_name='resampled',
        #                           ftile=Sig0tname,
        #                           src_res=10)
        #         Sig0fname = [xs for xs in S0tile._tile_files]
        #         Sig0fname = S0tile.dir + '/' + Sig0fname[0] + '.tif'
        #         s0ds = gdal.Open(Sig0fname, gdal.GA_ReadOnly)
        #         s0_proj = s0ds.GetProjection()
        #         s0_geotrans = s0ds.GetGeoTransform()
        #         wide = s0ds.RasterXSize
        #         high = s0ds.RasterYSize
        #
        #         # resample
        #         resLCds = gdal.GetDriverByName('GTiff').Create(resLCpath, wide, high, 1, gdalconst.GDT_Byte)
        #         resLCds.SetGeoTransform(s0_geotrans)
        #         resLCds.SetProjection(s0_proj)
        #
        #         gdal.ReprojectImage(LC, resLCds, LC_proj, s0_proj, gdalconst.GRA_NearestNeighbour)
        #
        #         del resLCds
        #
        #     LC = gdal.Open(resLCpath, gdal.GA_ReadOnly)
        #     LC_geotrans = LC.GetGeoTransform()
        #     LCband = LC.GetRasterBand(1)
        #     LCpx = LCband.ReadAsArray(xoff=int((x-LC_geotrans[0])/10.0), yoff=int((LC_geotrans[3]-y)/10.0), win_xsize=dx, win_ysize=dy)
        #
        #     if dx==1 and dy==1:
        #         return LCpx[0][0]
        #     else:
        #         return LCpx

        if source == 'Globcover':
            ee.Initialize()
            globcover_image = ee.Image("ESA/GLOBCOVER_L4_200901_200912_V2_3")
            roi = ee.Geometry.Point(x, y).buffer(buffer)

            return globcover_image.reduceRegion(ee.Reducer.mode(), roi).getInfo()

        if source == 'Copernicus':
            ee.Initialize()
            copernicus_collection = ee.ImageCollection('COPERNICUS/Landcover/100m/Proba-V/Global')
            copernicus_image = ee.Image(copernicus_collection.toList(1000).get(0))
            roi = ee.Geometry.Point(x, y).buffer(buffer)

            class_info = {'classid': copernicus_image.select('discrete_classification').reduceRegion(ee.Reducer.mode(),
                                                                                                     roi).getInfo()[
                'discrete_classification'],
                          'forestType': copernicus_image.select('forest_type').reduceRegion(ee.Reducer.mean(),
                                                                                            roi).getInfo()[
                              'forest_type'],
                          'bare': copernicus_image.select('bare-coverfraction').reduceRegion(ee.Reducer.mean(),
                                                                                             roi).getInfo()[
                              'bare-coverfraction'],
                          'crops': copernicus_image.select('crops-coverfraction').reduceRegion(ee.Reducer.mean(),
                                                                                               roi).getInfo()[
                              'crops-coverfraction'],
                          'grass': copernicus_image.select('grass-coverfraction').reduceRegion(ee.Reducer.mean(),
                                                                                               roi).getInfo()[
                              'grass-coverfraction'],
                          'moss': copernicus_image.select('moss-coverfraction').reduceRegion(ee.Reducer.mean(),
                                                                                             roi).getInfo()[
                              'moss-coverfraction'],
                          'shrub': copernicus_image.select('shrub-coverfraction').reduceRegion(ee.Reducer.mean(),
                                                                                               roi).getInfo()[
                              'shrub-coverfraction'],
                          'tree': copernicus_image.select('tree-coverfraction').reduceRegion(ee.Reducer.mean(),
                                                                                             roi).getInfo()[
                              'tree-coverfraction'],
                          'urban': copernicus_image.select('urban-coverfraction').reduceRegion(ee.Reducer.mean(),
                                                                                               roi).getInfo()[
                              'urban-coverfraction'],
                          'waterp': copernicus_image.select('water-permanent-coverfraction').reduceRegion(
                              ee.Reducer.mean(), roi).getInfo()['water-permanent-coverfraction'],
                          'waters': copernicus_image.select('water-seasonal-coverfraction').reduceRegion(
                              ee.Reducer.mean(), roi).getInfo()['water-seasonal-coverfraction']}
            return class_info

    # def get_slope(self, x, y, dx=1, dy=1):
    #
    #     # set up parameter grid
    #     Eq7Par = Equi7Grid(10)
    #
    #     # get tile name of Equi7 10 grid
    #     tilename = Eq7Par.identfy_tile(self.subgrid, (x,y))
    #     Stile = SgrtTile(dir_root=self.sgrt_root,
    #                      product_id='S1AIWGRDH',
    #                      soft_id='B0212',
    #                      product_name='sig0m',
    #                      ftile=tilename,
    #                      src_res=10)
    #
    #     SfilenameVV = [xs for xs in Stile._tile_files if 'SIGSL' in xs and 'VV' in xs and '_qlook' not in xs]
    #     SfilenameVH = [xs for xs in Stile._tile_files if 'SIGSL' in xs and 'VH' in xs and '_qlook' not in xs]
    #     SfilenameVV = Stile.dir + '/' + SfilenameVV[0] + '.tif'
    #     SfilenameVH = Stile.dir + '/' + SfilenameVH[0] + '.tif'
    #
    #     SVV = gdal.Open(SfilenameVV, gdal.GA_ReadOnly)
    #     SVH = gdal.Open(SfilenameVH, gdal.GA_ReadOnly)
    #     SVVband = SVV.GetRasterBand(1)
    #     SVHband = SVH.GetRasterBand(1)
    #     slopeVV = SVVband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), dx, dy)[0][0]
    #     slopeVH = SVHband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), dx, dy)[0][0]
    #
    #     return (slopeVV, slopeVH)

    # def get_sig0mean(self, x, y, path=None):
    #
    #     import math
    #
    #     # set up parameter grid
    #     Eq7Par = Equi7Grid(10)
    #
    #     # get tile name of Equi7 10 grid
    #     tilename = Eq7Par.identfy_tile(self.subgrid, (x,y))
    #     Stile = SgrtTile(dir_root=self.sgrt_root,
    #                      product_id='S1AIWGRDH',
    #                      soft_id='B0212',
    #                      product_name='sig0m',
    #                      ftile=tilename,
    #                      src_res=10)
    #     if path != None:
    #         SfilenameVV = [xs for xs in Stile._tile_files if 'SIG0M' in xs and 'VV' in xs and path in xs and '_qlook' not in xs]
    #         SfilenameVH = [xs for xs in Stile._tile_files if 'SIG0M' in xs and 'VH' in xs and path in xs and'_qlook' not in xs]
    #     else:
    #         SfilenameVV = [xs for xs in Stile._tile_files if 'SIG0M' in xs and 'VV' in xs and '_qlook' not in xs]
    #         SfilenameVH = [xs for xs in Stile._tile_files if 'SIG0M' in xs and 'VH' in xs and '_qlook' not in xs]
    #     if len(SfilenameVH) == 0 | len(SfilenameVV) == 0:
    #         return (-9999, -9999)
    #     SfilenameVV = Stile.dir + '/' + SfilenameVV[0] + '.tif'
    #     SfilenameVH = Stile.dir + '/' + SfilenameVH[0] + '.tif'
    #
    #     SVV = gdal.Open(SfilenameVV, gdal.GA_ReadOnly)
    #     SVH = gdal.Open(SfilenameVH, gdal.GA_ReadOnly)
    #     SVVband = SVV.GetRasterBand(1)
    #     SVHband = SVH.GetRasterBand(1)
    #
    #     img_x = int(math.floor((x-Stile.geotags['geotransform'][0])/10))
    #     img_y = int(math.floor((Stile.geotags['geotransform'][3]-y)/10))
    #
    #     if img_x == 10000 or img_y == 10000:
    #         meanVH = -9999
    #         meanVV = -9999
    #     else:
    #         meanVV = SVVband.ReadAsArray(img_x, img_y, 1, 1)[0][0]
    #         meanVH = SVHband.ReadAsArray(img_x, img_y, 1, 1)[0][0]
    #
    #     return (meanVV, meanVH)
    #
    #
    # def get_sig0sd(self, x, y, path):
    #
    #     # set up parameter grid
    #     Eq7Par = Equi7Grid(10)
    #
    #     # get tile name of Equi7 10 grid
    #     tilename = Eq7Par.identfy_tile(self.subgrid, (x,y))
    #     Stile = SgrtTile(dir_root=self.sgrt_root,
    #                      product_id='S1AIWGRDH',
    #                      soft_id='B0212',
    #                      product_name='sig0m',
    #                      ftile=tilename,
    #                      src_res=10)
    #
    #     if path != None:
    #         SfilenameVV = [xs for xs in Stile._tile_files if 'SIGSD' in xs and 'VV' in xs and path in xs and '_qlook' not in xs]
    #         SfilenameVH = [xs for xs in Stile._tile_files if 'SIGSD' in xs and 'VH' in xs and path in xs and '_qlook' not in xs]
    #     else:
    #         SfilenameVV = [xs for xs in Stile._tile_files if 'SIGSD' in xs and 'VV' in xs and '_qlook' not in xs]
    #         SfilenameVH = [xs for xs in Stile._tile_files if 'SIGSD' in xs and 'VH' in xs and '_qlook' not in xs]
    #
    #     if len(SfilenameVH) == 0 | len(SfilenameVV) == 0:
    #         return (-9999, -9999)
    #
    #     SfilenameVV = Stile.dir + '/' + SfilenameVV[0] + '.tif'
    #     SfilenameVH = Stile.dir + '/' + SfilenameVH[0] + '.tif'
    #
    #     SVV = gdal.Open(SfilenameVV, gdal.GA_ReadOnly)
    #     SVH = gdal.Open(SfilenameVH, gdal.GA_ReadOnly)
    #     SVVband = SVV.GetRasterBand(1)
    #     SVHband = SVH.GetRasterBand(1)
    #     sdVV = SVVband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), 1, 1)[0][0]
    #     sdVH = SVHband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), 1, 1)[0][0]
    #
    #     return (sdVV, sdVH)
    #
    #
    # def get_kN(self, x, y, n, path=None):
    #
    #     import math
    #
    #     # set up parameter grid
    #     Eq7Par = Equi7Grid(10)
    #
    #     # get tile name of Equi7 10 grid
    #     tilename = Eq7Par.identfy_tile(self.subgrid, (x, y))
    #     Stile = SgrtTile(dir_root=self.sgrt_root,
    #                      product_id='S1AIWGRDH',
    #                      soft_id='B0212',
    #                      product_name='sig0m',
    #                      ftile=tilename,
    #                      src_res=10)
    #
    #     knr = 'K' + str(n)
    #     if path != None:
    #         SfilenameVV = [xs for xs in Stile._tile_files if
    #                        knr in xs and 'VV' in xs and path in xs and '_qlook' not in xs]
    #         SfilenameVH = [xs for xs in Stile._tile_files if
    #                        knr in xs and 'VH' in xs and path in xs and '_qlook' not in xs]
    #     else:
    #         SfilenameVV = [xs for xs in Stile._tile_files if knr in xs and 'VV' in xs and '_qlook' not in xs]
    #         SfilenameVH = [xs for xs in Stile._tile_files if knr in xs and 'VH' in xs and '_qlook' not in xs]
    #     if len(SfilenameVH) == 0 | len(SfilenameVV) == 0:
    #         return (-9999, -9999)
    #     SfilenameVV = Stile.dir + '/' + SfilenameVV[0] + '.tif'
    #     SfilenameVH = Stile.dir + '/' + SfilenameVH[0] + '.tif'
    #
    #     SVV = gdal.Open(SfilenameVV, gdal.GA_ReadOnly)
    #     SVH = gdal.Open(SfilenameVH, gdal.GA_ReadOnly)
    #     SVVband = SVV.GetRasterBand(1)
    #     SVHband = SVH.GetRasterBand(1)
    #
    #     img_x = int(math.floor((x - Stile.geotags['geotransform'][0]) / 10))
    #     img_y = int(math.floor((Stile.geotags['geotransform'][3] - y) / 10))
    #
    #     if img_x == 10000 or img_y == 10000:
    #         kVH = -9999
    #         kVV = -9999
    #     else:
    #         kVV = SVVband.ReadAsArray(img_x, img_y, 1, 1)[0][0]
    #         kVH = SVHband.ReadAsArray(img_x, img_y, 1, 1)[0][0]
    #
    #     return (kVV, kVH)

    def get_ssm(self, x, y, source='SMAP', stname=None, network=None):
        import math
        import datetime as dt

        success = 1

        # grid = Equi7Grid(10)
        # poi_lonlat = grid.equi7xy2lonlat(self.subgrid, x, y)

        # if source == 'SMAP':
        #     #load file stack
        #     h5file = self.sgrt_root + '/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/SMAPL4/SMAPL4_SMC_2015.h5'
        #     ssm_stack = h5py.File(h5file, 'r')
        #
        #     # find the nearest gridpoint
        #     lat = ssm_stack['LATS']
        #     lon = ssm_stack['LONS']
        #     mindist = 10
        #
        #     dist = np.sqrt(np.power(x - lon, 2) + np.power(y - lat, 2))
        #     mindist_loc = np.unravel_index(dist.argmin(), dist.shape)
        #
        #     # stack time series of the nearest grid-point
        #     ssm = np.array(ssm_stack['SM_array'][:,mindist_loc[0], mindist_loc[1]])
        #     # create the time vector
        #     time_sec = np.array(ssm_stack['time'])
        #     time_dt = [dt.datetime(2000,1,1,11,58,55,816) + dt.timedelta(seconds=x) for x in time_sec]
        #     ssm_series = pd.Series(data=ssm, index=time_dt)
        #
        #     ssm_stack.close()
        #
        # elif source == 'ASCAT':
        #
        #     ascat_db = ascat.AscatH109_SSM('/mnt/SAT/Workspaces/GrF/01_Data/ASCAT/h109/SM_ASCAT_TS12.5_DR2016/',
        #                                    '/mnt/SAT/Workspaces/GrF/01_Data/ASCAT/h109_auxiliary/grid/',
        #                                    grid_info_filename = 'TUW_WARP5_grid_info_2_1.nc',
        #                                    static_path = '/mnt/SAT/Workspaces/GrF/01_Data/ASCAT/h109_auxiliary/static_layers/')
        #
        #     ascat_series = ascat_db.read_ssm(x, y)
        #     if ascat_series.wetland_frac > 20:
        #         success = 0
        #     valid = np.where((ascat_series.data['proc_flag'] == 0) & (ascat_series.data['ssf'] == 1) & (ascat_series.data['snow_prob'] < 20))
        #     ssm_series = pd.Series(data=ascat_series.data['sm'][valid[0]], index=ascat_series.data.index[valid[0]])
        #
        #     ascat_db.close()

        if source == 'ISMN':

            if network == 'Mazia':
                m_station_paths = '/mnt/CEPH_PROJECTS/ESA_TIGER/S1_SMC_DEV/01_Data/InSitu/MaziaValley_SWC_2015_16/'

                full_path2015 = m_station_paths + stname + '_YEAR_2015.csv'
                full_path2016 = m_station_paths + stname + '_YEAR_2016.csv'

                insitu2015 = pd.read_csv(
                    full_path2015,
                    header=0,
                    skiprows=[0, 2, 3],
                    index_col=0,
                    parse_dates=True,
                    sep=',')

                insitu2016 = pd.read_csv(
                    full_path2016,
                    header=0,
                    skiprows=[0, 2, 3],
                    index_col=0,
                    parse_dates=True,
                    sep=',')

                insitu = insitu2015.append(insitu2016)

                ssm_series = pd.Series(insitu[['SWC_02_A_Avg', 'SWC_02_B_Avg', 'SWC_02_C_Avg']].mean(axis=1))
                success = 1
            else:

                station = self.ismn.get_station(stname, network)
                station_vars = station.get_variables()

                station_depths = station.get_depths('soil moisture')

                if 0.0 in station_depths[0]:
                    did = np.where(station_depths[0] == 0.0)
                    dto = station_depths[1][did]
                    sm_sensors = station.get_sensors('soil moisture', depth_from=0, depth_to=dto[0])
                    print(sm_sensors[0])
                    station_ts = station.read_variable('soil moisture', depth_from=0, depth_to=dto[0],
                                                       sensor=sm_sensors[0])
                elif 0.05 in station_depths[0]:
                    sm_sensors = station.get_sensors('soil moisture', depth_from=0.05, depth_to=0.05)
                    station_ts = station.read_variable('soil moisture', depth_from=0.05, depth_to=0.05,
                                                       sensor=sm_sensors[0])
                else:
                    return (0, None)

                sm_valid = np.where(station_ts.data['soil moisture_flag'] == 'G')
                ssm_series = station_ts.data['soil moisture']
                ssm_series = ssm_series[sm_valid[0]]
                if len(ssm_series) < 5:
                    success = 0

        elif source == 'Mazia':
            m_station_paths = '/mnt/CEPH_PROJECTS/ESA_TIGER/S1_SMC_DEV/01_Data/InSitu/MaziaValley_SWC_2015_16/'

            full_path2015 = m_station_paths + stname + '_YEAR_2015.csv'
            full_path2016 = m_station_paths + stname + '_YEAR_2016.csv'

            insitu2015 = pd.read_csv(
                full_path2015,
                header=0,
                skiprows=[0, 2, 3],
                index_col=0,
                parse_dates=True,
                sep=',')

            insitu2016 = pd.read_csv(
                full_path2016,
                header=0,
                skiprows=[0, 2, 3],
                index_col=0,
                parse_dates=True,
                sep=',')

            insitu = insitu2015.append(insitu2016)

            ssm_series = pd.Series(insitu[['SWC_02_A_Avg', 'SWC_02_B_Avg', 'SWC_02_C_Avg']].mean(axis=1))
            success = 1

        return (success, ssm_series)

    def get_gldas(self, x, y, date, varname=['SoilMoi0_10cm_inst']):

        def get_ts(image):
            return image.reduceRegion(ee.Reducer.median(), roi, 50)

        ee.Initialize()
        doi = ee.Date(date)
        gldas = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H") \
            .select(varname) \
            .filterDate(doi, doi.advance(3, 'hour'))

        gldas_img = ee.Image(gldas.first())
        roi = ee.Geometry.Point(x, y).buffer(100)
        try:
            return (gldas_img.reduceRegion(ee.Reducer.median(), roi, 50).getInfo())
        except:
            return dict([(k, None) for k in varname])

    def get_USDASM(self, x, y, date):
        ee.Initialize()
        doi = ee.Date(date)
        sm = ee.ImageCollection('NASA_USDA/HSL/soil_moisture') \
            .select('ssm') \
            .filterDate(doi.advance(-2, 'day'), doi.advance(2, 'day'))
        sm_img = ee.Image(sm.first())
        roi = ee.Geometry.Point(x, y).buffer(100)
        try:
            tmp = sm_img.reduceRegion(ee.Reducer.mean(), roi, 50).getInfo()
            return tmp['ssm']
        except:
            return None

    def get_tree_cover(self, x, y, buffer=20):

        ee.Initialize()
        roi = ee.Geometry.Point(x, y).buffer(buffer)
        tree_cover_image = ee.ImageCollection("GLCF/GLS_TCC").filterBounds(roi).filter(
            ee.Filter.eq('year', 2010)).mosaic()
        return (tree_cover_image.reduceRegion(ee.Reducer.mode(), roi, 10).getInfo())

    def get_soil_texture_class(self, x, y, buffer=20):
        ee.Initialize()
        roi = ee.Geometry.Point(x, y).buffer(buffer)
        steximg = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0')
        return (steximg.reduceRegion(ee.Reducer.mode(), roi, 10).getInfo())

    def get_bulk_density(self, x, y, buffer=20):
        ee.Initialize()
        roi = ee.Geometry.Point(x, y).buffer(buffer)
        steximg = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select('b0')
        return (steximg.reduceRegion(ee.Reducer.mode(), roi, 10).getInfo())

    def get_clay_content(self, x, y, buffer=20):
        ee.Initialize()
        roi = ee.Geometry.Point(x, y).buffer(buffer)
        steximg = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0')
        return (steximg.reduceRegion(ee.Reducer.mode(), roi, 10).getInfo())

    def get_sand_content(self, x, y, buffer=20):
        ee.Initialize()
        roi = ee.Geometry.Point(x, y).buffer(buffer)
        steximg = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b0')
        return (steximg.reduceRegion(ee.Reducer.mode(), roi, 10).getInfo())


class Estimationset(object):

    def __init__(self, sgrt_root, tile, sig0mpath, dempath, outpath, mlmodel, mlmodel_avg, subgrid="EU", uselc=True,
                 track=None):

        self.sgrt_root = sgrt_root
        self.outpath = outpath
        self.sig0mpath = sig0mpath
        self.dempath = dempath
        self.subgrid = subgrid
        self.uselc = uselc

        # get list of available parameter tiles
        # tiles = os.listdir(sig0mpath)
        # self.tiles = ['E048N014T1']
        self.tiles = tile
        self.mlmodel = mlmodel
        self.mlmodel_avg = mlmodel_avg
        self.track = track

    def ssm_ts(self, x, y, fdim, name=None):

        from extr_TS import read_NORM_SIG0
        from extr_TS import extr_SIG0_LIA_ts
        from scipy.stats import moment

        # extract parameters
        siglia_ts = extr_SIG0_LIA_ts(self.sgrt_root, 'S1AIWGRDH', 'A0111', 'resampled', 10, x, y, fdim, fdim,
                                     pol_name=['VV', 'VH'], grid='Equi7', sat_pass='A',
                                     monthmask=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        terr_arr = self.get_terrain(self.tiles[0])
        param_arr = self.get_params(self.tiles[0])
        bac_arr = self.get_sig0_lia(self.tiles[0], siglia_ts[0][0].strftime("D%Y%m%d_%H%M%S"))
        lc_arr = self.create_LC_mask(self.tiles[0], bac_arr)

        aoi_pxdim = [int((x - terr_arr['h'][1][0]) / 10),
                     int((terr_arr['h'][1][3] - y) / 10),
                     int((x - terr_arr['h'][1][0]) / 10) + fdim,
                     int((terr_arr['h'][1][3] - y) / 10) + fdim]
        a = terr_arr['a'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        s = terr_arr['s'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        h = terr_arr['h'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        sig0mVV = param_arr['sig0mVV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        sig0mVH = param_arr['sig0mVH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        sig0sdVV = param_arr['sig0sdVV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        sig0sdVH = param_arr['sig0sdVH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # slpVV = param_arr['slpVV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # slpVH = param_arr['slpVH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k1VV = param_arr['k1VV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k1VH = param_arr['k1VH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k2VV = param_arr['k2VV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k2VH = param_arr['k2VH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k3VV = param_arr['k3VV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k3VH = param_arr['k3VH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k4VV = param_arr['k4VV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k4VH = param_arr['k4VH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]

        # calculate k1,...,kN
        k1VV = np.full((fdim, fdim), -9999, dtype=np.float32)
        k2VV = np.full((fdim, fdim), -9999, dtype=np.float32)
        k3VV = np.full((fdim, fdim), -9999, dtype=np.float32)
        k4VV = np.full((fdim, fdim), -9999, dtype=np.float32)
        k1VH = np.full((fdim, fdim), -9999, dtype=np.float32)
        k2VH = np.full((fdim, fdim), -9999, dtype=np.float32)
        k3VH = np.full((fdim, fdim), -9999, dtype=np.float32)
        k4VH = np.full((fdim, fdim), -9999, dtype=np.float32)

        for ix in range(fdim):
            for iy in range(fdim):
                temp_ts1 = siglia_ts[1]['sig0'][:, iy, ix] / 100.
                temp_ts2 = siglia_ts[1]['sig02'][:, iy, ix] / 100.
                temp_mask1 = np.where(temp_ts1 != -99.99)
                temp_mask2 = np.where(temp_ts2 != -99.99)

                k1VV[iy, ix] = np.mean(temp_ts1[temp_mask1])
                k1VH[iy, ix] = np.mean(temp_ts2[temp_mask2])
                k2VV[iy, ix] = moment(temp_ts1[temp_mask1], moment=2, nan_policy='omit')
                k2VH[iy, ix] = moment(temp_ts2[temp_mask2], moment=2, nan_policy='omit')
                k3VV[iy, ix] = moment(temp_ts1[temp_mask1], moment=3, nan_policy='omit')
                k3VH[iy, ix] = moment(temp_ts2[temp_mask2], moment=3, nan_policy='omit')
                k4VV[iy, ix] = moment(temp_ts1[temp_mask1], moment=4, nan_policy='omit')
                k4VH[iy, ix] = moment(temp_ts2[temp_mask2], moment=4, nan_policy='omit')

        lc_mask = lc_arr[aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        terr_mask = (a != -9999) & \
                    (s != -9999) & \
                    (h != -9999)

        param_mask = (sig0mVH != -9999) & \
                     (sig0mVV != -9999) & \
                     (sig0sdVH != -9999) & \
                     (sig0sdVV != -9999)

        ssm_ts_out = (siglia_ts[0], np.full((len(siglia_ts[0])), -9999, dtype=np.float32))

        # average sig0 time-series based on selected pixel footprint
        for i in range(len(siglia_ts[0])):
            sig0_mask = (siglia_ts[1]['sig0'][i, :, :] != -9999) & \
                        (siglia_ts[1]['sig0'][i, :, :] >= -2000) & \
                        (siglia_ts[1]['sig02'][i, :, :] != -9999) & \
                        (siglia_ts[1]['sig02'][i, :, :] >= -2000) & \
                        (siglia_ts[1]['lia'][i, :, :] >= 1000) & \
                        (siglia_ts[1]['lia'][i, :, :] <= 5000)

            mask = lc_mask & terr_mask & sig0_mask

            tmp_smc_arr = np.full((fdim, fdim), -9999, dtype=np.float32)

            # estimate smc for each pixel in the 10x10 footprint
            # create a list of aeach feature
            sig0_l = list()
            sig02_l = list()
            # sigvvssd_l = list()
            # sigvhssd_l = list()
            lia_l = list()
            # liassd_l = list()
            sig0mvv_l = list()
            sig0sdvv_l = list()
            sig0mvh_l = list()
            sig0sdvh_l = list()
            # slpvv_l = list()
            # slpvh_l = list()
            h_l = list()
            a_l = list()
            s_l = list()
            k1vv_l = list()
            k1vh_l = list()
            k2vv_l = list()
            k2vh_l = list()
            k3vv_l = list()
            k3vh_l = list()
            k4vv_l = list()
            k4vh_l = list()

            for ix in range(fdim):
                for iy in range(fdim):

                    if mask[iy, ix] == True:

                        if param_mask[iy, ix] == True:
                            # fvect = [np.float32(siglia_ts[1]['sig0'][i,iy,ix])/100.,
                            #          np.float32(siglia_ts[1]['sig02'][i, iy, ix])/100.,
                            #          np.float32(siglia_ts[1]['lia'][i, iy, ix])/100.,
                            #          sig0mVV[iy, ix] / 100.,
                            #          sig0sdVV[iy,ix] / 100.,
                            #          sig0mVH[iy,ix] / 100.,
                            #          sig0sdVH[iy,ix] / 100.,
                            #          slpVV[iy,ix],
                            #          slpVH[iy,ix], #]
                            #          h[iy,ix],
                            #          a[iy,ix],
                            #          s[iy,ix]]
                            # fvect = self.mlmodel[1].transform(fvect)
                            # tmp_smc_arr[iy,ix] = self.mlmodel[0].predict(fvect)
                            sig0_l.append(np.float32(siglia_ts[1]['sig0'][i, iy, ix]) / 100.)
                            sig02_l.append(np.float32(siglia_ts[1]['sig02'][i, iy, ix]) / 100.)
                            lia_l.append(np.float32(siglia_ts[1]['lia'][i, iy, ix]) / 100.)
                            sig0mvv_l.append(sig0mVV[iy, ix] / 100.)
                            sig0sdvv_l.append(sig0sdVV[iy, ix] / 100.)
                            sig0mvh_l.append(sig0mVH[iy, ix] / 100.)
                            sig0sdvh_l.append(sig0sdVH[iy, ix] / 100.)
                            # slpvv_l.append(slpVV[iy,ix])
                            # slpvh_l.append(slpVH[iy,ix])
                            h_l.append(h[iy, ix])
                            a_l.append(a[iy, ix])
                            s_l.append(s[iy, ix])
                            k1vv_l.append(k1VV[iy, ix])
                            k1vh_l.append(k1VH[iy, ix])
                            k2vv_l.append(k2VV[iy, ix])
                            k2vh_l.append(k2VH[iy, ix])
                            k3vv_l.append(k3VV[iy, ix])
                            k3vh_l.append(k3VH[iy, ix])
                            k4vv_l.append(k4VV[iy, ix])
                            k4vh_l.append(k4VH[iy, ix])

            if len(sig0_l) > 0:
                # calculate average of features
                fvect = [  # np.mean(np.array(lia_l)),
                    10 * np.log10(np.mean(np.power(10, (np.array(sig0mvv_l) / 10.)))),
                    10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvv_l) / 10.)))),
                    10 * np.log10(np.mean(np.power(10, (np.array(sig0mvh_l) / 10.)))),
                    10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvh_l) / 10.)))),
                    10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.)))),
                    10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.)))),
                    np.mean(k1vv_l),
                    np.mean(k1vh_l),
                    np.mean(k2vv_l),
                    np.mean(k2vh_l)]
                # fvect = [#np.mean(np.array(lia_l)),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.))))) - (10 * np.log10(np.mean(np.power(10, (np.array(sig0mvv_l) / 10.))))),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.))))) - (10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvv_l) / 10.))))),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.))))) - (10 * np.log10(np.mean(np.power(10, (np.array(sig0mvh_l) / 10.))))),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.))))) - (10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvh_l) / 10.))))),
                #          10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.)))),
                #          10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.)))),
                #          #10 * np.log10(np.std(np.power(10, (np.array(sig0_l) / 10.)))),
                #          #10 * np.log10(np.std(np.power(10, (np.array(sig02_l) / 10.)))),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.))))) - np.mean(k1vv_l),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.))))) - np.mean(k1vh_l),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.))))) - np.mean(k2vv_l),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.))))) - np.mean(k2vh_l),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.))))) - np.mean(k3vv_l),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.))))) - np.mean(k3vh_l),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.))))) - np.mean(k4vv_l),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.))))) - np.mean(k4vh_l)]
                # np.mean(np.array(lia_l)),
                # np.std(np.array(lia_l)),
                # 10 * np.log10(np.mean(np.power(10, (np.array(sig0mvv_l) / 10.)))),
                # 10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvv_l) / 10.)))),
                # 10 * np.log10(np.mean(np.power(10, (np.array(sig0mvh_l) / 10.)))),
                # 10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvh_l) / 10.)))),
                # np.mean(np.array(slpvv_l)),
                # np.mean(np.array(slpvh_l))]#,
                # np.mean(np.array(h_l)),
                # np.mean(np.array(a_l)),
                # np.mean(np.array(s_l))]

                ## val_ssm = tmp_smc_arr[tmp_smc_arr != -9999]
                # if len(val_ssm) > 0: ssm_ts_out[1][i] = np.mean(val_ssm)

                fvect = self.mlmodel[1].transform(fvect)
                ssm_ts_out[1][i] = self.mlmodel[0].predict(fvect)

        valid = np.where(ssm_ts_out[1] != -9999)
        xx = ssm_ts_out[0][valid]
        yy = ssm_ts_out[1][valid]

        plt.figure(figsize=(18, 6))
        plt.plot(xx, yy)
        plt.show()
        if name == None:
            outfile = self.outpath + 'ts' + str(x) + '_' + str(y)
        else:
            outfile = self.outpath + 'ts_' + name
        plt.savefig(outfile + '.png')
        plt.close()
        csvout = np.array([m.strftime("%d/%m/%Y") for m in ssm_ts_out[0][valid]], dtype=np.str)
        csvout2 = np.array(ssm_ts_out[1][valid], dtype=np.str)
        # np.savetxt(self.outpath + 'ts.csv', np.hstack((csvout, csvout2)), fmt="%-10c", delimiter=",")
        with open(outfile + '.csv', 'w') as text_file:
            [text_file.write(csvout[i] + ', ' + csvout2[i] + '\n') for i in range(len(csvout))]

        print(ssm_ts_out[0][valid])

        print("Done")

    def ssm_ts_alternative(self, x, y, fdim, name=None):

        from extr_TS import read_NORM_SIG0
        from extr_TS import extr_SIG0_LIA_ts
        from scipy.stats import moment

        # extract model attributes
        # model_attrs = np.append(self.mlmodel[0]['model_attr'], self.mlmodel[0]['r'][0,1])
        class_model = self.mlmodel[1]
        reg_model = self.mlmodel[0]
        # model_rs = self.mlmodel[0]['r'][0,1]

        # for mit in range(1,len(self.mlmodel)):
        #    model_attrs = np.vstack((model_attrs, np.append(self.mlmodel[mit]['model_attr'], self.mlmodel[mit]['r'][0,1])))
        #    #model_rs = np.vstack((model_rs, self.mlmodel[mit]['r'][0,1]))

        # extract parameters
        siglia_ts = extr_SIG0_LIA_ts(self.sgrt_root, 'S1AIWGRDH', 'A0111', 'resampled', 10, x, y, fdim, fdim,
                                     pol_name=['VV', 'VH'], grid='Equi7', sat_pass='A',
                                     monthmask=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        terr_arr = self.get_terrain(self.tiles[0])
        param_arr = self.get_params(self.tiles[0])
        bac_arr = self.get_sig0_lia(self.tiles[0], siglia_ts[0][0].strftime("D%Y%m%d_%H%M%S"))
        lc_arr = self.create_LC_mask(self.tiles[0], bac_arr)

        aoi_pxdim = [int((x - terr_arr['h'][1][0]) / 10),
                     int((terr_arr['h'][1][3] - y) / 10),
                     int((x - terr_arr['h'][1][0]) / 10) + fdim,
                     int((terr_arr['h'][1][3] - y) / 10) + fdim]
        a = terr_arr['a'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        s = terr_arr['s'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        h = terr_arr['h'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        sig0mVV = param_arr['sig0mVV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        sig0mVH = param_arr['sig0mVH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        sig0sdVV = param_arr['sig0sdVV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        sig0sdVH = param_arr['sig0sdVH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]

        # calculate k1,...,kN
        k1VV = np.full((fdim, fdim), -9999, dtype=np.float32)
        k2VV = np.full((fdim, fdim), -9999, dtype=np.float32)
        k3VV = np.full((fdim, fdim), -9999, dtype=np.float32)
        k4VV = np.full((fdim, fdim), -9999, dtype=np.float32)
        k1VH = np.full((fdim, fdim), -9999, dtype=np.float32)
        k2VH = np.full((fdim, fdim), -9999, dtype=np.float32)
        k3VH = np.full((fdim, fdim), -9999, dtype=np.float32)
        k4VH = np.full((fdim, fdim), -9999, dtype=np.float32)

        for ix in range(fdim):
            for iy in range(fdim):
                temp_ts1 = siglia_ts[1]['sig0'][:, iy, ix] / 100.
                temp_ts2 = siglia_ts[1]['sig02'][:, iy, ix] / 100.
                temp_mask1 = np.where(temp_ts1 != -99.99)
                temp_mask2 = np.where(temp_ts2 != -99.99)

                k1VV[iy, ix] = np.mean(temp_ts1[temp_mask1])
                k1VH[iy, ix] = np.mean(temp_ts2[temp_mask2])
                k2VV[iy, ix] = moment(temp_ts1[temp_mask1], moment=2, nan_policy='omit')
                k2VH[iy, ix] = moment(temp_ts2[temp_mask2], moment=2, nan_policy='omit')
                k3VV[iy, ix] = moment(temp_ts1[temp_mask1], moment=3, nan_policy='omit')
                k3VH[iy, ix] = moment(temp_ts2[temp_mask2], moment=3, nan_policy='omit')
                k4VV[iy, ix] = moment(temp_ts1[temp_mask1], moment=4, nan_policy='omit')
                k4VH[iy, ix] = moment(temp_ts2[temp_mask2], moment=4, nan_policy='omit')

        lc_mask = lc_arr[aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        terr_mask = (a != -9999) & \
                    (s != -9999) & \
                    (h != -9999)

        param_mask = (sig0mVH != -9999) & \
                     (sig0mVV != -9999) & \
                     (sig0sdVH != -9999) & \
                     (sig0sdVV != -9999)

        ssm_ts_out = (siglia_ts[0], np.full((len(siglia_ts[0])), -9999, dtype=np.float32))

        # average sig0 time-series based on selected pixel footprint
        for i in range(len(siglia_ts[0])):
            sig0_mask = (siglia_ts[1]['sig0'][i, :, :] != -9999) & \
                        (siglia_ts[1]['sig0'][i, :, :] >= -2000) & \
                        (siglia_ts[1]['sig02'][i, :, :] != -9999) & \
                        (siglia_ts[1]['sig02'][i, :, :] >= -2000) & \
                        (siglia_ts[1]['lia'][i, :, :] >= 1000) & \
                        (siglia_ts[1]['lia'][i, :, :] <= 5000)

            mask = lc_mask & terr_mask & sig0_mask

            tmp_smc_arr = np.full((fdim, fdim), -9999, dtype=np.float32)

            # estimate smc for each pixel in the 10x10 footprint
            # create a list of aeach feature
            sig0_l = list()
            sig02_l = list()
            lia_l = list()
            sig0mvv_l = list()
            sig0sdvv_l = list()
            sig0mvh_l = list()
            sig0sdvh_l = list()
            h_l = list()
            a_l = list()
            s_l = list()
            k1vv_l = list()
            k1vh_l = list()
            k2vv_l = list()
            k2vh_l = list()
            k3vv_l = list()
            k3vh_l = list()
            k4vv_l = list()
            k4vh_l = list()

            for ix in range(fdim):
                for iy in range(fdim):

                    if mask[iy, ix] == True:

                        if param_mask[iy, ix] == True:
                            sig0_l.append(np.float32(siglia_ts[1]['sig0'][i, iy, ix]) / 100.)
                            sig02_l.append(np.float32(siglia_ts[1]['sig02'][i, iy, ix]) / 100.)
                            lia_l.append(np.float32(siglia_ts[1]['lia'][i, iy, ix]) / 100.)
                            sig0mvv_l.append(sig0mVV[iy, ix] / 100.)
                            sig0sdvv_l.append(sig0sdVV[iy, ix] / 100.)
                            sig0mvh_l.append(sig0mVH[iy, ix] / 100.)
                            sig0sdvh_l.append(sig0sdVH[iy, ix] / 100.)

                            h_l.append(h[iy, ix])
                            a_l.append(a[iy, ix])
                            s_l.append(s[iy, ix])
                            k1vv_l.append(k1VV[iy, ix])
                            k1vh_l.append(k1VH[iy, ix])
                            k2vv_l.append(k2VV[iy, ix])
                            k2vh_l.append(k2VH[iy, ix])
                            k3vv_l.append(k3VV[iy, ix])
                            k3vh_l.append(k3VH[iy, ix])
                            k4vv_l.append(k4VV[iy, ix])
                            k4vh_l.append(k4VH[iy, ix])

            if len(sig0_l) > 0:
                # calculate average of features
                model_attr = [10 * np.log10(np.mean(np.power(10, (np.array(sig0mvv_l) / 10.)))),
                              10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvv_l) / 10.)))),
                              10 * np.log10(np.mean(np.power(10, (np.array(sig0mvh_l) / 10.)))),
                              10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvh_l) / 10.))))]
                fvect = [10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.)))),
                         10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.))))]

                # def calc_nn(a, poi=None):
                #     if a[4] > 0.5:
                #         dist = np.sqrt(np.square(poi[0]-a[0]) +
                #                        np.square(poi[1]-a[1]) +
                #                        np.square(poi[2]-a[2]) +
                #                        np.square(poi[3]-a[3]))
                #     else:
                #         dist = 9999
                #
                #     return(dist)

                # find the best model
                # nn = np.argmin(np.apply_along_axis(calc_nn, 1, model_attrs, poi=np.array(model_attr)))
                nn = class_model.predict(model_attr)
                nn = int(nn[0])
                nn_model = reg_model[nn]['model']
                nn_scaler = reg_model[nn]['scaler']

                fvect = nn_scaler.transform(fvect)
                ssm_ts_out[1][i] = nn_model.predict(fvect)

        valid = np.where(ssm_ts_out[1] != -9999)
        xx = ssm_ts_out[0][valid]
        yy = ssm_ts_out[1][valid]

        plt.figure(figsize=(18, 6))
        plt.plot(xx, yy)
        plt.show()
        if name == None:
            outfile = self.outpath + 'ts' + str(x) + '_' + str(y)
        else:
            outfile = self.outpath + 'ts_' + name
        plt.savefig(outfile + '.png')
        plt.close()
        csvout = np.array([m.strftime("%d/%m/%Y") for m in ssm_ts_out[0][valid]], dtype=np.str)
        csvout2 = np.array(ssm_ts_out[1][valid], dtype=np.str)
        # np.savetxt(self.outpath + 'ts.csv', np.hstack((csvout, csvout2)), fmt="%-10c", delimiter=",")
        with open(outfile + '.csv', 'w') as text_file:
            [text_file.write(csvout[i] + ', ' + csvout2[i] + '\n') for i in range(len(csvout))]

        print(ssm_ts_out[0][valid])

        print("Done")

    def ssm_ts_gee2step(self, lon, lat, x, y, fdim, name=None, plotts=False, calcstd=False, desc=False,
                        feature_vect1=None, feature_vect2=None):

        # from extr_TS import read_NORM_SIG0
        from sgrt_devels.extr_TS import extr_SIG0_LIA_ts_GEE
        # from extr_TS import extr_MODIS_MOD13Q1_ts_GEE
        # from extr_TS import extr_L8_median
        from sgrt_devels.extr_TS import extr_L8_ts_GEE
        from scipy.stats import moment

        # extract parameters
        siglia_ts = extr_SIG0_LIA_ts_GEE(lon, lat,
                                         bufferSize=fdim,
                                         lcmask=False,
                                         trackflt=self.track,
                                         masksnow=False,
                                         maskwinter=False,
                                         desc=desc,
                                         tempfilter=False,
                                         returnLIA=True,
                                         S1B=True,
                                         treemask=True)

        # siglia_ts_1k = extr_SIG0_LIA_ts_GEE(lon, lat,
        #                                  bufferSize=1000,
        #                                  lcmask=False,
        #                                  trackflt=self.track,
        #                                  masksnow=False,
        #                                  maskwinter=False,
        #                                  desc=desc,
        #                                  tempfilter=False,
        #                                  returnLIA=True,
        #                                  S1B=True,
        #                                  treemask=True)

        # get lc
        ee.Initialize()
        globcover_image = ee.Image("ESA/GLOBCOVER_L4_200901_200912_V2_3")
        roi = ee.Geometry.Point(lon, lat).buffer(fdim)
        land_cover = globcover_image.reduceRegion(ee.Reducer.mode(), roi).getInfo()
        land_cover = land_cover['landcover']
        # get elevation
        elev = ee.Image("CGIAR/SRTM90_V4").reduceRegion(ee.Reducer.median(), roi).getInfo()
        aspe = ee.Terrain.aspect(ee.Image("CGIAR/SRTM90_V4")).reduceRegion(ee.Reducer.median(), roi).getInfo()
        slop = ee.Terrain.slope(ee.Image("CGIAR/SRTM90_V4")).reduceRegion(ee.Reducer.median(), roi).getInfo()
        elev = elev['elevation']
        aspe = aspe['aspect']
        slop = slop['slope']

        # valid_lc = [11, 14, 20, 30, 120, 140, 150]
        #
        # # if the current land cover class is not in the training-set then continue
        # if (land_cover not in valid_lc):
        #     return None

        epty = 1
        keylist = list()
        for track_id in siglia_ts.keys():
            if len(siglia_ts[track_id]) != 0:
                epty = 0
            else:
                keylist.append(track_id)

        if len(keylist) != 0:
            for ki in keylist:
                del siglia_ts[ki]

        if epty == 1:
            return (None)

        # create ts stack
        cntr = 0
        print(siglia_ts.keys())
        for track_id in siglia_ts.keys():

            mlmodelpath = self.mlmodel
            # mlmodelpath = self.mlmodel + 'gee_' + str(track_id) + '/mlmodel' + str(track_id) + '.p'
            if not os.path.exists(mlmodelpath):
                continue

            # extract model attributes
            mlmodelfile = pickle.load(open(mlmodelpath, 'rb'), encoding='latin-1')
            # mlmodelfile_avg = pickle.load(open(self.mlmodel_avg, 'rb'))
            reg_model1 = (mlmodelfile[0], mlmodelfile[1])
            reg_model2 = (mlmodelfile[2], mlmodelfile[3])
            # outldetector1 = mlmodelfile[4]
            # outldetector2 = mlmodelfile[5]

            # calculate k1,...,kN and sig0m
            temp_ts1 = siglia_ts[track_id]['sig0'].astype(np.float)
            temp_ts2 = siglia_ts[track_id]['sig02'].astype(np.float)
            temp_tslia = siglia_ts[track_id]['lia'].astype(np.float)

            # temp_ts1_1k = siglia_ts_1k[track_id]['sig0'].astype(np.float)
            # temp_ts2_1k = siglia_ts_1k[track_id]['sig02'].astype(np.float)
            # temp_tslia_1k = siglia_ts_1k[track_id]['lia'].astype(np.float)
            valid = np.isfinite(temp_ts1) & np.isfinite(
                temp_ts2)  # & np.isfinite(temp_ts1_1k) & np.isfinite(temp_ts2_1k)
            temp_ts1 = temp_ts1[valid]
            temp_ts2 = temp_ts2[valid]
            temp_tslia = temp_tslia[valid]
            # temp_ts1_1k = temp_ts1_1k[valid]
            # temp_ts2_1k = temp_ts2_1k[valid]
            # temp_tslia_1k = temp_tslia_1k[valid]

            ts_length = len(temp_ts1)

            if ts_length < 10:
                continue

            temp_ts1_lin = np.power(10, temp_ts1 / 10.)
            temp_ts2_lin = np.power(10, temp_ts2 / 10.)
            # temp_ts1_lin_1k = np.power(10, temp_ts1_1k / 10.)
            # temp_ts2_lin_1k = np.power(10, temp_ts2_1k / 10.)

            # # get ndvi
            # tmpndvi, ndvi_success = extr_MODIS_MOD13Q1_ts_GEE(lon, lat,
            #                                                  datefilter=[
            #                                                         np.min(temp_ts1.index).strftime('%Y-%m-%d'),
            #                                                         np.max(temp_ts1.index).strftime('%Y-%m-%d')])
            # if ndvi_success == 0:
            #     print('No valid NDVI for given location')
            #     continue

            l8_tries = 1

            try:
                l8_tmp = extr_L8_ts_GEE(lon, lat, fdim)
                # l8_1k_tmp = extr_L8_ts_GEE(lon, lat, 1000)
            except:
                print('Landsat extraction failed!')
                continue

            # get gldas
            gldas_series = pd.Series(index=siglia_ts[track_id][valid].index)
            # gldas_veg_water = pd.Series(index=siglia_ts[track_id][valid].index)
            # gldas_et = pd.Series(index=siglia_ts[track_id][valid].index)
            gldas_swe = pd.Series(index=temp_ts1.index)
            gldas_soilt = pd.Series(index=temp_ts1.index)
            ndvi_series = pd.Series(index=temp_ts1.index)
            # gldas_precip = pd.Series(index=siglia_ts[track_id][valid].index)
            # gldas_snowmelt = pd.Series(index=siglia_ts[track_id][valid].index)

            l8_series_b1 = pd.Series(index=temp_ts1.index)
            l8_series_b2 = pd.Series(index=temp_ts1.index)
            l8_series_b3 = pd.Series(index=temp_ts1.index)
            l8_series_b4 = pd.Series(index=temp_ts1.index)
            l8_series_b5 = pd.Series(index=temp_ts1.index)
            l8_series_b6 = pd.Series(index=temp_ts1.index)
            l8_series_b7 = pd.Series(index=temp_ts1.index)
            l8_series_b10 = pd.Series(index=temp_ts1.index)
            l8_series_b11 = pd.Series(index=temp_ts1.index)
            l8_series_timediff = pd.Series(index=temp_ts1.index)
            # l8_series_b1_1k = pd.Series(index=temp_ts1.index)
            # l8_series_b2_1k = pd.Series(index=temp_ts1.index)
            # l8_series_b3_1k = pd.Series(index=temp_ts1.index)
            # l8_series_b4_1k = pd.Series(index=temp_ts1.index)
            # l8_series_b5_1k = pd.Series(index=temp_ts1.index)
            # l8_series_b6_1k = pd.Series(index=temp_ts1.index)
            # l8_series_b7_1k = pd.Series(index=temp_ts1.index)
            # l8_series_b10_1k = pd.Series(index=temp_ts1.index)
            # l8_series_b11_1k = pd.Series(index=temp_ts1.index)
            # l8_series_timediff_1k = pd.Series(index=temp_ts1.index)

            for i in l8_series_b1.index:
                # ndvi_timediff = np.min(np.abs(tmpndvi.index - i))
                # if ndvi_timediff > dt.timedelta(days=32):
                #     ndvi_series[i] = np.nan
                # else:
                #     ndvi_series[i] = tmpndvi.iloc[np.argmin(np.abs(tmpndvi.index - i))]

                tmp_gldas = self.get_gldas(float(lon), float(lat),
                                           i.strftime('%Y-%m-%d'),
                                           varname=['SWE_inst', 'SoilTMP0_10cm_inst', 'SoilMoi0_10cm_inst'])
                gldas_series[i] = tmp_gldas['SoilMoi0_10cm_inst']
                # gldas_veg_water[i] = tmp_gldas['CanopInt_inst']
                # gldas_et[i] = tmp_gldas['Evap_tavg']
                gldas_swe[i] = tmp_gldas['SWE_inst']
                gldas_soilt[i] = tmp_gldas['SoilTMP0_10cm_inst']
                # gldas_precip[i] = tmp_gldas['Rainf_f_tavg']
                # gldas_snowmelt[i] = tmp_gldas['Qsm_acc']

                l8_timediff = np.min(np.abs(l8_tmp.index - i))

                l8_series_b1[i] = l8_tmp['B1'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b2[i] = l8_tmp['B2'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b3[i] = l8_tmp['B3'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b4[i] = l8_tmp['B4'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b5[i] = l8_tmp['B5'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b6[i] = l8_tmp['B6'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b7[i] = l8_tmp['B7'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b10[i] = l8_tmp['B10'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b11[i] = l8_tmp['B11'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_timediff[i] = l8_timediff.total_seconds()

                # l8_series_b1_1k[i] = l8_1k_tmp['B1'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                # l8_series_b2_1k[i] = l8_1k_tmp['B2'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                # l8_series_b3_1k[i] = l8_1k_tmp['B3'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                # l8_series_b4_1k[i] = l8_1k_tmp['B4'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                # l8_series_b5_1k[i] = l8_1k_tmp['B5'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                # l8_series_b6_1k[i] = l8_1k_tmp['B6'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                # l8_series_b7_1k[i] = l8_1k_tmp['B7'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                # l8_series_b10_1k[i] = l8_1k_tmp['B10'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                # l8_series_b11_1k[i] = l8_1k_tmp['B11'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                # l8_series_timediff_1k[i] = l8_timediff.total_seconds()

            # check for nans in the gldas time-series
            valid2 = (gldas_soilt > 275) & (gldas_swe == 0)  # & np.isfinite(ndvi_series) & np.isfinite(gldas_series)

            ts_length = len(np.where(valid2 == True)[0])

            meanVV = np.median(temp_ts1_lin[valid2])
            meanVH = np.median(temp_ts2_lin[valid2])
            stdVV = np.std(temp_ts1_lin[valid2])
            stdVH = np.std(temp_ts2_lin[valid2])
            k1VV = np.mean(np.log(temp_ts1_lin[valid2]))
            k1VH = np.mean(np.log(temp_ts2_lin[valid2]))
            k2VV = np.std(np.log(temp_ts1_lin[valid2]))
            k2VH = np.std(np.log(temp_ts2_lin[valid2]))
            k3VV = moment(temp_ts1[valid2], moment=3)
            k3VH = moment(temp_ts2[valid2], moment=3)
            k4VV = moment(temp_ts1[valid2], moment=4)
            k4VH = moment(temp_ts2[valid2], moment=4)

            # get tree cover
            tmp_trees = self.get_tree_cover(lon, lat)
            trees = tmp_trees['tree_canopy_cover']

            fmat_tmp1 = np.array([k1VV, k1VH,
                                  k2VV, k2VH,
                                  k3VV, k3VH,
                                  k4VV, k4VH,
                                  trees,
                                  temp_tslia[valid2].median(),
                                  l8_series_b1[valid2].median(),
                                  l8_series_b2[valid2].median(),
                                  l8_series_b3[valid2].median(),
                                  l8_series_b4[valid2].median(),
                                  l8_series_b5[valid2].median(),
                                  l8_series_b6[valid2].median(),
                                  l8_series_b7[valid2].median(),
                                  l8_series_b10[valid2].median(),
                                  l8_series_b11[valid2].median()])

            if feature_vect1 is not None:
                fmat_tmp1 = fmat_tmp1[feature_vect1]

            nn_model1 = reg_model1[0]
            nn_scaler1 = reg_model1[1]

            if nn_scaler1 is not None:
                sub_center1 = nn_scaler1.center_  # [best_features1]
                sub_scaler1 = nn_scaler1.scale_  # [best_features1]
                fvect1 = nn_scaler1.transform(fmat_tmp1.reshape(1, -1))
            else:
                fvect1 = fmat_tmp1.reshape(1, -1)

            if calcstd == False:
                avg_ssm_estimated_tmp = nn_model1.predict(fvect1)
            else:
                avg_ssm_estimated_tmp, std_avg_estimated_tmp = nn_model1.predict_dist(fvect1)

            fmat_tmp2 = np.hstack((temp_ts1.values[valid2].reshape(ts_length, 1),
                                   temp_ts2.values[valid2].reshape(ts_length, 1),
                                   temp_ts1_lin.values[valid2].reshape(ts_length, 1) - meanVV,
                                   # .repeat(ts_length).reshape(ts_length, 1),
                                   temp_ts2_lin.values[valid2].reshape(ts_length, 1) - meanVH,
                                   # .repeat(ts_length).reshape(ts_length, 1),
                                   np.array(lon).repeat(ts_length).reshape(ts_length, 1),
                                   np.array(lat).repeat(ts_length).reshape(ts_length, 1),
                                   np.array(trees).repeat(ts_length).reshape(ts_length, 1),
                                   np.array(temp_tslia[valid2]).reshape(ts_length, 1),
                                   k1VV.repeat(ts_length).reshape(ts_length, 1),
                                   k1VH.repeat(ts_length).reshape(ts_length, 1),
                                   k2VV.repeat(ts_length).reshape(ts_length, 1),
                                   k2VH.repeat(ts_length).reshape(ts_length, 1),
                                   k3VV.repeat(ts_length).reshape(ts_length, 1),
                                   k3VH.repeat(ts_length).reshape(ts_length, 1),
                                   k4VV.repeat(ts_length).reshape(ts_length, 1),
                                   k4VH.repeat(ts_length).reshape(ts_length, 1),
                                   l8_series_b1.values[valid2].reshape(ts_length, 1),
                                   np.array(np.median(l8_series_b1.values[valid2])).repeat(ts_length).reshape(ts_length,
                                                                                                              1),
                                   l8_series_b2.values[valid2].reshape(ts_length, 1),
                                   np.array(np.median(l8_series_b2.values[valid2])).repeat(ts_length).reshape(ts_length,
                                                                                                              1),
                                   l8_series_b3.values[valid2].reshape(ts_length, 1),
                                   np.array(np.median(l8_series_b3.values[valid2])).repeat(ts_length).reshape(ts_length,
                                                                                                              1),
                                   l8_series_b4.values[valid2].reshape(ts_length, 1),
                                   np.array(np.median(l8_series_b4.values[valid2])).repeat(ts_length).reshape(ts_length,
                                                                                                              1),
                                   l8_series_b5.values[valid2].reshape(ts_length, 1),
                                   np.array(np.median(l8_series_b5.values[valid2])).repeat(ts_length).reshape(ts_length,
                                                                                                              1),
                                   l8_series_b6.values[valid2].reshape(ts_length, 1),
                                   np.array(np.median(l8_series_b6.values[valid2])).repeat(ts_length).reshape(ts_length,
                                                                                                              1),
                                   l8_series_b7.values[valid2].reshape(ts_length, 1),
                                   np.array(np.median(l8_series_b7.values[valid2])).repeat(ts_length).reshape(ts_length,
                                                                                                              1),
                                   l8_series_b10.values[valid2].reshape(ts_length, 1),
                                   np.array(np.median(l8_series_b10.values[valid2])).repeat(ts_length).reshape(
                                       ts_length, 1),
                                   l8_series_b11.values[valid2].reshape(ts_length, 1),
                                   np.array(np.median(l8_series_b11.values[valid2])).repeat(ts_length).reshape(
                                       ts_length, 1),
                                   l8_series_timediff.values[valid2].reshape(ts_length, 1),
                                   gldas_series.values[valid2].reshape(ts_length, 1) - np.median(gldas_series[valid2])))

            if feature_vect2 is not None:
                fmat_tmp2 = fmat_tmp2[:, feature_vect2]

            # apply masks to dates and fmat
            dates_tmp = temp_ts1[valid2].index
            # remove nans
            t = np.unique(np.where(np.isnan(fmat_tmp2))[0])
            fmat_tmp2 = np.delete(fmat_tmp2, t, axis=0)
            dates_tmp = np.delete(dates_tmp, t)

            nn_model2 = reg_model2[0]
            nn_scaler2 = reg_model2[1]

            if nn_scaler2 is not None:
                sub_center2 = nn_scaler2.center_  # [best_features2]
                sub_scaler2 = nn_scaler2.scale_  # [best_features2]

                fvect2 = nn_scaler2.transform(fmat_tmp2)
            else:
                fvect2 = fmat_tmp2

            if calcstd == False:
                diff_ssm_estimated_tmp = nn_model2.predict(fvect2)
                ssm_estimated_tmp = diff_ssm_estimated_tmp + avg_ssm_estimated_tmp
            else:
                diff_ssm_estimated_tmp, std_diff_estimated_tmp = nn_model2.predict_dist(fvect2)
                ssm_estimated_tmp = diff_ssm_estimated_tmp + avg_ssm_estimated_tmp
                std_estimated_tmp = std_diff_estimated_tmp + std_avg_estimated_tmp

            # detect outliers
            # outl1 = (outldetector1.predict(fvect1) != 1)
            # outl2 = (outldetector2.predict(fvect2) != 1)
            outl1 = None
            outl2 = None

            if cntr == 0:
                avg_ssm_estimated = np.repeat(avg_ssm_estimated_tmp, len(ssm_estimated_tmp))
                ssm_estimated = np.copy(ssm_estimated_tmp)
                dates = np.copy(dates_tmp)
                # outl_comb = np.copy(outl1 | outl2)
                if calcstd == True:
                    std_estimated = np.copy(std_estimated_tmp)
            else:
                # combine tss performing linear cdf maching
                avg_ssm_estimated = np.append(avg_ssm_estimated,
                                              np.repeat(avg_ssm_estimated_tmp, len(ssm_estimated_tmp)))
                # statmask1 = (ssm_estimated > 0) & (ssm_estimated < 1)
                # statmask2 = (ssm_estimated_tmp > 0) & (ssm_estimated_tmp < 1)
                # p2 = np.std(ssm_estimated[statmask1]) / np.std(ssm_estimated_tmp[statmask2])
                # p1 = np.mean(ssm_estimated[statmask1]) - (p2 * np.mean(ssm_estimated_tmp[statmask2]))
                # ssm_estimated = np.append(ssm_estimated, p1 + (p2*np.copy(ssm_estimated_tmp)))
                ssm_estimated = np.append(ssm_estimated, np.copy(ssm_estimated_tmp))
                dates = np.append(dates, np.copy(dates_tmp))
                # outl_comb = np.append(outl_comb, np.copy(outl1 | outl2))
                if calcstd == True:
                    std_estimated = np.append(std_estimated, np.copy(std_estimated_tmp))

            # retrieve SSM for "failed retrievals"  --------------------------------------------
            ts_length = len(np.where(valid2 == False)[0])
            if ts_length >= 2:
                invalid = (valid2 == False)

                meanVV = np.nanmedian(temp_ts1_lin)
                meanVH = np.nanmedian(temp_ts2_lin)
                stdVV = np.std(temp_ts1_lin)
                stdVH = np.std(temp_ts2_lin)
                k1VV = np.mean(np.log(temp_ts1_lin))
                k1VH = np.mean(np.log(temp_ts2_lin))
                k2VV = np.std(np.log(temp_ts1_lin))
                k2VH = np.std(np.log(temp_ts2_lin))
                k3VV = moment(temp_ts1, moment=3)
                k3VH = moment(temp_ts2, moment=3)
                k4VV = moment(temp_ts1, moment=4)
                k4VH = moment(temp_ts2, moment=4)

                fmat_tmp1 = np.array([k1VV, k1VH,
                                      k2VV, k2VH,
                                      k3VV, k3VH,
                                      k4VV, k4VH,
                                      trees,
                                      temp_tslia.median(),
                                      l8_series_b1.median(),
                                      l8_series_b2.median(),
                                      l8_series_b3.median(),
                                      l8_series_b4.median(),
                                      l8_series_b5.median(),
                                      l8_series_b6.median(),
                                      l8_series_b7.median(),
                                      l8_series_b10.median(),
                                      l8_series_b11.median()])

                if feature_vect1 is not None:
                    fmat_tmp1 = fmat_tmp1[feature_vect1]

                nn_model1 = reg_model1[0]
                nn_scaler1 = reg_model1[1]

                if nn_scaler1 is not None:
                    sub_center1 = nn_scaler1.center_  # [best_features1]
                    sub_scaler1 = nn_scaler1.scale_  # [best_features1]
                    fvect1 = nn_scaler1.transform(fmat_tmp1.reshape(1, -1))
                else:
                    fvect1 = fmat_tmp1.reshape(1, -1)

                if calcstd == False:
                    avg_ssm_estimated_tmp = nn_model1.predict(fvect1)
                else:
                    avg_ssm_estimated_tmp, std_avg_estimated_tmp = nn_model1.predict_dist(fvect1)

                fmat_tmp2 = np.hstack((temp_ts1.values[invalid].reshape(ts_length, 1),
                                       temp_ts2.values[invalid].reshape(ts_length, 1),
                                       temp_ts1_lin.values[invalid].reshape(ts_length, 1) - meanVV,
                                       # .repeat(ts_length).reshape(ts_length, 1),
                                       temp_ts2_lin.values[invalid].reshape(ts_length, 1) - meanVH,
                                       # .repeat(ts_length).reshape(ts_length, 1),
                                       np.array(lon).repeat(ts_length).reshape(ts_length, 1),
                                       np.array(lat).repeat(ts_length).reshape(ts_length, 1),
                                       np.array(trees).repeat(ts_length).reshape(ts_length, 1),
                                       np.array(temp_tslia[invalid]).reshape(ts_length, 1),
                                       k1VV.repeat(ts_length).reshape(ts_length, 1),
                                       k1VH.repeat(ts_length).reshape(ts_length, 1),
                                       k2VV.repeat(ts_length).reshape(ts_length, 1),
                                       k2VH.repeat(ts_length).reshape(ts_length, 1),
                                       k3VV.repeat(ts_length).reshape(ts_length, 1),
                                       k3VH.repeat(ts_length).reshape(ts_length, 1),
                                       k4VV.repeat(ts_length).reshape(ts_length, 1),
                                       k4VH.repeat(ts_length).reshape(ts_length, 1),
                                       l8_series_b1.values[invalid].reshape(ts_length, 1),
                                       np.array(np.median(l8_series_b1.values)).repeat(ts_length).reshape(ts_length,
                                                                                                          1),
                                       l8_series_b2.values[invalid].reshape(ts_length, 1),
                                       np.array(np.median(l8_series_b2.values)).repeat(ts_length).reshape(ts_length,
                                                                                                          1),
                                       l8_series_b3.values[invalid].reshape(ts_length, 1),
                                       np.array(np.median(l8_series_b3.values)).repeat(ts_length).reshape(ts_length,
                                                                                                          1),
                                       l8_series_b4.values[invalid].reshape(ts_length, 1),
                                       np.array(np.median(l8_series_b4.values)).repeat(ts_length).reshape(ts_length,
                                                                                                          1),
                                       l8_series_b5.values[invalid].reshape(ts_length, 1),
                                       np.array(np.median(l8_series_b5.values)).repeat(ts_length).reshape(ts_length,
                                                                                                          1),
                                       l8_series_b6.values[invalid].reshape(ts_length, 1),
                                       np.array(np.median(l8_series_b6.values)).repeat(ts_length).reshape(ts_length,
                                                                                                          1),
                                       l8_series_b7.values[invalid].reshape(ts_length, 1),
                                       np.array(np.median(l8_series_b7.values)).repeat(ts_length).reshape(ts_length,
                                                                                                          1),
                                       l8_series_b10.values[invalid].reshape(ts_length, 1),
                                       np.array(np.median(l8_series_b10.values)).repeat(ts_length).reshape(
                                           ts_length, 1),
                                       l8_series_b11.values[invalid].reshape(ts_length, 1),
                                       np.array(np.median(l8_series_b11.values)).repeat(ts_length).reshape(
                                           ts_length, 1),
                                       l8_series_timediff.values[invalid].reshape(ts_length, 1),
                                       gldas_series.values[invalid].reshape(ts_length, 1) - np.nanmedian(gldas_series)))

                if feature_vect2 is not None:
                    fmat_tmp2 = fmat_tmp2[:, feature_vect2]

                # apply masks to dates and fmat
                dates_tmp = temp_ts1[invalid].index
                # remove nans
                t = np.unique(np.where(np.isnan(fmat_tmp2))[0])
                fmat_tmp2 = np.delete(fmat_tmp2, t, axis=0)
                dates_tmp = np.delete(dates_tmp, t)

                nn_model2 = reg_model2[0]
                nn_scaler2 = reg_model2[1]

                if nn_scaler2 is not None:
                    sub_center2 = nn_scaler2.center_  # [best_features2]
                    sub_scaler2 = nn_scaler2.scale_  # [best_features2]

                    fvect2 = nn_scaler2.transform(fmat_tmp2)
                else:
                    fvect2 = fmat_tmp2

                if calcstd == False:
                    diff_ssm_estimated_tmp = nn_model2.predict(fvect2)
                    ssm_estimated_tmp = diff_ssm_estimated_tmp + avg_ssm_estimated_tmp
                else:
                    diff_ssm_estimated_tmp, std_diff_estimated_tmp = nn_model2.predict_dist(fvect2)
                    ssm_estimated_tmp = diff_ssm_estimated_tmp + avg_ssm_estimated_tmp
                    std_estimated_tmp = std_diff_estimated_tmp + std_avg_estimated_tmp

                # detect outliers
                # outl1 = (outldetector1.predict(fvect1) != 1)
                # outl2 = (outldetector2.predict(fvect2) != 1)
                outl1 = None
                outl2 = None

                if cntr == 0:
                    avg_fail_estimated = np.repeat(avg_ssm_estimated_tmp, len(ssm_estimated_tmp))
                    ssm_failed = np.copy(ssm_estimated_tmp)
                    dates_failed = np.copy(dates_tmp)
                    # outl_comb = np.copy(outl1 | outl2)
                    if calcstd == True:
                        std_failed = np.copy(std_estimated_tmp)
                else:
                    avg_fail_estimated = np.append(avg_fail_estimated,
                                                   np.repeat(avg_ssm_estimated_tmp, len(ssm_estimated_tmp)))
                    ssm_failed = np.append(ssm_failed, np.copy(ssm_estimated_tmp))
                    dates_failed = np.append(dates_failed, np.copy(dates_tmp))
                    # outl_comb = np.append(outl_comb, np.copy(outl1 | outl2))
                    if calcstd == True:
                        std_failed = np.append(std_failed, np.copy(std_estimated_tmp))

                cntr = cntr + 1

        if 'ssm_estimated' in locals():

            ssm_ts = pd.Series(ssm_estimated, index=dates)
            ssm_ts.sort_index(inplace=True)
            avg_ssm_ts = pd.Series(avg_ssm_estimated, index=dates)
            avg_ssm_ts.sort_index(inplace=True)
            avg_avg = avg_ssm_ts.mean()
            avg_ssm_ts = avg_ssm_ts - avg_avg
            ssm_ts = ssm_ts - avg_ssm_ts

            if calcstd == True:
                std_ts = pd.Series(std_estimated, index=dates)
                std_ts.sort_index(inplace=True)

            ssm_failed = pd.Series(ssm_failed, index=dates_failed)
            ssm_failed.sort_index(inplace=True)
            avg_fail_ts = pd.Series(avg_fail_estimated, index=dates_failed)
            avg_fail_ts.sort_index(inplace=True)
            ssm_failed = ssm_failed - (avg_fail_ts - avg_avg)

            # valid = np.where(ssm_ts_out[1] != -9999)
            # xx = ssm_ts_out[0][valid]
            # yy = ssm_ts_out[1][valid]
            if plotts == True:
                plt.figure(figsize=(18, 6))
                # plt.plot(xx,yy)
                # plt.show()
                ssm_ts.plot()
                plt.show()

                if name == None:
                    outfile = self.outpath + 's1ts' + str(x) + '_' + str(y)
                else:
                    outfile = self.outpath + 's1ts_' + name

                plt.savefig(outfile + '.png')
                plt.close()
                csvout = np.array([m.strftime("%d/%m/%Y") for m in ssm_ts.index], dtype=np.str)
                csvout2 = np.array(ssm_ts, dtype=np.str)
                # np.savetxt(self.outpath + 'ts.csv', np.hstack((csvout, csvout2)), fmt="%-10c", delimiter=",")
                with open(outfile + '.csv', 'w') as text_file:
                    [text_file.write(csvout[i] + ', ' + csvout2[i] + '\n') for i in range(len(csvout))]

            # print(ssm_ts)

            print("Done")
            if calcstd == True:
                return (ssm_ts, std_ts)
            else:
                return (ssm_ts, None, None, ssm_failed)
        else:
            return (None, None, None, None)

    def ssm_ts_gee(self, lon, lat, x, y, fdim, name=None, plotts=False, calcstd=False, desc=False):

        from extr_TS import read_NORM_SIG0
        from extr_TS import extr_SIG0_LIA_ts_GEE
        from extr_TS import extr_MODIS_MOD13Q1_ts_GEE
        from extr_TS import extr_L8_median
        from extr_TS import extr_L8_ts_GEE
        from scipy.stats import moment

        # extract parameters
        siglia_ts = extr_SIG0_LIA_ts_GEE(lon, lat,
                                         bufferSize=fdim,
                                         lcmask=False,
                                         trackflt=self.track,
                                         masksnow=False,
                                         maskwinter=False,
                                         desc=desc,
                                         tempfilter=False,
                                         returnLIA=True,
                                         S1B=True)

        # get lc
        ee.Initialize()
        globcover_image = ee.Image("ESA/GLOBCOVER_L4_200901_200912_V2_3")
        roi = ee.Geometry.Point(lon, lat).buffer(fdim)
        land_cover = globcover_image.reduceRegion(ee.Reducer.mode(), roi).getInfo()
        land_cover = land_cover['landcover']
        # get elevation
        elev = ee.Image("CGIAR/SRTM90_V4").reduceRegion(ee.Reducer.median(), roi).getInfo()
        aspe = ee.Terrain.aspect(ee.Image("CGIAR/SRTM90_V4")).reduceRegion(ee.Reducer.median(), roi).getInfo()
        slop = ee.Terrain.slope(ee.Image("CGIAR/SRTM90_V4")).reduceRegion(ee.Reducer.median(), roi).getInfo()
        elev = elev['elevation']
        aspe = aspe['aspect']
        slop = slop['slope']

        epty = 0
        for track_id in siglia_ts.keys():
            if len(siglia_ts[track_id]) == 0:
                epty = 1

        if epty == 1:
            return (None)

        # create ts stack
        cntr = 0
        print(siglia_ts.keys())
        for track_id in siglia_ts.keys():

            mlmodelpath = self.mlmodel
            # mlmodelpath = self.mlmodel + 'gee_' + str(track_id) + '/mlmodel' + str(track_id) + '.p'
            if not os.path.exists(mlmodelpath):
                continue

            # extract model attributes
            mlmodelfile = pickle.load(open(mlmodelpath, 'rb'))
            reg_model1 = (mlmodelfile[0], mlmodelfile[1])

            # calculate k1,...,kN and sig0m
            temp_ts1 = siglia_ts[track_id]['sig0'].astype(np.float)
            temp_ts2 = siglia_ts[track_id]['sig02'].astype(np.float)
            temp_tslia = siglia_ts[track_id]['lia'].astype(np.float)
            valid = np.isfinite(temp_ts1) & np.isfinite(temp_ts2)
            temp_ts1 = temp_ts1[valid]
            temp_ts2 = temp_ts2[valid]
            temp_tslia = temp_tslia[valid]

            ts_length = len(temp_ts1)

            if ts_length < 10:
                continue

            temp_ts1_lin = np.power(10, temp_ts1 / 10.)
            temp_ts2_lin = np.power(10, temp_ts2 / 10.)
            meanVV = np.mean(temp_ts1_lin)
            meanVH = np.mean(temp_ts2_lin)
            stdVV = np.std(temp_ts1_lin)
            stdVH = np.std(temp_ts2_lin)
            k1VV = np.mean(np.log(temp_ts1_lin))
            k1VH = np.mean(np.log(temp_ts2_lin))
            k2VV = np.std(np.log(temp_ts1_lin))
            k2VH = np.std(np.log(temp_ts1_lin))

            # get ndvi
            # tmpndvi, ndvi_success = extr_MODIS_MOD13Q1_ts_GEE(lon, lat,
            #                                                   datefilter=[
            #                                                       np.min(temp_ts1.index).strftime(
            #                                                           '%Y-%m-%d'),
            #                                                       np.max(temp_ts1.index).strftime(
            #                                                           '%Y-%m-%d')])
            # if ndvi_success == 0:
            #     print('No valid NDVI for given location')
            #     continue

            l8_tries = 1
            while l8_tries < 4:
                try:
                    l8_tmp = extr_L8_ts_GEE(lon, lat, fdim)
                except:
                    l8_tries = l8_tries + 1
                else:
                    break

            if l8_tries > 3:
                print('Failed to extract Landsat-8')
                continue

            # get gldas
            gldas_series = pd.Series(index=siglia_ts[track_id][valid].index)
            gldas_veg = pd.Series(index=siglia_ts[track_id][valid].index)
            gldas_et = pd.Series(index=siglia_ts[track_id][valid].index)
            gldas_swe = pd.Series(index=siglia_ts[track_id][valid].index)
            gldas_soilt = pd.Series(index=siglia_ts[track_id][valid].index)
            ndvi_series = pd.Series(index=siglia_ts[track_id][valid].index)

            l8_series_b1 = pd.Series(index=temp_ts1.index)
            l8_series_b2 = pd.Series(index=temp_ts1.index)
            l8_series_b3 = pd.Series(index=temp_ts1.index)
            l8_series_b4 = pd.Series(index=temp_ts1.index)
            l8_series_b5 = pd.Series(index=temp_ts1.index)
            l8_series_b6 = pd.Series(index=temp_ts1.index)
            l8_series_b7 = pd.Series(index=temp_ts1.index)
            l8_series_b10 = pd.Series(index=temp_ts1.index)
            l8_series_b11 = pd.Series(index=temp_ts1.index)
            l8_series_timediff = pd.Series(index=temp_ts1.index)

            for i in gldas_series.index:
                # ndvi_timediff = np.min(np.abs(tmpndvi.index - i))
                # if (ndvi_timediff > dt.timedelta(days=32)):
                #     continue
                gldas_tmp = self.get_gldas(lon, lat,
                                           i.strftime('%Y-%m-%d'),
                                           varname=['SoilMoi0_10cm_inst', 'CanopInt_inst', 'Evap_tavg',
                                                    'SWE_inst', 'SoilTMP0_10cm_inst'])

                gldas_series[i] = gldas_tmp['SoilMoi0_10cm_inst']
                gldas_veg[i] = gldas_tmp['CanopInt_inst']
                gldas_et[i] = gldas_tmp['Evap_tavg']
                gldas_swe[i] = gldas_tmp['SWE_inst']
                gldas_soilt[i] = gldas_tmp['SoilTMP0_10cm_inst']
                # ndvi_series[i] = tmpndvi.iloc[np.argmin(np.abs(tmpndvi.index - i))]

                l8_timediff = np.min(np.abs(l8_tmp.index - i))

                l8_series_b1[i] = l8_tmp['B1'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b2[i] = l8_tmp['B2'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b3[i] = l8_tmp['B3'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b4[i] = l8_tmp['B4'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b5[i] = l8_tmp['B5'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b6[i] = l8_tmp['B6'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b7[i] = l8_tmp['B7'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b10[i] = l8_tmp['B10'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_b11[i] = l8_tmp['B11'].iloc[np.argmin(np.abs(l8_tmp.index - i))]
                l8_series_timediff[i] = l8_timediff.total_seconds()

            # check for nans in the gldas time-series
            valid2 = (gldas_soilt > 275) & (gldas_swe == 0) & np.isfinite(gldas_soilt)

            ts_length = len(np.where(valid2 == True)[0])

            meanVV = np.mean(temp_ts1_lin[valid2])
            meanVH = np.mean(temp_ts2_lin[valid2])
            stdVV = np.std(temp_ts1_lin[valid2])
            stdVH = np.std(temp_ts2_lin[valid2])
            k1VV = np.mean(np.log(temp_ts1_lin[valid2]))
            k1VH = np.mean(np.log(temp_ts2_lin[valid2]))
            k2VV = np.std(np.log(temp_ts1_lin[valid2]))
            k2VH = np.std(np.log(temp_ts1_lin[valid2]))
            k3VV = moment(temp_ts1[valid2], moment=3)
            k3VH = moment(temp_ts2[valid2], moment=3)
            k4VV = moment(temp_ts1[valid2], moment=4)
            k4VH = moment(temp_ts2[valid2], moment=4)

            # get tree cover
            tmp_trees = self.get_tree_cover(lon, lat)
            trees = tmp_trees['tree_canopy_cover']

            fmat_tmp2 = np.hstack((temp_ts1.values[valid2].reshape(ts_length, 1),
                                   temp_ts2.values[valid2].reshape(ts_length, 1),
                                   np.repeat(meanVV, ts_length).reshape(ts_length, 1),
                                   np.repeat(meanVH, ts_length).reshape(ts_length, 1),
                                   np.repeat(stdVV, ts_length).reshape(ts_length, 1),
                                   np.repeat(stdVH, ts_length).reshape(ts_length, 1),
                                   np.repeat(land_cover, ts_length).reshape(ts_length, 1),
                                   np.repeat(trees, ts_length).reshape(ts_length, 1),
                                   gldas_series.values[valid2].reshape(ts_length, 1),
                                   np.repeat(lon, ts_length).reshape(ts_length, 1),
                                   np.repeat(lat, ts_length).reshape(ts_length, 1),
                                   l8_series_b1.values[valid2].reshape(ts_length, 1),
                                   l8_series_b2.values[valid2].reshape(ts_length, 1),
                                   l8_series_b3.values[valid2].reshape(ts_length, 1),
                                   l8_series_b4.values[valid2].reshape(ts_length, 1),
                                   l8_series_b5.values[valid2].reshape(ts_length, 1),
                                   l8_series_b6.values[valid2].reshape(ts_length, 1),
                                   l8_series_b7.values[valid2].reshape(ts_length, 1),
                                   l8_series_b10.values[valid2].reshape(ts_length, 1),
                                   l8_series_b11.values[valid2].reshape(ts_length, 1),
                                   l8_series_timediff.values[valid2].reshape(ts_length, 1)))

            dates_tmp = temp_ts1[valid2].index

            nn_model = reg_model1[0]
            nn_scaler = reg_model1[1]
            if nn_scaler is not None:
                fvect = nn_scaler.transform(fmat_tmp2)
            else:
                fvect = fmat_tmp2

            if calcstd == False:
                ssm_estimated_tmp = nn_model.predict(fvect)
            else:
                ssm_estimated_tmp, std_estimated_tmp = nn_model.predict_dist(fvect)

            if cntr == 0:
                ssm_estimated = np.copy(ssm_estimated_tmp)
                dates = np.copy(dates_tmp)
                if calcstd == True:
                    std_estimated = np.copy(std_estimated_tmp)
            else:
                ssm_estimated = np.append(ssm_estimated, np.copy(ssm_estimated_tmp))
                dates = np.append(dates, np.copy(dates_tmp))
                if calcstd == True:
                    std_estimated = np.append(std_estimated, np.copy(std_estimated_tmp))

            cntr = cntr + 1

        # valid = np.where(ssm_estimated != -9999)
        ssm_ts = pd.Series(ssm_estimated, index=dates)
        ssm_ts.sort_index(inplace=True)
        if calcstd == True:
            std_ts = pd.Series(std_estimated, index=dates)
            std_ts.sort_index(inplace=True)

        # valid = np.where(ssm_ts_out[1] != -9999)
        # xx = ssm_ts_out[0][valid]
        # yy = ssm_ts_out[1][valid]
        if plotts == True:
            plt.figure(figsize=(18, 6))
            # plt.plot(xx,yy)
            # plt.show()
            ssm_ts.plot()
            plt.show()

            if name == None:
                outfile = self.outpath + 's1ts' + str(x) + '_' + str(y)
            else:
                outfile = self.outpath + 's1ts_' + name

            plt.savefig(outfile + '.png')
            plt.close()
            csvout = np.array([m.strftime("%d/%m/%Y") for m in ssm_ts.index], dtype=np.str)
            csvout2 = np.array(ssm_ts, dtype=np.str)
            # np.savetxt(self.outpath + 'ts.csv', np.hstack((csvout, csvout2)), fmt="%-10c", delimiter=",")
            with open(outfile + '.csv', 'w') as text_file:
                [text_file.write(csvout[i] + ', ' + csvout2[i] + '\n') for i in range(len(csvout))]

        # print(ssm_ts)

        print("Done")
        if calcstd == True:
            return (ssm_ts, std_ts)
        else:
            return (ssm_ts, None, None)

    def ssm_ts_gee_with_target(self, lon, lat, x, y, fdim, target, name=None, plotts=False, calcstd=False, desc=False):

        from extr_TS import read_NORM_SIG0
        from extr_TS import extr_SIG0_LIA_ts_GEE
        from extr_TS import extr_MODIS_MOD13Q1_ts_GEE
        from extr_TS import extr_L8_median
        from scipy.stats import moment

        # extract parameters
        siglia_ts = extr_SIG0_LIA_ts_GEE(lon, lat,
                                         bufferSize=fdim,
                                         lcmask=False,
                                         trackflt=self.track,
                                         masksnow=False,
                                         maskwinter=False,
                                         desc=desc,
                                         tempfilter=False,
                                         returnLIA=True,
                                         treemask=True)

        # get lc
        ee.Initialize()
        globcover_image = ee.Image("ESA/GLOBCOVER_L4_200901_200912_V2_3")
        roi = ee.Geometry.Point(lon, lat).buffer(fdim)
        land_cover = globcover_image.reduceRegion(ee.Reducer.mode(), roi).getInfo()
        land_cover = land_cover['landcover']
        # get elevation
        elev = ee.Image("CGIAR/SRTM90_V4").reduceRegion(ee.Reducer.median(), roi).getInfo()
        aspe = ee.Terrain.aspect(ee.Image("CGIAR/SRTM90_V4")).reduceRegion(ee.Reducer.median(), roi).getInfo()
        slop = ee.Terrain.slope(ee.Image("CGIAR/SRTM90_V4")).reduceRegion(ee.Reducer.median(), roi).getInfo()
        elev = elev['elevation']
        aspe = aspe['aspect']
        slop = slop['slope']

        epty = 0
        for track_id in siglia_ts.keys():
            if len(siglia_ts[track_id]) == 0:
                epty = 1

        if epty == 1:
            return (None)

        # create ts stack
        cntr = 0
        print(siglia_ts.keys())
        for track_id in siglia_ts.keys():

            # calculate k1,...,kN and sig0m
            temp_ts1 = siglia_ts[track_id]['sig0'].astype(np.float)
            temp_ts2 = siglia_ts[track_id]['sig02'].astype(np.float)
            temp_tslia = siglia_ts[track_id]['lia'].astype(np.float)
            valid = np.isfinite(temp_ts1) & np.isfinite(temp_ts2)
            temp_ts1 = temp_ts1[valid]
            temp_ts2 = temp_ts2[valid]
            temp_tslia = temp_tslia[valid]

            ts_length = len(temp_ts1)

            if ts_length < 10:
                continue

            temp_ts1_lin = np.power(10, temp_ts1 / 10.)
            temp_ts2_lin = np.power(10, temp_ts2 / 10.)
            meanVV = np.mean(temp_ts1_lin)
            meanVH = np.mean(temp_ts2_lin)
            stdVV = np.std(temp_ts1_lin)
            stdVH = np.std(temp_ts2_lin)
            k1VV = np.mean(np.log(temp_ts1_lin))
            k1VH = np.mean(np.log(temp_ts2_lin))
            k2VV = np.std(np.log(temp_ts1_lin))
            k2VH = np.std(np.log(temp_ts1_lin))

            # get ndvi
            tmpndvi, ndvi_success = extr_MODIS_MOD13Q1_ts_GEE(lon, lat, bufferSize=fdim,
                                                              datefilter=[
                                                                  np.min(siglia_ts[track_id][valid].index).strftime(
                                                                      '%Y-%m-%d'),
                                                                  np.max(siglia_ts[track_id][valid].index).strftime(
                                                                      '%Y-%m-%d')])
            if ndvi_success == 0:
                print('No valid NDVI for given location')
                continue

            l8_tries = 1
            while l8_tries < 4:
                try:
                    l8_medians = extr_L8_median(lon, lat,
                                                startDate=np.min(siglia_ts[track_id][valid].index).strftime('%Y-%m-%d'),
                                                endDate=np.max(siglia_ts[track_id][valid].index).strftime('%Y-%m-%d'))
                except:
                    l8_tries = l8_tries + 1
                else:
                    break

            if l8_tries > 3:
                print('Failed to extract Landsat-8')
                continue

            # get gldas
            gldas_series = pd.Series(index=siglia_ts[track_id][valid].index)
            # gldas_veg = pd.Series(index=siglia_ts[track_id][valid].index)
            # gldas_et = pd.Series(index=siglia_ts[track_id][valid].index)
            gldas_swe = pd.Series(index=siglia_ts[track_id][valid].index)
            gldas_soilt = pd.Series(index=siglia_ts[track_id][valid].index)
            ndvi_series = pd.Series(index=siglia_ts[track_id][valid].index)

            for i in gldas_series.index:
                ndvi_timediff = np.min(np.abs(tmpndvi.index - i))
                if (ndvi_timediff > dt.timedelta(days=32)):
                    continue
                gldas_tmp = self.get_gldas(lon, lat,
                                           i.strftime('%Y-%m-%d'),
                                           varname=[  # 'SoilMoi0_10cm_inst', 'CanopInt_inst', 'Evap_tavg',
                                               'SWE_inst', 'SoilTMP0_10cm_inst'])

                # gldas_series[i] = gldas_tmp['SoilMoi0_10cm_inst']
                # gldas_veg[i] = gldas_tmp['CanopInt_inst']
                # gldas_et[i] = gldas_tmp['Evap_tavg']
                gldas_swe[i] = gldas_tmp['SWE_inst']
                gldas_soilt[i] = gldas_tmp['SoilTMP0_10cm_inst']
                ndvi_series[i] = tmpndvi.iloc[np.argmin(np.abs(tmpndvi.index - i))]

            # check for nans in the gldas time-series
            valid2 = (gldas_soilt > 275) & (gldas_swe == 0) & np.isfinite(ndvi_series)

            ts_length = len(np.where(valid2 == True)[0])

            meanVV = np.mean(temp_ts1_lin[valid2])
            meanVH = np.mean(temp_ts2_lin[valid2])
            stdVV = np.std(temp_ts1_lin[valid2])
            stdVH = np.std(temp_ts2_lin[valid2])
            k1VV = np.mean(np.log(temp_ts1_lin[valid2]))
            k1VH = np.mean(np.log(temp_ts2_lin[valid2]))
            k2VV = np.std(np.log(temp_ts1_lin[valid2]))
            k2VH = np.std(np.log(temp_ts1_lin[valid2]))
            k3VV = moment(temp_ts1[valid2], moment=3)
            k3VH = moment(temp_ts2[valid2], moment=3)
            k4VV = moment(temp_ts1[valid2], moment=4)
            k4VH = moment(temp_ts2[valid2], moment=4)

            # get tree cover
            tmp_trees = self.get_tree_cover(lon, lat)
            trees = tmp_trees['tree_canopy_cover']

            fmat_tmp2 = np.hstack((temp_ts1[valid2].reshape(ts_length, 1),
                                   temp_ts2[valid2].reshape(ts_length, 1),
                                   # temp_tslia[valid2].reshape(ts_length, 1),
                                   ndvi_series[valid2].reshape(ts_length, 1)))  # ,
            # np.repeat(ndvi_series[valid2].median(), ts_length).reshape(ts_length, 1),
            # gldas_veg[valid2].reshape(ts_length, 1),
            # gldas_et[valid2].reshape(ts_length, 1)))

            dates_tmp = temp_ts1[valid2].index

            train_target = np.full(len(dates_tmp), np.nan)
            for i in range(len(dates_tmp)):
                if np.min(np.abs(target.index - dates_tmp[i])) < dt.timedelta(days=1):
                    train_target[i] = target[np.argmin(np.abs(target.index - dates_tmp[i]))]

            valid_training = np.where(np.isfinite(train_target))
            reg_model = self.train_model(fmat_tmp2[valid_training, :].squeeze(), train_target[valid_training].ravel())

            nn_model = reg_model[0]
            nn_scaler = reg_model[1]
            # fvect = nn_scaler.transform(fmat_tmp2)
            fvect = fmat_tmp2
            if calcstd == False:
                ssm_estimated_tmp = nn_model.predict(fvect)
            else:
                ssm_estimated_tmp, std_estimated_tmp = nn_model.predict_dist(fvect)

            if cntr == 0:
                ssm_estimated = np.copy(ssm_estimated_tmp)
                dates = np.copy(dates_tmp)
                if calcstd == True:
                    std_estimated = np.copy(std_estimated_tmp)
            else:
                ssm_estimated = np.append(ssm_estimated, np.copy(ssm_estimated_tmp))
                dates = np.append(dates, np.copy(dates_tmp))
                if calcstd == True:
                    std_estimated = np.append(std_estimated, np.copy(std_estimated_tmp))

            cntr = cntr + 1

        # valid = np.where(ssm_estimated != -9999)
        ssm_ts = pd.Series(ssm_estimated, index=dates)
        ssm_ts.sort_index(inplace=True)
        if calcstd == True:
            std_ts = pd.Series(std_estimated, index=dates)
            std_ts.sort_index(inplace=True)

        # valid = np.where(ssm_ts_out[1] != -9999)
        # xx = ssm_ts_out[0][valid]
        # yy = ssm_ts_out[1][valid]
        if plotts == True:
            plt.figure(figsize=(18, 6))
            # plt.plot(xx,yy)
            # plt.show()
            ssm_ts.plot()
            plt.show()

            if name == None:
                outfile = self.outpath + 's1ts' + str(x) + '_' + str(y)
            else:
                outfile = self.outpath + 's1ts_' + name

            plt.savefig(outfile + '.png')
            plt.close()
            csvout = np.array([m.strftime("%d/%m/%Y") for m in ssm_ts.index], dtype=np.str)
            csvout2 = np.array(ssm_ts, dtype=np.str)
            # np.savetxt(self.outpath + 'ts.csv', np.hstack((csvout, csvout2)), fmt="%-10c", delimiter=",")
            with open(outfile + '.csv', 'w') as text_file:
                [text_file.write(csvout[i] + ', ' + csvout2[i] + '\n') for i in range(len(csvout))]

        # print(ssm_ts)

        print("Done")
        if calcstd == True:
            return (ssm_ts, std_ts)
        else:
            return (ssm_ts, None)

    def train_model(self, x, y):

        import scipy.stats
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.decomposition import PCA
        from sklearn.linear_model import TheilSenRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.svm import NuSVR

        # # filter nan values
        # valid = ~np.any(np.isinf(x))
        # track_target = x[valid]
        # track_features = y[valid,:]
        # filter nan
        valid = ~np.any(np.isnan(x), axis=1)
        track_target = y[valid]
        track_features = x[valid, :]
        # filter bad ssm values
        valid = np.where(track_target > 0)  # & (self.features[:,-1] == itrack))
        track_target = track_target[valid[0]]
        track_features = track_features[valid[0], :]

        # scaling
        scaler = sklearn.preprocessing.RobustScaler().fit(track_features)
        features = scaler.transform(track_features)
        # fatures = track_features

        # ...----...----...----...----...----...----...----...----...----...
        # SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR
        # ---....---....---....---....---....---....---....---....---....---

        dictCV = dict(C=np.logspace(-2, 2, 5),
                      gamma=np.logspace(-2, -0.5, 5),
                      epsilon=np.logspace(-2, -0.5, 5),
                      # nu = [0.1,0.25,0.5,0.75,1.0],
                      # degree =  [2,3],
                      kernel=['rbf'])

        # specify kernel
        svr_rbf = SVR()

        # parameter tuning -> grid search
        start = time()
        print('Model training startet at: ' + str(start))
        #
        # SVR --- SVR --- SVR --- SVR --- SVR --- SVR --- SVR
        #
        gdCV = GridSearchCV(estimator=svr_rbf,
                            param_grid=dictCV,
                            n_jobs=4,
                            verbose=1,
                            # cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=42),# shuffle=True),
                            # cv = GroupKFold(n_splits=5).split(x_train, y_train, self.loc_id[valid]),
                            cv=KFold(n_splits=3, random_state=42),
                            scoring='r2')
        # gdCV = SVR(kernel='poly', C=1e3, degree=2, cache_size=500)

        gdCV.fit(x, y)
        # print(gdCV.best_score_)
        # print(gdCV.best_estimator_)

        return (gdCV, scaler)

    def get_gldas(self, x, y, date, varname=['SoilMoi0_10cm_inst']):

        def get_ts(image):
            return image.reduceRegion(ee.Reducer.median(), roi, 50)

        ee.Initialize()
        doi = ee.Date(date)
        gldas = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H") \
            .select(varname) \
            .filterDate(doi, doi.advance(3, 'hour'))

        gldas_img = ee.Image(gldas.first())
        roi = ee.Geometry.Point(x, y).buffer(100)
        try:
            return (gldas_img.reduceRegion(ee.Reducer.median(), roi, 50).getInfo())
        except:
            return dict([(k, None) for k in varname])

    def get_tree_cover(self, x, y, buffer=20):

        ee.Initialize()
        roi = ee.Geometry.Point(x, y).buffer(buffer)
        tree_cover_image = ee.ImageCollection("GLCF/GLS_TCC").filterBounds(roi).filter(
            ee.Filter.eq('year', 2010)).mosaic()
        return (tree_cover_image.reduceRegion(ee.Reducer.mode(), roi, 10).getInfo())

    def ssm_ts_gee_alternative(self, lon, lat, x, y, fdim, name=None, plotts=False):

        from extr_TS import read_NORM_SIG0
        from extr_TS import extr_SIG0_LIA_ts_GEE
        from scipy.stats import moment

        # ee.Initialize()

        # extract model attributes
        class_model = self.mlmodel[1]
        reg_model = self.mlmodel[0]

        # extract parameters
        siglia_ts = extr_SIG0_LIA_ts_GEE(lon, lat, bufferSize=fdim)
        # siglia_ts_alldays = extr_SIG0_LIA_ts_GEE(lon, lat, bufferSize=fdim, maskwinter=False)

        # create ts stack
        cntr = 0
        for track_id in siglia_ts.keys():

            # calculate k1,...,kN and sig0m
            temp_ts1 = siglia_ts[track_id][1]['sig0'].astype(np.float)
            temp_ts2 = siglia_ts[track_id][1]['sig02'].astype(np.float)
            valid = np.where(np.isfinite(temp_ts1) & np.isfinite(temp_ts2))
            temp_ts1 = temp_ts1[valid]
            temp_ts2 = temp_ts2[valid]

            ts_length = len(temp_ts1)

            if ts_length < 10:
                continue

            temp_ts1_lin = np.power(10, temp_ts1 / 10.)
            temp_ts2_lin = np.power(10, temp_ts2 / 10.)
            sig0mVV = 10 * np.log10(np.mean(temp_ts1_lin))
            sig0mVH = 10 * np.log10(np.mean(temp_ts2_lin))
            sig0sdVV = 10 * np.log10(np.std(temp_ts1_lin))
            sig0sdVH = 10 * np.log10(np.std(temp_ts2_lin))
            k1VV = np.mean(np.log(temp_ts1_lin))
            k1VH = np.mean(np.log(temp_ts2_lin))
            k2VV = np.std(np.log(temp_ts1_lin))
            k2VH = np.std(np.log(temp_ts1_lin))
            k3VV = moment(temp_ts1, moment=3, nan_policy='omit')
            k3VH = moment(temp_ts2, moment=3, nan_policy='omit')
            k4VV = moment(temp_ts1, moment=4, nan_policy='omit')
            k4VH = moment(temp_ts2, moment=4, nan_policy='omit')

            # siglia_ts = None
            # temp_ts1 = siglia_ts_alldays[track_id][1]['sig0'].astype(np.float)
            # temp_ts2 = siglia_ts_alldays[track_id][1]['sig02'].astype(np.float)
            # valid = np.where(np.isfinite(temp_ts1) & np.isfinite(temp_ts2))
            # temp_ts1 = temp_ts1[valid]
            # temp_ts2 = temp_ts2[valid]

            # ts_length = len(temp_ts1)

            # fmat_tmp = np.hstack((np.repeat(sig0mVV, ts_length).reshape(ts_length, 1),
            #                       np.repeat(sig0sdVV, ts_length).reshape(ts_length, 1),
            #                       np.repeat(sig0mVH, ts_length).reshape(ts_length, 1),
            #                       np.repeat(sig0sdVH, ts_length).reshape(ts_length, 1),
            #                       temp_ts1.reshape(ts_length, 1),
            #                       temp_ts2.reshape(ts_length, 1),
            #                       np.repeat(k1VV, ts_length).reshape(ts_length, 1),
            #                       np.repeat(k1VH, ts_length).reshape(ts_length, 1)))
            #                       #np.repeat(k2VV, ts_length).reshape(ts_length, 1),
            #                       #np.repeat(k2VH, ts_length).reshape(ts_length, 1)))

            model_attr_tmp = np.hstack((np.repeat(k1VV, ts_length).reshape(ts_length, 1),
                                        np.repeat(k1VH, ts_length).reshape(ts_length, 1),
                                        np.repeat(k2VV, ts_length).reshape(ts_length, 1),
                                        np.repeat(k2VH, ts_length).reshape(ts_length, 1)))
            fmat_tmp = np.hstack((temp_ts1.reshape(ts_length, 1),
                                  temp_ts2.reshape(ts_length, 1)))

            # dates_tmp = siglia_ts_alldays[track_id][0][valid]
            dates_tmp = siglia_ts[track_id][0][valid]

            if cntr == 0:
                model_attr = model_attr_tmp
                fmat = fmat_tmp
                dates = dates_tmp
            else:
                model_attr = np.vstack((model_attr, model_attr_tmp))
                fmat = np.vstack((fmat, fmat_tmp))
                dates = np.concatenate((dates, dates_tmp))

            cntr = cntr + 1

        ssm_estimated = np.full(len(dates), -9999, dtype=np.float)
        ssm_error = np.full(len(dates), -9999, dtype=np.float)
        for i in range(len(dates)):

            nn = class_model.predict(model_attr[i, :].reshape(1, -1))
            nn = int(nn[0])
            if reg_model[nn]['quality'] == 'good':
                nn_model = reg_model[nn]['model']
                nn_scaler = reg_model[nn]['scaler']
                fvect = nn_scaler.transform(fmat[i, :].reshape(1, -1))
                ssm_estimated[i] = nn_model.predict(fvect)
                ssm_error[i] = reg_model[nn]['rmse']
            else:
                nn_model = reg_model[nn]['model']
                nn_scaler = reg_model[nn]['scaler']
                fvect = nn_scaler.transform(fmat[i, :].reshape(1, -1))
                ssm_estimated[i] = nn_model.predict(fvect)
                ssm_error[i] = reg_model[nn]['rmse']
                # ssm_estimated[i] = -50

        # nn_model = self.mlmodel[0]
        # nn_scaler = self.mlmodel[1]
        # fmat = nn_scaler.transform(fmat)
        # ssm_estimated = nn_model.predict(fmat)

        valid = np.where(ssm_estimated != -9999)
        ssm_ts = pd.Series(ssm_estimated[valid], index=dates[valid])
        error_ts = pd.Series(ssm_error[valid], index=dates[valid])
        ssm_ts.sort_index(inplace=True)
        error_ts.sort_index(inplace=True)

        # valid = np.where(ssm_ts_out[1] != -9999)
        # xx = ssm_ts_out[0][valid]
        # yy = ssm_ts_out[1][valid]
        if plotts == True:
            plt.figure(figsize=(18, 6))
            # plt.plot(xx,yy)
            # plt.show()
            ssm_ts.plot()
            plt.show()

            if name == None:
                outfile = self.outpath + 's1ts' + str(x) + '_' + str(y)
            else:
                outfile = self.outpath + 's1ts_' + name

            plt.savefig(outfile + '.png')
            plt.close()
            csvout = np.array([m.strftime("%d/%m/%Y") for m in ssm_ts.index], dtype=np.str)
            csvout2 = np.array(ssm_ts, dtype=np.str)
            # np.savetxt(self.outpath + 'ts.csv', np.hstack((csvout, csvout2)), fmt="%-10c", delimiter=",")
            with open(outfile + '.csv', 'w') as text_file:
                [text_file.write(csvout[i] + ', ' + csvout2[i] + '\n') for i in range(len(csvout))]

        print(ssm_ts)
        print(error_ts)

        print("Done")
        return (ssm_ts, error_ts)

    def ssm_map(self, date=None, path=None):

        from joblib import Parallel, delayed, load, dump
        import sys
        import tempfile
        import shutil

        for tname in self.tiles:

            # for date in ['D20160628_170640']:

            for date in self.get_filenames(tname):

                # check if file already exists
                if os.path.exists(self.outpath + tname + '_SSM_' + date + '.tif'):
                    print(tname + ' / ' + date + ' allready processed')
                    continue

                print("Retrieving soil moisture for " + tname + " / " + date)

                # get sig0 image to derive ssm
                bacArrs = self.get_sig0_lia(tname, date)
                # terrainArrs = self.get_terrain(tname)
                paramArr = self.get_params(tname, path)

                # create masks
                if self.uselc == True:
                    lc_mask = self.create_LC_mask(tname, bacArrs)
                sig0_mask = (bacArrs['sig0vv'][0] != -9999) & \
                            (bacArrs['sig0vv'][0] >= -2500) & \
                            (bacArrs['sig0vh'][0] != -9999) & \
                            (bacArrs['sig0vh'][0] >= -2500) & \
                            (bacArrs['lia'][0] >= 1000) & \
                            (bacArrs['lia'][0] <= 5000)
                # terr_mask = (terrainArrs['h'][0] != -9999) & \
                #             (terrainArrs['a'][0] != -9999) & \
                #             (terrainArrs['s'][0] != -9999)
                param_mask = (paramArr['k1VH'][0] != -9999) & \
                             (paramArr['k1VV'][0] != -9999) & \
                             (paramArr['k2VH'][0] != -9999) & \
                             (paramArr['k2VV'][0] != -9999)

                # extrapolation mask
                # sig0vv_extr = ((((bacArrs['sig0vv'][0]/100.0) - self.mlmodel[1].mean_[4]) / self.mlmodel[1].std_[4]) > self.mlmodel[0].best_estimator_.support_vectors_[:,4].min()) & \
                #             ((((bacArrs['sig0vv'][0] / 100.0) - self.mlmodel[1].mean_[4]) / self.mlmodel[1].std_[4]) < self.mlmodel[0].best_estimator_.support_vectors_[:, 4].max())
                # sig0vh_extr = ((((bacArrs['sig0vh'][0] / 100.0) - self.mlmodel[1].mean_[5]) / self.mlmodel[1].std_[5]) >
                #                self.mlmodel[0].best_estimator_.support_vectors_[:, 5].min()) & \
                #               ((((bacArrs['sig0vh'][0] / 100.0) - self.mlmodel[1].mean_[5]) / self.mlmodel[1].std_[5]) <
                #                self.mlmodel[0].best_estimator_.support_vectors_[:, 5].max())
                # k1vv_extr = ((((paramArr['k1VV'][0] / 100.0) - self.mlmodel[1].mean_[0]) / self.mlmodel[1].std_[0]) >
                #                self.mlmodel[0].best_estimator_.support_vectors_[:, 0].min()) & \
                #               ((((paramArr['k1VV'][0] / 100.0) - self.mlmodel[1].mean_[0]) / self.mlmodel[1].std_[0]) <
                #                self.mlmodel[0].best_estimator_.support_vectors_[:, 0].max())
                # k1vh_extr = ((((paramArr['k1VH'][0] / 100.0) - self.mlmodel[1].mean_[1]) / self.mlmodel[1].std_[1]) >
                #              self.mlmodel[0].best_estimator_.support_vectors_[:, 1].min()) & \
                #             ((((paramArr['k1VH'][0] / 100.0) - self.mlmodel[1].mean_[1]) / self.mlmodel[1].std_[1]) <
                #              self.mlmodel[0].best_estimator_.support_vectors_[:, 1].max())
                # k2vv_extr = ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[2]) / self.mlmodel[1].std_[2]) >
                #              self.mlmodel[0].best_estimator_.support_vectors_[:, 2].min()) & \
                #             ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[2]) / self.mlmodel[1].std_[2]) <
                #              self.mlmodel[0].best_estimator_.support_vectors_[:, 2].max())
                # k2vh_extr = ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[3]) / self.mlmodel[1].std_[3]) >
                #              self.mlmodel[0].best_estimator_.support_vectors_[:, 3].min()) & \
                #             ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[3]) / self.mlmodel[1].std_[3]) <
                #              self.mlmodel[0].best_estimator_.support_vectors_[:, 3].max())
                # extr_mask = sig0vv_extr & sig0vh_extr & k1vv_extr & k1vh_extr & k2vv_extr & k2vh_extr

                # combined mask
                if self.uselc == True:
                    # mask = lc_mask & sig0_mask & terr_mask & param_mask
                    mask = lc_mask & sig0_mask & param_mask
                else:
                    # mask = sig0_mask & terr_mask & param_mask
                    mask = sig0_mask & param_mask

                # resample
                # bacArrs, paramArr, terrainArrs = self.resample(bacArrs,terrainArrs,paramArr, mask, 5)
                bacArrs, paramArr = self.resample(bacArrs, paramArr, mask, 5)
                print('Resampled: check')

                sig0_mask = (bacArrs['sig0vv'][0] != -9999) & \
                            (bacArrs['sig0vv'][0] >= -2500) & \
                            (bacArrs['sig0vh'][0] != -9999) & \
                            (bacArrs['sig0vh'][0] >= -2500) & \
                            (bacArrs['lia'][0] >= 1000) & \
                            (bacArrs['lia'][0] <= 5000)
                # terr_mask = (terrainArrs['h'][0] != -9999) & \
                #             (terrainArrs['a'][0] != -9999) & \
                #             (terrainArrs['s'][0] != -9999)
                param_mask = (paramArr['k1VH'][0] != -9999) & \
                             (paramArr['k1VV'][0] != -9999) & \
                             (paramArr['k2VH'][0] != -9999) & \
                             (paramArr['k2VV'][0] != -9999)

                # extrapolation mask
                sig0vv_extr = ((((bacArrs['sig0vv'][0] / 100.0) - self.mlmodel[1].mean_[4]) / math.sqrt(
                    self.mlmodel[1].var_[4])) >
                               self.mlmodel[0].best_estimator_.support_vectors_[:, 4].min()) & \
                              ((((bacArrs['sig0vv'][0] / 100.0) - self.mlmodel[1].mean_[4]) / math.sqrt(
                                  self.mlmodel[1].var_[4])) <
                               self.mlmodel[0].best_estimator_.support_vectors_[:, 4].max())
                sig0vh_extr = ((((bacArrs['sig0vh'][0] / 100.0) - self.mlmodel[1].mean_[5]) / math.sqrt(
                    self.mlmodel[1].var_[5])) >
                               self.mlmodel[0].best_estimator_.support_vectors_[:, 5].min()) & \
                              ((((bacArrs['sig0vh'][0] / 100.0) - self.mlmodel[1].mean_[5]) / math.sqrt(
                                  self.mlmodel[1].var_[5])) <
                               self.mlmodel[0].best_estimator_.support_vectors_[:, 5].max())
                k1vv_extr = ((((paramArr['k1VV'][0] / 100.0) - self.mlmodel[1].mean_[0]) / math.sqrt(
                    self.mlmodel[1].var_[0])) >
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 0].min()) & \
                            ((((paramArr['k1VV'][0] / 100.0) - self.mlmodel[1].mean_[0]) / math.sqrt(
                                self.mlmodel[1].var_[0])) <
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 0].max())
                k1vh_extr = ((((paramArr['k1VH'][0] / 100.0) - self.mlmodel[1].mean_[1]) / math.sqrt(
                    self.mlmodel[1].var_[1])) >
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 1].min()) & \
                            ((((paramArr['k1VH'][0] / 100.0) - self.mlmodel[1].mean_[1]) / math.sqrt(
                                self.mlmodel[1].var_[1])) <
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 1].max())
                k2vv_extr = ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[2]) / math.sqrt(
                    self.mlmodel[1].var_[2])) >
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 2].min()) & \
                            ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[2]) / math.sqrt(
                                self.mlmodel[1].var_[2])) <
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 2].max())
                k2vh_extr = ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[3]) / math.sqrt(
                    self.mlmodel[1].var_[3])) >
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 3].min()) & \
                            ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[3]) / math.sqrt(
                                self.mlmodel[1].var_[3])) <
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 3].max())
                extr_mask = sig0vv_extr & sig0vh_extr & k1vv_extr & k1vh_extr & k2vv_extr & k2vh_extr

                # create masks
                if self.uselc == True:
                    lc_mask = self.create_LC_mask(tname, bacArrs, res=1000)

                # combined mask
                if self.uselc == True:
                    # mask = lc_mask & sig0_mask & terr_mask & param_mask
                    mask = lc_mask & sig0_mask & param_mask & extr_mask
                else:
                    # mask = sig0_mask & terr_mask & param_mask
                    mask = sig0_mask & param_mask & extr_mask

                valid_ind = np.where(mask == True)
                valid_ind = np.ravel_multi_index(valid_ind, (1000, 1000))

                # vv_sstd = _local_std(bacArrs['sig0vv'][0], -9999, valid_ind)
                # vh_sstd = _local_std(bacArrs['sig0vh'][0], -9999, valid_ind)
                # lia_sstd = _local_std(bacArrs['lia'][0], -9999, valid_ind, "lia")

                # bacStats = {"vv": vv_sstd, "vh": vh_sstd, 'lia': lia_sstd}
                bacStats = {'vv': bacArrs['sig0vv'][0], 'vh': bacArrs['sig0vh'][0], 'lia': bacArrs['lia'][0]}

                ssm_out = np.full((1000, 1000), -9999, dtype=np.float32)

                # parallel prediction
                if not hasattr(sys.stdin, 'close'):
                    def dummy_close():
                        pass

                    sys.stdin.close = dummy_close

                ind_splits = np.array_split(valid_ind, 8)

                # prepare multi processing
                # dump arrays to temporary folder
                temp_folder = tempfile.mkdtemp()
                filename_in = os.path.join(temp_folder, 'joblib_dump1.mmap')
                if os.path.exists(filename_in): os.unlink(filename_in)
                # _ = dump((bacArrs, terrainArrs, paramArr, bacStats), filename_in)
                _ = dump((bacArrs, paramArr, bacStats), filename_in)
                large_memmap = load(filename_in, mmap_mode='r+')
                # output
                filename_out = os.path.join(temp_folder, 'joblib_dump2.mmap')
                ssm_out = np.memmap(filename_out, dtype=np.float32, mode='w+', shape=(1000, 1000))
                ssm_out[:] = -9999

                # extract model attributes
                # model_attrs = self.mlmodel[0][0]['model_attr']

                # for mit in range(1, len(self.mlmodel[0])):
                #    model_attrs = np.vstack((model_attrs, self.mlmodel[0][mit]['model_attr']))
                # reg_model = self.mlmodel[0]
                # class_model = self.mlmodel[1]

                # predict SSM
                # Parallel(n_jobs=8, verbose=5, max_nbytes=None)(delayed(_estimate_ssm_alternative)(large_memmap[0],large_memmap[1],large_memmap[2], large_memmap[3],ssm_out,i,reg_model,class_model) for i in ind_splits)
                Parallel(n_jobs=8, verbose=5, max_nbytes=None)(
                    delayed(_estimate_ssm)(large_memmap[0], large_memmap[1], large_memmap[2], ssm_out, i, self.mlmodel)
                    for i in ind_splits)

                # _estimate_ssm(bacArrs, terrainArrs, paramArr, bacStats, ssm_out, valid_ind, self.mlmodel)

                # write ssm out
                self.write_ssm(tname, date, bacArrs, ssm_out)

                try:
                    shutil.rmtree(temp_folder)
                except:
                    print("Failed to delete: " + temp_folder)

                print('HELLO')

    def get_sig0_lia(self, tname, date):

        # read sig0 vv/vh and lia in arrays
        tile = SgrtTile(dir_root=self.sgrt_root,
                        product_id='S1AIWGRDH',
                        soft_id='A0111',
                        product_name='resampled',
                        ftile=self.subgrid + '010M_' + tname,
                        src_res=10)
        tile_lia = SgrtTile(dir_root=self.sgrt_root,
                            product_id='S1AIWGRDH',
                            soft_id='A0111',
                            product_name='resampled',
                            ftile=self.subgrid + '010M_' + tname,
                            src_res=10)

        sig0vv = tile.read_tile(pattern=date + '.*VV.*')
        sig0vh = tile.read_tile(pattern=date + '.*VH.*')
        lia = tile_lia.read_tile(pattern=date + '.*_LIA.*')

        return {'sig0vv': sig0vv, 'sig0vh': sig0vh, 'lia': lia}

    def get_terrain(self, tname):

        if self.subgrid == 'AF':
            h = np.full([10000, 10000], 0)
            elevGeo = 0
            a = np.full([10000, 10000], 0)
            aspGeo = 0
            s = np.full([10000, 10000], 0)
            slpGeo = 0
        else:

            # elevation
            filename = glob.glob(self.dempath + '*' + tname + '.tif')
            elev = gdal.Open(filename[0], gdal.GA_ReadOnly)
            elevBand = elev.GetRasterBand(1)
            elevGeo = elev.GetGeoTransform()
            h = elevBand.ReadAsArray()
            elev = None

            # aspect
            filename = glob.glob(self.dempath + '*' + tname + '_aspect.tif')
            asp = gdal.Open(filename[0], gdal.GA_ReadOnly)
            aspBand = asp.GetRasterBand(1)
            aspGeo = asp.GetGeoTransform()
            a = aspBand.ReadAsArray()
            asp = None

            # slope
            filename = glob.glob(self.dempath + '*' + tname + '_slope.tif')
            slp = gdal.Open(filename[0], gdal.GA_ReadOnly)
            slpBand = slp.GetRasterBand(1)
            slpGeo = slp.GetGeoTransform()
            s = slpBand.ReadAsArray()
            slp = None

        return {'h': (h, elevGeo), 'a': (a, aspGeo), 's': (s, slpGeo)}

    def get_params(self, tname, path=None):

        # get slope and sig0 mean and standard deviation
        Stile = SgrtTile(dir_root=self.sgrt_root,
                         product_id='S1AIWGRDH',
                         soft_id='B0212',
                         product_name='sig0m',
                         ftile=self.subgrid + '010M_' + tname,
                         src_res=10)

        if path == None:
            path = ''

        # slpVV = Stile.read_tile(pattern='.*SIGSL.*VV'+path+'.*T1$')
        # slpVH = Stile.read_tile(pattern='.*SIGSL.*VH'+path+'.*T1$')
        # sig0mVV = Stile.read_tile(pattern='.*SIG0M.*VV'+path+'.*T1$')
        # sig0mVH = Stile.read_tile(pattern='.*SIG0M.*VH'+path+'.*T1$')
        # sig0sdVV = Stile.read_tile(pattern='.*SIGSD.*VV'+path+'.*T1$')
        # sig0sdVH = Stile.read_tile(pattern='.*SIGSD.*VH'+path+'.*T1$')
        k1VV = Stile.read_tile(pattern='.*K1.*VV' + path + '.*T1$')
        k1VH = Stile.read_tile(pattern='.*K1.*VH' + path + '.*T1$')
        k2VV = Stile.read_tile(pattern='.*K2.*VV' + path + '.*T1$')
        k2VH = Stile.read_tile(pattern='.*K2.*VH' + path + '.*T1$')
        # k3VV = Stile.read_tile(pattern='.*K3.*VV' + path + '.*T1$')
        # k3VH = Stile.read_tile(pattern='.*K3.*VH' + path + '.*T1$')
        # k4VV = Stile.read_tile(pattern='.*K4.*VV' + path + '.*T1$')
        # k4VH = Stile.read_tile(pattern='.*K4.*VH' + path + '.*T1$')

        return {  # 'slpVV': slpVV,
            # 'slpVH': slpVH,
            # 'sig0mVV': sig0mVV,
            # 'sig0mVH': sig0mVH,
            # 'sig0sdVV': sig0sdVV,
            # 'sig0sdVH': sig0sdVH,
            'k1VV': k1VV,
            'k1VH': k1VH,
            'k2VV': k2VV,
            'k2VH': k2VH}
        # 'k3VV': k3VV,
        # 'k3VH': k3VH,
        # 'k4VV': k4VV,
        # 'k4VH': k4VH}

    def create_LC_mask(self, tname, bacArrs, res=10000):

        # get tile name in 75m lc grid
        eq7tile = Equi7Tile(self.subgrid + '010M_' + tname)
        tname75 = eq7tile.find_family_tiles(res=75)
        # load lc array, resampled to 10m
        lcArr = self.get_lc(tname75[0], bacArrs, res)

        # generate mask
        tmp = np.array(lcArr[0])
        mask = (tmp == 10) | (tmp == 12) | (tmp == 13) | (tmp == 18) | (tmp == 26) | (tmp == 29) | (tmp == 32) | \
               (tmp == 11) | (tmp == 19) | (tmp == 20) | (tmp == 21) | (tmp == 27) | (tmp == 28)

        return mask

    def get_lc(self, tname, bacArrs, res=10000):

        # get tile name of 75 Equi7 grid to check land-cover
        LCtile = SgrtTile(dir_root=self.sgrt_root,
                          product_id='S1AIWGRDH',
                          soft_id='E0110',
                          product_name='CORINE06',
                          ftile=self.subgrid + '075M_' + tname,
                          src_res=75)

        LCfilename = [xs for xs in LCtile._tile_files]
        LCfilename = LCtile.dir + '/' + LCfilename[0] + '.tif'
        LC = gdal.Open(LCfilename, gdal.GA_ReadOnly)
        # LCband = LC.GetRasterBand(1)
        LCgeo = LC.GetGeoTransform()
        LCproj = LC.GetProjection()
        # LC = LCband.ReadAsArray()

        # resample to 10m grid
        dst_proj = bacArrs['sig0vv'][1]['spatialreference']
        dst_geotrans = bacArrs['sig0vv'][1]['geotransform']
        dst_width = res
        dst_height = res
        # define output
        LCres_filename = self.outpath + 'tmp_LCres.tif'
        LCres = gdal.GetDriverByName('GTiff').Create(LCres_filename, dst_width, dst_height, 1, gdalconst.GDT_Int16)
        LCres.SetGeoTransform(dst_geotrans)
        LCres.SetProjection(dst_proj)
        # resample
        gdal.ReprojectImage(LC, LCres, LCproj, dst_proj, gdalconst.GRA_NearestNeighbour)

        del LC, LCres

        LC = gdal.Open(LCres_filename, gdal.GA_ReadOnly)
        LCband = LC.GetRasterBand(1)
        LCgeo = LC.GetGeoTransform()
        LC = LCband.ReadAsArray()

        return (LC, LCgeo)

    def write_ssm(self, tname, date, bacArrs, outarr):

        # write ssm map
        dst_proj = bacArrs['sig0vv'][1]['spatialreference']
        dst_geotrans = bacArrs['sig0vv'][1]['geotransform']
        dst_width = 1000
        dst_height = 1000

        # set up output file
        ssm_path = self.outpath + tname + '_SSM_' + date + '.tif'
        ssm_map = gdal.GetDriverByName('GTiff').Create(ssm_path, dst_width, dst_height, 1, gdalconst.GDT_Int16)
        ssm_map.SetGeoTransform(dst_geotrans)
        ssm_map.SetProjection(dst_proj)

        # transform array to byte
        valid = np.where(outarr != -9999)
        novalid = np.where(outarr == -9999)
        outarr[valid] = np.around(outarr[valid] * 100)
        outarr[novalid] = -9999
        outarr = outarr.astype(dtype=np.int16)

        # write data
        ssm_outband = ssm_map.GetRasterBand(1)
        ssm_outband.WriteArray(outarr)
        ssm_outband.FlushCache()
        ssm_outband.SetNoDataValue(-9999)

        del ssm_map

    # def get_gldas(self, x, y, date):
    #
    #     def get_ts(image):
    #         return image.reduceRegion(ee.Reducer.median(), roi, 50)
    #
    #     ee.Initialize()
    #     doi = ee.Date(date)
    #     gldas = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H") \
    #               .select('SoilMoi0_10cm_inst') \
    #               .filterDate(doi, doi.advance(3, 'hour'))
    #
    #     gldas_img = ee.Image(gldas.first())
    #     roi = ee.Geometry.Point(x, y).buffer(100)
    #     try:
    #         return (gldas_img.reduceRegion(ee.Reducer.median(), roi, 50).getInfo())
    #     except:
    #         return {'SoilMoi0_10cm_inst': np.nan}

    def moving_average(self, a, n=3):
        return median_filter(a, size=n, mode='constant', cval=-9999)

    def resample(self, bacArrs, paramArrs, mask, resolution=10):

        # bacArrs
        # bacArrsRes = dict()
        for key in bacArrs.keys():
            tmparr = np.array(bacArrs[key][0], dtype=np.float32)
            tmpmeta = bacArrs[key][1]
            tmpmeta['geotransform'] = (
                tmpmeta['geotransform'][0], 100.0, tmpmeta['geotransform'][2], tmpmeta['geotransform'][3],
                tmpmeta['geotransform'][4], -100.0)
            tmparr[mask == False] = np.nan
            # tmparr = self.moving_average(tmparr, resolution)
            if key == 'sig0vv' or key == 'sig0vh':
                tmparr[mask] = np.power(10, tmparr[mask] / 1000.0)
                # tmparr[mask == False] = np.nan
                tmparr_res = np.full((1000, 1000), fill_value=np.nan)
                for ix in range(1000):
                    for iy in range(1000):
                        tmparr_res[iy, ix] = np.nanmean(tmparr[(iy * 10):(iy * 10 + 10), (ix * 10):(ix * 10 + 10)])

                tmparr_res = 1000. * np.log10(tmparr_res)
            else:
                # tmparr_res = (tmparr[::10, ::10] + tmparr[::10, 1::10] + tmparr[1::10, ::10] + tmparr[1::10,
                #                                                                               1::10]) / 100.
                tmparr_res = np.full((1000, 1000), fill_value=np.nan)
                for ix in range(1000):
                    for iy in range(1000):
                        tmparr_res[iy, ix] = np.nanmean(tmparr[(iy * 10):(iy * 10 + 10), (ix * 10):(ix * 10 + 10)])

            # bacArrs[key][0][:,:] = np.array(tmparr, dtype=np.int16)
            tmparr_res[np.isnan(tmparr_res)] = -9999
            bacArrs[key] = (np.array(tmparr_res, dtype=np.int16), tmpmeta)

        # paramArrs
        for key in paramArrs.keys():
            tmparr = np.array(paramArrs[key][0], dtype=np.float32)
            tmpmeta = paramArrs[key][1]
            tmpmeta['geotransform'] = (
                tmpmeta['geotransform'][0], 100.0, tmpmeta['geotransform'][2], tmpmeta['geotransform'][3],
                tmpmeta['geotransform'][4], -100.0)
            tmparr[mask == False] = np.nan
            # tmparr = self.moving_average(tmparr, resolution)
            # tmparr_res = (tmparr[::10, ::10] + tmparr[::10, 1::10] + tmparr[1::10, ::10] + tmparr[1::10, 1::10]) / 100.
            tmparr_res = np.full((1000, 1000), fill_value=np.nan)
            for ix in range(1000):
                for iy in range(1000):
                    tmparr_res[iy, ix] = np.nanmean(tmparr[(iy * 10):(iy * 10 + 10), (ix * 10):(ix * 10 + 10)])

            tmparr_res[np.isnan(tmparr_res)] = -9999
            paramArrs[key] = (tmparr_res, tmpmeta)

        # terrainArrs
        # for key in terrainArrs.keys():
        #     tmparr = np.array(terrainArrs[key][0], dtype=np.float32)
        #     tmparr[mask == False] = np.nan
        #     tmparr = self.moving_average(tmparr, resolution)
        #     terrainArrs[key][0][:,:] = tmparr

        return (bacArrs, paramArrs)

    def get_filenames(self, tname):

        tile = SgrtTile(dir_root=self.sgrt_root,
                        product_id='S1AIWGRDH',
                        soft_id='A0111',
                        product_name='resampled',
                        ftile=self.subgrid + '010M_' + tname,
                        src_res=10)

        filelist = tile._tile_files.keys()
        datelist = [x[0:16] for x in filelist]

        return (datelist)


def _calc_nn(a, poi=None):
    dist = np.sqrt(np.square(poi[0] - a[0]) +
                   np.square(poi[1] - a[1]) +
                   np.square(poi[2] - a[2]) +
                   np.square(poi[3] - a[3]))
    return (dist)


def _estimate_ssm(bacArrs, paramArr, bacStats, ssm_out, valid_ind, mlmodel):
    for i in valid_ind:
        ind = np.unravel_index(i, (1000, 1000))
        try:
            # compile feature vector
            sig0vv = bacArrs['sig0vv'][0][ind] / 100.0
            sig0vh = bacArrs['sig0vh'][0][ind] / 100.0
            # lia = bacArrs['lia'][0][ind] / 100.0
            # h = terrainArrs['h'][0][ind]
            # a = terrainArrs['a'][0][ind]
            # s = terrainArrs['s'][0][ind]
            # slpvv = paramArr['slpVV'][0][ind]
            # slpvh = paramArr['slpVH'][0][ind]
            # sig0mvv = paramArr['sig0mVV'][0][ind] / 100.0
            # sig0mvh = paramArr['sig0mVH'][0][ind] / 100.0
            # sig0sdvv = paramArr['sig0sdVV'][0][ind] / 100.0
            # sig0sdvh = paramArr['sig0sdVH'][0][ind] / 100.0
            # vvsstd = bacStats['vv'][ind]
            # vhsstd = bacStats['vh'][ind]
            # liasstd = bacStats['lia'][ind]
            k1VV = paramArr['k1VV'][0][ind] / 100.
            k1VH = paramArr['k1VH'][0][ind] / 100.
            k2VV = paramArr['k2VV'][0][ind] / 100.
            k2VH = paramArr['k2VH'][0][ind] / 100.
            # k3VV = paramArr['k3VV'][0][ind]/100.
            # k3VH = paramArr['k3VH'][0][ind]/100.
            # k4VV = paramArr['k4VV'][0][ind]/100.
            # k4VH = paramArr['k4VH'][0][ind]/100.

            fvect = [  # sig0mvv,
                # sig0sdvv,
                # sig0mvh,
                # sig0sdvh,
                k1VV,
                k1VH,
                k2VV,
                k2VH,
                sig0vv,
                sig0vh]

            fvect = mlmodel[1].transform(np.reshape(fvect, (1, -1)))
            # predict ssm
            predssm = mlmodel[0].predict(fvect)
            if predssm < 0:
                ssm_out[ind] = 0
            else:
                ssm_out[ind] = predssm
        except:
            ssm_out[ind] = -9999


def _estimate_ssm_alternative(bacArrs, terrainArrs, paramArr, bacStats, ssm_out, valid_ind, mlmodel, model_attrs):
    for i in valid_ind:
        ind = np.unravel_index(i, (10000, 10000))
        try:
            # compile feature vector
            sig0vv = bacArrs['sig0vv'][0][ind] / 100.0
            sig0vh = bacArrs['sig0vh'][0][ind] / 100.0
            lia = bacArrs['lia'][0][ind] / 100.0
            h = terrainArrs['h'][0][ind]
            a = terrainArrs['a'][0][ind]
            s = terrainArrs['s'][0][ind]
            # slpvv = paramArr['slpVV'][0][ind]
            # slpvh = paramArr['slpVH'][0][ind]
            sig0mvv = paramArr['sig0mVV'][0][ind] / 100.0
            sig0mvh = paramArr['sig0mVH'][0][ind] / 100.0
            sig0sdvv = paramArr['sig0sdVV'][0][ind] / 100.0
            sig0sdvh = paramArr['sig0sdVH'][0][ind] / 100.0
            vvsstd = bacStats['vv'][ind]
            vhsstd = bacStats['vh'][ind]
            liasstd = bacStats['lia'][ind]
            k1VV = paramArr['k1VV'][0][ind] / 100.
            k1VH = paramArr['k1VH'][0][ind] / 100.
            k2VV = paramArr['k2VV'][0][ind] / 100.
            k2VH = paramArr['k2VH'][0][ind] / 100.
            k3VV = paramArr['k3VV'][0][ind] / 100.
            k3VH = paramArr['k3VH'][0][ind] / 100.
            k4VV = paramArr['k4VV'][0][ind] / 100.
            k4VH = paramArr['k4VH'][0][ind] / 100.

            attr = [sig0mvv,
                    sig0sdvv,
                    sig0mvh,
                    sig0sdvh]

            fvect = [sig0vv,
                     sig0vh]

            # find the best model
            # nn = np.argmin(np.apply_along_axis(_calc_nn, 1, model_attrs, poi=np.array(attr)))
            nn = model_attrs.predict(np.reshape(attr, (1, -1)))
            nn = int(nn[0])

            # load the best model
            nn_model = mlmodel[nn]['model']
            nn_scaler = mlmodel[nn]['scaler']

            fvect = nn_scaler.transform(np.reshape(fvect, (1, -1)))
            # predict ssm
            predssm = nn_model.predict(fvect)
            ssm_out[ind] = predssm
        except:
            ssm_out[ind] = -9999


def _local_std(arr, nanval, valid_ind, parameter="sig0"):
    # calculate local variance of image

    from scipy import ndimage
    from joblib import Parallel, delayed, load, dump
    import sys
    import tempfile
    import shutil

    # conver to float, then from db to linear
    arr = np.float32(arr)
    valid = np.where(arr != nanval)
    arr[valid] = arr[valid] / 100.0
    if parameter == "sig0":
        arr[valid] = np.power(10, arr[valid] / 10)
    arr[arr == nanval] = np.nan

    # prepare multi processing
    if not hasattr(sys.stdin, 'close'):
        def dummy_close():
            pass

        sys.stdin.close = dummy_close

    # prepare multi processing
    # dump arrays to temporary folder
    temp_folder = tempfile.mkdtemp()
    filename_in = os.path.join(temp_folder, 'joblib_dump1.mmap')
    if os.path.exists(filename_in): os.unlink(filename_in)
    _ = dump(arr, filename_in)
    inarr_map = load(filename_in, mmap_mode='r+')
    # output
    filename_out = os.path.join(temp_folder, 'joblib_dump2.mmap')
    outStd = np.memmap(filename_out, dtype=np.float32, mode='w+', shape=arr.shape)
    outStd[:] = np.nan

    # split arrays
    # inarr_splits = np.array_split(inarr_map, 8)
    # outarr_splits = np.array_split(outStd, 8)

    # get valid indices
    # valid_ind = np.where(np.isfinite(arr))
    # valid_ind = np.ravel_multi_index(valid_ind, arr.shape)
    valid_splits = np.array_split(valid_ind, 8)

    Parallel(n_jobs=8, verbose=5, max_nbytes=None)(
        delayed(_calc_std)(inarr_map, outStd, valid_splits[i], arr.shape) for i in range(8))

    # convert from linear to db
    if parameter == "sig0":
        valid = np.where(np.isfinite(outStd))
        outStd[valid] = 10 * np.log10(outStd[valid])
    outStd[np.isnan(outStd)] = nanval

    try:
        shutil.rmtree(temp_folder)
    except:
        print("Failed to delete: " + temp_folder)

    return outStd


def _calc_std(inarr, outarr, valid_ind, shape):
    for i in valid_ind:
        ind = np.unravel_index(i, shape)

        if (ind[0] >= 5) and (ind[0] <= shape[0] - 6) and (ind[1] >= 5) and (ind[1] <= shape[1] - 6):
            outarr[ind] = np.nanstd(inarr[ind[0] - 5:ind[0] + 5, ind[1] - 5:ind[1] + 5])


def _local_mean(arr, nanval):
    # calculate local variance of image

    from scipy import ndimage

    # conver to float, then from db to linear
    arr = np.float32(arr)
    valid = np.where(arr != nanval)
    arr[valid] = arr[valid] / 100.0
    arr[valid] = np.power(10, arr[valid] / 10)
    arr[arr == nanval] = np.nan

    outStd = ndimage.generic_filter(arr, np.nanmean, size=3)

    # convert from linear to db
    valid = np.where(outStd != np.nan)
    outStd[valid] = 10 * np.log10(outStd[valid])
    outStd[outStd == np.nan] = nanval

    return outStd


def _generate_model(features, target, urowidx):
    rows_select = np.where((features[:, 0] == features[urowidx, 0]) &
                           (features[:, 1] == features[urowidx, 1]))  # &
    # (features[:, 2] == features[urowidx, 2]) &
    # (features[:, 3] == features[urowidx, 3]))  # &
    # (self.features[:, 4] == self.features[0, 4]) &
    # (self.features[:, 5] == self.features[0, 5]))

    # if (len(rows_select) <= 9):
    #    # delete selected features from array
    #    # self.target = np.delete(self.target, rows_select)
    #    # self.features = np.delete(self.features, rows_select, axis=0)
    #    return

    utarg = np.copy(target[rows_select].squeeze())
    ufeat = np.copy(features[rows_select, 6::].squeeze())
    point_model = {'model_attr': features[0, 2:6]}

    # scaling
    scaler = sklearn.preprocessing.StandardScaler().fit(ufeat)
    ufeat = scaler.transform(ufeat)

    point_model['scaler'] = scaler

    # split into independent training data and test data
    x_train = ufeat
    y_train = utarg
    x_test = ufeat
    y_test = utarg

    # ...----...----...----...----...----...----...----...----...----...
    # SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR
    # ---....---....---....---....---....---....---....---....---....---

    dictCV = dict(C=scipy.stats.expon(scale=100),
                  gamma=scipy.stats.expon(scale=.1),
                  epsilon=scipy.stats.expon(scale=.1),
                  kernel=['rbf'])

    # specify kernel
    svr_rbf = SVR()

    # parameter tuning -> grid search

    gdCV = RandomizedSearchCV(estimator=svr_rbf,
                              param_distributions=dictCV,
                              n_iter=200,  # iterations used to be 200
                              n_jobs=1,
                              # pre_dispatch='all',
                              cv=KFold(n_splits=2, shuffle=True),  # cv used to be 10
                              # cv=sklearn.model_selection.TimeSeriesSplit(n_splits=3),
                              verbose=0,
                              scoring='r2')

    gdCV.fit(x_train, y_train)
    # prediction on test set
    tmp_est = gdCV.predict(x_test)
    # estimate point accuracies
    tmp_r = np.corrcoef(y_test, tmp_est)
    error = np.sqrt(np.sum(np.square(y_test - tmp_est)) / len(y_test))

    point_model['model'] = gdCV
    point_model['rmse'] = error
    point_model['r'] = tmp_r

    if (tmp_r[0, 1] > 0.5) & (error < 20):

        # set quality flag
        point_model['quality'] = 'good'
        # add to overall list
        # estimated = np.append(estimated, tmp_est)
        # true = np.append(true, y_test)
    else:
        # return
        point_model['quality'] = 'bad'

    # add model to model list
    return (point_model)


def _generate_model_linear(features, target, urowidx):
    from sklearn.linear_model import TheilSenRegressor

    rows_select = np.where((features[:, 0] == features[urowidx, 0]) &
                           (features[:, 1] == features[urowidx, 1]))  # &
    # (features[:, 2] == features[urowidx, 2]) &
    # (features[:, 3] == features[urowidx, 3]))  # &
    # (self.features[:, 4] == self.features[0, 4]) &
    # (self.features[:, 5] == self.features[0, 5]))

    # if (len(rows_select) <= 9):
    #    # delete selected features from array
    #    # self.target = np.delete(self.target, rows_select)
    #    # self.features = np.delete(self.features, rows_select, axis=0)
    #    return

    utarg = np.copy(target[rows_select].squeeze())
    ufeat = np.copy(features[rows_select, 6::].squeeze())
    point_model = {'model_attr': features[rows_select[0][0], 2:6]}

    # scaling
    scaler = sklearn.preprocessing.StandardScaler().fit(ufeat)
    ufeat = scaler.transform(ufeat)

    point_model['scaler'] = scaler

    # split into independent training data and test data
    x_train = ufeat
    y_train = utarg
    x_test = ufeat
    y_test = utarg

    # ...----...----...----...----...----...----...----...----...----...
    # Linear regression
    # ---....---....---....---....---....---....---....---....---....---

    # specify estimator
    estimator = TheilSenRegressor(random_state=42)

    # fit the model

    estimator.fit(x_train, y_train)
    # prediction on test set
    tmp_est = estimator.predict(x_test)
    # estimate point accuracies
    tmp_r = np.corrcoef(y_test, tmp_est)
    error = np.sqrt(np.sum(np.square(y_test - tmp_est)) / len(y_test))

    point_model['model'] = estimator
    point_model['rmse'] = error
    point_model['r'] = tmp_r

    if (tmp_r[0, 1] > 0.75) & (error < 5):

        # set quality flag
        point_model['quality'] = 'good'
        # add to overall list
        # estimated = np.append(estimated, tmp_est)
        # true = np.append(true, y_test)
    else:
        # return
        point_model['quality'] = 'bad'

    # add model to model list
    return (point_model)
