__author__ = 'felix'

import os
import sklearn.preprocessing
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
import ee
import pytesmo.io.ismn.interface as ismn_interface
import copy
from sklearn.tree import _tree
from scipy.stats import moment
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error


def tree_to_code_GEE(mlmodel, feature_names, step, suffix, outpath):
    '''
    Outputs a decision tree model as a Python function

    Parameters:
    -----------
    tree: decision tree model
        The decision tree to represent as a function
    feature_names: list
        The feature names of the dataset used for building the decision tree
    '''

    outfile = open(outpath + 'w_GLDAS_decisiontree_GEE_' + suffix + '.py', 'w')
    mlmodel = pickle.load(open(mlmodel, 'rb'))
    if step == 1:
        #mlmodel = mlmodel[0]
        pass
    elif step == 2:
        mlmodel = mlmodel[2]

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


class Trainingset(object):

    def __init__(self, outpath, uselc=True,
                 track=117, desc=False,
                 footprint=None):

        self.outpath = outpath
        self.uselc = uselc
        self.track = track
        self.desc = desc
        self.footprint = footprint
        self.ismn = None
        self.points = None
        self.trainingdb = None
        self.sub_loc = None
        self.target1 = None
        self.features1 = None
        self.target2 = None
        self.target2_2 = None
        self.features2 = None
        self.testtarget1 = None
        self.testfeatures1 = None
        self.testtarget2 = None
        self.testtarget2_2 = None
        self.testfeatures2 = None

        # define ismn path
        self.ismn_path = '/mnt/CEPH_PROJECTS/ECOPOTENTIAL/reference_data/ISMN/'

    def create_trainingset(self):

        if os.path.isfile(self.outpath + 'trainingdb' + str(self.track) + '.p'):
            trainingdb = pickle.load(open(self.outpath + 'trainingdb' + str(self.track) + '.p', 'rb'))
            self.trainingdb = trainingdb
        else:
            ee.Initialize()

            # # Get the locations of training points
            self.points = self.get_ismn_locations()

            # # # # # # extract parameters
            self.trainingdb = self.build_db()

            # save to disk
            pickle.dump(self.trainingdb, open(self.outpath + 'trainingdb' + str(self.track) + '.p', 'wb'))

        # apply additional asking
        self.apply_l8ndvi_mask()

        # compute tempora statistics
        self.compute_temporal_statistics()

        # create training arrays
        self.create_training_and_testing_array()

    def create_training_and_testing_array(self):

        step1_subs = self.trainingdb.drop_duplicates(['locid'])
        self.sub_loc = step1_subs['locid'].values
        target1 = np.array(step1_subs['ssm_mean'], dtype=np.float32)
        features1 = np.vstack((
            np.array(step1_subs['vv_gamma_v_k1'], dtype=np.float32),  # 0
            np.array(step1_subs['vh_gamma_v_k1'], dtype=np.float32),  # 1
            np.array(step1_subs['vv_gamma_v_k2'], dtype=np.float32),  # 2
            np.array(step1_subs['vh_gamma_v_k2'], dtype=np.float32),  # 3
            np.array(step1_subs['vv_gamma_v_k3'], dtype=np.float32),  # 4
            np.array(step1_subs['vh_gamma_v_k3'], dtype=np.float32),  # 5
            np.array(step1_subs['vv_gamma_v_k4'], dtype=np.float32),  # 6
            np.array(step1_subs['vh_gamma_v_k4'], dtype=np.float32),  # 7
            np.array(step1_subs['vv_gamma_v_tmean'], dtype=np.float32),  # 8
            np.array(step1_subs['vh_gamma_v_tmean'], dtype=np.float32),  # 9
            np.array(step1_subs['vv_gamma_s_k1'], dtype=np.float32),  # 10
            np.array(step1_subs['vh_gamma_s_k1'], dtype=np.float32),  # 11
            np.array(step1_subs['vv_gamma_s_k2'], dtype=np.float32),  # 12
            np.array(step1_subs['vh_gamma_s_k2'], dtype=np.float32),  # 13
            np.array(step1_subs['vv_gamma_s_k3'], dtype=np.float32),  # 14
            np.array(step1_subs['vh_gamma_s_k3'], dtype=np.float32),  # 15
            np.array(step1_subs['vv_gamma_s_k4'], dtype=np.float32),  # 16
            np.array(step1_subs['vh_gamma_s_k4'], dtype=np.float32),  # 17
            np.array(step1_subs['vv_gamma_s_tmean'], dtype=np.float32),  # 18
            np.array(step1_subs['vh_gamma_s_tmean'], dtype=np.float32),  # 19
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
            np.array(step1_subs['texture'], dtype=np.float32),  # 44
            np.array(step1_subs['lon'], dtype=np.float32),  # 45
            np.array(step1_subs['lat'], dtype=np.float32),  # 46
            np.array(step1_subs['gldas_mean'], dtype=np.float32),  # 47
            np.array(step1_subs['usdasm_mean'], dtype=np.float32))).transpose()  # 48

        target2 = np.array(self.trainingdb['ssm'], dtype=np.float32) - np.array(self.trainingdb['ssm_mean'],
                                                                                dtype=np.float32)
        target2_2 = np.array(self.trainingdb['ssm'], dtype=np.float32)
        features2 = np.vstack((
            np.array(self.trainingdb['gamma0_v_vv'], dtype=np.float32),  # 0
            np.array(self.trainingdb['gamma0_v_vh'], dtype=np.float32),  # 1
            np.array(self.trainingdb['gamma0_v_vv'], dtype=np.float32) -
            np.array(self.trainingdb['vv_gamma_v_tmean'], dtype=np.float32),  # 2
            np.array(self.trainingdb['gamma0_v_vh'], dtype=np.float32) -
            np.array(self.trainingdb['vh_gamma_v_tmean'], dtype=np.float32),  # 3
            np.array(self.trainingdb['gamma0_s_vv'], dtype=np.float32),  # 4
            np.array(self.trainingdb['gamma0_s_vh'], dtype=np.float32),  # 5
            np.array(self.trainingdb['gamma0_s_vv'], dtype=np.float32) -
            np.array(self.trainingdb['vv_gamma_s_tmean'], dtype=np.float32),  # 6
            np.array(self.trainingdb['gamma0_s_vh'], dtype=np.float32) -
            np.array(self.trainingdb['vh_gamma_s_tmean'], dtype=np.float32),  # 7
            np.array(self.trainingdb['lia'], dtype=np.float32),  # 8
            np.array(self.trainingdb['vv_gamma_v_k1'], dtype=np.float32),  # 9
            np.array(self.trainingdb['vh_gamma_v_k1'], dtype=np.float32),  # 10
            np.array(self.trainingdb['vv_gamma_v_k2'], dtype=np.float32),  # 11
            np.array(self.trainingdb['vh_gamma_v_k2'], dtype=np.float32),  # 12
            np.array(self.trainingdb['vv_gamma_v_k3'], dtype=np.float32),  # 13
            np.array(self.trainingdb['vh_gamma_v_k3'], dtype=np.float32),  # 14
            np.array(self.trainingdb['vv_gamma_v_k4'], dtype=np.float32),  # 15
            np.array(self.trainingdb['vh_gamma_v_k4'], dtype=np.float32),  # 16
            np.array(self.trainingdb['vv_gamma_s_k1'], dtype=np.float32),  # 17
            np.array(self.trainingdb['vh_gamma_s_k1'], dtype=np.float32),  # 18
            np.array(self.trainingdb['vv_gamma_s_k2'], dtype=np.float32),  # 19
            np.array(self.trainingdb['vh_gamma_s_k2'], dtype=np.float32),  # 20
            np.array(self.trainingdb['vv_gamma_s_k3'], dtype=np.float32),  # 21
            np.array(self.trainingdb['vh_gamma_s_k3'], dtype=np.float32),  # 22
            np.array(self.trainingdb['vv_gamma_s_k4'], dtype=np.float32),  # 23
            np.array(self.trainingdb['vh_gamma_s_k4'], dtype=np.float32),  # 24
            np.array(self.trainingdb['lc'], dtype=np.float32),  # 25
            np.array(self.trainingdb['bare_perc'], dtype=np.float32),  # 26
            np.array(self.trainingdb['crops_perc'], dtype=np.float32),  # 27
            np.array(self.trainingdb['trees'], dtype=np.float32),  # 28
            np.array(self.trainingdb['forest_type'], dtype=np.float32),  # 29
            np.array(self.trainingdb['grass_perc'], dtype=np.float32),  # 30
            np.array(self.trainingdb['moss_perc'], dtype=np.float32),  # 31
            np.array(self.trainingdb['urban_perc'], dtype=np.float32),  # 32
            np.array(self.trainingdb['waterp_perc'], dtype=np.float32),  # 33
            np.array(self.trainingdb['waters_perc'], dtype=np.float32),  # 34
            np.array(self.trainingdb['L8_b1'], dtype=np.float32),  # 35
            np.array(self.trainingdb['L8_b1_median'], dtype=np.float32),  # 36
            np.array(self.trainingdb['L8_b2'], dtype=np.float32),  # 37
            np.array(self.trainingdb['L8_b2_median'], dtype=np.float32),  # 38
            np.array(self.trainingdb['L8_b3'], dtype=np.float32),  # 39
            np.array(self.trainingdb['L8_b3_median'], dtype=np.float32),  # 40
            np.array(self.trainingdb['L8_b4'], dtype=np.float32),  # 41
            np.array(self.trainingdb['L8_b4_median'], dtype=np.float32),  # 42
            np.array(self.trainingdb['L8_b5'], dtype=np.float32),  # 43
            np.array(self.trainingdb['L8_b5_median'], dtype=np.float32),  # 44
            np.array(self.trainingdb['L8_b6'], dtype=np.float32),  # 45
            np.array(self.trainingdb['L8_b6_median'], dtype=np.float32),  # 46
            np.array(self.trainingdb['L8_b7'], dtype=np.float32),  # 47
            np.array(self.trainingdb['L8_b7_median'], dtype=np.float32),  # 48
            np.array(self.trainingdb['L8_b10'], dtype=np.float32),  # 49
            np.array(self.trainingdb['L8_b10_median'], dtype=np.float32),  # 50
            np.array(self.trainingdb['L8_b11'], dtype=np.float32),  # 51
            np.array(self.trainingdb['L8_b11_median'], dtype=np.float32),  # 52
            np.array(self.trainingdb['L8_timediff'], dtype=np.float32),  # 53
            np.array(self.trainingdb['ndvi'], dtype=np.float32),  # 54
            np.array(self.trainingdb['ndvi_mean'], dtype=np.float32),  # 55
            np.array(self.trainingdb['bulk'], dtype=np.float32),  # 56
            np.array(self.trainingdb['clay'], dtype=np.float32),  # 57
            np.array(self.trainingdb['sand'], dtype=np.float32),  # 58
            np.array(self.trainingdb['texture'], dtype=np.float32),  # 59
            np.array(self.trainingdb['orbit_direction'], dtype=np.float32), # 60
            np.array(self.trainingdb['gldas'], dtype=np.float32),  # 61
            np.array(self.trainingdb['gldas_mean'], dtype=np.float32),  # 62
            np.array(self.trainingdb['usdasm'], dtype=np.float32),  # 63
            np.array(self.trainingdb['usdasm_mean'], dtype=np.float32),  # 64
        )).transpose()

        self.target1 = target1
        self.features1 = features1
        self.target2 = target2
        self.target2_2 = target2_2
        self.features2 = features2

    def apply_l8ndvi_mask(self, threshold=0.5):
        # remove samples with ndvi above threshold
        ndvi = (np.array(self.trainingdb['L8_b5']) - np.array(self.trainingdb['L8_b4'])) / (
                np.array(self.trainingdb['L8_b5']) + np.array(self.trainingdb['L8_b4']))
        self.trainingdb = self.trainingdb.apply(lambda x: x[ndvi < threshold], axis=0)

    def compute_temporal_statistics(self, min_ts_len=10):
        # Compute temporal statistics

        # identify the unique sites
        locations_array = np.array(
            [self.trainingdb.index.get_level_values(0), self.trainingdb.index.get_level_values(1)]).transpose()
        unique_locations, indices, unique_counts = np.unique(locations_array, axis=0, return_index=True,
                                                             return_counts=True)
        # create a location index
        tmp_loc_id = np.full(self.trainingdb.shape[0], fill_value=0)
        unique_locations_tracks, tindices, tunique_c = np.unique(np.array(
            [self.trainingdb.index.get_level_values(0), self.trainingdb.index.get_level_values(1),
             self.trainingdb.index.get_level_values(2)]).transpose(), axis=0, return_index=True,
                                                                 return_counts=True)
        for uniqu_idx in range(len(tindices)):
            tmp_loc_id[tindices[uniqu_idx]:tindices[uniqu_idx] + tunique_c[uniqu_idx]] = uniqu_idx
        self.loc_id = tmp_loc_id
        self.trainingdb['locid'] = self.loc_id

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
            sttn_tracks = self.trainingdb.loc[unique_locations[i_sttn, 0],
                                              unique_locations[i_sttn, 1]].index.get_level_values(0).unique()
            for i_sttn_tracks in sttn_tracks:
                if self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                        unique_locations[i_sttn, 1],
                                        i_sttn_tracks)].shape[0] < min_ts_len:
                    tslong = False
                else:
                    tslong = True
                for var_i in range(len(tmp_avg_targets)):
                    self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                         unique_locations[i_sttn, 1],
                                         i_sttn_tracks), tmp_avg_targets[var_i]] = self.trainingdb.loc[
                        (unique_locations[i_sttn, 0],
                         unique_locations[i_sttn, 1],
                         i_sttn_tracks),
                        tmp_avg_sources[var_i]].median() if tslong else np.nan
                # sig0
                tmp_vv_lin = np.power(10, self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                                               unique_locations[i_sttn, 1],
                                                               i_sttn_tracks), 'sig0vv'] / 10)
                tmp_vh_lin = np.power(10, self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                                               unique_locations[i_sttn, 1],
                                                               i_sttn_tracks), 'sig0vv'] / 10)
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vv_tmean'] = 10 * np.log10(
                    tmp_vv_lin.median()) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vh_tmean'] = 10 * np.log10(
                    tmp_vh_lin.median()) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vv_k1'] = np.mean(np.log(tmp_vv_lin)) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vh_k1'] = np.mean(np.log(tmp_vh_lin)) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vv_k2'] = np.std(np.log(tmp_vv_lin)) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vh_k2'] = np.std(np.log(tmp_vh_lin)) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vv_k3'] = moment(np.log(tmp_vv_lin),
                                                                       moment=3) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vh_k3'] = moment(np.log(tmp_vh_lin),
                                                                       moment=3) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vv_k4'] = moment(np.log(tmp_vv_lin),
                                                                       moment=4) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vh_k4'] = moment(np.log(tmp_vh_lin),
                                                                       moment=4) if tslong else np.nan
                # gamma0vol
                tmp_gammavv_lin = np.power(10, self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                                                    unique_locations[i_sttn, 1],
                                                                    i_sttn_tracks), 'gamma0_v_vv'] / 10)
                tmp_gammavh_lin = np.power(10, self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                                                    unique_locations[i_sttn, 1],
                                                                    i_sttn_tracks), 'gamma0_v_vh'] / 10)
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vv_gamma_v_tmean'] = 10 * np.log10(
                    tmp_gammavv_lin.median()) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vh_gamma_v_tmean'] = 10 * np.log10(
                    tmp_gammavh_lin.median()) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vv_gamma_v_k1'] = np.mean(
                    np.log(tmp_gammavv_lin)) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vh_gamma_v_k1'] = np.mean(
                    np.log(tmp_gammavh_lin)) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vv_gamma_v_k2'] = np.std(
                    np.log(tmp_gammavv_lin)) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vh_gamma_v_k2'] = np.std(
                    np.log(tmp_gammavh_lin)) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vv_gamma_v_k3'] = moment(np.log(tmp_gammavv_lin),
                                                                               moment=3) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vh_gamma_v_k3'] = moment(np.log(tmp_gammavh_lin),
                                                                               moment=3) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vv_gamma_v_k4'] = moment(np.log(tmp_gammavv_lin),
                                                                               moment=4) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vh_gamma_v_k4'] = moment(np.log(tmp_gammavh_lin),
                                                                               moment=4) if tslong else np.nan

                # gamma0surf
                tmp_gammavv_lin = np.power(10, self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                                                    unique_locations[i_sttn, 1],
                                                                    i_sttn_tracks), 'gamma0_s_vv'] / 10)
                tmp_gammavh_lin = np.power(10, self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                                                    unique_locations[i_sttn, 1],
                                                                    i_sttn_tracks), 'gamma0_s_vh'] / 10)
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vv_gamma_s_tmean'] = 10 * np.log10(
                    tmp_gammavv_lin.median()) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vh_gamma_s_tmean'] = 10 * np.log10(
                    tmp_gammavh_lin.median()) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vv_gamma_s_k1'] = np.mean(
                    np.log(tmp_gammavv_lin)) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vh_gamma_s_k1'] = np.mean(
                    np.log(tmp_gammavh_lin)) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vv_gamma_s_k2'] = np.std(
                    np.log(tmp_gammavv_lin)) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vh_gamma_s_k2'] = np.std(
                    np.log(tmp_gammavh_lin)) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vv_gamma_s_k3'] = moment(np.log(tmp_gammavv_lin),
                                                                               moment=3) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vh_gamma_s_k3'] = moment(np.log(tmp_gammavh_lin),
                                                                               moment=3) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vv_gamma_s_k4'] = moment(np.log(tmp_gammavv_lin),
                                                                               moment=4) if tslong else np.nan
                self.trainingdb.loc[(unique_locations[i_sttn, 0],
                                     unique_locations[i_sttn, 1],
                                     i_sttn_tracks), 'vh_gamma_s_k4'] = moment(np.log(tmp_gammavh_lin),
                                                                               moment=4) if tslong else np.nan

    def create_test_set(self, steps=1):
        if steps == 1:
            # filter nan
            valid = ~np.any(np.isnan(self.features2), axis=1)
            self.target2_2 = self.target2_2[valid]
            self.features2 = self.features2[valid, :]
            self.loc_id = self.loc_id[valid]
            self.features2, self.testfeatures2,  \
                self.target2_2, self.testtarget2_2,\
                self.loc_id, self.testloc_id = train_test_split(self.features2,
                                                  self.target2_2,
                                                  self.loc_id,
                                                  test_size=0.2,
                                                  random_state=25)

    def create_learning_curve(self, modelpath, feature_vect=None):
        from sgrt_devels.utils import plot_learning_curve
        from sklearn.model_selection import ShuffleSplit

        # specify training data
        if feature_vect is not None:
            x = self.features2[:, feature_vect].copy()
            #x_test = self.testfeatures2[:, feature_vect].copy()
        else:
            x = self.features2.copy()
            #x_test = self.testfeatures2.copy()
        y = self.target2_2.copy()
        #y_test = self.testtarget2_2.copy()

        loc_id = self.loc_id.copy()

        # filter nan
        valid = ~np.any(np.isnan(x), axis=1)
        y = y[valid]
        x = x[valid, :]
        loc_id = loc_id[valid]

        trainedmodel = pickle.load(open(modelpath, 'rb'))

        mlmodel = GradientBoostingRegressor(random_state=12, learning_rate=trainedmodel.learning_rate,
                                            n_estimators=trainedmodel.n_estimators,
                                            subsample=trainedmodel.subsample,
                                            max_depth=trainedmodel.max_depth,
                                            n_iter_no_change=10)

        title = 'GBRT learning curve'
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        plot_learning_curve(mlmodel,
                            title,
                            x,
                            y,
                            cv=cv,
                            n_jobs=-1)
        plt.savefig(self.outpath + 'learning_curve.png', dpi=300)


    def train_GBR_LOGO_2step(self,
                             feature_vect1=None,
                             feature_vect2=None,
                             prefix='',
                             export_results=False):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.ensemble import HistGradientBoostingRegressor

        # specify training data
        if feature_vect1 is not None:
            x1 = self.features1[:, feature_vect1].copy()
        else:
            x1 = self.features1.copy()
        y1 = self.target1.copy()

        if feature_vect2 is not None:
            x2 = self.features2[:, feature_vect2].copy()
        else:
            x2 = self.features2.copy()
        y2 = self.target2.copy()
        sub_loc = self.sub_loc.copy()
        loc_id = self.loc_id.copy()

        # filter nan
        valid = ~np.any(np.isnan(x1), axis=1)
        y1 = y1[valid]
        x1 = x1[valid, :]
        sub_loc = sub_loc[valid]

        # filter nan
        valid = ~np.any(np.isnan(x2), axis=1)
        y2 = y2[valid]
        x2 = x2[valid, :]
        loc_id = loc_id[valid]

        # perform leave-one-group out cross validation to estimate the prediction accuracy
        # in each iteration the left out group corresponds to all measurements of one ismn station

        true_vect1 = list()
        pred_vect1 = list()
        true_vect2 = list()
        pred_vect2 = list()
        true_vect_tot = list()
        pred_vect_tot = list()
        r2_list = list()
        rmse1_list = list()
        rmse2_list = list()

        mlmodel1 = HistGradientBoostingRegressor(max_iter=500, verbose=False, random_state=42)
        mlmodel2 = HistGradientBoostingRegressor(max_iter=500, verbose=False, random_state=42)

        for iid in np.unique(loc_id):
            print(iid)
            tmp_x1_train = x1[sub_loc != iid, :]
            tmp_y1_train = y1[sub_loc != iid]
            tmp_x1_test = x1[sub_loc == iid, :]
            tmp_y1_test = y1[sub_loc == iid]

            tmp_x2_train = x2[loc_id != iid, :]
            tmp_y2_train = y2[loc_id != iid]
            tmp_x2_test = x2[loc_id == iid, :]
            tmp_y2_test = y2[loc_id == iid]

            mlmodel1.fit(tmp_x1_train, tmp_y1_train)
            mlmodel2.fit(tmp_x2_train, tmp_y2_train)

            # prediction of the average and relative sm
            tmp_pred1 = mlmodel1.predict(tmp_x1_test)
            pred_vect1 += list(tmp_pred1)
            true_vect1 += list(tmp_y1_test)
            tmp_pred2 = mlmodel2.predict(tmp_x2_test)
            pred_vect2 += list(tmp_pred2)
            true_vect2 += list(tmp_y2_test)
            pred_vect_tot += list(tmp_pred1 + tmp_pred2)
            true_vect_tot += list(tmp_y1_test + tmp_y2_test)

            # collect scores
            r2_list.append(r2_score(tmp_y2_test, tmp_pred2))
            rmse1_list.append(mean_squared_error(tmp_y1_test, tmp_pred1, squared=False))
            rmse2_list.append(mean_squared_error(tmp_y2_test, tmp_pred2, squared=False))

        print('Average scores: \n')
        print('RMSE - MODEL1: ' + str(np.nanmedian(rmse1_list)))
        print('R2 - MODEL2: ' + str(np.nanmedian(r2_list)))
        print('RMSE - MODEL2: ' + str(np.nanmedian(rmse2_list)))

        # create the estimation models
        mlmodel1 = GradientBoostingRegressor(n_estimators=500,
                                             verbose=True,
                                             random_state=42)

        mlmodel2 = GradientBoostingRegressor(n_estimators=500,
                                             verbose=True,
                                             random_state=42)
        mlmodel1.fit(x1, y1)
        mlmodel2.fit(x2, y2)

        pickle.dump((mlmodel1, mlmodel2),
                    open(self.outpath + prefix + 'GBRmlmodel_2step.p', 'wb'))

        pred_vect1 = np.array(pred_vect1)
        true_vect1 = np.array(true_vect1)
        pred_vect2 = np.array(pred_vect2)
        true_vect2 = np.array(true_vect2)
        pred_vect_tot = np.array(pred_vect_tot)
        true_vect_tot = np.array(true_vect_tot)
        if export_results:
            r2_list = np.array(r2_list)
            rmse1_list = np.array(rmse1_list)
            rmse2_list = np.array(rmse2_list)
            np.savez(self.outpath + 'loo_tmp.npz', pred_vect1, true_vect1, pred_vect2, true_vect2, pred_vect_tot,
                     true_vect_tot)
            np.savez(self.outpath + 'loo_RF_2step_scores.npz', r2_list, rmse1_list, rmse2_list)

        # Overall scores
        r = r2_score(true_vect_tot, pred_vect_tot)
        bias = np.nanmedian(pred_vect_tot - true_vect_tot)
        error = mean_squared_error(true_vect_tot, pred_vect_tot, squared=False)

        r_avg = r2_score(true_vect1, pred_vect1)
        r_rel = r2_score(true_vect2, pred_vect2)
        rmse_avg = mean_squared_error(true_vect1, pred_vect1, squared=False)
        rmse_rel = mean_squared_error(true_vect2, pred_vect2, squared=False)
        bias_avg = np.nanmedian(pred_vect1 - true_vect1)
        bias_rel = np.nanmedian(pred_vect2 - true_vect2)

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
        ax1.text(0.1, 0.6, 'R2=' + '{:03.2f}'.format(r_avg) +
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
        ax2.text(-0.4, 0.12, 'R2=' + '{:03.2f}'.format(r_rel) +
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
        ax3.text(0.1, 0.6, 'R2=' + '{:03.2f}'.format(r) +
                 '\nRMSE=' + '{:03.2f}'.format(error) +
                 '\nBias=' + '{:03.2f}'.format(bias), fontsize=8)
        ax3.set_aspect('equal', 'box')
        plt.tight_layout()
        plt.savefig(self.outpath + prefix + 'GBR_LOO_2step_combined.png', dpi=600)
        plt.close()

    def train_GBR_LOGO_1step(self,
                             feature_vect=None,
                             prefix='',
                             export_results=False):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.metrics import mean_absolute_error

        # specify training data
        if feature_vect is not None:
            x = self.features2[:, feature_vect].copy()
            x_test = self.testfeatures2[:, feature_vect].copy()
        else:
            x = self.features2.copy()
            x_test = self.testfeatures2.copy()
        y = self.target2_2.copy()
        y_test = self.testtarget2_2.copy()

        loc_id = self.loc_id.copy()
        testloc_id = self.testloc_id.copy()

        # filter nan
        valid = ~np.any(np.isnan(x), axis=1)
        y = y[valid]
        x = x[valid, :]
        loc_id = loc_id[valid]

        # perform leave-one-group out cross validation to estimate the prediction accuracy
        # in each iteration the left out group corresponds to all measurements of one ismn station

        mlmodel = GradientBoostingRegressor(random_state=12)
        params_gbr = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 500, 1000], 'subsample': [0.2, 0.5, 1],
                      'max_depth': [3, 5, 10], 'n_iter_no_change': [10]}
        gcv_gbr = GridSearchCV(mlmodel, params_gbr, scoring=['r2', 'neg_root_mean_squared_error'],
                               n_jobs=-1, cv=GroupKFold(), verbose=1, refit='r2')
        gcv_gbr.fit(x, y, groups=loc_id)
        print(gcv_gbr.best_params_)

        print('Average scores: \n')
        print('RMSE: ' + str(gcv_gbr.cv_results_['mean_test_neg_root_mean_squared_error'].max() * -1))
        print('R2: ' + str(gcv_gbr.cv_results_['mean_test_r2'].max()))

        pickle.dump(gcv_gbr.best_estimator_,
                    open(self.outpath + prefix + 'GBRmlmodel_1step.p', 'wb'))

        predictions = gcv_gbr.predict(x_test)

        if export_results:
            np.savez(self.outpath + 'loo_tmp.npz', np.array(predictions), y_test)
            # pickle.dump(scores,
            #             open(self.outpath + 'loo_RF_1step_scores.npz', 'wb'))

        # Overall scores
        r = r2_score(y_test, predictions)
        bias = np.mean(predictions - y_test)
        error = mean_squared_error(y_test, predictions, squared=False)
        mae = mean_absolute_error(y_test, predictions)

        pltlims = 0.6

        # create plots
        fig, ax = plt.subplots(1, 1, sharex=False, figsize=(3.5, 3), dpi=600, squeeze=True)
        ax.scatter(y_test, predictions, c='k', label='True vs Est', s=1, marker='*')
        ax.set_xlim(0, pltlims)
        ax.set_ylim(0, pltlims)
        ax.set_xlabel("$SMC$ [m$^3$m$^{-3}$]", size=8)
        ax.set_ylabel("$SMC^*$ [m$^3$m$^{-3}$]", size=8)
        ax.plot([0, pltlims], [0, pltlims], 'k--', linewidth=0.8)
        ax.text(0.1, 0.4, 'R2=' + '{:03.2f}'.format(r) +
                '\nRMSE=' + '{:03.2f}'.format(error) +
                '\nMAE=' + '{:03.2f}'.format(mae), fontsize=8)
        ax.set_aspect('equal', 'box')
        plt.tight_layout()
        plt.savefig(self.outpath + prefix + 'GBR_LOO_1step.png', dpi=600)
        plt.close()

    def train_SVR_LOGO_2step(self,
                             feature_vect1=None,
                             feature_vect2=None,
                             prefix='',
                             export_results=False):
        # specify training data
        if feature_vect1 is not None:
            x1 = self.features1[:, feature_vect1].copy()
        else:
            x1 = self.features1.copy()
        y1 = self.target1.copy()

        if feature_vect2 is not None:
            x2 = self.features2[:, feature_vect2].copy()
        else:
            x2 = self.features2.copy()
        y2 = self.target2.copy()
        sub_loc = self.sub_loc.copy()
        loc_id = self.loc_id.copy()

        # filter nan
        valid = ~np.any(np.isnan(x1), axis=1)
        y1 = y1[valid]
        x1 = x1[valid, :]
        sub_loc = sub_loc[valid]

        # filter nan
        valid = ~np.any(np.isnan(x2), axis=1)
        y2 = y2[valid]
        x2 = x2[valid, :]
        loc_id = loc_id[valid]

        # perform leave-one-group out cross validation to estimate the prediction accuracy
        # in each iteration the left out group corresponds to all measurements of one ismn station

        # scaling
        scaler1 = sklearn.preprocessing.RobustScaler().fit(x1)
        x1 = scaler1.transform(x1)
        scaler2 = sklearn.preprocessing.RobustScaler().fit(x2)
        x2 = scaler2.transform(x2)

        init_c = [-2, 2]
        init_g = [-2, -0.5]
        init_e = [-2, -0.5]

        # find the best parameters for the two models
        dictCV = dict(C=np.logspace(init_c[0], init_c[1], 3),
                      gamma=np.logspace(init_g[0], init_g[1], 3),
                      epsilon=np.logspace(init_e[0], init_e[1], 3),
                      kernel=['rbf'])

        model1 = GridSearchCV(estimator=SVR(),
                              param_grid=dictCV,
                              n_jobs=-1,
                              verbose=1,
                              cv=KFold(),
                              scoring=['r2', 'neg_root_mean_squared_error'],
                              refit='r2')
        model2 = GridSearchCV(estimator=SVR(),
                              param_grid=dictCV,
                              n_jobs=-1,
                              verbose=1,
                              cv=GroupKFold(),
                              scoring=['r2', 'neg_root_mean_squared_error'],
                              refit='r2')
        model1.fit(x1, y1)
        model2.fit(x2, y2, groups=loc_id)

        print('Average scores: \n')
        print('MODEL1: ' + str(model1.best_score_))
        print('MODEL2: ' + str(model2.best_score_))

        pickle.dump((model1.best_estimator_, scaler1, model2.best_estimator_, scaler2),
                    open(self.outpath + prefix + 'SVRmlmodel_2step.p', 'wb'))

        # Create estimations for plot
        true_vect1 = list()
        pred_vect1 = list()
        true_vect2 = list()
        pred_vect2 = list()
        true_vect_tot = list()
        pred_vect_tot = list()
        r2_list = list()
        rmse_list = list()

        loomodel1 = SVR(C=model1.best_params_['C'],
                        epsilon=model1.best_params_['epsilon'],
                        gamma=model1.best_params_['gamma'])
        loomodel2 = SVR(C=model2.best_params_['C'],
                        epsilon=model2.best_params_['epsilon'],
                        gamma=model2.best_params_['gamma'])

        for iid in np.unique(loc_id):
            print(iid)
            tmp_features_train1 = x1[sub_loc != iid, :]
            tmp_target_train1 = y1[sub_loc != iid]
            tmp_features_test1 = x1[sub_loc == iid, :]
            tmp_target_test1 = y1[sub_loc == iid]

            tmp_features_train2 = x2[loc_id != iid, :]
            tmp_target_train2 = y2[loc_id != iid]
            tmp_features_test2 = x2[loc_id == iid, :]
            tmp_target_test2 = y2[loc_id == iid]

            loomodel1.fit(tmp_features_train1, tmp_target_train1)
            loomodel2.fit(tmp_features_train2, tmp_target_train2)

            # prediction of the average and relative sm
            tmp_pred1 = loomodel1.predict(tmp_features_test1)
            pred_vect1 += list(tmp_pred1)
            true_vect1 += list(tmp_target_test1)
            tmp_pred2 = loomodel2.predict(tmp_features_test2)
            pred_vect2 += list(tmp_pred2)
            true_vect2 += list(tmp_target_test2)
            pred_vect_tot += list(tmp_pred1 + tmp_pred2)
            true_vect_tot += list(tmp_target_test1 + tmp_target_test2)

            # collect average scores
            r2_list.append(r2_score(tmp_target_test2, tmp_pred2))
            rmse_list.append(mean_squared_error(tmp_target_test2, tmp_pred2, squared=False))

        pred_vect1 = np.array(pred_vect1)
        true_vect1 = np.array(true_vect1)
        pred_vect2 = np.array(pred_vect2)
        true_vect2 = np.array(true_vect2)
        pred_vect_tot = np.array(pred_vect_tot)
        true_vect_tot = np.array(true_vect_tot)
        if export_results:
            r2_list = np.array(r2_list)
            rmse_list = np.array(rmse_list)
            np.savez(self.outpath + 'loo_tmp.npz', pred_vect1, true_vect1, pred_vect2, true_vect2, pred_vect_tot,
                     true_vect_tot)
            np.savez(self.outpath + 'loo_SVR_2step_scores.npz', r2_list, rmse_list)

        r = r2_score(true_vect_tot, pred_vect_tot)
        bias = np.nanmedian(pred_vect_tot - true_vect_tot)
        error = mean_squared_error(true_vect_tot, pred_vect_tot, squared=False)

        r_avg = r2_score(true_vect1, pred_vect1)
        r_rel = r2_score(true_vect2, pred_vect2)
        rmse_avg = mean_squared_error(true_vect1, pred_vect1, squared=False)
        rmse_rel = mean_squared_error(true_vect2, pred_vect2, squared=False)
        bias_avg = np.median(pred_vect1 - true_vect1)
        bias_rel = np.median(pred_vect2 - true_vect2)

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
        ax1.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r_avg) +
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
        ax2.text(-0.15, 0.12, 'R=' + '{:03.2f}'.format(r_rel) +
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
        ax3.text(0.1, 0.6, 'R=' + '{:03.2f}'.format(r) +
                 '\nRMSE=' + '{:03.2f}'.format(error) +
                 '\nBias=' + '{:03.2f}'.format(bias), fontsize=8)
        ax3.set_aspect('equal', 'box')
        plt.tight_layout()
        plt.savefig(self.outpath + prefix + 'SVR_LOO_2step.png', dpi=600)
        plt.close()

    def train_SVR_LOGO_1step(self,
                             feature_vect=None,
                             prefix='',
                             export_results=False):
        # specify training data
        if feature_vect is not None:
            x = self.features2[:, feature_vect].copy()
        else:
            x = self.features2.copy()
        y = self.target2_2.copy()
        loc_id = self.loc_id.copy()

        # filter nan
        valid = ~np.any(np.isnan(x), axis=1)
        y = y[valid]
        x = x[valid, :]
        loc_id = loc_id[valid]

        # perform leave-one-group out cross validation to estimate the prediction accuracy
        # in each iteration the left out group corresponds to all measurements of one ismn station

        # scaling
        scaler = sklearn.preprocessing.RobustScaler().fit(x)
        x = scaler.transform(x)

        init_c = [-2, 2]
        init_g = [-2, -0.5]
        init_e = [-2, -0.5]

        # find the best parameters for the two models
        dictCV = dict(C=np.logspace(init_c[0], init_c[1], 3),
                      gamma=np.logspace(init_g[0], init_g[1], 3),
                      epsilon=np.logspace(init_e[0], init_e[1], 3),
                      kernel=['rbf'])

        model = GridSearchCV(estimator=SVR(),
                             param_grid=dictCV,
                             n_jobs=-1,
                             verbose=1,
                             cv=GroupKFold(),
                             scoring=['r2', 'neg_root_mean_squared_error'],
                             refit='r2')

        model.fit(x, y, groups=loc_id)

        print('Average scores: \n')
        print('MODEL: ' + str(model.cv_results_))

        pickle.dump((model.best_estimator_, scaler),
                    open(self.outpath + prefix + 'SVRmlmodel_1step.p', 'wb'))

        # Create accuracy estimations
        loomodel = SVR(C=model.best_params_['C'],
                       epsilon=model.best_params_['epsilon'],
                       gamma=model.best_params_['gamma'])

        scores = cross_validate(loomodel,
                                x, y,
                                groups=loc_id,
                                cv=LeaveOneGroupOut(),
                                scoring=['r2', 'neg_root_mean_squared_error'],
                                n_jobs=-1,
                                verbose=1)

        predictions = cross_val_predict(loomodel,
                                        x, y,
                                        groups=loc_id,
                                        cv=LeaveOneGroupOut(),
                                        n_jobs=-1,
                                        verbose=1)

        print('Average scores: \n')
        print('RMSE: ' + str(np.nanmedian(scores['test_r2'])))
        print('R2: ' + str(np.nanmedian(scores['test_neg_root_mean_squared_error']) * -1))

        if export_results:
            np.savez(self.outpath + 'loo_tmp.npz', np.array(predictions), y)
            pickle.dump(scores,
                        open(self.outpath + 'loo_SVR_2step_scores.npz', 'wb'))

        # Overall scores
        r = r2_score(y, predictions)
        bias = np.mean(predictions - y)
        error = mean_squared_error(y, predictions, squared=False)

        pltlims = 1.0

        # create plots
        fig, ax = plt.subplots(1, 1, sharex=False, figsize=(3.5, 3), dpi=600, squeeze=True)
        ax.scatter(y, predictions, c='k', label='True vs Est', s=1, marker='*')
        ax.set_xlim(0, pltlims)
        ax.set_ylim(0, pltlims)
        ax.set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=8)
        ax.set_ylabel("$SMC^*_{Tot}$ [m$^3$m$^{-3}$]", size=8)
        ax.plot([0, pltlims], [0, pltlims], 'k--', linewidth=0.8)
        ax.text(0.1, 0.6, 'R2=' + '{:03.2f}'.format(r) +
                '\nRMSE=' + '{:03.2f}'.format(error) +
                '\nBias=' + '{:03.2f}'.format(bias), fontsize=8)
        ax.set_aspect('equal', 'box')
        plt.tight_layout()
        plt.savefig(self.outpath + prefix + 'SVR_LOO_1step.png', dpi=600)
        plt.close()

    def parameter_selection_2step(self, ml='SVR'):

        # specify training data
        x1 = self.features1.copy()
        y1 = self.target1.copy()

        x2 = self.features2.copy()
        y2 = self.target2.copy()
        sub_loc = self.sub_loc.copy()
        loc_id = self.loc_id.copy()

        # filter nan
        valid = ~np.any(np.isnan(x1), axis=1)
        y1 = y1[valid]
        x1 = x1[valid, :]
        sub_loc = sub_loc[valid]

        # filter nan
        valid = ~np.any(np.isnan(x2), axis=1)
        y2 = y2[valid]
        x2 = x2[valid, :]
        loc_id = loc_id[valid]

        if ml == 'GBR':
            mlmodel1 = GradientBoostingRegressor(n_estimators=500,
                                                 verbose=False,
                                                 random_state=42)

            mlmodel2 = GradientBoostingRegressor(n_estimators=500,
                                                 verbose=False,
                                                 random_state=42)
        elif ml == 'SVR':
            mlmodel1 = SVR()
            mlmodel2 = SVR()

        rfecv1 = RFECV(estimator=mlmodel1, cv=KFold(random_state=10), scoring='r2', verbose=1, n_jobs=-1)
        rfecv1.fit(x1, y1)

        rfecv2 = RFECV(estimator=mlmodel2, cv=GroupKFold(), scoring='r2', verbose=1, n_jobs=-1)
        rfecv2.fit(x2, y2, groups=loc_id)
        print("Optimal number of features M1: %d" % rfecv1.n_features_)
        print(rfecv1.grid_scores_.max())
        print(np.where(rfecv1.support_))
        print("Optimal number of features M2: %d" % rfecv2.n_features_)
        print(rfecv2.grid_scores_.max())
        print(np.where(rfecv2.support_))

        return np.where(rfecv1.support_), np.where(rfecv2.support_)

    def parameter_selection_1step(self, ml='SVR', frange=59):

        # specify training data
        x = self.features2.copy()
        y = self.target2_2.copy()
        sub_loc = self.sub_loc.copy()
        loc_id = self.loc_id.copy()

        # filter nan
        valid = ~np.any(np.isnan(x), axis=1)
        y = y[valid]
        x = x[valid, 0:frange]
        loc_id = loc_id[valid]

        if ml == 'GBR':
            mlmodel = GradientBoostingRegressor(random_state=12)
            params = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 500, 1000],
                      'subsample': [0.2, 0.5, 1],
                      'max_depth': [3, 5, 10], 'n_iter_no_change': [10]}
        elif ml == 'SVR':
            mlmodel = SVR()

        best_score = 0
        for i in ParameterGrid(params):
            mlmodel.set_params(**i)
            rfecv = RFECV(estimator=mlmodel, cv=GroupKFold(), scoring='r2', verbose=1, n_jobs=-1)
            rfecv.fit(x, y, groups=loc_id)
            if rfecv.grid_scores_.max() > best_score:
                best_score = rfecv.grid_scores_.max()
                best_nr = rfecv.n_features_
                best_ids = rfecv.support_
        print("Optimal number of features: %d" % best_nr)
        print(best_score)
        print(np.where(best_ids))
        return np.where(best_ids)

    def build_db(self):

        import sgrt_devels.extr_TS as exTS
        import datetime as dt
        from scipy.stats import moment

        ee.Initialize()

        cntr = 0
        cntr2 = 0

        if self.uselc:
            # COPERNICUS
            val_lc = [20, 30, 40, 60, 90, 125, 126, 121, 122, 13, 124]
        trainingdb_samples = dict()
        used_stations = list()

        # check if tmp files exist to continue extraction after crash
        if os.path.exists(self.outpath + 'siglia.tmp'):
            trainingdb_samples = pickle.load(open(self.outpath + 'siglia.tmp', 'rb'))
            cntr = trainingdb_samples.shape[0]
            used_stations = pickle.load(open(self.outpath + 'stations.tmp', 'rb'))
            cntr2 = len(used_stations)

        # cycle through all points
        for px in self.points:

            if (px[3], px[2]) in used_stations:
                continue

            print('Grid-Point ' + str(cntr2 + 1) + '/' + str(len(self.points)))
            print(px[3] + ', ' + px[2])

            # extract time series
            tries = 1

            while tries < 4:
                try:
                    # extract ISMN measutement
                    ssm_success, tmp_ssm = self.get_ssm(stname=px[2], network=px[3])
                    if ssm_success == 0:
                        print('Failed to read ISMN data')
                        tries = 4
                        continue

                    # get land-cover
                    tmplc = self.get_lc(px[0], px[1])
                    tmplc_disc = tmplc['classid'] if tmplc['classid'] is not None else 0
                    tmplc_forestT = tmplc['forestType'] if tmplc['forestType'] is not None else 0
                    tmplc_bare = tmplc['bare'] if tmplc['bare'] is not None else 0
                    tmplc_crops = tmplc['crops'] if tmplc['crops'] is not None else 0
                    tmplc_grass = tmplc['grass'] if tmplc['grass'] is not None else 0
                    tmplc_moss = tmplc['shrub'] if tmplc['shrub'] is not None else 0
                    tmplc_urban = tmplc['urban'] if tmplc['urban'] is not None else 0
                    tmplc_waterp = tmplc['waterp'] if tmplc['waterp'] is not None else 0
                    tmplc_waters = tmplc['waters'] if tmplc['waters'] is not None else 0
                    trees = tmplc['tree']

                    if self.uselc and (tmplc_disc not in val_lc):
                        print('Location belongs to a masked land cover class')
                        tries = 4
                        continue

                    # get soil info
                    tmp_bulc = self.get_bulk_density(px[0], px[1])
                    tmp_clay = self.get_clay_content(px[0], px[1])
                    tmp_sand = self.get_sand_content(px[0], px[1])
                    tmp_text = self.get_soil_texture_class(px[0], px[1])

                    # get topography
                    elev, aspe, slop = self.get_topo(px[0], px[1])

                    tmp_series, tmp_dirs = exTS.extr_SIG0_LIA_ts_GEE(float(px[0]), float(px[1]),
                                                           bufferSize=self.footprint,
                                                           trackflt=self.track,
                                                           desc=self.desc,
                                                           tempfilter=False,
                                                           returnLIA=True,
                                                           datefilter=[np.min(tmp_ssm.index).strftime('%Y-%m-%d'),
                                                                       np.max(tmp_ssm.index).strftime('%Y-%m-%d')],
                                                           S1B=True,
                                                           radcor=True)
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

            px_counter = 0

            if cntr != 0:
                out_df = copy.deepcopy(trainingdb_samples)

            for track_key in tmp_series.keys():
                vv_series = np.array(tmp_series[track_key]['vv_sig0'], dtype=np.float32)
                vh_series = np.array(tmp_series[track_key]['vh_sig0'], dtype=np.float32)
                lia_series = np.array(tmp_series[track_key]['lia'], dtype=np.float32)
                vv_gamma_s = np.array(tmp_series[track_key]['vv_g0surf'], dtype=np.float32)
                vh_gamma_s = np.array(tmp_series[track_key]['vh_g0surf'], dtype=np.float32)
                vv_gamma_v = np.array(tmp_series[track_key]['vv_g0vol'], dtype=np.float32)
                vh_gamma_v = np.array(tmp_series[track_key]['vh_g0vol'], dtype=np.float32)

                # get ndvi
                try:
                    tmpndvi, ndvi_success = exTS.extr_MODIS_MOD13Q1_ts_GEE(px[0], px[1],
                                                                           bufferSize=self.footprint,
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
                    l8_tmp = exTS.extr_L8_ts_GEE(px[0], px[1], self.footprint)
                except:
                    print('Landsat extraction failed!')
                    continue

                if l8_tmp is None:
                    print('No Landsat data')
                    continue

                # initialize in-situ series with S1 dates
                ssm_series = pd.Series(index=tmp_series[track_key].index)
                # initialize gdal series
                gldas_series = pd.Series(index=tmp_series[track_key].index)
                gldas_veg_water = pd.Series(index=tmp_series[track_key].index)
                gldas_et = pd.Series(index=tmp_series[track_key].index)
                gldas_swe = pd.Series(index=tmp_series[track_key].index)
                gldas_soilt = pd.Series(index=tmp_series[track_key].index)
                gldas_precip = pd.Series(index=tmp_series[track_key].index)
                gldas_snowmelt = pd.Series(index=tmp_series[track_key].index)
                # initialize usdasm series
                usdasm_series = pd.Series(index=tmp_series[track_key].index)
                # initialize l8 series
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
                # initialize ndvi time-series
                ndvi_series = pd.Series(index=tmp_series[track_key].index)

                # match time-series of ismn and auxiliary GEE data to S1
                for i in range(len(ssm_series.index)):
                    current_day = ssm_series.index[i]
                    if not isinstance(current_day, dt.datetime):
                        continue
                    timediff = np.min(np.abs(tmp_ssm.index - current_day))
                    ndvi_timediff = np.min(np.abs(tmpndvi.index - current_day))
                    if timediff > dt.timedelta(days=1):
                        continue
                    ssm_series.iloc[i] = tmp_ssm.iloc[np.argmin(np.abs(tmp_ssm.index - current_day))]

                    if ndvi_timediff > dt.timedelta(days=16):
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

                    tmp_gldas = self.get_gldas(float(px[0]), float(px[1]),
                                               current_day.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                               varname=['SoilMoi0_10cm_inst', 'CanopInt_inst', 'Evap_tavg',
                                                        'SWE_inst', 'SoilTMP0_10cm_inst', 'Rainf_f_tavg', 'Qsm_acc'])
                    gldas_series.iloc[i] = tmp_gldas['SoilMoi0_10cm_inst']
                    gldas_veg_water.iloc[i] = tmp_gldas['CanopInt_inst']
                    gldas_et.iloc[i] = tmp_gldas['Evap_tavg']
                    gldas_swe.iloc[i] = tmp_gldas['SWE_inst']
                    gldas_soilt.iloc[i] = tmp_gldas['SoilTMP0_10cm_inst']
                    gldas_precip.iloc[i] = tmp_gldas['Rainf_f_tavg']
                    gldas_snowmelt.iloc[i] = tmp_gldas['Qsm_acc']

                    usdasm_series.iloc[i] = self.get_USDASM(px[0], px[1],
                                                            current_day.strftime('%Y-%m-%dT%H:%M:%SZ'))

                # check the valid ISMN - S1 overlap
                vld = np.isfinite(ssm_series) & (gldas_soilt > 275) & (gldas_swe == 0)
                overlap = len(np.where(vld)[0])
                print('Valid S1 - ISMN overlap ' + str(overlap))
                if overlap < 2:
                    print('S1 - ISMN overlap too short!')
                    continue

                # build the training data frame
                mindexlist = [(px[0], px[1], int(track_key), ix) for ix in ssm_series.index[vld]]
                mindex = pd.MultiIndex.from_tuples(mindexlist)
                ll = len(vld[vld])
                tmp_dframe = pd.DataFrame({'ssm': list(np.array(ssm_series[vld]).squeeze()),
                                           'sig0vv': list(vv_series[vld]),
                                           'sig0vh': list(vh_series[vld]),
                                           'gamma0_s_vv': list(vv_gamma_s[vld]),
                                           'gamma0_s_vh': list(vh_gamma_s[vld]),
                                           'gamma0_v_vv': list(vv_gamma_v[vld]),
                                           'gamma0_v_vh': list(vh_gamma_v[vld]),
                                           'gldas': list(np.array(gldas_series[vld]).squeeze()),
                                           'usdasm': list(np.array(usdasm_series[vld]).squeeze()),
                                           'plant_water': list(gldas_veg_water[vld]),
                                           'gldas_et': list(gldas_et[vld]),
                                           'gldas_swe': list(gldas_swe[vld]),
                                           'gldas_soilt': list(gldas_soilt[vld]),
                                           'gldas_precip': list(gldas_precip[vld]),
                                           'gldas_snowmelt': list(gldas_snowmelt[vld]),
                                           'ndvi': list(ndvi_series[vld]),
                                           'lia': list(lia_series[vld]),
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
                                           'texture': [tmp_text] * ll,
                                           'lon': [px[0]] * ll,
                                           'lat': [px[1]] * ll,
                                           'track': [int(track_key)] * ll,
                                           'trees': [trees] * ll,
                                           'network': [px[3]] * ll,
                                           'station': [px[2]] * ll,
                                           'sensor': [px[4]] * ll,
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
                                           'overlap': [overlap] * ll,
                                           'orbit_direction': tmp_dirs[int(track_key)]}, index=mindex)

                if cntr == 0:
                    out_df = tmp_dframe
                else:
                    out_df = pd.concat([out_df, tmp_dframe], axis=0)

                cntr = cntr + 1
                px_counter = px_counter + 1

            if px_counter >= 1:
                trainingdb_samples = copy.deepcopy(out_df)
                used_stations.append((px[3], px[2]))
                # save temporary data
                trainingdb_samples.to_pickle(self.outpath + 'siglia.tmp')
                pickle.dump(used_stations, open(self.outpath + 'stations.tmp', 'wb'))
            else:
                if cntr == px_counter:
                    cntr = 0

            cntr2 = cntr2 + 1

        return trainingdb_samples

    def get_ismn_locations(self):

        # initialise point set
        points = set()

        # initialise available ISMN data
        ismn = ismn_interface.ISMN_Interface(self.ismn_path)
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

                points.add((station.longitude, station.latitude, st_name, ntwk, sensor[0]))

        return points

    def get_lc(self, x, y):

        #ee.Initialize()
        copernicus_collection = ee.ImageCollection('COPERNICUS/Landcover/100m/Proba-V/Global')
        copernicus_image = ee.Image(copernicus_collection.toList(100).get(0))
        roi = ee.Geometry.Point(x, y).buffer(self.footprint)

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

    def get_ssm(self, stname=None, network=None):

        if (stname is None) or (network is None):
            return 0, None

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
            return 0, None

        sm_valid = np.where(station_ts.data['soil moisture_flag'] == 'G')
        ssm_series = station_ts.data['soil moisture']
        ssm_series = ssm_series[sm_valid[0]]
        if len(ssm_series) < 5:
            return 0, None

        return 1, ssm_series

    def get_gldas(self, x, y, date, varname=['SoilMoi0_10cm_inst']):

        def get_ts(image):
            return image.reduceRegion(ee.Reducer.median(), roi, 50)

        #ee.Initialize()
        doi = ee.Date(date)
        gldas = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H") \
            .select(varname) \
            .filterDate(doi, doi.advance(3, 'hour'))

        gldas_img = ee.Image(gldas.first())
        roi = ee.Geometry.Point(x, y).buffer(100)
        try:
            return gldas_img.reduceRegion(ee.Reducer.median(), roi, 50).getInfo()
        except:
            return dict([(k, None) for k in varname])

    def get_USDASM(self, x, y, date):
        #ee.Initialize()
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

    def get_soil_texture_class(self, x, y):
        #ee.Initialize()
        roi = ee.Geometry.Point(x, y).buffer(self.footprint)
        steximg = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0')
        tmp = steximg.reduceRegion(ee.Reducer.mode(), roi, 10).getInfo()
        return tmp['b0']

    def get_bulk_density(self, x, y):
        #ee.Initialize()
        roi = ee.Geometry.Point(x, y).buffer(self.footprint)
        steximg = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select('b0')
        tmp = steximg.reduceRegion(ee.Reducer.mode(), roi, 10).getInfo()
        return tmp['b0']

    def get_clay_content(self, x, y):
        #ee.Initialize()
        roi = ee.Geometry.Point(x, y).buffer(self.footprint)
        steximg = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0')
        tmp = steximg.reduceRegion(ee.Reducer.mode(), roi, 10).getInfo()
        return tmp['b0']

    def get_sand_content(self, x, y):
        #ee.Initialize()
        roi = ee.Geometry.Point(x, y).buffer(self.footprint)
        steximg = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b0')
        tmp = steximg.reduceRegion(ee.Reducer.mode(), roi, 10).getInfo()
        return tmp['b0']

    def get_topo(self, x, y):
        #ee.Initialize()
        roi = ee.Geometry.Point(x, y).buffer(self.footprint)
        elev = ee.Image('USGS/SRTMGL1_003').reduceRegion(ee.Reducer.median(), roi).getInfo()
        aspe = ee.Terrain.aspect(ee.Image('USGS/SRTMGL1_003')).reduceRegion(ee.Reducer.median(),
                                                                            roi).getInfo()
        slop = ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003')).reduceRegion(ee.Reducer.median(),
                                                                           roi).getInfo()
        elev = elev['elevation']
        aspe = aspe['aspect']
        slop = slop['slope']

        return elev, aspe, slop
