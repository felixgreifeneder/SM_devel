# collection of plots for publication 3
import os
os.environ['PROJ_LIB'] = r'/home/fgreifeneder@eurac.edu/anaconda3/pkgs/proj-7.2.0-h8b9fe22_0/share/proj/'
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pytesmo.io.ismn.interface as ismn_interface
import numpy as np
import pickle
import pandas as pd


# PLOT BOXPLOT VV-SSM and VH-SSM
# res_names = ['5km','1km', '500m', '250m', '100m', '50m', '10m']
# basepath = '//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/'
# correlations_vv = dict()
# correlations_vh = dict()
# for j in range(7):
#     sig0lia_path = basepath + 'S1AB_' + res_names[j] + '_reprocess_lt_05/sig0lia_dictNone.p'
#     sig0lia = pickle.load(open(sig0lia_path, 'rb'))
#     uniq_sttn, uniq_idx, uniqu_cnt = np.unique(sig0lia['station'], return_index=True, return_counts=True)
#     correlations_vv[res_names[j]] = [np.corrcoef(sig0lia['sig0vv'][uniq_idx[i]:uniq_idx[i]+uniqu_cnt[i]],
#                                                  sig0lia['ssm'][uniq_idx[i]:uniq_idx[i]+uniqu_cnt[i]])[0,1] for i in range(len(uniq_idx))]
#     correlations_vh[res_names[j]] = [np.corrcoef(sig0lia['sig0vh'][uniq_idx[i]:uniq_idx[i]+uniqu_cnt[i]],
#                                                  sig0lia['ssm'][uniq_idx[i]:uniq_idx[i]+uniqu_cnt[i]])[0,1] for i in range(len(uniq_idx))]
#
# #plt.figure(figsize=(7.16, 8.5))
# plt.figure(figsize=(3.5, 2.5))
# labels, data = correlations_vv.keys(), correlations_vv.values()
# neworder = [5,2,6,0,4,1,3]
# labels = [labels[i] for i in neworder]
# data = [data[i] for i in neworder]
# plt.boxplot(data)
# plt.xticks(range(1, len(labels) + 1), labels, size=8)
# plt.tick_params(labelsize=8)
# plt.ylabel("Correlation coefficient", size=8)
# plt.title(r'$\sigma^0 VV$', size=8)
# plt.tight_layout()
# plt.savefig('C:/Users/FGreifeneder/OneDrive - Scientific Network South Tyrol/1_THESIS/pub3/images_submission2/ssm_vv_correlations_test.png', dpi=600)
# plt.close()
# #plt.figure(figsize=(7.16, 8.5))
# plt.figure(figsize=(3.5, 2.5))
# labels, data = correlations_vh.keys(), correlations_vh.values()
# neworder = [5,2,6,0,4,1,3]
# labels = [labels[i] for i in neworder]
# data = [data[i] for i in neworder]
# plt.boxplot(data)
# plt.xticks(range(1, len(labels) + 1), labels, size=8)
# plt.tick_params(labelsize=8)
# plt.ylabel("Correlation coefficient", size=8)
# plt.title(r'$\sigma^0 VH$', size=8)
# plt.tight_layout()
# plt.savefig(
#     'C:/Users/FGreifeneder/OneDrive - Scientific Network South Tyrol/1_THESIS/pub3/images_submission2/ssm_vh_correlations_test.png',
#     dpi=600)
# plt.close()

#CORRELATION PLOT BETWEEN TRUE AND ESTIMATED SMC
# def cdf(data):
#
#     data_size = len(data)
#
#     # Set bins edges
#     data_set = sorted(set(data))
#     bins = np.append(data_set, data_set[-1] + 1)
#
#     # Use the histogram function to bin the data
#     # counts, bin_edges = np.histogram(data, bins=bins, density=False)
#     counts, bin_edges = np.histogram(data, bins=100, range=(0, 0.65), density=False)
#
#     counts = counts.astype(float) / data_size
#
#     # Find the cdf
#     cdf = np.cumsum(counts)
#
#     # Plot the cdf
#     return (cdf, bin_edges)
#
# s1_ts_list, s1_ub_list, s1_failed, station_ts_list, gldas_ts_list, station_name_list = pickle.load(
#     open("C:/Users/FGreifeneder/OneDrive - Scientific Network South Tyrol/1_THESIS/pub3/images_submission2/no_GLDAS_validation_tss_50m.p", "rb"))
# #plt.interactive(True)
# fig, axs = plt.subplots(nrows=2, ncols=2, dpi=600, figsize=(3.5,4.5))
# #fig.subplots_adjust(bottom=0.8)
# #fig.set_size_inches(7, 6)
# j = -1
# gldas_ub_list = list()
# for i in range(len(s1_ts_list)):
#     j = j + 1
#     s1_ts_res = s1_ts_list[i].resample('D').mean().rename('s1')
#     gldas_ts_res = gldas_ts_list[i].resample('D').mean().rename('gldas')
#     station_ts_res = station_ts_list[i].resample('D').mean().rename('ismn')
#     tobemerged = [s1_ts_res.dropna(), station_ts_res.dropna(), gldas_ts_res.dropna()]
#     s1_and_station = pd.concat(tobemerged, axis=1, join='inner')
#     statmask = np.where((s1_and_station['ismn'] > 0) & (s1_and_station['ismn'] < 1) & (s1_and_station['s1'] > 0) & (
#                 s1_and_station['s1'] < 1))
#     p2 = s1_and_station['ismn'][statmask[0]].std() / s1_and_station['gldas'][statmask[0]].std()
#     p1 = s1_and_station['ismn'][statmask[0]].mean() - (p2 * s1_and_station['gldas'][statmask[0]].mean())
#     gldas_ub_list.append(p1 + (p2 * gldas_ts_list[i]))
#     s1_and_station = s1_and_station.iloc[statmask[0],:]
#     ts_bias = s1_and_station['gldas'].subtract(s1_and_station['ismn']).median()
#     xytmp = pd.concat({'y': s1_and_station['gldas'], 'x': s1_and_station['ismn']}, join='inner',
#                       axis=1)
#     if j == 0:
#         xyplot = xytmp
#     else:
#         xyplot = pd.concat([xyplot, xytmp], axis=0)
#
# j=-1
# for i in range(len(s1_ts_list)):
#     j = j + 1
#     s1_ub_res = s1_ub_list[i].resample('D').mean().rename('s1')
#     gldas_ub_res = gldas_ub_list[i].resample('D').mean().rename('gldas')
#     station_ts_res = station_ts_list[i].resample('D').mean().rename('ismn')
#     tobemerged = [s1_ub_res.dropna(), station_ts_res.dropna(), gldas_ub_res.dropna()]
#     s1_and_station = pd.concat(tobemerged, axis=1, join='inner')
#     statmask = np.where((s1_and_station['ismn'] > 0) & (s1_and_station['ismn'] < 1) & (s1_and_station['s1'] > 0) & (
#                 s1_and_station['s1'] < 1))
#     s1_and_station = s1_and_station.iloc[statmask[0],:]
#     ts_bias = s1_and_station['gldas'].subtract(s1_and_station['ismn']).median()
#     xytmp = pd.concat({'y': s1_and_station['s1'], 'x': s1_and_station['ismn']}, join='inner',
#                       axis=1)
#     if j == 0:
#         xyplot_ub = xytmp
#     else:
#         xyplot_ub = pd.concat([xyplot_ub, xytmp], axis=0)
#
# # plot smc true vs est (biased)
# rmse_scatter = np.sqrt(np.nanmean(np.square(xyplot['x'].subtract(xyplot['y']))))
# r_scatter = xyplot['x'].corr(xyplot['y'])
#
# xyplot.plot.scatter(x='x', y='y', color='k', xlim=(0, 1), ylim=(0, 1), s=1, marker='*', ax=axs[0,0])
# axs[0,0].set_xlim(0, 1.0)
# axs[0,0].set_ylim(0, 1.0)
# axs[0,0].set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=6)
# axs[0,0].set_ylabel("$SMC^*_{Tot}$ [m$^3$m$^{-3}$]", size=6)
# axs[0,0].plot([0, 1.0], [0, 1.0], 'k--', linewidth=0.8)
# axs[0,0].text(0.1, 0.8, 'R=' + '{:03.2f}'.format(r_scatter) +
#          '\nRMSE=' + '{:03.2f}'.format(rmse_scatter), fontsize=6)  # +
# # '\nRMSE=' + '{:03.2f}'.format(rmse_scatter), fontsize=8)
# #axs[0,0].set_tick_params(labelsize=8)
# axs[0,0].set_aspect('equal', 'box')
# axs[0,0].set_title('a)', size=6)
# axs[0,0].tick_params(labelsize=6)
#
# # plot smc true vs unbiased est
# rmse_scatter = np.sqrt(np.nanmean(np.square(xyplot_ub['x'].subtract(xyplot_ub['y']))))
# r_scatter = xyplot_ub['x'].corr(xyplot_ub['y'])
# xyplot_ub.plot.scatter(x='x', y='y', color='k', xlim=(0, 1), ylim=(0, 1),  s=1, marker='*',
#                     ax=axs[0, 1])
# axs[0, 1].set_xlim(0, 1.0)
# axs[0, 1].set_ylim(0, 1.0)
# axs[0, 1].set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=6)
# axs[0, 1].set_ylabel("$SMC^*_{Tot}$ [m$^3$m$^{-3}$]", size=6)
# axs[0, 1].plot([0, 1.0], [0, 1.0], 'k--', linewidth=0.8)
# axs[0, 1].text(0.1, 0.8, 'R=' + '{:03.2f}'.format(r_scatter) +
#                '\nuRMSE=' + '{:03.2f}'.format(rmse_scatter), fontsize=6)  # +
# # '\nRMSE=' + '{:03.2f}'.format(rmse_scatter), fontsize=8)
# #axs[0, 1].set_tick_params(labelsize=8)
# axs[0, 1].set_aspect('equal', 'box')
# axs[0, 1].set_title('b)', size=6)
# axs[0, 1].tick_params(labelsize=6)
#
# # plot cdfs
# cdf_y_tot = cdf(xyplot['x'])
# cdf_y_tot_pred = cdf(xyplot['y'])
# cdf_y_tot_pred_ub = cdf(xyplot_ub['y'])
# axs[1, 0].set_title('c)', size=6)
# axs[1,0].plot(cdf_y_tot[1][0:-1], cdf_y_tot[0], linewidth=1, color='k', label='$SMC_{Tot}$ [m$^3$m$^{-3}$]')
# axs[1,0].plot(cdf_y_tot_pred[1][0:-1], cdf_y_tot_pred[0], linewidth=0.8, color='k', linestyle='--', label='$SMC^*_{Tot}$ [m$^3$m$^{-3}$]')
# axs[1, 0].plot(cdf_y_tot_pred_ub[1][0:-1], cdf_y_tot_pred_ub[0], linewidth=0.8, color='k', linestyle='-.',
#                label='$SMC^*_{Tot}$ [m$^3$m$^{-3}$] cdf lin')
# axs[1,0].set_ylabel('Cum. freq.', size=6)
# axs[1,0].set_xlim((0, 0.7))
# axs[1,0].set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=6)
# axs[1,0].grid(b=True)
# #axs[1,0].legend(fontsize=8)
# axs[1,0].tick_params(labelsize=6)
# axs[1,0].legend(fontsize=6, bbox_to_anchor=(0.5,-0.6), loc='center', ncol=1, fancybox=False)
#
# # ismn_sorted = np.argsort(xyplot['x'])
# # y_tot_s = xyplot['x'][ismn_sorted]
# # y_pred_tot_s = xyplot['y'][ismn_sorted]
# # y_pred_tot_ub_s = xyplot_ub['y'][ismn_sorted]
# # ax2.plot(y_tot_s, y_pred_tot_s - y_tot_s, color='k', linestyle='', marker='*')
# axs[1, 1].set_title('d)', size=6)
# axs[1,1].plot(cdf_y_tot[1][0:-1], cdf_y_tot[0] - cdf_y_tot_pred[0], color='k', linewidth=0.8, label='orig.', marker='', linestyle='--')
# axs[1,1].plot(cdf_y_tot[1][0:-1], cdf_y_tot[0] - cdf_y_tot_pred_ub[0], color='k', linewidth=0.8, label='cdf lin', marker='', linestyle='-.')
# axs[1,1].set_ylabel('Cumu. freq. diff. (true - est.)', size=6)
# axs[1,1].set_xlabel("$SMC_{Tot}$ [m$^3$m$^{-3}$]", size=6)
# axs[1,1].set_xlim((0, 0.7))
# axs[1,1].set_ylim((-0.25, 0.25))
# axs[1,1].grid(b=True)
# axs[1,1].tick_params(labelsize=6)
# axs[1,1].legend(fontsize=6, bbox_to_anchor=(0.5,-0.6), loc='center', ncol=1, fancybox=False)
#
# #plt.tick_params(labelsize=6)
# plt.tight_layout()
# plt.savefig('C:/Users/FGreifeneder/OneDrive - Scientific Network South Tyrol/1_THESIS/pub3/images_submission2/1_scatter_cdf_no_GLDAS_50m.png', dpi=600)
# plt.close()


# # PLOT THE TIME SERIES FOR THE VALIDATION DATASET
# res_names = ['1km', '500m', '250m', '100m', '50m']
# colors = ['b', 'g', 'r', 'm', 'b']
# all_station_names = ['MOONEYCYN', 'Goodwell-2-E', 'SANPETEVAL', '2.10', 'marshland-soil-11', 'HAYSPEAK', 'St.-Mary-1-SSW', 'DELVALLE', 'Crossville-7-NW', 'Boulder-14-W']
# fig, axs = plt.subplots(nrows=10, ncols=1, sharex=True, squeeze=True)
# fig.set_size_inches(7.16, 9)
# s1_lines = list()
# s1_cdf_linesA = list()
# insitu_lines = list()
# s1_cdf_linesB = list()
#
# for i_res in range(4,5):
#     i_name = res_names[i_res]
#     i_color = colors[i_res]
#     s1_ts_listB, s1_ub_listB, s1_failedB, station_ts_listB, gldas_ts_listB, station_name_listB = pickle.load(open("C:/Users/FGreifeneder/OneDrive - Scientific Network South Tyrol/1_THESIS/pub3/images_submission2/w_GLDAS_validation_tss_" + i_name + ".p", "rb"))
#     s1_ts_listA, s1_ub_listA, s1_failedA, station_ts_listA, gldas_ts_listA, station_name_listA = pickle.load(open(
#         "C:/Users/FGreifeneder/OneDrive - Scientific Network South Tyrol/1_THESIS/pub3/images_submission2/no_GLDAS_validation_tss_" + i_name + ".p",
#         "rb"))
#     cntr = 0
#     for i in range(len(s1_ts_listA)):
#         print(i)
#         print('ModelA' + station_name_listA[i] + ' ModelB' + station_name_listB[i])
#         j = np.where(np.array(all_station_names) == station_name_listA[i])
#         if len(j[0]) == 0:
#             continue
#         j = j[0][0]
#         #j = i
#         s1_ts_resA = s1_ts_listA[i].resample('D').mean().rename('S1_A')
#         s1_ts_ub_resA = s1_ub_listA[i].resample('D').mean().rename('S1_cdf_A')
#         s1_ts_resB = s1_ts_listB[i].resample('D').mean().rename('S1_B')
#         s1_ts_ub_resB = s1_ub_listB[i].resample('D').mean().rename('S1_cdf_B')
#         station_ts_res = station_ts_listA[i].resample('D').mean().rename('ISMN')
#         tobemerged = [s1_ts_resA.dropna(), s1_ts_ub_resA.dropna(), s1_ts_resB.dropna(), s1_ts_ub_resB.dropna(), station_ts_res.dropna()]
#         s1_and_station = pd.concat(tobemerged, axis=1, join='inner')
#         ts_bias = s1_and_station['ISMN'].subtract(s1_and_station['S1_A']).median()
#         ts_corA = s1_and_station['S1_A'].corr(s1_and_station['ISMN'])
#         ts_corB = s1_and_station['S1_B'].corr(s1_and_station['ISMN'])
#         ts_rmseA = np.sqrt(np.nanmean(np.square(s1_and_station['S1_A'].subtract(s1_and_station['ISMN']))))
#         ts_rmseB = np.sqrt(np.nanmean(np.square(s1_and_station['S1_B'].subtract(s1_and_station['ISMN']))))
#         # ts_ubrmse = np.sqrt(np.sum(np.square((s1_and_station['S1'] - s1_and_station['S1'].mean()) - (
#         #             s1_and_station['ISMN'] - s1_and_station['ISMN'].mean()))) / len(
#         #     s1_and_station['S1']))
#         ts_ubrmseA = np.sqrt(np.nanmean(np.square(s1_and_station['S1_cdf_A'].subtract(s1_and_station['ISMN']))))
#         ts_ubrmseB = np.sqrt(np.nanmean(np.square(s1_and_station['S1_cdf_B'].subtract(s1_and_station['ISMN']))))
#
#         axs[j].set_title(station_name_listA[i], size=8)
#         if cntr == 0:
#             # s1_lines.append(axs[j].plot(s1_ts_list[i].index, s1_ts_list[i], color=i_color, linestyle='', marker='*',
#             #                             label='$SMC^*_{Tot}$ [m$^3$m$^{-3}$]', linewidth=0.2))
#             s1_cdf_linesA.append(axs[j].plot(s1_ub_listA[i].index, s1_ub_listA[i], color='r', linestyle='', marker='*',
#                                             label='Model A $SMC^*_{Tot}$ [m$^3$m$^{-3}$] cdf lin', linewidth=0.2))
#             s1_cdf_linesB.append(
#                 axs[j].plot(s1_ub_listB[i].index, s1_ub_listB[i], color='g', linestyle='', marker='+',
#                             label='Model B $SMC^*_{Tot}$ [m$^3$m$^{-3}$] cdf lin', linewidth=0.2))
#             insitu_lines.append(axs[j].plot(station_ts_listA[i].index, station_ts_listA[i],
#                                             label='$SMC_{Tot}$ [m$^3$m$^{-3}$]', color='k', linestyle='-', linewidth=0.4))
#         else:
#             axs[j].plot(s1_ub_listA[i].index, s1_ub_listA[i], color='r', linestyle='', marker='*',
#                             label='Model A $SMC^*_{Tot}$ [m$^3$m$^{-3}$] cdf lin', linewidth=0.2)
#             axs[j].plot(s1_ub_listB[i].index, s1_ub_listB[i], color='g', linestyle='', marker='+',
#                             label='Model B $SMC^*_{Tot}$ [m$^3$m$^{-3}$] cdf lin', linewidth=0.2)
#             axs[j].plot(station_ts_listA[i].index, station_ts_listA[i],
#                                             label='$SMC_{Tot}$ [m$^3$m$^{-3}$]', color='k', linestyle='-',
#                                             linewidth=0.4)
#
#         smc_max = np.max([s1_ts_listB[i].max(), station_ts_listA[i].max()])
#         cntr = cntr + 1
#         if smc_max <= 0.5:
#             smc_max = 0.5
#         axs[j].set_ylim((0, smc_max))
#         if (j == 1) or (j==4) or (j==6) or (j==7) or (j==8) or (j==9):
#             textx = 0.01
#         else:
#             textx = 0.80
#         axs[j].text(textx, 0.1, '\N{GREEK SMALL LETTER RHO}' + '=' + '{:03.2f}'.format(ts_corA) + '/' + '{:03.2f}'.format(ts_corB) +
#                 '\nRMSE=' + '{:03.2f}'.format(ts_rmseA) + '/' + '{:03.2f}'.format(ts_rmseB) +
#                 '\nuRMSE=' + '{:03.2f}'.format(ts_ubrmseA) + '/' + '{:03.2f}'.format(ts_ubrmseB), transform=axs[j].transAxes, fontsize=8)
#
# #plt.title(st_name, fontsize=8)
# #plt.legend(loc=8, fancybox=False, shadow=False, ncol=2, fontsize=8)
# #plt.ylabel("SMC [m$^3$m$^{-3}$]", size=8)
# #plt.legend(loc=8, ncol=2)
# plt.tight_layout()
# fig.subplots_adjust(left=0.08, bottom=0.1)
# fig.text(0.01, 0.5, "SMC [m$^3$m$^{-3}$]", size=8, ha="center", va="center", rotation="vertical")
# plt.legend(tuple([i[0] for i in s1_cdf_linesA] + [i[0] for i in s1_cdf_linesB] + [insitu_lines[0][0]]),
#            tuple([i[0]._label for i in s1_cdf_linesA] + [i[0]._label for i in s1_cdf_linesB] + [insitu_lines[0][0]._label]), loc=8, fancybox=False,
#            shadow=False, ncol=3, fontsize=8, bbox_to_anchor=(0.5, -1.9))
# plt.savefig("C:/Users/FGreifeneder/OneDrive - Scientific Network South Tyrol/1_THESIS/pub3/images_submission2/w_GLDAS_validation_tss_50.png", dpi=300)
# plt.close()

# # PLOT PREDICTION ACCURACY RELATED TO SPATIAL RESOLUTION
# NO GDAL
# res = np.array([1000,500,250,100,50])
# ub_rmse_avg = np.array([0.04,0.06,0.04,0.03,0.06])
# rmse_avg = np.array([0.04, 0.06, 0.04, 0.03, 0.06])
# r_avg = np.array([0.94,0.86,0.93,0.95,0.85])
#
# ub_rmse_rel = np.array([0.05,0.05,0.05,0.05,0.06])
# rmse_rel = np.array([0.05, 0.05, 0.05, 0.05, 0.06])
# r_rel = np.array([0.75,0.73,0.73,0.72,0.68])
# # WITH GDAL
# # res = np.array([1000, 500, 250, 100, 50])
# # ub_rmse_avg = np.array([0.03,0.04,0.04,0.03,0.06])
# # rmse_avg = np.array([0.03, 0.05, 0.04, 0.03, 0.06])
# # r_avg = np.array([0.97, 0.91, 0.95, 0.95, 0.87])
# #
# # ub_rmse_rel = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
# # rmse_rel = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
# # r_rel = np.array([0.79, 0.78, 0.78, 0.77, 0.76])
#
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
# ax1.set_title('RMSE', size=8)
# ax1.plot(res, rmse_avg, marker='x', linestyle='--', color='k', label='Avg.', linewidth=0.5, markersize=6)
# ax1.plot(res, rmse_rel, marker='+', linestyle='--', color='k', label='Rel', linewidth=0.5, markersize=6)
# ax1.set_xlabel('Resolution [m]', size = 8)
# ax1.set_ylabel('RMSE [m$^3$m$^{-3}$]', size= 8)
# ax1.tick_params(labelsize=8)
#
# #fig, ax1 = plt.subplots()
# ax2.set_title('Correlation coefficient', size=8)
# ax2.plot(res, r_avg, marker='x', linestyle='--', color='k', label='_nolegend_', linewidth=0.5, markersize=6)
# ax2.plot(res, r_rel, marker='+', linestyle='--', color='k', label='_nolegend_', linewidth=0.5, markersize=6)
# ax2.set_xlabel('Resolution [m]', size=8)
# ax2.set_ylabel('Correlation coefficient', size=8)
# ax2.tick_params(labelsize=8)
# fig.set_size_inches(3.5, 5.5)
# fig.legend(loc=2, fontsize=8, bbox_to_anchor=(0.7, 0.92))
# fig.tight_layout()
# fig.savefig('C:/Users/FGreifeneder/OneDrive - Scientific Network South Tyrol/1_THESIS/pub3/images_submission2/res_vs_acc.png', dpi=600)
# plt.close()


# PLOT MAP WITH THE LOCATIONS OF ISMN STATIONS
def plot_ismn_map():
    stations = pickle.load(open('/mnt/CEPH_PROJECTS/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/newrun_ft_descending/stations.tmp',
                                'rb'), encoding='latin-1')

    ismn = ismn_interface.ISMN_Interface('/mnt/CEPH_PROJECTS/ECOPOTENTIAL/reference_data/ISMN/')

    fig = plt.figure(figsize=(5,4))
    ax = plt.subplot(111)
    #plt.figure(figsize=(3.5, 3))
    m = Basemap(projection='mill', ax=ax)
    m.drawcoastlines(linewidth=0.5)
    #m.fillcontinents()

    # plot stations from the different networks with different markers
    markerlist = ['o', 'v', '*', 'h', 'p', '^', '+', 'x', 'd', '8', 's',
                  '*', 'v', 'o', 'p', 'h', '^', 'x', '+', '8', 'd', 's',
                  's', '8', 'd', 'x', '+', 'p', '^', 'p', 'h', '*', 'o']
    colours = ['g','b','r','k','y','c','m','g','k','b','c',
               'c','b','k','g','m','c','y','k','b','r','g',
               'g','b','r','k','y','c','m','g','k','b','c']

    # create a networklist
    allntwks = list()
    for i_ntwk, _ in stations:
        allntwks.append(i_ntwk)
    allntwks = np.unique(allntwks)
    for i_ntwk, i_sttn in stations:
        style_ind = np.where(allntwks == i_ntwk)[0][0]
        lon = ismn.get_station(i_sttn, i_ntwk).longitude
        lat = ismn.get_station(i_sttn, i_ntwk).latitude
        x, y = m(lon, lat)
        m.plot(x, y, marker='.', color='C' + str(style_ind), linestyle='', markersize=1.5, label=str(i_ntwk))

    m.drawmeridians([-180, -90, 0, 90, 180], labels=[1,1,0,1], fontsize="x-small")
    m.drawparallels([-60, 0, 60],labels=[1,1,0,1], fontsize="x-small")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.9])

    #create legend without dublicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Put a legend below current axis
    ax.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=False, ncol=4, fontsize='x-small', markerscale=5)

    #plt.tight_layout()
    plt.savefig("/home/fgreifeneder@eurac.edu/Documents/sm_paper/sub3mdpi/used_ismn_stations.png", dpi=600)
    plt.close()

#plot correlation plots - create plots between SMC and static feature (mean sig0, lc, topo, etc ...)
# sig0lia = pickle.load(open("//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_250m/sig0lia_dictNone.p", 'rb'))
# invalid_col = pickle.load(open('//projectdata.eurac.edu/projects/ESA_TIGER/S1_SMC_DEV/Processing/S1ALPS/ISMN/S1AB_1km/invalid_col.p', 'rb'))
#
# vv = np.array(sig0lia['vv_tmean'])
# #vv = 10 * np.log10(np.exp(vv))
# vh = np.array(sig0lia['vh_tmean'])
# #vh = 10*np.log10(np.exp(vh))
#
# valid = np.where(np.isfinite(sig0lia['ssm']) & (np.array(sig0lia['sig0vv']) > -22) & (np.array(sig0lia['sig0vh']) > -22))
# valid2 = np.where(np.isfinite(sig0lia['ssm_mean']) & (vv > -22) & (vh > -22) & (np.array(sig0lia['ssm_mean']) < 0.5))
#
# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# f.set_size_inches(3.5,3)
# f.subplots_adjust(wspace=0.5, hspace=0.5)
# ssm = np.array(sig0lia['ssm'])[valid]
# vv = np.array(sig0lia['sig0vv'])[valid]
# ax1.scatter(ssm, vv, marker='.', s=1, c="k")
# ax1.plot([[0,0]])
# ax1.set_xlabel("SMC [m$^3$m$^{-3}$]", size=8)
# ax1.set_ylabel('SIG0 VV [dB]', size=8)
# #ax1.set_xlim(0,0.5)
# r=np.corrcoef(ssm,vv)[0,1]
# ax1.text(0.1, 0.8, 'R=' + '{:03.2f}'.format(r), transform=ax1.transAxes, fontsize=8)
#
# vh = np.array(sig0lia['sig0vh'])[valid]
# ax2.scatter(ssm, vh, marker='.', s=1, c="k")
# ax2.set_xlabel('SMC [m$^3$m$^{-3}$]', size=8)
# ax2.set_ylabel('SIG0 VH [dB]', size=8)
# #ax2.set_xlim(0,0.5)
# r=np.corrcoef(ssm,vh)[0,1]
# ax2.text(0.1, 0.8, 'R=' + '{:03.2f}'.format(r), transform=ax2.transAxes, fontsize=8)
#
# ssm = np.array(sig0lia['ssm_mean'])[valid2]
# vv = np.array(sig0lia['vv_tmean'])[valid2]
# #vv = 10 * np.log10(np.exp(vv))
# ax3.scatter(ssm, vv, marker='.', s=1, c="k")
# ax3.set_xlabel('Mean SMC [m$^3$m$^{-3}$]', size=8)
# ax3.set_ylabel('Mean SIG0 VV [dB]', size=8)
# #ax3.set_xlim(0,0.5)
# #ax3.set_yticks([-8,-10,-12,-14,-16,-18])
# ax3.set_yticklabels(['-8','-10','-12','-14','-16','-18'])
# r=np.corrcoef(ssm,vv)[0,1]
# ax3.text(0.1, 0.8, 'R=' + '{:03.2f}'.format(r), transform=ax3.transAxes, fontsize=8)
#
# ssm = np.array(sig0lia['ssm_mean'])[valid2]
# vh = np.array(sig0lia['vh_tmean'])[valid2]
# #vh = 10 * np.log10(np.exp(vh))
# ax4.scatter(ssm, vh, marker='.', s=1, c="k")
# ax4.set_xlabel('Mean SMC [m$^3$m$^{-3}$]', size=8)
# ax4.set_ylabel('Mean SIG0 VH [dB]', size=8)
# #ax4.set_xlim(0,0.5)
# r=np.corrcoef(ssm,vh)[0,1]
# ax4.text(0.1, 0.8, 'R=' + '{:03.2f}'.format(r), transform=ax4.transAxes, fontsize=8)
#
# plt.tick_params(labelsize=8)
# plt.tight_layout()
# plt.savefig('X:/Workspaces/GrF/02_Documents/1_THESIS/pub3/images_submission2/sig0_vs_ssm.png', dpi=600)
# plt.close()

# # plot the comparison of valid vs invalid ISMN
# data = np.loadtxt('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ISMN/tmp/invalidcols.csv', skiprows=1, delimiter=',')
# invalid_col = pickle.load(open("X:/Workspaces/GrF/Processing/S1ALPS/ISMN/gee_global_r_optim/invalid_col.p", 'rb'))
# sdepth = list()
# lon=list()
# lat=list()
# ismn = ismn_interface.ISMN_Interface('T:/ECOPOTENTIAL/reference_data/ISMN/')
# for i in range(len(invalid_col['ntwkname'])):
#     tmpst = ismn.get_station(invalid_col['stname'][i], invalid_col['ntwkname'][i])
#     print(tmpst.get_depths('soil moisture')[1][0])
#     sdepth.append(tmpst.get_depths('soil moisture')[1][0])
#     lon.append(tmpst.longitude)
#     lat.append(tmpst.latitude)
# f, axes = plt.subplots(2, 3)
# f.set_size_inches(9,5)
# f.subplots_adjust(wspace=0.7, hspace=0.5)
#
# valid = np.where(np.array(invalid_col['label']) == 'valid')
# invalid = np.where(np.array(invalid_col['label']) == 'invalid')
#
# # SSM Mean
# tmp = [np.array(invalid_col['ssm_mean'])[valid], np.array(invalid_col['ssm_mean'])[invalid]]
# axes[0,0].boxplot(tmp, labels=['Valid', 'Invalid'])
# axes[0,0].set_ylabel('Avg. SMC [m3m-3]')
#
# # sig0 mean VV
# tmp = [np.array(invalid_col['vv_k1'])[valid], np.array(invalid_col['vv_k1'])[invalid]]
# tmp = 10*np.log10(np.exp(tmp) / 10)
# axes[0,1].boxplot(tmp, labels=['Valid', 'Invalid'])
# axes[0,1].set_ylabel('Avg. SIG0-VV [dB]')
#
# # sig0 mean VH
# tmp = [np.array(invalid_col['vh_k1'])[valid], np.array(invalid_col['vh_k1'])[invalid]]
# tmp = 10 * np.log10(np.exp(tmp) / 10)
# axes[0,2].boxplot(tmp, labels=['Valid', 'Invalid'])
# axes[0,2].set_ylabel('Avg. SIG0-VH [dB]')
#
# # elevation
# [np.array(invalid_col['vv_k1'])[valid], np.array(invalid_col['vv_k1'])[invalid]]
# axes[0,3].boxplot(tmp, labels=['Valid', 'Invalid'])
# axes[0,3].set_ylabel('Elevation [m]')
#
# # slope
# tmp = [data[valid, 0], data[invalid, 0]]
# axes[1,0].boxplot(tmp, labels=['Valid', 'Invalid'])
# axes[1,0].set_ylabel('Slope [deg]')
#
# # aspect
# tmp = [data[valid, 11], data[invalid, 11]]
# axes[1,1].boxplot(tmp, labels=['Valid', 'Invalid'])
# axes[1,1].set_ylabel('Aspect [deg]')
#
# lc
# vldidity = np.full(len(invalid_col['label']), 0)
# vldidity[invalid_col['label'] == 'invalid'] = 1
# axes[1,0].scatter(vldidity, invalid_col['lc'])
# axes[1,0].set_ylabel('Land-Cover')
# axes[1,0].set_xlabel('Validity')
#
# #sensor depth / ndvi
# from sgrt_devels.extr_TS import extr_USGS_LC
# lc=list()
# for i in range(len(lon)):
#     tmp = extr_USGS_LC(lon[i],lat[i], 40)
#     lc.append(tmp['landcover'])
#
# lc = np.array(lc)
# #evi[np.where(np.isnan(evi))] = 0
# axes[1,1].scatter(vldidity, lc)
# axes[1,1].set_ylabel('LC')
#
# axes[1,3].boxplot([np.array(sdepth)[valid], np.array(sdepth)[invalid]], labels=['Valid', 'Invalid'])
# axes[1,3].set_ylabel('Sensor depth [m]')
# axes[1,3].set_xlabel('Validity')
#
# plt.savefig('X:/Workspaces/GrF/Processing/S1ALPS/ISMN/gee_global_r_optim/validinvalidstations2.png', dpi=600)
# plt.close()

# plot the correlations between SMC sensitivity and different features
# invalid_col = pickle.load(open("X:/Workspaces/GrF/Processing/S1ALPS/ISMN/gee_global_all_stations/invalid_col.p", 'rb'))
# sdepth = list()
# lon = list()
# lat = list()
# ismn = ismn_interface.ISMN_Interface('T:/ECOPOTENTIAL/reference_data/ISMN/')
# for i in range(len(invalid_col['ntwkname'])):
#     tmpst = ismn.get_station(invalid_col['stname'][i], invalid_col['ntwkname'][i])
#     print(tmpst.get_depths('soil moisture')[1][0])
#     sdepth.append(tmpst.get_depths('soil moisture')[1][0])
#     lon.append(tmpst.longitude)
#     lat.append(tmpst.latitude)
# f, axes = plt.subplots(3, 4)
# f.set_size_inches(12, 9)
# f.subplots_adjust(wspace=0.7, hspace=0.5)
#
# valid = np.isfinite(invalid_col['ssm_mean'])
#
# # SSM Mean
# axes[0, 0].scatter(np.array(invalid_col['ssm_mean'])[valid], np.array(invalid_col['multi_cor'])[valid])
# axes[0, 0].set_ylabel('Sensitivity')
# axes[0, 0].set_xlabel('Mean SMC')
#
# # vv k1
# axes[0, 1].scatter(np.array(invalid_col['vv_k1'])[valid], np.array(invalid_col['multi_cor'])[valid])
# axes[0, 1].set_ylabel('Sensitivity')
# axes[0, 1].set_xlabel('Mean SIG0 VV')
#
# # vh k1
# axes[0, 2].scatter(np.array(invalid_col['vh_k1'])[valid], np.array(invalid_col['multi_cor'])[valid])
# axes[0, 2].set_ylabel('Sensitivity')
# axes[0, 2].set_xlabel('Mean SIG0 VH')
#
# # vv k2
# axes[0, 3].scatter(np.array(invalid_col['vv_k2'])[valid], np.array(invalid_col['multi_cor'])[valid])
# axes[0, 3].set_ylabel('Sensitivity')
# axes[0, 3].set_xlabel('Std SIG0 VV')
#
# # vh k2
# axes[1, 0].scatter(np.array(invalid_col['vh_k2'])[valid], np.array(invalid_col['multi_cor'])[valid])
# axes[1, 0].set_ylabel('Sensitivity')
# axes[1, 0].set_xlabel('Std SIG0 VH')
#
# # lia
# axes[1, 1].scatter(np.array(invalid_col['lia'])[valid], np.array(invalid_col['multi_cor'])[valid])
# axes[1, 1].set_ylabel('Sensitivity')
# axes[1, 1].set_xlabel('LIA')
#
# # gldas swe
# #from sgrt_devels.extr_TS import extr_gldas_avrgs
# #swe = extr_gldas_avrgs(lon, lat, 'swe')
#
# axes[1, 2].scatter(np.log10(np.array(invalid_col['lc'])[valid]), np.array(invalid_col['multi_cor'])[valid])
# axes[1, 2].set_ylabel('Sensitivity')
# axes[1, 2].set_xlabel('LC')
#
# # plant water
# # evi
# # from sgrt_devels.extr_TS import extr_MODIS_MOD13Q1_ts_GEE
# # evi = list()
# # for i in range(len(lon)):
# #     evi.append(extr_MODIS_MOD13Q1_ts_GEE(lon[i],lat[i]))
# # evi = np.array(evi)
# #evi[np.where(np.isnan(evi))] = 0
#
# # soil temp
# #stemp = extr_gldas_avrgs(lon, lat, 'tmmn')
#
# axes[1, 3].scatter(np.array(invalid_col['sig_cor'])[valid], np.array(invalid_col['multi_cor'])[valid])
# #axes[1, 3].scatter(evi[valid], np.array(invalid_col['multi_cor'])[valid])
# axes[1, 3].set_ylabel('Sensitivity')
# axes[1, 3].set_xlabel('VV-VH cor')
# #axes[1, 3].set_xscale('log')
#
# # land cover
# #from sgrt_devels.extr_TS import extr_USGS_LC
#
# #lc = extr_USGS_LC(lon,lat)
# #lc = np.array(lc)
#
# # ET
# #et = extr_gldas_avrgs(lon, lat, 'aet')
#
# axes[2, 0].scatter(np.array(invalid_col['gldas_et'])[valid], np.array(invalid_col['multi_cor'])[valid])
# #axes[2, 0].scatter(lc[valid], np.array(invalid_col['multi_cor'])[valid])
# axes[2, 0].set_ylabel('Sensitivity')
# axes[2, 0].set_xlabel('GLDAS ET')
# axes[2, 0].set_xlim(0,0.0001)
#
# # height
# axes[2, 1].scatter(np.array(invalid_col['height'])[valid], np.array(invalid_col['multi_cor'])[valid])
# axes[2, 1].set_ylabel('Sensitivity')
# axes[2, 1].set_xlabel('Height')
#
# # lon
# axes[2, 2].scatter(np.array(lon)[valid], np.array(invalid_col['multi_cor'])[valid])
# axes[2, 2].set_ylabel('Sensitivity')
# axes[2, 2].set_xlabel('Longitude')
#
# # lat
# axes[2, 3].scatter(np.array(lat)[valid], np.array(invalid_col['multi_cor'])[valid])
# axes[2, 3].set_ylabel('Sensitivity')
# axes[2, 3].set_xlabel('Latitude')
#
#
# plt.savefig("X:/Workspaces/GrF/Processing/S1ALPS/ISMN/gee_global_all_stations/sensitivity2.png", dpi=600)
# plt.close()

# plot dependency between S2 reflectances and s1 sensitivity
# invalid_col = pickle.load(
#     open("X:/Workspaces/GrF/Processing/S1ALPS/ISMN/gee_global_all_stations/invalid_col.p", 'rb'))
# sdepth = list()
# lon = list()
# lat = list()
# ismn = ismn_interface.ISMN_Interface('T:/ECOPOTENTIAL/reference_data/ISMN/')
# for i in range(len(invalid_col['ntwkname'])):
#     tmpst = ismn.get_station(invalid_col['stname'][i], invalid_col['ntwkname'][i])
#     print(tmpst.get_depths('soil moisture')[1][0])
#     sdepth.append(tmpst.get_depths('soil moisture')[1][0])
#     lon.append(tmpst.longitude)
#     lat.append(tmpst.latitude)
# f, axes = plt.subplots(3, 4)
# f.set_size_inches(12, 9)
# f.subplots_adjust(wspace=0.7, hspace=0.5)
#
# # sample s2
# from sgrt_devels.extr_TS import extr_s2_avgs
#
# s2_samples = extr_s2_avgs(lon, lat)
#
# #valid = np.isfinite(invalid_col['ssm_mean'])
#
# # B2 - blue
# b2 = np.array([x['properties']['B2_mean'] for x in s2_samples['features']])
#
# axes[0, 0].scatter(b2, np.array(invalid_col['multi_cor']))
# axes[0, 0].set_ylabel('Sensitivity')
# axes[0, 0].set_xlabel('S2 B2')
#
# # B3 - Green
# b3 = np.array([x['properties']['B3_mean'] for x in s2_samples['features']])
#
# axes[0, 1].scatter(b3, np.array(invalid_col['multi_cor']))
# axes[0, 1].set_ylabel('Sensitivity')
# axes[0, 1].set_xlabel('S2 B3')
#
# # B4 - Green
# b4 = np.array([x['properties']['B4_mean'] for x in s2_samples['features']])
#
# axes[0, 2].scatter(b4, np.array(invalid_col['multi_cor']))
# axes[0, 2].set_ylabel('Sensitivity')
# axes[0, 2].set_xlabel('S2 B4')
#
# # B5 - Red Edge 1
# b5 = np.array([x['properties']['B5_mean'] for x in s2_samples['features']])
#
# axes[0, 3].scatter(b5, np.array(invalid_col['multi_cor']))
# axes[0, 3].set_ylabel('Sensitivity')
# axes[0, 3].set_xlabel('S2 B5')
#
# # B6 - Red Edge 2
# b6 = np.array([x['properties']['B6_mean'] for x in s2_samples['features']])
#
# axes[1, 0].scatter(b6, np.array(invalid_col['multi_cor']))
# axes[1, 0].set_ylabel('Sensitivity')
# axes[1, 0].set_xlabel('S2 B6')
#
# # B7 - Red Edge 3
# b7 = np.array([x['properties']['B7_mean'] for x in s2_samples['features']])
#
# axes[1, 1].scatter(b7, np.array(invalid_col['multi_cor']))
# axes[1, 1].set_ylabel('Sensitivity')
# axes[1, 1].set_xlabel('S2 B7')
#
# # B8 - NIR
# b8 = np.array([x['properties']['B8_mean'] for x in s2_samples['features']])
#
# axes[1, 2].scatter(b8, np.array(invalid_col['multi_cor']))
# axes[1, 2].set_ylabel('Sensitivity')
# axes[1, 2].set_xlabel('S2 B8')
#
# # B8a - Red Edge 4
# b8a = np.array([x['properties']['B8A_mean'] for x in s2_samples['features']])
#
# axes[1, 3].scatter(b8a, np.array(invalid_col['multi_cor']))
# axes[1, 3].set_ylabel('Sensitivity')
# axes[1, 3].set_xlabel('S2 B8a')
#
# # B11 - SWIR
# b11 = np.array([x['properties']['B11_mean'] for x in s2_samples['features']])
#
# axes[2, 0].scatter(b11, np.array(invalid_col['multi_cor']))
# axes[2, 0].set_ylabel('Sensitivity')
# axes[2, 0].set_xlabel('S2 B11')
#
# # B12 - SWIR 2
# b12 = np.array([x['properties']['B12_mean'] for x in s2_samples['features']])
#
# axes[2, 1].scatter(b12, np.array(invalid_col['multi_cor']))
# axes[2, 1].set_ylabel('Sensitivity')
# axes[2, 1].set_xlabel('S2 B12')
#
# plt.savefig("X:/Workspaces/GrF/Processing/S1ALPS/ISMN/gee_global_all_stations/sensitivity_s2.png", dpi=600)
# plt.close()

# #ismn station time-series
# ismn = ismn_interface.ISMN_Interface('/mnt/SAT4/DATA/S1_EODC/ISMN/')
# invalid_col = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ISMN/tmp/invalid_col.p', 'rb'))
#
# for i in range(len(invalid_col['label'])):
#     station = ismn.get_station(invalid_col['stname'][i], invalid_col['ntwkname'][i])
#     print(invalid_col['stname'][i] + ', ' + invalid_col['ntwkname'][i])
#     station_depths = station.get_depths('soil moisture')
#     did = np.where(station_depths[0] == 0.0)
#     dto = station_depths[1][did]
#     sm_sensors = station.get_sensors('soil moisture', depth_from=0, depth_to=dto[0])
#     station_ts = station.read_variable('soil moisture', depth_from=0, depth_to=dto[0], sensor=sm_sensors[0])
#
#     plt.figure(figsize=(2,4))
#     station_ts.plot()
#     if invalid_col['label'][i] == 'valid':
#         plt.savefig('/mnt/SAT/Workspaces/GrF/02_Documents/1_THESIS/pub3/plots/valid_stations/' + invalid_col['stname'][i] + '.png', dpi=600)
#         plt.close()
#     else:
#         plt.savefig('/mnt/SAT/Workspaces/GrF/02_Documents/1_THESIS/pub3/plots/invalid_stations/' + invalid_col['stname'][
#             i] + '.png', dpi=600)
#         plt.close()






