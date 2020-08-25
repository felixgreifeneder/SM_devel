import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os.path

if __name__ == '__main__':

    dateparse_insitu = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    dateparse_sat = lambda x: (pd.to_datetime('1900-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S') + pd.to_timedelta(np.round(x), unit='D'))

    for file in glob.glob('X:/Workspaces/GrF/Processing/SCA_paper/revision3/ascat_extracted/sig_slope/*.csv'):
        bname = os.path.basename(file)
        slon = float(bname[4:9])
        slat = float(bname[14:18])
        print(slon)
        print(slat)
        # insitu_file = '/mnt/SAT/Workspaces/GrF/Processing/SCA_paper/insitu_mariette/' + bname
        #
        # # load time series from csvs
        #
        # ts_insitu = pd.read_csv(insitu_file,
        #                         header=0,
        #                         index_col=0,
        #                         usecols=[0, 2],
        #                         names=['Date', 'SMC insitu'],
        #                         squeeze=True,
        #                         parse_dates=True,
        #                         date_parser=dateparse_insitu)
        #
        # ts_insitu = ts_insitu['20130101':'20131231']
        # ts_insitu = ts_insitu.resample('D').mean()

        ts_sat = pd.read_csv(file,
                             header=0,
                             index_col=3,
                             names=['lat', 'lon', 'lc', 'Date', 'ASCAT_front', 'ASCAT_mid', 'ASCAT_aft',
                                    'Slope', 'aquVV', 'aquVH', 'aquHH', 'VVVH', 'SMC ERA', 'VHVV', 'ob.ASCAT',
                                    'ob.ASCAT+Slope', 'ob.ASCAT+VHVV', 'ob.all', 'SMC(ASCAT)', 'SMC(ASCAT+Slope)',
                                    'SMC(ASCAT+VHVV)', 'SMC(all)'],
                             parse_dates=True)
                             #date_parser=dateparse_sat)

        ts_sat['VHVV'] = 10*np.log10(ts_sat['VHVV'])
        ts_sat.index = dateparse_sat(ts_sat.index)
        # ts_sat = ts_sat['20130101':'20131231']
        #ts_sat = ts_sat.resample('D').pad()

        #ts_combined = pd.concat([ts_insitu, ts_sat], axis=1, join='outer')

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax1 = fig.add_subplot(2, 1, 1)
        # ax2 = fig.add_subplot(2, 1, 2)
        fig, axes = plt.subplots(5, 1, sharex=True, figsize=(4,8))
        #plt.subplots_adjust(hspace=0.5)

        # if len(ts_insitu) > 0:
        #     ts_insitu.plot(legend=True, ax=axes[0,0])
        #     axes[0,0].set_ylabel('SMC [m3m-3]')

        # ts_sat.loc[:,['SMC ERA', 'ASCAT_front', 'ASCAT_mid', 'ASCAT_aft']].plot(subplots=False,
        #                                                                      ax=axes[0],
        #                                                                      legend=True,
        #                                                                      secondary_y=['SMC ERA'],
        #                                                                      mark_right=False)
        # axes[0].set_ylabel('SIG0 [dB]', fontsize=6)
        # axes[0].right_ax.set_ylabel('SMC [m3m-3]', fontsize=6)
        # axes[0].tick_params(labelsize = 5)
        # axes[0].right_ax.tick_params(labelsize=5)
        # axes[0].legend(fontsize=5)
        # axes[0].right_ax.legend(fontsize=5)

        ts_sat.loc[:, ['Slope', 'VHVV']].plot(subplots=False,
                                              ax=axes[0],
                                              legend=True,
                                              secondary_y=['VHVV'],
                                              mark_right=False,
                                              lw=0.7,
                                              style=['k-','k--'])

        axes[0].set_ylabel('Slope', fontsize=6)
        axes[0].right_ax.set_ylabel('VH/VV [dB]', fontsize=6)
        axes[0].tick_params(labelsize=5)
        axes[0].right_ax.tick_params(labelsize=5)
        axes[0].legend(fontsize=5, loc=4)
        axes[0].right_ax.legend(fontsize=5)

        # ts_sat.loc[:, ['SMC ERA', 'Slope']].plot(subplots=False,
        #                                                       ax=axes[2],
        #                                                       legend=True,
        #                                                       secondary_y=['SMC ERA'],
        #                                                       mark_right=False)
        # axes[2].set_ylabel('Slope', fontsize=6)
        # axes[2].right_ax.set_ylabel('SMC [m3m-3]', fontsize=6)
        # axes[2].tick_params(labelsize=5)
        # axes[2].right_ax.tick_params(labelsize=5)
        # axes[2].legend(fontsize=5, loc=4)
        # axes[2].right_ax.legend(fontsize=5)
        linewidths=[1,1,1,1,1.7]
        cols = ['SMC(ASCAT)', 'SMC(ASCAT+Slope)', 'SMC(ASCAT+VHVV)', 'SMC(all)']

        for i in range(4):
            ts_sat[['SMC ERA', cols[i]]].plot(subplots=False,
                                              legend=True,
                                              ax=axes[i+1],
                                              lw=0.7,
                                              ylim=(0,0.6),
                                              style=['k-','k--'])
            axes[i+1].set_ylabel('SMC [m3m-3]', fontsize=6)
            axes[i+1].tick_params(labelsize=5)
            axes[i+1].legend(fontsize=5)


        axes[4].set_xlabel('Date', fontsize=6)

        #ts_insitu.plot(subplots=False, legend=True)

        # ax.spines['top'].set_color('none')
        # ax.spines['bottom'].set_color('none')
        # ax.spines['left'].set_color('none')
        # ax.spines['right'].set_color('none')
        # ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        # ax.set_ylabel('SMC [m3m-3]\n')

        #plt.plot(ts_insitu.index, ts_insitu, 'b--', label='In-Situ SMC')
        # ax1.plot(ts_sat.index, ts_sat['ts.SMC.pred.ob.ascat'], 'g-', label='Pred. OB Ascat')
        # ax1.plot(ts_sat.index, ts_sat['ts.SMC.pred.ob.ascat.slope'], 'r-', label='Pred. OB Ascat+Slope')
        # ax1.plot(ts_sat.index, ts_sat['ts.SMC.pred.ob.ascat.vhvv'], 'c-', label='Pred. OB Ascat+Pol-Ratio')
        # ax1.plot(ts_sat.index, ts_sat['ts.SMC.pred.ob.ascat.all'], 'm-', label='Pred. OB Ascat+Slope+Pol-Ratio')
        # ax1.plot(ts_sat.index, ts_sat['ts.SMC.pred.ascat'], 'g--', label='Pred. Ascat')
        # ax1.plot(ts_sat.index, ts_sat['ts.SMC.pred.ascat.slope'], 'r--', label='Pred. Ascat+Slope')
        # ax1.plot(ts_sat.index, ts_sat['ts.SMC.pred.ascat.VHVV'], 'c--', label='Pred. Ascat+Pol-Ratio')
        # ax1.plot(ts_sat.index, ts_sat['ts.SMC.pred.ascat.all'], 'm--', label='Pred. Ascat+Slope+Pol-Ratio')
        #
        # ax2.plot(ts_sat.index, ts_sat['ts.SMC'], 'b-', label='ERA SMC', linewidth=2.0)

        plt.savefig('X:/Workspaces/GrF/Processing/SCA_paper/revision3/ts_plots/sig_slope/test/' + bname + '.png', dpi=600, )

        plt.close()