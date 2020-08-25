import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M')
dateparse2 = lambda x: pd.datetime.strptime(x, '%d/%m/%Y')
dateparse3 = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

# READ In-Situ data
# --------------------

nepas2000_insitu = pd.read_csv('/mnt/SAT/Workspaces/GrF/02_Documents/monalisa/NEPAS2000/obs_NEPAS2000_0001.txt',
                               header=0,
                               index_col=0,
                               usecols=[0,2],
                               names=['Date', 'Nepas2000.insitu'],
                               squeeze=True,
                               parse_dates=True,
                               date_parser=dateparse)

#nepas2000_insitu = nepas2000_insitu.asfreq('H')
nepas2000_insitu = nepas2000_insitu.resample('D').mean()

domef1500_insitu = pd.read_csv('/mnt/SAT/Workspaces/GrF/02_Documents/monalisa/NEPAS2000/obs_domef1500_nofr0001.txt',
                               header=0,
                               index_col=0,
                               usecols=[0,2],
                               names=['Date', 'Domef1500.insitu'],
                               squeeze=True,
                               parse_dates=True,
                               date_parser=dateparse,
                               na_values=-9999)

nemef1500_insitu = pd.read_csv('/mnt/SAT/Workspaces/GrF/02_Documents/monalisa/insitu/meadows/nemef1500/csv-zip-content.csv',
                               header=0,
                               index_col=0,
                               usecols=[3,4],
                               names=['Date', 'Nemef1500.insitu'],
                               squeeze=True,
                               parse_dates=True,
                               na_values=-9999,
                               sep=';')

nemef1500_insitu = nemef1500_insitu.resample('D').mean()

# --------------------
# Read simulated data
# --------------------

nepas2000_sim = pd.read_csv('/mnt/SAT/Workspaces/GrF/02_Documents/monalisa/NEPAS2000/thetaliq0001_NEPAS2000.txt',
                            header=0,
                            index_col=0,
                            usecols=[0,9],
                            names=['Date', 'Nepas2000.sim'],
                            squeeze=True,
                            parse_dates=True,
                            date_parser=dateparse)

#nepas2000_sim = nepas2000_sim.asfreq('H')
nepas2000_sim = nepas2000_sim.resample('D').mean()

domef1500_sim = pd.read_csv('/mnt/SAT/Workspaces/GrF/02_Documents/monalisa/NEPAS2000/thetaliq0001_domef1500.txt',
                            header=0,
                            index_col=0,
                            usecols=[0,7],
                            names=['Date', 'Domef1500.sim'],
                            squeeze=True,
                            parse_dates=True,
                            date_parser=dateparse,
                            na_values=-9999)

#domef1500_sim = domef1500_sim.asfreq('H')
domef1500_sim = domef1500_sim.resample('D').mean()

# ---------------------

df = pd.concat([nepas2000_insitu, nepas2000_sim, nemef1500_insitu, domef1500_sim], axis=1, join='outer')
#df = df.resample('D').mean()

# ---------------------
# Read S1 simulations

nepas2000_s1 = pd.read_csv('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ASCAT/gee_117/Monalisa/s1ts_Nepas2000.csv',
                           index_col=0,
                           names=['Date', 'Nepas2000.S1'],
                           squeeze=True,
                           parse_dates=True,
                           date_parser=dateparse2)

domef1500_s1 = pd.read_csv('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ASCAT/gee_117/Monalisa/s1ts_Domef1500.csv',
                           index_col=0,
                           names=['Date', 'Domef1500.S1'],
                           squeeze=True,
                           parse_dates=True,
                           date_parser=dateparse2)

Nemef1500_s1 = pd.read_csv('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ASCAT/gee_117/Monalisa/s1ts_Nemef1500.csv',
                           index_col=0,
                           names=['Date', 'Nemef1500.S1'],
                           squeeze=True,
                           parse_dates=True,
                           date_parser=dateparse2)

# ------------------

df = pd.concat([df, nepas2000_s1.resample('D').mean()], axis=1, join='outer')
df = pd.concat([df, Nemef1500_s1.resample('D').mean()], axis=1, join='outer')
df_2015 = df['20150401':'20161031']
#nepas2000_s1 = nepas2000_s1['20150401':'20161031'] + 0.2
#domef1500_s1 = domef1500_s1['20150401':'20161031'] + 0.2

df_2015.loc[:,'Nepas2000.S1'] = df_2015.loc[:,'Nepas2000.S1']
df_2015.loc[:,'Nemef1500.S1'] = df_2015.loc[:,'Nemef1500.S1']

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
ax.set_ylabel('Bodenfeuchte [m3m-3]\n')

ax1.plot(df_2015.index, df_2015['Nepas2000.insitu'], 'b-', label='in-situ')
ax1.plot(df_2015.index, df_2015['Nepas2000.sim'], 'r-', label='GEOtop')
ax1.plot(nepas2000_s1.index, nepas2000_s1/100, 'g-', label='Sentinel-1', mew=2.0)
ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
#ax1.axis(['2015-04-01', '2015-10-31', 0.1, 0.7])
#ax1.xaxis.set_ticks(['2015-05-01', '2015-06-01', '2015-07-01', '2015-08-01','2015-09-01', '2015-10-01'])
#ax1.set_xticklabels(['May 2015', 'Jun 2015', 'Jul 2015', 'Aug 2015', 'Sep 2015', 'Oct 2015'])
#ax1.text('2015-04-10', 0.6, 'NEPAS2000', bbox=dict(edgecolor='black', facecolor='white'), verticalalignment='center')
#plt.subplot(2,1,2)
ax2.plot(df_2015.index, df_2015['Nemef1500.insitu'], 'b-', label='in-situ')
ax2.plot(df_2015.index, df_2015['Domef1500.sim'], 'r-', label='GEOtop')
ax2.plot(Nemef1500_s1.index, Nemef1500_s1/100, 'g-', label='Sentinel-1', mew=2.0)
#ax2.axis(['2015-04-01', '2015-10-31', 0.1, 0.7])
#ax2.xaxis.set_ticks(['2015-05-01', '2015-06-01', '2015-07-01', '2015-08-01','2015-09-01', '2015-10-01'])
#ax2.set_xticklabels(['May 2015', 'Jun 2015', 'Jul 2015', 'Aug 2015', 'Sep 2015', 'Oct 2015'])
#ax2.text('2015-04-10', 0.6, 'DOMEF1500', bbox=dict(edgecolor='black', facecolor='white'), verticalalignment='center')

#plt.xlabel('Datum')
#plt.ylabel('Soil Moisture [m3m-3]')
plt.show()

#nepas2000_2.plot()
plt.savefig('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ASCAT/gee_117/Monalisa/nemef1500_geeall_pwise.png', dpi=600)
plt.close()