import matplotlib
matplotlib.use('PDF')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt
import time
import re
import itertools
import pandas as pd
import pandas as pd
import numpy as np


attr_list = []
time_list = []
prune_time_list = []
no_prune_time_list = []
source_list = []
test_id_list = []
size_legend = ['800k', '400k', '200k', '100k', ' 50k', ' 25k']
for test_id in range(1,2):
    sct_df_prune = pd.read_csv('../expl_perf_exp/time_record/crime_pruning_top3_exp_{}.csv'.format(str(test_id)),        
        names=['#attr', 'time'])
    sct_df_no_prune = pd.read_csv('../expl_perf_exp/time_record/crime_no_pruning_top3_exp_{}.csv'.format(str(test_id)),      
        names=['#attr', 'time'])
    
    for idx, row in sct_df_prune.iterrows():
        if row['time'] > 0:
            attr_list.append(str(int(row['#attr'])))
            time_list.append(row['time'])
            source_list.append('Pruning')
            prune_time_list.append(row['time'])
            # test_id_list.append(size_legend[test_id-1])
    for idx, row in sct_df_no_prune.iterrows():
        if row['time'] > 0:
    #     # attr_list.append(row['#attr'])
    #     # time_list.append(row['time'])
    #     # source_list.append('No Pruning')
            no_prune_time_list.append(row['time'])


#ix3 = pd.MultiIndex.from_arrays([
#    attr_list], names=['#attr'])


sct_df = pd.DataFrame({'#attr':attr_list, 'Prune':prune_time_list, 'No Prune':no_prune_time_list})
# sct_df = pd.DataFrame({'Prune':prune_time_list}, index=ix3)

gp = sct_df.groupby('#attr',as_index=False)
tot = gp.sum()
tempcol=[]
for string in list(tot['#attr']):
        tempcol.append(int(string))
tot['temp']=tempcol
tot=tot.sort_values('temp').reset_index(drop=True)
print(tot)
##means = gp.mean()
##errors = gp.std()
##min_data = gp.min()
##max_data = gp.max()
##err = [[[],[]],[[],[]]]
##mean_list = [[],[]]
##for idx, val in means.iterrows():
##    mean_list[0].append(val['Prune'])
##    mean_list[1].append(val['No Prune'])
##for idx, val in min_data.iterrows():  # Iterate over bar groups (represented as columns)
##    # err[int(idx)-2].append([])
##    err[1][0].append(mean_list[0][int(idx) - 2] - val['Prune'])
##    err[0][0].append(mean_list[1][int(idx) - 2] - val['No Prune'])
##for idx, val in max_data.iterrows():  # Iterate over bar groups (represented as columns)
##    err[1][1].append(val['Prune'] - mean_list[0][int(idx) - 2])
##    err[0][1].append(val['No Prune'] - mean_list[1][int(idx) - 2])
# err.append([min_data[col].values, errHi[col].values])
# err = np.abs(err)  # Absolute error values (you had some negatives)
# print(means)
# print(err)
# print(np.shape(err))    

##fig, ax = plt.subplots()
### means.plot.bar(yerr=errors, ax=ax)
##means.plot.bar(yerr=err, ax=ax, capsize=4.5)
### ax.set_title('a')
##fig.subplots_adjust(bottom=0.15, left=0.15)
#fig.suptitle('#Attributes vs. Time', fontsize=14, fontweight='bold')
# plot settings
mymarker=['s','o','v','x']
msize=80.0
mymarkerlw=1.0
mylinewd = 1.0


ax=tot.plot(x='#attr',y='No Prune',color='blue',marker=mymarker[0],lw=mymarkerlw)
ax=tot.plot(x='#attr',y='Prune',color='red',ax=ax,marker=mymarker[1],lw=mymarkerlw)
ax.set_xlabel('\#Attributes in User Question', fontsize=30)
ax.set_ylabel('time (sec)', fontsize=30)
ax.set_xticks(range(7))
ax.set_xticklabels(list(tot['#attr']))
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(30) 
        
for tick in ax.yaxis.get_major_ticks():
       tick.label.set_fontsize(30) 

# grid
ax.yaxis.grid(which='major',linewidth=3.0,linestyle=':')
ax.set_axisbelow(True)


ax.legend(prop={'size': 30},labels=['ExplGen-Naive', 'ExplGen-Opt'],loc=2,borderpad=0,labelspacing=0,handlelength=1,handletextpad=0.2,
              columnspacing=0.5)

plt.tight_layout()
plt.savefig('expl_crime_numatt.pdf')
