import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import time
import re
import itertools
import pandas as pd
#from matplotlib.backends.backend_pdf import PdfPages


attr_list = []
time_list = []
prune_time_list = []
no_prune_time_list = []
source_list = []
test_id_list = []
size_legend = ['800k', '400k', '200k', '100k', ' 50k', ' 25k']
for test_id in range(1, 7):
	sct_df_prune = pd.read_csv('../expl_perf_exp/time_record/crime_pruning_top3_exp{}.csv'.format(str(test_id)),		
		names=['#attr', 'time'])
	sct_df_no_prune = pd.read_csv('../expl_perf_exp/time_record/crime_no_pruning_top3_exp{}.csv'.format(str(test_id)),		
		names=['#attr', 'time'])

	
	for idx, row in sct_df_prune.iterrows():
		if row['time'] > 0:
			attr_list.append(str(int(row['#attr'])))
			time_list.append(row['time'])
			source_list.append('Pruning')
			prune_time_list.append(row['time'])
			test_id_list.append(size_legend[test_id-1])
	for idx, row in sct_df_no_prune.iterrows():
		if row['time'] > 0:
			no_prune_time_list.append(row['time'])


ix3 = pd.MultiIndex.from_arrays([
	test_id_list], names=['#patterns'])



sct_df = pd.DataFrame({'Prune':prune_time_list, 'No Prune':no_prune_time_list}, index=ix3)
# sct_df = pd.DataFrame({'Prune':prune_time_list}, index=ix3)

gp = sct_df.groupby(level=('#patterns'))
means = gp.mean()
errors = gp.std()
min_data = gp.min()
max_data = gp.max()
err = [[[],[]],[[],[]]]
mean_list = [[],[]]
for idx, val in means.iterrows():
    mean_list[0].append(val['Prune'])
    mean_list[1].append(val['No Prune'])
cnt = 0
for idx, val in min_data.iterrows():  # Iterate over bar groups (represented as columns)
    # err[int(idx)-2].append([])
    err[1][0].append(mean_list[0][cnt] - val['Prune'])
    err[0][0].append(mean_list[1][cnt] - val['No Prune'])
    cnt += 1
cnt = 0
for idx, val in max_data.iterrows():  # Iterate over bar groups (represented as columns)
    err[1][1].append(val['Prune'] - mean_list[0][cnt])
    err[0][1].append(val['No Prune'] - mean_list[1][cnt])
    cnt += 1
# err.append([min_data[col].values, errHi[col].values])
# err = np.abs(err)  # Absolute error values (you had some negatives)
# print(means)
# print(err)
# print(np.shape(err))    

fig, ax = plt.subplots()
# means.plot.bar(yerr=errors, ax=ax)
means.plot.bar(yerr=err, ax=ax, capsize=4.5)
# ax.set_title('a')
fig.subplots_adjust(bottom=0.15, left=0.15)
#fig.suptitle('#Attributes vs. Time', fontsize=14, fontweight='bold')
ax.set_xlabel('#RegModels', fontsize=22)
ax.set_ylabel('time (sec)', fontsize=22)
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22) 
        
for tick in ax.yaxis.get_major_ticks():
       tick.label.set_fontsize(22) 

# grid
ax.yaxis.grid(which='major',linewidth=1.0,linestyle=':')
ax.set_axisbelow(True)


ax.legend(prop={'size': 24},labels=['ExplGen-Naive', 'ExplGen-Opt'],loc=2,
	borderpad=0,labelspacing=0,handlelength=1,handletextpad=0.2,
              columnspacing=0.5)
    
plt.savefig('expl_crime_numpat.pdf')

