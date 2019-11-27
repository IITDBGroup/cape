import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import time
import re
import itertools
import pandas as pd
# from matplotlib.backends.backend_pdf import PdfPages

attr_list = []
time_list = []
prune_time_list_3 = []
prune_time_list_10 = []
no_prune_time_list = []
source_list = []
test_id_list = []
size_legend = ['800k', '400k', '200k', '100k', ' 40k', ' 20k', ' 10k', '  4k']

		
test_id_used = [1,2,3,4,5,7]

for test_id in test_id_used:
# sct_df = pandas.DataFrame({'#attr': att_size_list, 'time':sct_list})
	sct_df_prune_3 = pd.read_csv('../expl_perf_expcd/time_record/dblp_pruning_top3_exp_{}.csv'.format(str(test_id)),		
		names=['#attr', 'time'])
	sct_df_prune_10 = pd.read_csv('../expl_perf_expcd/time_record/dblp_pruning_top10_exp_{}.csv'.format(str(test_id)),		
		names=['#attr', 'time'])
	sct_df_no_prune = pd.read_csv('../expl_perf_expcd/time_record/dblp_no_pruning_top3_exp_{}.csv'.format(str(test_id)),		
		names=['#attr', 'time'])

	
	for idx, row in sct_df_prune_3.iterrows():
		if row['time'] > 0:
			prune_time_list_3.append(row['time'])
			test_id_list.append(size_legend[test_id-1])
	for idx, row in sct_df_prune_10.iterrows():
		if row['time'] > 0:
			prune_time_list_10.append(row['time'])
	for idx, row in sct_df_no_prune.iterrows():
		if row['time'] > 0:
			no_prune_time_list.append(row['time'])


ix3 = pd.MultiIndex.from_arrays([
	test_id_list], names=['#patterns'])


sct_df = pd.DataFrame({'No Prune':no_prune_time_list, 'Prune(Top-3)':prune_time_list_3, 'Prune(Top-10)':prune_time_list_10}, index=ix3)

gp = sct_df.groupby(level=('#patterns'))
means = gp.mean()
errors = gp.std()
min_data = gp.min()
max_data = gp.max()
err = [[[],[]],[[],[]],[[],[]]]
mean_list = [[],[],[]]
for idx, val in means.iterrows():
    mean_list[0].append(val['Prune(Top-10)'])
    mean_list[1].append(val['Prune(Top-3)'])
    mean_list[2].append(val['No Prune'])
cnt = 0
for idx, val in min_data.iterrows():  # Iterate over bar groups (represented as columns)
    # print(idx, val['Prune'])
    # err[int(idx)-2].append([])
    err[1][0].append(mean_list[0][cnt] - val['Prune(Top-10)'])
    err[2][0].append(mean_list[1][cnt] - val['Prune(Top-3)'])
    err[0][0].append(mean_list[2][cnt] - val['No Prune'])
    cnt += 1
cnt = 0
for idx, val in max_data.iterrows():  # Iterate over bar groups (represented as columns)
    # print(idx, val['Prune'])
    err[1][1].append(val['Prune(Top-10)'] - mean_list[0][cnt])
    err[2][1].append(val['Prune(Top-3)'] - mean_list[1][cnt])
    err[0][1].append(val['No Prune'] - mean_list[2][cnt])
    cnt += 1




# PLOTTING

fig, ax = plt.subplots()
# means.plot.bar(yerr=errors, ax=ax, capsize=4.5)
means.plot.bar(yerr=err, ax=ax, capsize=4.5)
# ax.set_title('a')
fig.subplots_adjust(bottom=0.15)
#fig.suptitle('#Patterns vs. Time', fontsize=14, fontweight='bold')
ax.set_xlabel('#RegModels', fontsize=22)
ax.set_ylabel('time (sec)', fontsize=22)

# axis labels and tics
for tick in ax.xaxis.get_major_ticks():
	tick.label.set_fontsize(22) 
        
for tick in ax.yaxis.get_major_ticks():
	tick.label.set_fontsize(22) 

# grid
ax.yaxis.grid(which='major',linewidth=1.0,linestyle=':')
ax.set_axisbelow(True)

# legend
ax.legend(prop={'size': 24},loc=2,labels=['ExplGen-Naive','ExplGen-Opt(Top-10)','ExplGen-Opt(Top-3)'], 
	borderpad=0,labelspacing=0,handlelength=1,handletextpad=0.2,
              columnspacing=0.5)

plt.xticks(rotation=0)    

plt.savefig('expl_DBLP_numpat.pdf')
