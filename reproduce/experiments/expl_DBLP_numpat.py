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
	sct_df_prune_3 = pd.read_csv('./expl_time_record/dblp_pruning_top3_exp_{}.csv'.format(str(test_id)),		
		names=['#attr', 'time'])
	sct_df_prune_10 = pd.read_csv('./expl_time_record/dblp_pruning_top10_exp_{}.csv'.format(str(test_id)),		
		names=['#attr', 'time'])
	sct_df_no_prune = pd.read_csv('./expl_time_record/dblp_no_pruning_top3_exp_{}.csv'.format(str(test_id)),		
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

			
#ix3 = pd.MultiIndex.from_arrays([
#	test_id_list], names=['#patterns'])

sct_df = pd.DataFrame({'#patterns':test_id_list, 'No Prune':no_prune_time_list, 'Prune(Top-3)':prune_time_list_3, 'Prune(Top-10)':prune_time_list_10})
print(sct_df)
gp = sct_df.groupby('#patterns',as_index=False)
tot = gp.sum()
tempcol=[]
for string in list(tot['#patterns']):
	tempcol.append(int(string[:-1]))
tot['temp']=tempcol
tot=tot.sort_values('temp').reset_index(drop=True)

# plot settings
mymarker=['s','o','v','x']
msize=80.0
mymarkerlw=1.0
mylinewd = 1.0

ax=tot.plot(x='#patterns',y='No Prune',color='blue',marker=mymarker[0],lw=mymarkerlw)
ax=tot.plot(x='#patterns',y='Prune(Top-3)',color='red',ax=ax,marker=mymarker[1],lw=mymarkerlw)
ax=tot.plot(x='#patterns',y='Prune(Top-10)',color='green',ax=ax,marker=mymarker[2],lw=mymarkerlw)
ax.set_xlabel('\#RegModels', fontsize=30)
ax.set_ylabel('time (sec)', fontsize=30)
ax.set_xticks(range(6))
ax.set_xticklabels(list(tot['#patterns']))

# axis labels and tics
for tick in ax.xaxis.get_major_ticks():
	tick.label.set_fontsize(30) 
        
for tick in ax.yaxis.get_major_ticks():
	tick.label.set_fontsize(30) 

# grid
ax.yaxis.grid(which='major',linewidth=3.0,linestyle=':')
ax.set_axisbelow(True)

# legend
ax.legend(prop={'size': 30},loc=2,labels=['ExplGen-Naive','ExplGen-Opt(Top-3)','ExplGen-Opt(Top-10)'], 
	borderpad=0,labelspacing=0,handlelength=1,handletextpad=0.2,
              columnspacing=0.5)

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('expl_DBLP_numpat.pdf')
