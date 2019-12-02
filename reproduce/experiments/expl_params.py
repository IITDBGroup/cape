import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib import cm
import psycopg2
import time
import re
import itertools
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set(style="ticks")
plt.style.use(u'ggplot')



def plot_heatmap(gs, hm, ax):
    
    # cmap = cm.get_cmap('Spectral')  # Colour map (there are many others)


    x = [i*0.1 for i in range(1, len(hm)+1)]
    y = [i*0.1 for i in range(1, len(hm[0])+1)]
    # setup the 2D grid with Numpy
    # x, y = np.meshgrid(x, y)
    # fig = plt.figure()

    # convert intensity (list of lists) to a numpy array for plotting
    intensity = np.array(hm)

    # now just plug the data into pcolormesh, it's that easy!
    # plt.pcolormesh(x, y, intensity)
    # plt.xticks(x, list(map(lambda t: str(t*0.05)[:4], x)))
    # plt.yticks(y, list(map(lambda t: str(t*0.05)[:4], y)))
    # plt.colorbar()  # need a colorbar to show the intensity scale

    # plt.show()  # boom
    # plt.show()
    # plt.savefig(df_key + '.png')
    # l1 = plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, fontsize='x-small')
    sns_hm = sns.heatmap(intensity,cmap="YlGnBu", ax=ax, cbar=gs == 1,
        vmin=0, vmax=100,
        yticklabels=list(map(lambda z: str(z)[:3], y)), 
        xticklabels=list(map(lambda z: str(z)[:3], x)), cbar_ax=None if i else cbar_ax)
    # sns.heatmap(df, ax=ax,
    #             cbar=i == 0,
    #             vmin=0, vmax=1,
    #             cbar_ax=None if i else cbar_ax)

    sns_hm.invert_yaxis()
    # ax.legend(prop={'size': 18},loc=2)
    patches, labels = sns_hm.get_legend_handles_labels()
    # fig.subplots_adjust(bottom=0.2)
    # sns_hm.set_xlabel('Local Model Quality', fontsize=22)
    sns_hm.set_xlabel('θ', fontsize=22)
    # sns_hm.set_ylabel('Global Confidence', fontsize=22)
    if gs == 1:
        sns_hm.set_ylabel('λ', fontsize=22)
    ax.set_title("Δ={},δ=5".format(str(gs)))
    ax.title.set_size(22)
    for tick in sns_hm.get_xticklabels():
        tick.set_rotation(0)
    for tick in sns_hm.xaxis.get_major_ticks():
        tick.label.set_fontsize(22) 
    for tick in sns_hm.yaxis.get_major_ticks():
        tick.label.set_fontsize(22) 
    sns_hm.figure.axes[-1].yaxis.label.set_size(22)
    # ax.tick_params(labelsize=20)
    if gs == 1:
        cbar = sns_hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=22)



inf = open('./expl_params_top_10_delta_5.txt', 'r')
param_line = inf.readline()

gt_scores = [[[0 for i in range(7)] for j in range(7)],
[[0 for i in range(7)] for j in range(7)],
[[0 for i in range(7)] for j in range(7)],
[[0 for i in range(7)] for j in range(7)]]
gs_list = [1,5,15,25]
gs_mapping = {'1':0, '5':1, '15':2, '25':3}
lam_mapping = {'0.1':0, '0.2':1, '0.3':2, '0.4':3, '0.5':4, '0.6':5, '0.7':6}
the_mapping = {'0.1':0, '0.2':1, '0.3':2, '0.4':3, '0.5':4, '0.6':5, '0.7':6}
while param_line:
    score_line = inf.readline()
    if len(score_line) < 2:
        break
    params = param_line.strip().split(',')
    scores = eval(score_line)
    print(scores)
    # if params[0] == '0.7':
    #     continue
    gt_scores[gs_mapping[params[2]]][lam_mapping[params[0]]][the_mapping[params[1]]] = sum(scores)
    time_line = inf.readline()
    param_line = inf.readline()

# for i in range(4):
#     plot_heatmap(i, gt_scores[i])

# fig, axn = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
fig, axn = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20, 5))
cbar_ax = fig.add_axes([.91, .3, .03, .4])
for i, ax in enumerate(axn.flat):
    # plot_heatmap(gs_list[i*2], gt_scores[i*2], ax)
    print(gt_scores[i])
    plot_heatmap(gs_list[i], gt_scores[i], ax)
fig.tight_layout(rect=[0, 0, .9, 1])
# cbar = cbar_ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=22)
plt.savefig('params_gs.pdf', bbox_inches='tight')

