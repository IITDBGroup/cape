import pandas as pd
import statsmodels.formula.api as sm
import matplotlib
matplotlib.use('PDF')
import matplotlib.patches as mpatches
import pylab as pl
from numpy import arange,power

def main():
    df_bar=pd.read_csv('crime_bar.csv',
                       index_col=['num_attribute'])
    pl.ioff()

    
    cube_bar=df_bar.query('algo==\'cube\'')[['query','regression','mining']]
    _1q1g_bar=df_bar.query('algo==\'share_grp\'')[['query','regression','mining']]
    arp_bar=df_bar.query('algo==\'optimized\'')[['query','regression','mining']]

    for index in cube_bar.index:
        norm=df_bar.query('algo==\'cube\'')['total'][index]
        for col in ['query','regression','mining']:
            cube_bar[col][index]/=norm
            _1q1g_bar[col][index]/=norm
            arp_bar[col][index]/=norm

    query_col = 'blue'
    regression_col = 'orange'
    mining_col = 'black'
    
    mycolors=[query_col, regression_col, mining_col]

    ax = pl.subplot()
    ax=cube_bar.plot.bar(stacked=True,position=-0.5,width=0.2,color=mycolors)
    ax=_1q1g_bar.plot.bar(ax=ax,stacked=True,position=0.5,width=0.2,color=mycolors)
    ax=arp_bar.plot.bar(ax=ax,stacked=True,position=1.5,width=0.2,color=mycolors)
    ax.set_xlim(-0.55, 7.55)
    ax.set_ylim(0,1.05)

    ratio = 0.5
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    
    rects = ax.patches

    query_patch = mpatches.Patch(color=query_col, label='query')
    regression_patch = mpatches.Patch(color=regression_col, label='regression')
    mining_patch = mpatches.Patch(color=mining_col, label='mining')
    
    legend = pl.legend(handles=[query_patch,regression_patch,mining_patch], bbox_to_anchor=(-0.15, 1., 1.2, .102),ncol=3,mode="expand", fontsize=20, handletextpad=0.2, framealpha=1)    
    legend.get_frame().set_edgecolor('black')
    # For each bar: Place a label
##    done_x=set()
##    for rect in rects:
##        if rect.get_x() in done_x:
##            continue
##        # Get X and Y placement of label from rect.
##        x_value = rect.get_x() + rect.get_width() / 2
##        y_value = 1
##
##        done_x.add(rect.get_x())
##
##        # Number of points between bar and label. Change to your liking.
##        space = 5
##        # Vertical alignment for positive values
##        va = 'bottom'
##        
##        x="{0:0.1f}".format(rect.get_x()%1)
##        if x[-1]=='7':
##            label='ARP-mine'
##        elif x[-1]=='9':
##            label='share-grp'
##        else:
##            label='cube'
##
##        # Create annotation
##        matplotlib.pyplot.annotate(
##            label,                      # Use `label` as label
##            (x_value, y_value),         # Place label at end of the bar
##            rotation=90,
##            xytext=(0, space),          # Vertically shift label by `space`
##            textcoords="offset points", # Interpret `xytext` as offset in points
##            ha='center',                # Horizontally center label
##            va=va)                      # Vertically align label differently for
##                                        # positive and negative values.

    # axis
    ax.set_ylabel('runtime (cube=1)', fontsize=30)
    ax.set_xlabel('#attributes', fontsize=30)
    pl.xticks(rotation=0)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(30) 

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(30) 

    pl.subplots_adjust(bottom=0.15)

    pl.show()
    pl.savefig("crime_bar.pdf", bbox_inches='tight')
if __name__=="__main__":
    main()
