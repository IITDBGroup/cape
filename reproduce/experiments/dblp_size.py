import pandas as pd
import statsmodels.formula.api as sm
import matplotlib
matplotlib.use('PDF')
import pylab as pl
from numpy import arange,power

def main():
    df=pd.read_csv('dblp_size.csv')

    # colors
    col_cube='blue'
    col_arp='green'
    col_1q1g='red'
    col_naive='black'

    
    cube=df.query('algo==\'cube\'')
    _1q1g=df.query('algo==\'1Q1G\'')
    arp=df.query('algo==\'ARP-mine\'')

    # non interactive
    pl.ioff()

    # plot settings
    mymarker=['o','v','x']
    msize=80.0
    mymarkerlw=1.0
    mylinewd = 1.0

    f = pl.plot()
    
    # lines plots
    ax=cube.plot(x='size',y='total',label='cube',c=col_cube,linestyle='-',marker=mymarker[0],lw=mymarkerlw)
    ax=_1q1g.plot(ax=ax,x='size',y='total',label='share-grp',c=col_1q1g,linestyle='-',marker=mymarker[1],lw=mymarkerlw)
    ax=arp.plot(ax=ax,x='size',y='total',label='ARP-mine',c=col_arp,linestyle='-',marker=mymarker[2],lw=mymarkerlw)

    # grid
    ax.yaxis.grid(which='major',linewidth=3.0,linestyle=':')

    # make axis log-scale
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    
    # axis labels and tics
    ax.set_ylabel('time (sec) - log', fontsize=30)
    ax.set_xlabel('#rows - logscale', fontsize=30)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(30) 

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(30) 

    # x-axis and y-axis range
    ax.set_xlim(9000,1100000)
    ax.set_ylim(10,5000)    
        
    # legend
    ax.legend(prop={'size': 30}, loc=2,
              borderpad=0,labelspacing=0,handlelength=1,handletextpad=0.2,
              columnspacing=0.5,borderaxespad=0)
    
    pl.show()
    pl.savefig("dblp_size.pdf", bbox_inches='tight')
if __name__=="__main__":
    main()
