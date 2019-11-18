import pandas as pd
import statsmodels.formula.api as sm
import matplotlib
matplotlib.use('PDF')
import pylab as pl
from numpy import arange,power

def main():
    df=pd.read_csv('crime_fd_on_off.csv',nrows=6)
    off=list(df.query('fd_on==\'f\'')['total'])
    on=list(df.query('fd_on==\'t\'')['total'])
    index=list(df['size'].unique())
    
    pl.ioff()
    

    compare=pd.DataFrame({'fd_check_on':on,'fd_check_off':off},index=index)
    ax=compare.plot.bar()

    legend = ax.legend(bbox_to_anchor=(0.005, 1.04),prop={'size': 30},labels=['detect FD','not detect FD'],loc=2,
              borderpad=0.1,labelspacing=0,handlelength=1,handletextpad=0.2,
              columnspacing=0.5,framealpha=1)
    legend.get_frame().set_edgecolor('black')
    
    # axis labels and tics
    ax.set_ylabel('time (sec)', fontsize=30)
    ax.set_xlabel('#rows', fontsize=30)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(30) 

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(30) 

    pl.xticks(rotation=0)    

    # grid
    ax.yaxis.grid(which='major',linewidth=3.0,linestyle=':')
    ax.set_axisbelow(True)

    
    pl.show()
    pl.savefig("crime_fd_on_off.pdf", bbox_inches='tight')
    
if __name__=="__main__":
    main()
