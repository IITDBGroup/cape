import pandas as pd
import statsmodels.formula.api as sm
import matplotlib
matplotlib.use('PDF')
import pylab as pl
from numpy import arange,power

def main():
    df=pd.read_csv('crime_num_att.csv')

    # colors
    col_cube='blue'
    col_arp='green'
    col_1q1g='red'
    col_naive='black'

    # DFs
    cube=df.query('algo==\'cube\'')
    _1q1g=df.query('algo==\'1Q1G\'')
    arp=df.query('algo==\'ARP-mine\'')
    naive=df.query('algo==\'naive\' and total<7000')

    # non interactive
    pl.ioff()
   
    # plot settings
    mymarker=['s','o','v','x']
    msize=80.0
    mymarkerlw=1.0
    mylinewd = 1.0
    
    # scatter plots
    f = pl.plot()
    
    ax=naive.plot(x='num_attribute',y='total',kind='scatter',label='naive',c=col_naive,s=msize,marker=mymarker[0],lw=mymarkerlw)
    ax=cube.plot(ax=ax,x='num_attribute',y='total',kind='scatter',label='cube',c=col_cube,s=msize,marker=mymarker[1],lw=mymarkerlw)
    ax=_1q1g.plot(ax=ax,x='num_attribute',y='total',kind='scatter',label='share-grp',c=col_1q1g,s=msize,marker=mymarker[2],lw=mymarkerlw)
    ax=arp.plot(ax=ax,x='num_attribute',y='total',kind='scatter',label='ARP-mine',c=col_arp,s=msize,marker=mymarker[3],lw=mymarkerlw)    

    # regression lines

    
    x=arange(4,11.01,0.01)
    lr=sm.ols('total~num_attribute+power(3,num_attribute)',data=cube).fit()
    p=lr.params
    print(lr.rsquared_adj)
    y=p['Intercept']+p['power(3, num_attribute)']*3**x+x*p['num_attribute']
    ax.plot(x,y,c=col_cube,lw=mylinewd)

    lr=sm.ols('total~num_attribute+power(num_attribute,2)+power(num_attribute,3)+power(num_attribute,4)',data=_1q1g).fit()
    p=lr.params
    print(lr.rsquared_adj)
    y=p['Intercept']+p['num_attribute']*x+p['power(num_attribute, 2)']*x**2+p['power(num_attribute, 3)']*x**3+p['power(num_attribute, 4)']*x**4
    ax.plot(x,y,c=col_1q1g,lw=mylinewd)

    lr=sm.ols('total~num_attribute+power(num_attribute,2)+power(num_attribute,3)+power(num_attribute,4)',data=arp).fit()
    p=lr.params
    print(lr.rsquared_adj)
    y=p['Intercept']+p['num_attribute']*x+p['power(num_attribute, 2)']*x**2+p['power(num_attribute, 3)']*x**3+p['power(num_attribute, 4)']*x**4
    ax.plot(x,y,c=col_arp,lw=mylinewd)

    # axis labels and tics
    ax.set_ylabel('time (sec)', fontsize=30)
    ax.set_xlabel('#attributes', fontsize=30)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(30) 

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(30) 

    # grid
    ax.yaxis.grid(which='major',linewidth=3.0,linestyle=':')

    # legend
    ax.legend(prop={'size': 30},borderpad=0,labelspacing=0,handlelength=0,
              columnspacing=0.5,loc=2,ncol=2)

    # plot
    pl.show()

    pl.savefig("crime_num_att.pdf", bbox_inches='tight')


if __name__=="__main__":
    main()
