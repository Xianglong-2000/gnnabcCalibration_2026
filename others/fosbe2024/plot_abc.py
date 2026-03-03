import os
import pyabc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

from pyabc.storage import History
from pyabc.transition import MultivariateNormalTransition

subdir = '2d_kde'
plt.rcParams['font.size'] = 24
colors = ['b','grey','r','g','chocolate','gold','m']

### COPIED FIRST TWO FUNCS TO ENABLE CHANGING COLORBAR SCALE 

def kde_2d(
    df,
    w,
    x,
    y,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    numx=50,
    numy=50,
    kde=None,
):
    """
    Calculates a 2 dimensional histogram from a Dataframe and weights.

    For example, a results distribution might be obtained from the history
    class and plotted as follows::

        df, w = history.get_distribution(0)
        X, Y, PDF = hist_2d(df, w, "x", "y")
        plt.pcolormesh(X, Y, PDF)


    Parameters
    ----------
    df: Pandas Dataframe
        The rows are the observations, the columns the variables
    w: The corresponding weights
    x: str
        The variable for the x-axis
    y: str
        The variable for the y-axis
    xmin: float, optional
        The lower limit in x for the histogram.
        If left empty, it is set to the minimum of the ovbservations of the
        variable to be plotted as x.
    xmax: float, optional
        The upper limit in x for the histogram.
        If left empty, it is set to the maximum of the ovbservations of the
        variable to be plotted as x.
    ymin: float, optional
        The lower limit in y for the histogram.
        If left empty, it is set to the minimum of the ovbservations of the
        variable to be plotted as y
    ymax: float, optional
        The upper limit in y for the histogram.
        If left empty, it is set to the maximum of the ovbservations of the
        variable to be plotted as y.
    numx: int, optional
        The number of bins in x direction.
        Defaults to 50.
    numy int, optional
        The number of bins in y direction.
        Defaults to 50.
    kde: pyabc.Transition, optional
        The kernel density estimator to use for creating a smooth density
        from the sample. If None, a multivariate normal kde with
        cross-validated scaling is used.

    Returns
    -------
    X, Y, PDF: (np.ndarray, np.ndarray, np.ndarray)
        The X, the Y and the densities at these points.
        These can be passed for plotting, for example as
        plt.pcolormesh(X, Y, PDF)

    """
    if kde is None:
        kde = MultivariateNormalTransition(scaling=1)
    kde.fit(df[[x, y]], w)

    if xmin is None:
        xmin = df[x].min()
    if xmax is None:
        xmax = df[x].max()
    if ymin is None:
        ymin = df[y].min()
    if ymax is None:
        ymax = df[y].max()
    X, Y = np.meshgrid(
        np.linspace(xmin, xmax, num=numx), np.linspace(ymin, ymax, num=numy)
    )
    test = pd.DataFrame({x: X.flatten(), y: Y.flatten()})
    pdf = kde.pdf(test)
    PDF = pdf.reshape(X.shape)
    return X, Y, PDF

cbar_lim = [0, 0.25] #-AY
def my_plot_kde_2d(
    df,
    w,
    x,
    y,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    numx=50,
    numy=50,
    ax=None,
    size=None,
    colorbar=True,
    title: str = None,
    refval=None,
    refval_color='C1',
    kde=None,
    xname: str = None,
    yname: str = None,
    **kwargs,
):
    """
    Plot a 2d kernel density estimate of parameter samples.

    Parameters
    ----------
    df: Pandas Dataframe
        The rows are the observations, the columns the variables
    w: The corresponding weights.

    For the other parameters, see `plot_kde_2d_highlevel`.

    Returns
    -------
    ax: matplotlib axis
        Axis of the plot.

    """
    X, Y, PDF = kde_2d(
        df,
        w,
        x,
        y,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        numx=numx,
        numy=numy,
        kde=kde,
    )
    if xname is None:
        xname = x
    if yname is None:
        yname = y
    if ax is None:
        _, ax = plt.subplots()
    mesh = ax.pcolormesh(X, Y, PDF, shading='auto', vmin=cbar_lim[0], vmax=cbar_lim[1], **kwargs) #-AY
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    if title is not None:
        ax.set_title(title)
    if colorbar:
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', ticks=np.linspace(cbar_lim[0], cbar_lim[1], 4), format='%.4f')
        #cbar.formatter = ScalarFormatter(useMathText=True)
        #cbar.formatter.set_powerlimits((-10,-9))
        #cbar.update_ticks()
        # cbar.set_label("PDF")
    if refval is not None:
        ax.scatter([refval[x]], [refval[y]], color=refval_color)

    # set size
    if size is not None:
        ax.get_figure().set_size_inches(size)

    return ax
    
### NOW FOR THE REAL PLOTTING CODE

def ABCsimulation(params):
    return None

def plot_1d(history, limits, exact, key, **kwargs):
    """
    Plot posterior distributions of an ABC run
    @param history  ABC database
    @param limits   dict containing upper/lower bounds of priors
    @param exact    Exact value of the observed data
    @param key      Key in exact dict
    
    The figure is saved in the same directory as this script
    """

    xmin = limits[key][0]
    xmax = limits[key][1]
    
    fig = plt.figure(figsize=(9,11))
   
    #Plot prior distributions
    y = 1/(xmax-xmin) # uniform prior is equal to 1/max(param) for each parameter 
    plt.plot(np.linspace(xmin,xmax,100), y*np.ones(100,), color='k', lw=3, label='Prior')
    
    #Plot exact value
    #plt.plot(exact[key], 0, ls=':', color='black', label='Observed Value')
    
    ax = plt.gca()
    c = 0
    for t in range(0,history.n_populations,1):
        df, w = history.get_distribution(t=t)
        pyabc.visualization.plot_kde_1d(
            df, w, 
            x=key, ax=ax,
            xmin=xmin,
            xmax=xmax,              
            numx=10000,
            label="Stage={}".format(t+1)
            #refval=exact, refval_color="black", lw=3, color=colors[c]
            )    
        plt.xlabel(key)
        ax.set_ylabel("") 
        c+=1
    ax.set_ylabel("")
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', ncol=1, prop={'size': 18})
    plt.savefig("%s_1D_KDE-t=%s.png" % (key,history.n_populations), bbox_inches='tight')
    
def plot_1d_logscale(history, limits, exact, key, **kwargs):
    """
    Plot posterior distributions of an ABC run
    @param history  ABC database
    @param limits   dict containing upper/lower bounds of priors
    @param exact    Exact value of the observed data
    @param key      Key in exact dict
    
    The figure is saved in the same directory as this script
    """

    xmin = limits[key][0]
    xmax = limits[key][1]
    
    fig = plt.figure(figsize=(7, 8))     
    c = 0
    for t in range(0,history.n_populations,1):
        ax = fig.add_subplot(1,1,1) # Adjust as necessary to fit panels
        df, w = history.get_distribution(t=t)
        df['gamma'] = np.exp(df['gamma'])
        df['reg_param'] = np.exp(df['reg_param'])       
        ax = pyabc.visualization.plot_kde_1d(
            df, w, 
            x=key, ax=ax,
            xmin=xmin,
            xmax=xmax,              
            numx=10000,
            label="Stage={}".format(t+1)
            #refval=exact, refval_color="black", lw=3, color=colors[c]
            )    
        #plt.xlabel(key)
        ax.set_ylabel("") 
        c+=1

        ax.set_ylabel("")
        ax.set_ylim(bottom=0)
        ax.set_xscale('log')
        ax.set_xlim([xmin, xmax])
    
    '''
    #Plot prior distributions
    ax = fig.add_subplot(1,1,1) # Adjust as necessary to fit panels
    y = 1/(xmax-xmin) # uniform prior is equal to 1/max(param) for each parameter 
    ax.plot(np.linspace(xmin,xmax,100), y*np.ones(100,), color='k', lw=3, label='Prior')
    '''
    plt.show()
        
    ax.legend(loc='upper right', ncol=1, prop={'size': 18})
    plt.savefig("%s_1D_KDE-t=%s.png" % (key,history.n_populations), bbox_inches='tight')
    
def plot_2d(h):
    fig = plt.figure(figsize=(20, 20))
    for t in range(history.max_t + 1):
        ax = fig.add_subplot(3, 2, t + 1) # Adjust as necessary to fit panels
        ax = pyabc.visualization.plot_kde_2d(
            *h.get_distribution(m=0, t=t),
            "gamma",
            "reg_param",
            xmin=0,
            xmax=1000,
            numx=200,
            ymin=0,
            ymax=0.1,
            numy=200,
            ax=ax,
        )
        ax.set_title(f"Calibration Round {t+1}")
    plt.savefig("2D_KDE-t=%s.png" % (t+1), bbox_inches='tight')
    
def plot_2d_log_scaled(h):
    if not os.path.exists(subdir):
        os.makedirs(subdir)
   
    rounds = list(range(history.max_t + 1))
    for t in rounds:
        t=4
        fig = plt.figure(figsize=(7, 8))
        ax = fig.add_subplot(1,1,1) # Adjust as necessary to fit panels
        df, w = h.get_distribution(m=0, t=t)
        print(f'gamma mean: {np.mean(df["gamma"])}') 
        print(f'gamma std: {np.std(df["gamma"])}') 
        print(f'reg_param mean: {np.mean(df["reg_param"])}') 
        print(f'reg_param std: {np.std(df["reg_param"])}') 
        xedges, yedges, pdf = kde_2d(df, w, 'gamma', 'reg_param', xmin=np.log(0.01), xmax=np.log(100), ymin=np.log(0.1), ymax=np.log(1000), numx=500, numy=500)
        
        xedges = np.exp(xedges)
        yedges = np.exp(yedges)

        #print(xedges[0,:])
        #print(yedges[:,0])
        mesh = ax.pcolormesh(xedges, yedges, pdf, shading='auto', cmap='viridis', vmin=cbar_lim[0], vmax=cbar_lim[1])
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', ticks=np.linspace(cbar_lim[0], cbar_lim[1], 4), format='%.4f')
        
        ax = my_plot_kde_2d(
            df,
            w,
            "gamma",
            "reg_param",
            xmin=0.01,
            xmax=100,
            numx=10000,
            ymin=0.1,
            ymax=1000,
            numy=10000,
            ax=ax,
            xname='γ',
            yname='α',
            colorbar=False
        )
        
        ax.set_title(f"Calibration Round {t+1}")
        plt.yscale('log')
        plt.xlim([0.01, 100])
        plt.xscale('log')
        plt.ylim([0.1,1000])
        plt.savefig(os.path.join(subdir, "2D_KDE-t=%s.svg" % (t+1)), bbox_inches='tight')

def plot_2d_prior(xmin,xmax,ymin,ymax):
    Z = np.ones((10,10))*1/((xmax - xmin) * (ymax - ymin))
    x = np.linspace(0,100,11) 
    y = np.linspace(0,1000,11)
    fig, ax = plt.subplots(figsize=(7,8))
    mesh = ax.pcolormesh(x,y,Z, shading='auto', vmin=cbar_lim[0], vmax=cbar_lim[1])
    ax.set_title(f"Prior")
    ax.set_xlabel('γ')
    ax.set_ylabel('α')
    plt.yscale('log')
    plt.xlim([0.01, 100])
    plt.xscale('log')
    plt.ylim([0.1,1000])
    #cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', ticks=np.linspace(cbar_lim[0], cbar_lim[1], 4), format='%.4f')
    plt.savefig(os.path.join(subdir, "Prior.png" ), bbox_inches='tight')
    
def plot_kde_matrix(history):
    fig = plt.figure(figsize=(20,20))
    ax = fig.gca()
    df, w = history.get_distribution(m=0, t=3)
    ax = pyabc.visualization.plot_kde_matrix(df, w, limits=limits)
    plt.savefig("KDE_matrix.png")
        
def plot_population_sizes(history):
    populations = history.get_all_populations()
    ax = populations[populations.t >= 1].plot("t", "particles", style="o-")
    ax.set_xlabel("Generation")
    plt.savefig("populuations.png")
    
def plot_epsilons(history):
    plt.rcParams['font.size'] = 24
    fig = plt.figure(figsize=(12,10))
    eps = np.array(history.get_all_populations()['epsilon'][1:])
    x = np.arange(1, history.max_t + 2)
    ax = plt.gca()
    ax.plot(x, eps, 'ko-')
    ax.set_xlabel("Calibration Round")
    ax.set_ylabel("Error")
    plt.xticks(x)
    plt.savefig("epsilons.png")
    

if __name__ == '__main__':
    # Create "Null" ABCSMC object that has the corresponding parameter priors used in ABC run 
    lower_bound = 0
    scale = 1
    prior = pyabc.Distribution(mu=pyabc.RV("uniform", lower_bound, scale))
    abc = pyabc.ABCSMC(ABCsimulation, prior)
    
    # Load database
    db_path = ("sqlite:///" + "results.db")
    run_id = 1
    history = abc.load(db_path, run_id)
    
    ### Plots ###  

    exact = {"gamma": 10, "reg_param": 0.1}
    limits = {"gamma": [0.01, 100], "reg_param": [0.1, 1000]}
    for key in exact.keys():
        plot_1d(history, limits, exact, key)

    
    plot_2d_prior(0.01,100,0.1,1000)
    plot_2d_log_scaled(history)
    
    plot_epsilons(history)
    #plot_kde_matrix(history)
    #plot_population_sizes(history)
    
    
