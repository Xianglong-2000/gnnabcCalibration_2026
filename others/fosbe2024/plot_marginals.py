import os
import pyabc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from pyabc.storage import History
from pyabc.transition import MultivariateNormalTransition

subdir = '2d_kde'
plt.rcParams['font.size'] = 10
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

cbar_lim = [0, 0.3575702248246986] #-AY
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
        pass
        #cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', ticks=np.linspace(cbar_lim[0], cbar_lim[1], 4), format='%.4f')
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

def ABCsimulation(params):
    return None

n_x = 200    
gamma_min = 0.1
gamma_max = 1000
alpha_min = 0.1
alpha_max = 100

# Import history
lower_bound = 0
scale = 1
prior = pyabc.Distribution(mu=pyabc.RV("uniform", lower_bound, scale))
abc = pyabc.ABCSMC(ABCsimulation, prior)
db_path = ("sqlite:///" + "results.db")
run_id = 1
history = abc.load(db_path, run_id)

# Get data
t=2
df, w = history.get_distribution(m=0, t=t)
print(f'Mean gamma: {np.exp(df["gamma"].mean())}')
print(f'std gamma: {np.exp(df["gamma"].std())}')
print(f'Mean alpha: {np.exp(df["reg_param"].mean())}')
print(f'std alpha: {np.exp(df["reg_param"].std())}')

# Define figure and gridspec layout
fig = plt.figure(figsize=(4, 4)) # 4,4 is good size
gs = fig.add_gridspec(4, 4)

# Main plot - pcolormesh
xedges, yedges, pdf = kde_2d(df, w, 'gamma', 'reg_param', xmin=np.log(gamma_min), xmax=np.log(1000), ymin=np.log(alpha_min), ymax=np.log(alpha_max), numx=500, numy=500)
xedges = np.exp(xedges)
yedges = np.exp(yedges)

#print(xedges[0,:])
#print(yedges[:,0])

ax_main = fig.add_subplot(gs[1:4, 0:3])
#pcm = ax_main.pcolormesh(xedges, yedges, pdf, cmap='viridis', vmin=cbar_lim[0], vmax=cbar_lim[1])
pcm = ax_main.pcolormesh(xedges, yedges, pdf, cmap='viridis')
#fig.colorbar(pcm, ax=ax_main, orientation='horizontal', ticks=np.linspace(0, cbar_lim[1], 4), format='%.2f')
ax_main.set_xlabel('γ')
ax_main.set_ylabel('α')
ax_main.set_yscale('log')
ax_main.set_xlim([gamma_min, gamma_max])
ax_main.set_xscale('log')
ax_main.set_ylim([alpha_min, alpha_max])

# Top KDE
x = df['gamma']
ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
kde_x = gaussian_kde(x)
x_vals = np.linspace(gamma_min, gamma_max, 1000)
ax_top.plot(x_vals, kde_x(np.log(x_vals)), color='skyblue')
ax_top.fill_between(x_vals, kde_x(np.log(x_vals)), alpha=0.3, color='skyblue')

ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)
ax_top.spines['left'].set_visible(False)
ax_top.spines['bottom'].set_visible(False)
ax_top.tick_params(axis='both', which='both', left=False, bottom=False, top=False, labelbottom=False, labelleft=False)

# Right KDE
y = df['reg_param']
ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
kde_y = gaussian_kde(y)
y_vals = np.linspace(alpha_min, alpha_max, 1000)
ax_right.plot(kde_y(np.log(y_vals)), y_vals, color='salmon')
ax_right.fill_betweenx(y_vals, kde_y(np.log(y_vals)), alpha=0.3, color='salmon')
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)
ax_right.spines['bottom'].set_visible(False)
ax_right.spines['left'].set_visible(False)
ax_right.tick_params(axis='both', which='both', left=False, bottom=False, right=False, labelleft=False, labelbottom=False)

fig.set_tight_layout(True)
plt.show()
