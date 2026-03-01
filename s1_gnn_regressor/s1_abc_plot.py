import pyabc
import matplotlib.pyplot as plt
import numpy as np
import pandas
import math

plt.rcParams['font.size'] = 24
colors = ['b','grey','r','g','chocolate','gold','m']
plt.rcParams["legend.fontsize"] = 10


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
    plt.plot(exact[key], 0, ls=':', color='black', label='Observed Value')
    
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
            label="Stage={}".format(t+1),
            refval=exact, refval_color="black", lw=3, color=colors[c])    
        plt.xlabel(key)
        ax.set_ylabel("") 
        c+=1
    ax.set_ylabel("")
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', ncol=1, prop={'size': 18})
    plt.show()
    plt.savefig("s1_%s_1D_KDE-t=%s.png" % (key,t), bbox_inches='tight')

    fig, arr_ax = plt.subplots(1, 3, figsize=(12, 7))
    pyabc.visualization.plot_sample_numbers(history, ax=arr_ax[0])
    pyabc.visualization.plot_epsilons(history, ax=arr_ax[1])
    pyabc.visualization.plot_effective_sample_sizes(history, ax=arr_ax[2])
    fig.tight_layout()

if __name__ == '__main__':
    # Create "Null" ABCSMC object that has the corresponding parameter priors used in ABC run 
    #lower_bound = 0
    #scale = 1
    #prior = pyabc.Distribution(mu=pyabc.RV("uniform", lower_bound, scale))

    param_config = {"gamma": [math.log(0.1), math.log(1000)], "reg_param": [math.log(0.01), math.log(100)]}  ### range of prior of log10 gamma and alpha
    prior_distributions = {}
    for parameter_name in param_config.keys():
        param_low = param_config[parameter_name][0]
        param_hi = param_config[parameter_name][1]
        width = abs(param_hi - param_low)
        prior_distributions[parameter_name] = {"type": "uniform", "args": (param_low, width), "kwargs": {}}
    prior = pyabc.Distribution.from_dictionary_of_dictionaries(prior_distributions)

    abc = pyabc.ABCSMC(ABCsimulation, prior)
    
    # Load the saved database
    db_path = "S1_test_1120.db"
    run_id = 1
    history = abc.load("sqlite:////work/x5bai/project/Data_Files/_abc_results/" + db_path, run_id)
    print(history.get_all_populations())
    
    # create plots
    gamma = 200
    alpha = 35
    exact = {"gamma": math.log(gamma), "reg_param": math.log(alpha)}
    limits = {"gamma": [math.log(0.1), math.log(1000)], "reg_param": [math.log(0.01), math.log(100)]}
    for key in exact.keys():
        plot_1d(history, limits, exact, key)
    
