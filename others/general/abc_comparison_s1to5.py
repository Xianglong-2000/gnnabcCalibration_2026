import numpy as np
import matplotlib.pyplot as plt
import pyabc
from sklearn.neighbors import KernelDensity
import math
from scipy.stats import gaussian_kde, norm, multivariate_normal

plt.rcParams['font.size'] = 24
colors = ['b', 'grey', 'r', 'g', 'chocolate', 'gold', 'm', 'c', 'orange', 'pink']
plt.rcParams["legend.fontsize"] = 10

def plot_1d(s1, s2, s3, s4, s5, limits, exact, key, **kwargs):
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

    df_s1, w_s1 = s1.get_distribution(t=s1.n_populations-1)
    df_s2, w_s2 = s2.get_distribution(t=s2.n_populations-1)
    df_s3, w_s3 = s3.get_distribution(t=s3.n_populations-1)
    df_s4, w_s4 = s4.get_distribution(t=s4.n_populations-1)
    df_s5, w_s5 = s5.get_distribution(t=s5.n_populations-1)

    pyabc.visualization.plot_kde_1d(
        df_s1, w_s1, 
        x=key, ax=ax,
        xmin=xmin,
        xmax=xmax,              
        numx=10000,
        label="Stage={} in s1".format(s1.n_populations),
        refval=exact, refval_color="black", lw=3, color=colors[0])    

    pyabc.visualization.plot_kde_1d(
        df_s2, w_s2, 
        x=key, ax=ax,
        xmin=xmin,
        xmax=xmax,              
        numx=10000,
        label="Stage={} in s2".format(s2.n_populations),
        refval=exact, refval_color="black", lw=3, color=colors[1])    
    
    pyabc.visualization.plot_kde_1d(
        df_s3, w_s3, 
        x=key, ax=ax,
        xmin=xmin,
        xmax=xmax,              
        numx=10000,
        label="Stage={} in s3".format(s3.n_populations),
        refval=exact, refval_color="black", lw=3, color=colors[2])    
    
    pyabc.visualization.plot_kde_1d(
        df_s4, w_s4, 
        x=key, ax=ax,
        xmin=xmin,
        xmax=xmax,              
        numx=10000,
        label="Stage={} in s4".format(s4.n_populations),
        refval=exact, refval_color="black", lw=3, color=colors[3])    
    
    pyabc.visualization.plot_kde_1d(
        df_s5, w_s5, 
        x=key, ax=ax,
        xmin=xmin,
        xmax=xmax,              
        numx=10000,
        label="Stage={} in s5".format(s5.n_populations),
        refval=exact, refval_color="black", lw=3, color=colors[4])    
    
    plt.xlabel(key)
    ax.set_ylabel("") 
    ax.set_ylabel("")
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', ncol=1, prop={'size': 18})
    plt.show()

def ABCsimulation(params):
    return None

def main1():

    param_config = {'gamma': [0, 3], 'reg_param': [0, 2]}  ### range of prior of log10 gamma and alpha
    prior_distributions = {}
    for parameter_name in param_config.keys():
        param_low = param_config[parameter_name][0]
        param_hi = param_config[parameter_name][1]
        width = abs(param_hi - param_low)
        prior_distributions[parameter_name] = {"type": "uniform", "args": (param_low, width), "kwargs": {}}
    prior = pyabc.Distribution.from_dictionary_of_dictionaries(prior_distributions)

    abc = pyabc.ABCSMC(ABCsimulation, prior)
    
    # Load s1
    db_path_s1 = "test01_gatregressor_2025-08-20.db"
    run_id_s1 = 1
    s1 = abc.load("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/" + db_path_s1, run_id_s1)
    print("s1 abc summary: ",s1.get_all_populations())

    # Load s2
    db_path_s2 = "test01_gatlstmregressor_2025-08-20.db"
    run_id_s2 = 1
    s2 = abc.load("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/" + db_path_s2, run_id_s2)
    print("s2 abc summary: ",s2.get_all_populations())

    # Load s3
    db_path_s3 = "test04_results_57f_2025-09-25_gatlearner.db"
    run_id_s3 = 1
    s3 = abc.load("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/" + db_path_s3, run_id_s3)
    print("s3 abc summary: ",s3.get_all_populations())

    # Load s4
    db_path_s4 = "test01_gatlstmlearner_2025-09-25.db"
    run_id_s4 = 1
    s4 = abc.load("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/" + db_path_s4, run_id_s4)
    print("s4 abc summary: ",s4.get_all_populations())

    # Load s5
    db_path_s5 = "test01_gatlstmlearner_2025-09-25.db"
    run_id_s5 = 1
    s5 = abc.load("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/" + db_path_s5, run_id_s5)
    print("s5 abc summary: ",s5.get_all_populations())

    # Plot
    gamma = 200
    alpha = 35
    exact = {"gamma": math.log10(gamma), "reg_param": math.log10(alpha)}
    limits = {"gamma": [0, 3], "reg_param": [0, 2]}
    for key in exact.keys():
        plot_1d(s1, s2, s3, s4, s5, limits, exact, key)


################################################################################################################
# KL part

def normalize(samples, ranges):
    return (samples - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0])

def marginal_kl(samples, weights, theta0, sigma):
    kls = []
    for i in range(samples.shape[1]):
        kde = gaussian_kde(samples[:, i], weights=weights)
        q = norm(loc=theta0[i], scale=sigma)
        p_vals = kde(samples[:, i])
        q_vals = q.pdf(samples[:, i])
        ratio = np.maximum(p_vals / q_vals, 1e-300)
        kl = np.sum(weights * np.log(ratio))
        kls.append(kl)
    return np.array(kls)

def joint_kl(samples, weights, theta0, sigma):
    kde = gaussian_kde(samples.T, weights=weights)
    Q = multivariate_normal(mean=theta0, cov=np.eye(len(theta0)) * sigma**2)
    p_vals = kde(samples.T)
    q_vals = Q.pdf(samples)
    ratio = np.maximum(p_vals / q_vals, 1e-300)
    kl = np.sum(weights * np.log(ratio))
    return kl

def mean_squared_distance(samples, weights, theta0):
    return np.sum(weights * np.sum((samples - theta0)**2, axis=1))


def main2():

    # Load posteriors from pyABC databases
    abc1 = pyabc.History("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/CVIS_S1.db")
    abc2 = pyabc.History("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/CVIS_S2.db")
    #abc3 = pyabc.History("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/CVIS_S3.db")
    abc4 = pyabc.History("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/CVIS_S4.db")
    abc5 = pyabc.History("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/CVIS_S5.db")

    df1, w1 = abc1.get_distribution()
    df2, w2 = abc2.get_distribution()
    #df3, w3 = abc3.get_distribution()
    df4, w4 = abc4.get_distribution()
    df5, w5 = abc5.get_distribution()

    # Convert to weighted samples
    samples1 = df1[['gamma', 'reg_param']].to_numpy()
    samples2 = df2[['gamma', 'reg_param']].to_numpy()
    #samples3 = df3[['gamma', 'reg_param']].to_numpy()
    samples4 = df4[['gamma', 'reg_param']].to_numpy()
    samples5 = df5[['gamma', 'reg_param']].to_numpy()

    weights1 = w1 / np.sum(w1)
    weights2 = w2 / np.sum(w2)
    #weights3 = w3 / np.sum(w3)
    weights4 = w4 / np.sum(w4)
    weights5 = w5 / np.sum(w5)

    theta0 = np.array([math.log10(200), math.log10(35)])  # your fixed point (can be in log10 scale)
    print("fixed params: ", theta0)

    # normalize parameters to [0, 1] ranges
    param_ranges = np.array([[0, 3], [0, 2]])  # [min, max] for each parameter

    samples1_norm = normalize(samples1, param_ranges)
    samples2_norm = normalize(samples2, param_ranges)
    #samples3_norm = normalize(samples3, param_ranges)
    samples4_norm = normalize(samples4, param_ranges)
    samples5_norm = normalize(samples5, param_ranges)

    theta0_norm = normalize(theta0, param_ranges)

    # delta function approximation setup
    sigma = 0.02

    # marginal KL
    kl1_marg = marginal_kl(samples1_norm, weights1, theta0_norm, sigma)
    kl2_marg = marginal_kl(samples2_norm, weights2, theta0_norm, sigma)
    #kl3_marg = marginal_kl(samples3_norm, weights3, theta0_norm, sigma)
    kl4_marg = marginal_kl(samples4_norm, weights4, theta0_norm, sigma)
    kl5_marg = marginal_kl(samples5_norm, weights5, theta0_norm, sigma)

    print("Marginal KL divergence results [gamma, alpha]:")
    print(f"Posterior 1: {kl1_marg}")
    print(f"Posterior 2: {kl2_marg}")
    #print(f"Posterior 3: {kl3_marg}")
    print(f"Posterior 4: {kl4_marg}")
    print(f"Posterior 5: {kl5_marg}")

    # joint KL
    kl1_joint = joint_kl(samples1_norm, weights1, theta0_norm, sigma)
    kl2_joint = joint_kl(samples2_norm, weights2, theta0_norm, sigma)
    #kl3_joint = joint_kl(samples3_norm, weights3, theta0_norm, sigma)
    kl4_joint = joint_kl(samples4_norm, weights4, theta0_norm, sigma)
    kl5_joint = joint_kl(samples5_norm, weights5, theta0_norm, sigma)

    print("\nJoint KL divergence results:")
    print(f"Posterior 1: {kl1_joint}")
    print(f"Posterior 2: {kl2_joint}")
    #print(f"Posterior 3: {kl3_joint}")
    print(f"Posterior 4: {kl4_joint}")
    print(f"Posterior 5: {kl5_joint}")

    # Mean-squared distance for sanity check
    msd1 = mean_squared_distance(samples1_norm, weights1, theta0_norm)
    msd2 = mean_squared_distance(samples2_norm, weights2, theta0_norm)
    #msd3 = mean_squared_distance(samples3_norm, weights3, theta0_norm)
    msd4 = mean_squared_distance(samples4_norm, weights4, theta0_norm)
    msd5 = mean_squared_distance(samples5_norm, weights5, theta0_norm)

    print("\nMean-squared distance to delta [gamma+alpha]:")
    print(f"Posterior 1: {msd1}")
    print(f"Posterior 2: {msd2}")
    #print(f"Posterior 3: {msd3}")
    print(f"Posterior 4: {msd4}")
    print(f"Posterior 5: {msd5}")


if __name__ == '__main__':
    ##main1()
    main2()