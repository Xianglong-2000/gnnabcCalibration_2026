import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def get_plot(data):
    data = np.array(data)
    mean_val = data.mean()

    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 1000)

    plt.hist(data, bins=100, density=True)
    plt.plot(x, kde(x))
    plt.axvline(mean_val, color='yellow', linestyle='--', linewidth=2, label=f"Mean = {mean_val:.2f}")

    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Histogram + KDE with Mean")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # get feature values
    exp_path = "D:/Projects/GNN Research/Data Files/_exp_data/node"
    exp_files = [f for f in os.listdir(exp_path)]
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    feature_names = ["radius","Elongation_Rate","targetVol","startVol"]  ### depending on which features we are checking
    for f in exp_files:
        f_path = exp_path + "/" + f
        exp_df = pd.read_csv(f_path)
        list_1 += exp_df[feature_names[0]].dropna().tolist()
        list_2 += exp_df[feature_names[1]].dropna().tolist()
        list_3 += exp_df[feature_names[2]].dropna().tolist()
        list_4 += exp_df[feature_names[3]].dropna().tolist()

    print("total points: ", exp_df.shape[0]*len(exp_files))
    print("non-empty points in 4 lists: ", len(list_1), len(list_2), len(list_3), len(list_4))

    # get plots
    get_plot(list_1)
    get_plot(list_2)
    get_plot(list_3)
    get_plot(list_4)