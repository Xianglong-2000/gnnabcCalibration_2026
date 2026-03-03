import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_data(metrics, groups, path1, path2, path3):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)

    value_list = []
    group_list = []
    metric_list = []

    for i in range(len(metrics)):
    
        value_list_i = list(np.concatenate([df1[metrics[i]], df2[metrics[i]], df3[metrics[i]]], axis=0))
        group_list_i = [groups[0]]*len(df1[metrics[i]])+[groups[1]]*len(df2[metrics[i]])+[groups[2]]*len(df3[metrics[i]])
        metric_list_i = [metrics[i]]*len(value_list_i)
    
        value_list += value_list_i
        group_list += group_list_i
        metric_list += metric_list_i
    
    df = pd.DataFrame(columns=["Metric","Value","Group"])

    df["Value"] = value_list
    df["Group"] = group_list
    df["Metric"] = metric_list

    return df

def get_plot(df):
    plt.figure(figsize=(8, 6))

    ax = sns.violinplot(
        data=df,
        x="Metric",
        y="Value",
        hue="Group",
        split=False,       
        inner="box",           
        cut=2,             
        density_norm="width",
        linewidth=1
    )

    sns.stripplot(
        data=df,
        x="Metric",
        y="Value",
        hue="Group",
        dodge=True,
        jitter=0.1,
        alpha=0.7,
        size=4
    )

    xticks = ax.get_xticks()

    for x in xticks[:-1]:
        ax.axvline(x + 0.5, color='gray', linestyle='-', linewidth=1, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:3], labels[:3],
           loc="upper center", bbox_to_anchor=(0.5, 1.1),
           ncol=3, frameon=True)
    #plt.legend(handles[:3], labels[:3])

    labels = [
    "Aspect\nratio",
    "Order\nparameter",
    "Convexity",
    "Density",
    "Agreement\nwith exponential\nmodel",
    "Normalized\ngrowth rate"
    ]

    plt.xticks(ticks=range(len(labels)), labels=labels)

    plt.tight_layout()
    plt.ylim(-3, 3)
    plt.show()
    plt.savefig("/work/x5bai/project/Data_Files/_figures/s3_expabc_re_sampling_test.png")
                    
if __name__ == "__main__": 

    ##metrics = ["Aspect ratio", "Order parameter", "Convexity", "Density", "Agreement with exponential growth", "Normalized growth rate"]  ## S5
    metrics = ["e1", "e2", "e3", "e4", "e5", "e6"]  ## S3 & S4
    groups = ["Experimental", "Calibrated", "Prior"]

    path1 = "/work/x5bai/project/Data_Files/_exp_data/exp_summary_stat_s3s4.csv"  ### experimental
    path2 = "/work/x5bai/project/Data_Files/_sampling/s3_expabc_re_ss_pos.csv"  ### posterior
    path3 = "/work/x5bai/project/Data_Files/_sampling/s3_expabc_re_ss_pri.csv"  ### prior

    df = get_data(metrics, groups, path1, path2, path3)
    get_plot(df)
