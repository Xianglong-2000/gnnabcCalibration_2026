import pandas as pd
import numpy as np
import re
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import matplotlib.pyplot as plt


def get_param_dict(parent_dir_tr):

    #folders = ["iteration 00"]  # for a test
    folders = [f for f in os.listdir(parent_dir_tr) if os.path.isdir(os.path.join(parent_dir_tr, f))]

    pattern = re.compile(r"gamma=(\d+\.\d+)_reg_param=(\d+\.\d+)_iter=(\d+)")
    param_dict = {}

    for f in folders:

        sub_parent_dir_tr = parent_dir_tr + "/" + f

        #sub_folders = ["gamma=11.96_reg_param=0.08_iter=0","gamma=11.96_reg_param=0.042_iter=0"]  # for a test only
        sub_folders = [f for f in os.listdir(sub_parent_dir_tr) if os.path.isdir(os.path.join(sub_parent_dir_tr, f))]

        for ff in sub_folders:
            match = pattern.search(ff)
            gamma, reg_param, average = match.groups()
            counter = float(average)
            if counter <= 9:
                param_pair_index = f"gamma_{gamma}_reg_param_{reg_param}-Average-00{average}"
            elif (counter > 9)&(counter <=99):
                param_pair_index = f"gamma_{gamma}_reg_param_{reg_param}-Average-0{average}"
            elif (counter > 99)&(counter <=999):
                param_pair_index = f"gamma_{gamma}_reg_param_{reg_param}-Average-{average}"

            csv_path = sub_parent_dir_tr + "/" + ff + "/Trackrefiner.Objects properties_Average_analysis.csv"  # node features
            param_dict[param_pair_index] = csv_path

    return param_dict  # {"gamma_reg_param_Average": [csv1 path, csv2 path]}



def stacking_graphs(param_dict):

    all_x = np.empty((0, 69))

    #lengths = []
    #params = []
    #gamma_list = []  # depending on your targets
    #reg_param_list = []  # depending on your targets
    for param in param_dict.keys():

        print("stacking ", param)

        #split_list = param[:-11].split('_')
        #y1 = float(split_list[1])  # depending on your targets
        #y2 = float(split_list[4])  # depending on your targets

        df = pd.read_csv(param_dict[param])
        #lengths.append(len(df))
        #params.append(param)

        x = df.to_numpy()
        #gamma_list += [y1]*len(df)  # depending on your targets
        #reg_param_list += [y2]*len(df)  # depending on your targets

        all_x_last = all_x.copy()
        all_x = np.concatenate((all_x_last, x), axis=0)

    columns = list(pd.read_csv(param_dict[list(param_dict.keys())[0]]).columns)
    df_all = pd.DataFrame(all_x, columns=columns)
    #df_all["gamma"] = gamma_list  # depending on your targets
    #df_all["reg_param"] = reg_param_list  # depending on your targets
    print("stacked data size: ", df_all.shape)
    # print(lengths)
    # print(params)
    # print(df_all.columns)

    #return df_all, lengths, params
    return df_all



def to_float(val):
    try:
        return float(val)
    except ValueError:
        if (val == 'True') | (val == 'TRUE'):
            return float(1)
        elif (val == 'False') | (val == 'FALSE'):
            return float(0)
        elif type(val) == int:
            return float(val)
        else:
            return val  # keep as string if not a number



def preprocessing(df):

    # Sub-step 0: turn data type to float
    for col in df.columns:
        df[col] = df[col].map(to_float)
    print("data size after sub-step 0: ", df.shape)

    # Sub_step 1: turn vector strings to vector tuples
    df['dir'] = df['dir'].apply(eval).apply(lambda x: tuple(map(float, x)))
    print("data size after sub-step 1: ", df.shape)

    # Sub_step 2: fill empty values with the column mean
    df = df.fillna(df.mean(numeric_only=True))
    print("data size after sub-step 2: ", df.shape)

    # Sub_step 3: flatten vector tuples
    #x_list = list(df.drop(columns=['dir']).columns[:-2])
    #x_list = list(df.drop(columns=['dir']).columns)
    #y_list = list(df.columns[-2:])
    vec_df = pd.DataFrame(df['dir'].tolist(), columns=['dir_1', 'dir_2'])
    df = pd.concat([df.drop(columns=['dir']), vec_df], axis=1)
    #df = df[x_list+['dir_1', 'dir_2']+y_list]
    #df = df[x_list + ['dir_1', 'dir_2']]
    print("data size after sub-step 3: ", df.shape)

    # Sub_step 4: drop constant features
    #constant_columns = [col for col in df.columns[:-2] if df[col].nunique() <= 1]
    constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=constant_columns)
    print("data size after sub-step 4: ", df.shape)
    print("constant columns: ", constant_columns)

    # Sub_step 5: do standard normalization
    # Sub_step 5: do min-max normalization
    #scaler = StandardScaler()
    #x_scaled = scaler.fit_transform(df)  # do standard normalization
    #df = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)
    #print("data size after sub-step 6: ", df.shape)
    df = (df - df.min()) / (df.max() - df.min())

    """
    # Sub_step 6: drop low variance features
    selector = VarianceThreshold(threshold=1e-2)
    #x_reduced = selector.fit_transform(df.iloc[:, :-2])
    x_reduced = selector.fit_transform(df)
    #df1 = pd.DataFrame(x_reduced, columns=df.iloc[:, :-2].columns[selector.get_support()], index=df.iloc[:, :-2].index)
    df1 = pd.DataFrame(x_reduced, columns=df.columns[selector.get_support()], index=df.index)
    #df = pd.concat([df1, df.iloc[:, -2:]], axis=1)
    #df = pd.concat([df1, df.iloc[:, -2:]], axis=1)
    df = df1
    print("data size after sub-step 5: ", df.shape)

    # Sub_step 7: remove redundant features
    #corr_matrix = df.iloc[:, :-2].corr().abs()
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
    df = df.drop(columns=to_drop)
    print("data size after sub-step 7: ", df.shape)
    """

    # Sub_step 8: print and check
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)  # Show all rows
    print(df.columns)
    # print(df.dtypes)
    # print(df.head())
    # print(df.mean())

    return df


"""
def KBest_model(np_x, np_y, num_features, columns):
    selector = SelectKBest(score_func=f_classif, k=num_features)
    x_selected = selector.fit_transform(np_x, np_y)

    selected_indices = selector.get_support(indices=True)
    selected_columns = list(columns[selected_indices])
    scores = selector.scores_
    p_values = selector.pvalues_

    ranked = pd.DataFrame({
        'feature': selected_columns,
        'score': scores,
        'p value': p_values
    }).sort_values(by='score', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(ranked['feature'], ranked['score'], color='skyblue')
    plt.xlabel('Score (e.g., F-value)')
    plt.title('Feature Importance Ranking')
    plt.gca().invert_yaxis()  # show highest score at top
    plt.tight_layout()
    plt.show()

    print("selected features: ", selected_features)

    return x_selected,selected_columns
"""

"""
def RFR_model(np_x, np_y, num_features, columns):

    # set up the model:
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1,verbose=1)
    ## model = lgb.LGBMRegressor(n_estimators=100, verbose=1, n_jobs=-1)
    ## model = ExtraTreesRegressor(n_estimators=100, verbose=1, n_jobs=-1)

    # option 1:
    model.fit(np_x, np_y)
    k = num_features
    importance = model.feature_importances_
    top_k_indices = np.argsort(importance)[-k:][::-1]
    selected_columns = columns[top_k_indices]
    selected_importance = importance[top_k_indices]
    x_selected = np_x[:, top_k_indices]

    # option 2:
    # selector = SelectFromModel(model,threshold="median")
    # selector.fit(np_x, np_y)
    # x_selected = selector.transform(np_x)
    # selected_indices = selector.get_support(indices=True)
    # selected_columns = columns[selected_indices]
    # selected_importance = importance[selected_indices]
    # print(columns)
    # print(columns[:-2])
    # print(len(columns[:-2]))
    # print(len(importance))
    # Rank all features and plot
    ranked = pd.DataFrame({
        'feature': columns[:-2],  ## change it to selected_columns
        'importance': importance  ## change it to selected_importance
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(ranked['feature'], ranked['importance'], color='skyblue')
    plt.xlabel('importance')
    plt.title('Feature Importance Ranking')
    plt.gca().invert_yaxis()  # show highest score at top
    plt.tight_layout()
    plt.show()

    return x_selected, selected_columns
"""

"""
def feature_selection(df, num_features):

    # get np arrays for x, y, and columns
    np_x = df.iloc[:, :-2].to_numpy()
    np_y = df.iloc[:, -2:].to_numpy()
    columns = df.columns

    # choose model and run
    # x_selected = KBest_model(np_x, np_y, num_features, columns)  ## only 1 target at one time
    x_selected, selected_features = RFR_model(np_x, np_y, num_features, columns)  ## multiple targets at one time

    new_array = np.concatenate((x_selected, np_y), axis=1)
    new_columns = list(selected_features) + ["gamma","reg_param"]

    print("data size after feature selection (features + targets) : ", new_array.shape)

    return new_array, new_columns
"""

"""
def get_new_csv_and_save(new_array, new_columns, lengths, params, param_dict):

    # cut the stacked x array into graphs
    splits = np.cumsum(lengths)[:-1]
    split_arrays = np.split(new_array, splits)
    dfs = [pd.DataFrame(arr, columns = new_columns) for arr in split_arrays]

    # loop and save
    for param, df in zip(params, dfs):
        new_csv_path = param_dict[param].replace("_sim_tr_csv_data_gnn","_sim_tr_csv_feature_selected_data_gnn")
        os.makedirs(new_csv_path.rsplit('/', 1)[0], exist_ok=True)
        df.to_csv(new_csv_path, index=False)
"""


if __name__ == "__main__":

    # Step 0: start timing
    start_time = time.time()

    # Step 1: get {"gamma_reg_param_Average": csv path}
    parent_dir_tr = "D:/Projects/GNN Research/Data Files/_sim_tr_csv_data_gnn/2025-08-20"  # me
    param_dict = get_param_dict(parent_dir_tr)  # me
    ## print(param_dict)

    # Step 2: stack all graphs with their targets
    #df, lengths, params = stacking_graphs(param_dict)
    df = stacking_graphs(param_dict)

    # Step 3: data preprocessing
    df = preprocessing(df)





    """
    # Step 4: feature selection
    number_of_features = 20
    new_array, new_columns = feature_selection(df, number_of_features)

    # Step 5: split back to graph csv files and save
    get_new_csv_and_save(new_array, new_columns, lengths, params, param_dict)

    # Step 6: check the duration of time
    end_time = time.time()
    duration = (end_time - start_time) / (60 * 60)
    print(f"Computing time: {duration:.3f} hours")
    """