import subprocess
import os
import sys
import time
import re
import pickle as pk
import matplotlib.pyplot as plt
import tr_csv2g_data_new


def main_1():

    parent_dir_tr = "D:/Projects/GNN Research/Data Files/_sim_tr_csv_data_gnn/2025-07-27"
    parent_dir = "D:/Projects/GNN Research/Data Files/_sim_csv_data_gnn/2025-07-27"
    param_dict = tr_csv2g_data_new.get_param_dict(parent_dir_tr, parent_dir) 
    print(param_dict)
    print(len(param_dict.keys()))
    # print(param_dict.keys())

    # Select required columns and ensure they are numeric
    include_columns = ["ImageNumber", "ObjectNumber", "id", "parent_id",
                       'Unnamed: 0', 'ImageName', 'AreaShape_Area',
                       'AreaShape_Center_X', 'AreaShape_Center_Y', 'AreaShape_Orientation',
                       'TrackObjects_Label_50', 'TrackObjects_ParentObjectNumber_50',
                        'cellAge', 'LifeHistory', 'TrajectoryX', 'TrajectoryY',
                       'Bacterium_Slope', 'Orientation_Angle_Between_Slopes',
                       'Direction_of_Motion', 'Motion_Alignment_Angle',
                       'Source_Neighbor_Avg_TrajectoryX', 'Source_Neighbor_Avg_TrajectoryY',
                       'divideFlag', 'Division_Family_Count', 'Daughter_Mother_Length_Ratio',
                       'Total_Daughter_Mother_Length_Ratio', 'Neighbor_Difference_Count',
                       'Neighbor_Shared_Count', 'Average_Length', 'Length_Change_Ratio',
                       'Avg_Length_Change_Ratio', 'Unexpected_End', 'Unexpected_Beginning',
                       'strainRate_rolling', 'startVol', 'targetVol', 'Prev_Bacterium_Slope',
                       'Bacterium_Movement'
                       ]
    num_features = len(include_columns)

    # for i in range(4,num_features):
    for i in range(4,num_features):  ## for a test
        sub_columns = include_columns[:i] + include_columns[i+1:]
        remove_col = str(include_columns[i]).replace(" ", "_").replace(":", "")
        print(remove_col)
        ## print(sub_columns)

        start_time = time.time()
        data_tuple = {}
        counter = 0

        for key in param_dict.keys():
            data_tuple[key] = tr_csv2g_data_new.load_graph_data(key, param_dict[key][0], param_dict[key][1], sub_columns)
            # data_tuple[key] = load_graph_data_sushma(key, param_dict[key][0], param_dict[key][1])
            # print(param_dict[key][0])
            # print(param_dict[key][1])
            # print(data_tuple[key])
            # print(data_tuple[key].y)
            counter += 1
            if counter == 1:
                print('Processing...')
                print(f"{key}: {data_tuple[key]}")
            elif counter == len(list(param_dict.keys())):
                print(f"{key}: {data_tuple[key]}")
                print('...Done')
            else:
                print(f"{key}: {data_tuple[key]}")

        ## print(data_tuple)

        # check the training duration
        end_time = time.time()
        duration = (end_time - start_time)/(60*60)
        print(f"Computing time: {duration:.3f} hours")

        # save the graph data
        folder_path = "D:/Projects/GNN Research/Data Files/_sim_graph_data_gnn/2025-07-27/"
        os.makedirs(folder_path, exist_ok=True)
        g_data_path = os.path.join(folder_path, f"graph_data_where_{remove_col}_removed.pkl")
        with open(g_data_path, 'wb') as f:
            pk.dump(data_tuple, f)



def main_2():

    parent_dir = "D:/Projects/GNN Research/Data Files/_sim_graph_data_gnn/2025-07-27/"
    folders = [f for f in os.listdir(parent_dir) if (os.path.isfile(os.path.join(parent_dir, f))) & (f[0]=="g")]
    ### selection part folder name starts with "g", and will be removed in a bit
    ### training part folder name starts with "n", and will stay there
    print(folders)

    scores = {}
    for f in folders:
        print("Evaluating ", f)
        g_data_path = os.path.join(parent_dir, f)
        path = "D:/Projects/GNN Research/Code Files/my_code_files/training_new.py"
        envr = sys.executable
        result = subprocess.run([envr, path, g_data_path], capture_output=True, text=True)

        print("STDOUT:\n", result.stdout)
        print()
        text_with_float = result.stdout.split("Testing R_2:")[-1]
        match = re.search(r"[-+]?\d*\.?\d+", text_with_float)
        if match:
            extracted_float_str = match.group(0)
            extracted_float = float(extracted_float_str)
            scores[f] = extracted_float
            print(extracted_float)
        else:
            print("No float found in the string.")

        # remove all graph data files for selection, which we can do it manually as well
        ## if os.path.exists(g_data_path):
        ##     os.remove(g_data_path)
        ##     print(f"File '{g_data_path}' deleted successfully.")
        ## else:
        ##     print(f"File '{g_data_path}' does not exist.")

    categories = [c.split("where_")[-1].split("_removed")[0] for c in list(scores.keys())]
    values = list(scores.values())
    plt.bar(categories, values)
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Data from Dictionary')
    plt.show()


def main_3():
    df = pd.read_csv(".csv")
    categories = df["Feature_removed"]
    values = df["E"]
    plt.bar(categories, values)
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Data from Dictionary')
    plt.show()


if __name__ == "__main__":  ## we can run them respectively as they take long
    # main_1()
    # main_2()
    main_3()