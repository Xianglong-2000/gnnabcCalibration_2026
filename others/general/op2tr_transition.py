import os
import pandas as pd
import shutil

def delete_columns(old_path,new_path):
    df = pd.read_csv(old_path)
    # print(df.columns)
    df = df[["ImageName",
            "ImageNumber",
            "ObjectNumber",
            "AreaShape_Area",
            "AreaShape_Center_X",
            "AreaShape_Center_Y",
            "AreaShape_MajorAxisLength",
            "AreaShape_MinorAxisLength",
            "AreaShape_Orientation",
            "Location_Center_X",
            "Location_Center_Y",
            "TrackObjects_Label_50",
            "TrackObjects_ParentImageNumber_50",
            "TrackObjects_ParentObjectNumber_50",
            "YFP"]]
    df.to_csv(new_path)

def main():
    num_iter = 31
    dt_sim = "2025-10-20"
    # iterations = [5,6,7,8,9]

    # for i in iterations:
    for i in range(num_iter):
        if i < 10:  # the date here is always the date when the pickle files were generated
            parent_dir = f"D:/Projects/GNN Research/Data Files/_sim_csv_data_gnn/{dt_sim}/iteration 00{i}/"
            new_dir = f"D:/Projects/GNN Research/Data Files/_sim_pre_tr_csv_data_gnn/{dt_sim}/iteration 00{i}"
        elif (i >= 10) and (i <100):
            parent_dir = f"D:/Projects/GNN Research/Data Files/_sim_csv_data_gnn/{dt_sim}/iteration 0{i}/"
            new_dir = f"D:/Projects/GNN Research/Data Files/_sim_pre_tr_csv_data_gnn/{dt_sim}/iteration 0{i}"
        elif (i >= 100) and (i <1000):
            parent_dir = f"D:/Projects/GNN Research/Data Files/_sim_csv_data_gnn/{dt_sim}/iteration {i}/"
            new_dir = f"D:/Projects/GNN Research/Data Files/_sim_pre_tr_csv_data_gnn/{dt_sim}/iteration {i}"
        else:
            print("We don't want the number of iterations greater than 999 since it's too computationally heavy")

        os.makedirs(new_dir, exist_ok=True)  # create a new folder if it doesn't exist and do nothing if it exists

        folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
        # folder_with_error = [folders[25],folders[26],folders[27],folders[28],folders[29]]

        for f in folders:
            sub_parent_dir = parent_dir + f
            input_dir = sub_parent_dir

            # The location of the CSV output files
            output_dir = new_dir + "/" + f
            os.makedirs(output_dir, exist_ok=True)  # create a new folder if it doesn't exist and do nothing if it exists

            print("Input Directory: " + input_dir)
            print("Output Directory: " + output_dir)

            # Start Processing
            delete_columns(input_dir + "/Objects properties.csv", output_dir + "/Objects properties.csv")
            shutil.copy(input_dir + "/Object relationships.csv", output_dir + "/")
            print()

if __name__ == "__main__":
    main()