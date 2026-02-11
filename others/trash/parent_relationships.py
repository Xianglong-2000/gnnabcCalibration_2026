import pandas as pd
import os

# Generating parent child relationship
def generate_parent_df(df):
    # ImageNumber = stepNum
    parent_df = df[["ImageNumber", "ObjectNumber", "TrackObjects_ParentImageNumber_50", "TrackObjects_ParentObjectNumber_50"]].copy(deep=True)
    parent_df['Relationship'] = 'Parent'
    parent_df.columns = ["First Image Number", "First Object Number", "Second Image Number", "Second Object Number", "Relationship"]
    parent_df = parent_df[["Relationship", "First Image Number", "First Object Number", "Second Image Number", "Second Object Number"]].copy(deep=True)
    return parent_df

def main():
    parent_dir = "D:/Projects/GNN Research/Data Files/_sim_csv_data_gnn/2025-06-29"
    folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    for i in range(len(folders)):
        sub_parent_dir = parent_dir + "/" + folders[i]  
        sub_folders = [f for f in os.listdir(sub_parent_dir) if os.path.isdir(os.path.join(sub_parent_dir, f))]
        for j in range(len(sub_folders)):
            sub_sub_parent_dir = sub_parent_dir + "/" + sub_folders[j]  
            df_path = sub_sub_parent_dir + "/" + os.listdir(sub_sub_parent_dir)[1]
            df = pd.read_csv(df_path)
            parent_df = generate_parent_df(df)
            parent_df.to_csv(sub_sub_parent_dir + "/parent_relationships.csv")
        print(f"-- iteration {i} completed --")
if __name__ == '__main__':
    main()