import numpy as np
import pickle
import CellModeller
from neighborsFinding import neighbor_finders
import pandas as pd
import glob


def assign_label_from_nearest_disappeared_bacterium(current_position, previous_positions, disappeared_ids,
                                                    labels_dict, last_occurrence_map):
    """
    Assigns a label to a new bacterium by finding the closest disappeared bacterium
    from the previous time step (i.e., a 'grandmother').

    Parameters:
    - current_position (np.ndarray): 1D array representing the (x, y) position of the new bacterium.
    - previous_positions (np.ndarray): 2D array of shape (n, 2) with (x, y) positions of disappeared bacteria.
    - disappeared_ids (list): List of IDs corresponding to `previous_positions`.
    - labels_dict (dict): Dictionary mapping bacterium IDs to their assigned labels.
    - last_occurrence_map (dict): Dictionary mapping bacterium IDs to their last index in the features dict.

    Returns:
    - label (int): Assigned label from the closest disappeared bacterium.
    - last_index (int): Index of the last appearance of that bacterium in the features dict.
    """
    distances = np.linalg.norm(previous_positions - current_position, axis=1)
    closest_idx = np.argmin(distances)
    nearest_id = disappeared_ids[closest_idx]
    label = labels_dict[nearest_id]
    last_index = last_occurrence_map[nearest_id]
    return label, last_index


def extract_bacterial_features(current_time_step_num, current_time_step_bac, prev_time_step_bac, bac_features_dict,
                               bacteria_labels_dict, neighbor_records, cell_type_mapping, use_grandmother_as_parent):
    """
    Extracts and updates bacterial features for a specific time step from CellModeller simulation data.

    Parameters:
    - current_time_step_num (int): The current simulation time step number.
    - current_time_step_bac (dict): Dictionary containing 'cellStates' and 'lineage' at the current step.
    - prev_time_step_bac (dict): Dictionary containing 'cellStates' and 'lineage' at the previous step.
    - bac_features_dict (dict): Dictionary accumulating features across time steps.
    - bacteria_labels_dict (dict): Mapping from bacteria IDs to assigned tracking labels.
    - neighbor_records (list): List to accumulate neighboring bacteria relationships.
    - cell_type_mapping (dict): Mapping of cell type names to integer IDs.
    - use_grandmother_as_parent (bool):
        If True, approximates the parent bacterium by selecting the nearest disappeared bacterium
        from the previous time step. Useful when large time step gaps cause the actual parent
        to no longer appear in the previous step.

    Returns:
    - bac_features_dict (dict): Updated features dictionary.
    - bacteria_labels_dict (dict): Updated bacteria label mapping.
    - neighbor_records (list): Updated list of neighbor records.
    """

    # bacteria information
    cs = current_time_step_bac['cellStates']
    cs_prev_time_step = prev_time_step_bac['cellStates']

    # Cellmodeller pickle file structure:
    # ['cellStates', 'stepNum', 'lineage', 'moduleStr', 'moduleName']
    # important keys: 1: 'cellStates' (type: dictionary)  2: 'lineage' (dictionary)
    # 'cellStates' dictionary keys: bacteria id
    # 'lineage' dictionary: daughter id: parent id

    # find Bacteria whose life has ended.
    if len(cs_prev_time_step) > 0:
        bac_ids_prev_time_step = set([cs_prev_time_step[it].id for it in cs_prev_time_step.keys()])
    else:
        bac_ids_prev_time_step = set()
    bac_ids_current_time_step = set([cs[it].id for it in cs.keys()])
    life_ended_bacteria_in_previous_time_step = list(bac_ids_prev_time_step - bac_ids_current_time_step)

    # now center coordinate of them
    if len(life_ended_bacteria_in_previous_time_step) > 0:
        life_ended_bac_center_pos_in_previous_time_step = np.array([cs_prev_time_step[b_id].pos[:2] for b_id in
                                                                    life_ended_bacteria_in_previous_time_step])
    else:
        life_ended_bac_center_pos_in_previous_time_step = np.array([])

    bacteria_ids = np.array(bac_features_dict['id'])
    # Get last occurrence of each unique id:
    # Reverse the array and use return_index on it
    unique_bac_ids, reverse_bac_idx = np.unique(bacteria_ids[::-1], return_index=True)

    # Convert reverse indices to forward indices (relative to original)
    bac_last_indices = len(bacteria_ids) - 1 - reverse_bac_idx

    # Build a mapping: id → last index
    bac_id_to_last_idx = dict(zip(unique_bac_ids, bac_last_indices))

    if len(bacteria_labels_dict) > 0:
        max_bacteria_labels = max(bacteria_labels_dict.values()) + 1
    else:
        max_bacteria_labels = 1

    this_time_step_bac_parent_image_number = []
    this_time_step_bac_parent_obj_number = []
    this_time_step_bac_label = []
    this_time_step_bac_cell_age = []

    bac_it = cs.keys()
    prev_time_steps_bac_ids = bacteria_labels_dict.keys()

    for this_bac_key, this_bac_features in cs.items():
        # find neighbors
        this_bac_id = this_bac_features.id
        this_bac_neighbours = this_bac_features.neighbours
        if len(this_bac_neighbours) > 0:
            neighbor_records.extend(
                [(current_time_step_num, this_bac_id, neighbor_id) for neighbor_id in this_bac_neighbours])

        if this_bac_features.id in prev_time_steps_bac_ids:  # it means: Life has continued for bacterium

            # last occurrence of element in list
            last_occurrence_index_in_list = bac_id_to_last_idx[this_bac_features.id]

            # cell age
            this_time_step_bac_cell_age.append(bac_features_dict['CellAge'][last_occurrence_index_in_list] + 1)

            # parent information
            this_time_step_bac_parent_image_number.append(
                bac_features_dict['ImageNumber'][last_occurrence_index_in_list])

            this_time_step_bac_parent_obj_number.append(
                bac_features_dict['ObjectNumber'][last_occurrence_index_in_list])

            # assign label
            this_time_step_bac_label.append(bacteria_labels_dict[this_bac_features.id])

        else:  # it means: A bacterium has been born or a cell division has taken place
            if current_time_step_num == 1:  # birth

                if len(bacteria_labels_dict) == 0:
                    this_bacterium_label = max_bacteria_labels
                    max_bacteria_labels += 1
                else:
                    this_bacterium_label = max_bacteria_labels
                    max_bacteria_labels += 1

                # cell age
                this_time_step_bac_cell_age.append(0)
                # parent information
                this_time_step_bac_parent_image_number.append(0)
                this_time_step_bac_parent_obj_number.append(0)

            else:
                # find parent id
                if current_time_step_bac['lineage'][this_bac_features.id]:  # parent bacteria has been found
                    parent_id = current_time_step_bac['lineage'][this_bac_features.id]
                    # assign label
                    if parent_id in prev_time_steps_bac_ids:
                        this_bacterium_label = bacteria_labels_dict[parent_id]
                        # last occurrence of element in list
                        last_occurrence_index_in_list = bac_id_to_last_idx[parent_id]
                    else:
                        '''
                            Find the nearest bacterium from the previous timestep that does not exist in the current
                             timestep (since the founded bacterium is the grandmother of the daughter)                    
                        '''
                        if len(life_ended_bacteria_in_previous_time_step) > 0 and use_grandmother_as_parent:

                            # calculate distance
                            this_bacterium_center = np.array([this_bac_features.pos[0], this_bac_features.pos[1]])
                            this_bacterium_label, last_occurrence_index_in_list = \
                                assign_label_from_nearest_disappeared_bacterium(
                                    this_bacterium_center, life_ended_bac_center_pos_in_previous_time_step,
                                    life_ended_bacteria_in_previous_time_step, bacteria_labels_dict,
                                    bac_id_to_last_idx)
                        else:
                            this_bacterium_label = max_bacteria_labels
                            max_bacteria_labels += 1
                            last_occurrence_index_in_list = -1

                else:
                    '''
                        Find the nearest bacterium from the previous timestep that does not exist in the 
                        current timestep (since the founded bacterium is the grandmother of the daughter)                    
                    '''

                    if use_grandmother_as_parent:
                        # calculate distance
                        this_bacterium_center = np.array((this_bac_features.pos[0], this_bac_features.pos[1]))
                        this_bacterium_label, last_occurrence_index_in_list = \
                            assign_label_from_nearest_disappeared_bacterium(
                                this_bacterium_center, life_ended_bac_center_pos_in_previous_time_step,
                                life_ended_bacteria_in_previous_time_step, bacteria_labels_dict,
                                bac_id_to_last_idx)
                    else:
                        this_bacterium_label = max_bacteria_labels
                        max_bacteria_labels += 1
                        last_occurrence_index_in_list = -1

                if last_occurrence_index_in_list != -1:
                    # cell age
                    this_time_step_bac_cell_age.append(0)

                    # parent information
                    this_time_step_bac_parent_image_number.append(
                        bac_features_dict['ImageNumber'][last_occurrence_index_in_list])
                    this_time_step_bac_parent_obj_number.append(
                        bac_features_dict['ObjectNumber'][last_occurrence_index_in_list])
                else:
                    # cell age
                    this_time_step_bac_cell_age.append(0)

                    # parent information
                    this_time_step_bac_parent_image_number.append(0)
                    this_time_step_bac_parent_obj_number.append(0)

            # assign label
            this_time_step_bac_label.append(this_bacterium_label)
            # add new key: value to dictionary
            bacteria_labels_dict[this_bac_features.id] = this_bacterium_label

    bac_features_dict['ImageName'].extend([current_time_step_bac['stepNum']] * len(cs))
    bac_features_dict['ImageNumber'].extend([current_time_step_num] * len(cs))
    bac_features_dict['ObjectNumber'].extend(range(1, len(cs) + 1))
    bac_features_dict['AreaShape_Center_X'].extend([cs[it].pos[0] for it in bac_it])
    bac_features_dict['AreaShape_Center_Y'].extend([cs[it].pos[1] for it in bac_it])
    bac_features_dict['AreaShape_MajorAxisLength'].extend([cs[it].length for it in bac_it])
    bac_features_dict['AreaShape_MinorAxisLength'].extend([cs[it].radius for it in bac_it])
    bac_features_dict['AreaShape_Orientation'].extend([np.arctan2(cs[it].dir[1], cs[it].dir[0]) for it in bac_it])
    bac_features_dict['Node_x1_x'].extend([cs[it].ends[0][0] for it in bac_it])
    bac_features_dict['Node_x1_y'].extend([cs[it].ends[0][1] for it in bac_it])
    bac_features_dict['Node_x2_x'].extend([cs[it].ends[1][0] for it in bac_it])
    bac_features_dict['Node_x2_y'].extend([cs[it].ends[1][1] for it in bac_it])

    # Surface area of a capsule:
    # S = 2πr(2r + a)
    bac_features_dict['AreaShape_Area'].extend(
        [2 * np.pi * cs[it].radius * (cs[it].length + 2 * cs[it].radius) for it in bac_it])

    bac_features_dict['TrackObjects_ParentImageNumber_50'].extend(this_time_step_bac_parent_image_number)
    bac_features_dict['TrackObjects_ParentObjectNumber_50'].extend(this_time_step_bac_parent_obj_number)
    bac_features_dict["TrackObjects_Label_50"].extend(this_time_step_bac_label)
    bac_features_dict['CellAge'].extend(this_time_step_bac_cell_age)
    bac_features_dict['id'].extend([cs[it].id for it in bac_it])

    # cell Type
    # In CellModeller CellTypes are: 0,1,2,3,...
    for index, cell_type_id in enumerate(cell_type_mapping.values()):
        bac_features_dict['Type'][index].extend(
            [int(cs[it].cellType == cell_type_id) for it in bac_it]
        )

    return bac_features_dict, bacteria_labels_dict, neighbor_records


def annotate_parent_daughter_relationship(df):
    """
    Annotates a bacterial tracking DataFrame with parent-daughter relationships and division events.

    This function detects and labels division events by identifying unexpected lineage beginnings,
    assigning parent IDs to newly divided (daughter) cells, and marking division time steps. It also
    computes daughter-to-mother length ratios and flags unexpected lineage terminations.

    Parameters:
        df (pd.DataFrame): A DataFrame containing tracked bacterial cell data. It must include the following columns:
            - 'id': unique track ID for each cell
            - 'ImageNumber', 'ObjectNumber': object indexing info
            - 'TrackObjects_ParentImageNumber_50', 'TrackObjects_ParentObjectNumber_50': parent linkage info
            - 'AreaShape_MajorAxisLength': length measurement of the cell
            - 'CellAge': age of the cell (0 for new cells)

    Returns:
        pd.DataFrame: The input DataFrame with added columns:
            - 'Unexpected_Beginning': True if cell appears without a known parent mid-sequence
            - 'Unexpected_End': True if a mother cell doesn't divide before its track ends
            - 'parent_id': ID of the mother cell (if applicable)
            - 'Division_TimeStep': frame number when division occurred
            - 'divideFlag': True if the cell divided
            - 'Daughter_Mother_Length_Ratio': ratio of daughter to mother cell lengths
            - 'Total_Daughter_Mother_Length_Ratio': sum of both daughters' lengths divided by mother length
            - 'Max_Daughter_Mother_Length_Ratio': max daughter-to-mother length ratio
    """

    df['index'] = df.index

    df['Unexpected_Beginning'] = False
    df['Unexpected_End'] = False
    df['parent_id'] = 0
    df['Division_TimeStep'] = 0
    df['divideFlag'] = False
    df['Daughter_Mother_Length_Ratio'] = np.nan
    df['Total_Daughter_Mother_Length_Ratio'] = np.nan
    df['Max_Daughter_Mother_Length_Ratio'] = np.nan

    ub_bac = df.loc[(df['TrackObjects_ParentImageNumber_50'] == 0) &
                    (df['TrackObjects_ParentObjectNumber_50'] == 0) & (df['ImageNumber'] != 1)]

    daughters_bac = df.loc[(df['TrackObjects_ParentImageNumber_50'] != 0) &
                           (df['TrackObjects_ParentObjectNumber_50'] != 0) & (df['CellAge'] == 0)]

    df.loc[ub_bac.index, ['Unexpected_Beginning']] = [True]

    sel_cols = ['index', 'ImageNumber', 'ObjectNumber', 'id', 'TrackObjects_ParentImageNumber_50',
                'TrackObjects_ParentObjectNumber_50', 'AreaShape_MajorAxisLength']
    daughters_mother_bac = daughters_bac[sel_cols].merge(df[sel_cols],
                                                         left_on=['TrackObjects_ParentImageNumber_50',
                                                                  'TrackObjects_ParentObjectNumber_50'],
                                                         right_on=['ImageNumber', 'ObjectNumber'], how='inner',
                                                         suffixes=('_daughter', '_mother'))

    daughters_mother_bac['Daughter_Mother_Length_Ratio'] = \
        (daughters_mother_bac['AreaShape_MajorAxisLength_daughter'] /
         daughters_mother_bac['AreaShape_MajorAxisLength_mother'])

    daughters_mother_bac['Division_TimeStep'] = \
        daughters_mother_bac.groupby('id_mother')['ImageNumber_daughter'].transform('max')
    daughters_mother_bac['Total_Daughter_Mother_Length_Ratio'] = \
        daughters_mother_bac.groupby('id_mother')['AreaShape_MajorAxisLength_daughter'].transform('sum')
    daughters_mother_bac['Max_Daughter_Mother_Length_Ratio'] = \
        daughters_mother_bac.groupby('id_mother')['AreaShape_MajorAxisLength_daughter'].transform('max')

    mothers_bac = daughters_mother_bac.drop_duplicates(subset='id_mother', keep='last')

    # Only the first time step of each daughter bacterium has parent_id.
    df.loc[daughters_mother_bac['index_daughter'].values, 'parent_id'] = daughters_mother_bac['id_mother'].values
    df['parent_id'] = df.groupby('id')['parent_id'].transform('sum')

    df.loc[daughters_mother_bac['index_daughter'].values, 'Daughter_Mother_Length_Ratio'] = \
        daughters_mother_bac['Daughter_Mother_Length_Ratio'].values

    # Only the last time step of each mother bacterium has divideFlag.
    df.loc[mothers_bac['index_mother'].values, 'divideFlag'] = True
    df['divideFlag'] = df.groupby('id')['divideFlag'].transform('any')

    df.loc[mothers_bac['index_mother'].values, 'Division_TimeStep'] = mothers_bac['Division_TimeStep'].values
    df['Division_TimeStep'] = df.groupby('id')['Division_TimeStep'].transform('sum')

    df.loc[mothers_bac['index_mother'].values, 'Total_Daughter_Mother_Length_Ratio'] = \
        mothers_bac['Total_Daughter_Mother_Length_Ratio'].values
    df.loc[mothers_bac['index_mother'].values, 'Max_Daughter_Mother_Length_Ratio'] = \
        mothers_bac['Max_Daughter_Mother_Length_Ratio'].values

    last_time_point_bac = df.groupby('id', group_keys=False).tail(1)
    ue_bac = last_time_point_bac.loc[last_time_point_bac['divideFlag'] == False]
    df.loc[ue_bac['index'].values, 'Unexpected_End'] = True

    df = df.drop('index', axis=1)

    return df


def process_simulation_directory(input_directory, cell_type_mapping, output_directory, assign_cell_type=True,
                                 use_grandmother_as_parent=False, find_neighbors=True):
    """
    Processes a directory of CellModeller pickle files to extract cell features and track relationships.

    Parameters:
    - input_directory (str): Path to the directory containing CellModeller `.pickle` files.
    - cell_type_mapping (dict): Dictionary mapping cell type names to CellModeller IDs.
    - output_directory (str): Path to save output CSV files.
    - assign_cell_type (bool): If True, infer and assign cell types to tracked bacteria.
    - use_grandmother_as_parent (bool):
        If True, approximates the parent bacterium by selecting the nearest disappeared bacterium
        from the previous time step. Useful when large time step gaps cause the actual parent
        to no longer appear in the previous step.
    - find_neighbors (bool):
        If True, computes neighbor relationships between bacteria based on spatial proximity.
        Specifically, it defines two bacteria as neighbors if their expanded pixel boundaries
        touch — consistent with CellProfiler's "MeasureObjectNeighbors" module.
        Note: Even if this parameter is set to False, neighbor relationships will still be computed
        if the original pickle files do not already include neighbor data.

    Returns:
    - None. Writes two CSV files to the output directory:
        - 'Objects properties.csv' containing feature data.
        - 'Object relationships.csv' containing neighbor relationships.
    """

    # firstly I create a dictionary and append extracted features to corresponding key list
    dataframe = {'id': [], 'ImageNumber': [], 'ObjectNumber': [], 'Type': [], 'AreaShape_Area': [],
                 'AreaShape_Center_X': [], 'AreaShape_Center_Y': [], 'AreaShape_MajorAxisLength': [],
                 'AreaShape_MinorAxisLength': [], 'AreaShape_Orientation': [], 'Node_x1_x': [], 'Node_x1_y': [],
                 'Node_x2_x': [], 'Node_x2_y': [], 'CellAge': [], 'TrackObjects_ParentImageNumber_50': [],
                 'TrackObjects_ParentObjectNumber_50': [], 'validID': [], 'ImageName': [], 'TrackObjects_Label_50': []}

    rows_neighbors = []

    # keys: bacteria id
    # values: assigned bacteria labels
    bacteria_id_label = {}

    if assign_cell_type:
        dataframe['Type'] = [[] for _ in cell_type_mapping]

    # read pickle files
    path = input_directory + "/*.pickle"
    filename_list = [filename for filename in sorted(glob.glob(path))]

    for cnt, filename in enumerate(filename_list):

        # read current pickle file
        current_bacteria_info = pickle.load(open(filename, 'rb'))
        time_step = cnt + 1

        # read previous pickle file
        if cnt > 0:
            previous_bacteria = pickle.load(open(filename_list[cnt - 1], 'rb'))
        else:
            previous_bacteria = {'lineage': [], 'cellStates': []}

        # extract features
        dataframe, bacteria_id_label, rows_neighbors = \
            extract_bacterial_features(time_step, current_bacteria_info, previous_bacteria, dataframe,
                                       bacteria_id_label,
                                       rows_neighbors, cell_type_mapping, use_grandmother_as_parent)

    # create data frame
    df = pd.DataFrame({'ImageName': dataframe['ImageName'],
                       'ImageNumber': dataframe['ImageNumber'], 'ObjectNumber': dataframe['ObjectNumber'],
                       'AreaShape_Area': dataframe['AreaShape_Area'],
                       'AreaShape_Center_X': dataframe['AreaShape_Center_X'],
                       'AreaShape_Center_Y': dataframe['AreaShape_Center_Y'],
                       'AreaShape_MajorAxisLength': dataframe['AreaShape_MajorAxisLength'],
                       'AreaShape_MinorAxisLength': dataframe['AreaShape_MinorAxisLength'],
                       'AreaShape_Orientation': dataframe['AreaShape_Orientation'],
                       'Location_Center_X': dataframe['AreaShape_Center_X'],
                       'Location_Center_Y': dataframe['AreaShape_Center_Y'],
                       "TrackObjects_Label_50": dataframe["TrackObjects_Label_50"],
                       'TrackObjects_ParentImageNumber_50': dataframe['TrackObjects_ParentImageNumber_50'],
                       'TrackObjects_ParentObjectNumber_50': dataframe['TrackObjects_ParentObjectNumber_50'],
                       'id': dataframe['id'],
                       'CellAge': dataframe['CellAge'], 'Node_x1_x': dataframe['Node_x1_x'],
                       'Node_x1_y': dataframe['Node_x1_y'], 'Node_x2_x': dataframe['Node_x2_x'],
                       'Node_x2_y': dataframe['Node_x2_y']})

    if assign_cell_type:
        cell_type_names = cell_type_mapping.keys()
        for cnt, CellType in enumerate(cell_type_names):
            df[CellType] = dataframe['Type'][cnt]

        df['cellType'] = df[cell_type_names].idxmax(axis=1)
        df.loc[df[cell_type_names].max(axis=1) == 0, 'cellType'] = 0
        df.loc[(df[cell_type_names] == 1).sum(axis=1) > 1, 'cellType'] = 3

    df['LifeHistory'] = df.groupby('id')['id'].transform('size')
    df = annotate_parent_daughter_relationship(df)

    # write to csv
    df.to_csv(output_directory + "/Objects properties.csv", index=False)

    # now we should check dataframes
    if len(rows_neighbors) == 0 or find_neighbors:
        # This means that the neighbors were not found in the CellModeler
        print('Finding Neighbors')
        df_neighbors = neighbor_finders(input_directory)
    else:
        df_neighbors = pd.DataFrame(rows_neighbors, columns=['Image Number', 'First Object id',
                                                             'Second Object id'])

    # Convert 'Image Number' and 'First Object id' columns to string in both dataframes
    df_neighbors['Image Number'] = df_neighbors['Image Number'].astype(str)
    df_neighbors['First Object id'] = df_neighbors['First Object id'].astype(str)
    df_neighbors['Second Object id'] = df_neighbors['Second Object id'].astype(str)

    df['ImageNumber'] = df['ImageNumber'].astype(str)
    df['id'] = df['id'].astype(str)

    df_merge = df_neighbors.merge(df, left_on=['Image Number', 'First Object id'],
                                  right_on=['ImageNumber', 'id'], how='inner', suffixes=('_node', '_info'))

    df_final_merge = df_merge.merge(df, left_on=['Image Number', 'Second Object id'],
                                    right_on=['ImageNumber', 'id'], how='inner',
                                    suffixes=('_node_info', '_neighbor_info'))

    findl_df_neighbors = \
        df_final_merge[['Image Number', 'ObjectNumber_node_info', 'ObjectNumber_neighbor_info']].copy()

    findl_df_neighbors = findl_df_neighbors.rename({'Image Number': 'First Image Number',
                                                    'ObjectNumber_node_info': 'First Object Number',
                                                    'ObjectNumber_neighbor_info': 'Second Object Number'}, axis=1)

    findl_df_neighbors.insert(0, 'Relationship', 'Neighbors')
    findl_df_neighbors.insert(3, 'Second Image Number', findl_df_neighbors['First Image Number'].values)

    findl_df_neighbors.to_csv(output_directory + "/Object relationships.csv", index=False)
