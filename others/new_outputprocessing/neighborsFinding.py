import numpy as np
import cv2
import glob
import pickle
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation, distance_transform_edt
import matplotlib.pyplot as plt
from skimage.transform import resize
import pandas as pd


def generate_new_color(existing_colors, seed=None):
    """
    Generate a new random color that is not in the existing_colors array.
    """
    np.random.seed(seed)
    while True:
        # Generate a random color (R, G, B)
        new_color = np.random.randint(0, 256, size=3)
        # Ensure the color is not black and not already in existing_colors
        if not np.any(np.all(existing_colors == new_color, axis=1)) and not np.all(new_color == [0, 0, 0]):
            return new_color


def draw_bacteria_on_array(bacteria, colors, array, x_min_val, y_min_val, min_length=1, min_width=1, margin=50):
    """
    Draw the bacteria on a numpy array (image) using given colors.
    """
    for idx, bacterium in enumerate(bacteria):
        center, direction, length, width, endpoint1, endpoint2 = (
            bacterium['center'], bacterium['direction'], bacterium['length'], bacterium['width'],
            bacterium['endpoint1'], bacterium['endpoint2'])

        # Ensure minimum values for length and width
        length = max(length, min_length)
        width = max(width, min_width)

        # Adjust the endpoints based on minimum x and y values and a margin
        endpoint1[0] = endpoint1[0] - x_min_val + margin
        endpoint2[0] = endpoint2[0] - x_min_val + margin
        endpoint1[1] = endpoint1[1] - y_min_val + margin
        endpoint2[1] = endpoint2[1] - y_min_val + margin

        # Calculate the perpendicular direction for the width of the bacterium
        perpendicular = np.array([-direction[1], direction[0]])

        # Ensure the endpoints are within image bounds
        endpoint1 = np.maximum(endpoint1, 0).astype(int)
        endpoint2 = np.maximum(endpoint2, 0).astype(int)

        # Define the points that form the shape of the bacterium
        points = np.array([
            endpoint1 - width / 2 * perpendicular,
            endpoint1 + width / 2 * perpendicular,
            endpoint2 + width / 2 * perpendicular,
            endpoint2 - width / 2 * perpendicular
        ], dtype=np.int32)

        # Draw the bacterium polygon on the array using the corresponding color
        color = tuple(int(c) for c in colors[idx])  # Ensure color is a tuple of integers
        cv2.fillPoly(array, [points], color=color)

        # Draw semi-circles at the endpoints to complete the bacterium shape
        if width > 0:
            cv2.ellipse(array, tuple(endpoint1), (int(width / 2), int(width / 2)), 0, 0, 360, color, -1)
            cv2.ellipse(array, tuple(endpoint2), (int(width / 2), int(width / 2)), 0, 0, 360, color, -1)


def downscale_image(image_array, scale_factor):
    """
    Downscale the image by a factor to reduce its size for faster processing.
    """
    return resize(image_array,
                  (image_array.shape[0] // scale_factor, image_array.shape[1] // scale_factor),
                  order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8)


def upscale_labels(labeled_array, original_shape, scale_factor):
    """
    Upscale the labeled array back to the original image shape after processing.
    """
    return resize(labeled_array,
                  original_shape,
                  order=0, anti_aliasing=False, preserve_range=True).astype(np.int32)


def fast_color_to_label(image_array, colors, scale_factor=10):
    """
    Convert the image array's colors to unique labels using downscaling for faster processing.
    Each unique color is mapped to a unique label.
    """
    # Downscale the image to reduce processing time
    downscaled_image = downscale_image(image_array, scale_factor)

    # Initialize the labeled array for the downscaled image
    downscaled_labeled_array = np.zeros(downscaled_image.shape[:2], dtype=np.int32)

    # Label each color in the downscaled image
    for label_id, color in enumerate(colors, start=1):
        # Create a mask where the color matches
        mask = np.all(downscaled_image == color, axis=-1)
        # Apply the label to the masked regions
        downscaled_labeled_array[mask] = label_id

    # Upscale the labeled array back to the original image size
    labeled_array = upscale_labels(downscaled_labeled_array, image_array.shape[:2], scale_factor)

    return labeled_array


def fast_expand_labels(labeled_array):
    """
    Expand labels until they touch each other using the distance transform.
    """
    # Compute the distance transform and nearest label indices
    distances, nearest_label = distance_transform_edt(labeled_array == 0, return_indices=True)

    # Create a copy of the labeled array for expansion
    expanded_labels = labeled_array.copy()

    # Assign the nearest labels to expand the regions
    expanded_labels[distances > 0] = labeled_array[tuple(nearest_label[:, distances > 0])]

    return expanded_labels


def find_neighbors(bacteria, time, rows, margin=50):
    # List to hold the rows of the dataframe

    # Generate unique colors for each bacterium
    existing_colors = np.array([[0, 0, 0]])
    colors = []
    for _ in range(len(bacteria)):
        new_color = generate_new_color(existing_colors)
        colors.append(new_color)
        existing_colors = np.vstack([existing_colors, new_color])

    # Determine the minimum and maximum x and y values
    x_min_val = min(
        [bacterium['endpoint1'][0] for bacterium in bacteria] + [bacterium['endpoint2'][0] for bacterium in
                                                                 bacteria])
    y_min_val = min(
        [bacterium['endpoint1'][1] for bacterium in bacteria] + [bacterium['endpoint2'][1] for bacterium in
                                                                 bacteria])

    max_x = max([bacterium['endpoint1'][0] for bacterium in bacteria] + [bacterium['endpoint2'][0] for bacterium in
                                                                         bacteria])
    max_y = max([bacterium['endpoint1'][1] for bacterium in bacteria] + [bacterium['endpoint2'][1] for bacterium in
                                                                         bacteria])

    # Calculate the dimensions of the image
    image_width = int(max_x - x_min_val + 2 * margin)
    image_height = int(max_y - y_min_val + 2 * margin)
    image_shape = (image_height + 400, image_width + 400, 3)

    # Initialize a blank image array
    image_array = np.zeros(image_shape, dtype=np.uint8)

    # Draw bacteria on the image array using their assigned colors
    draw_bacteria_on_array(bacteria, colors, image_array, x_min_val, y_min_val, margin=margin)

    # Efficiently label the image based on unique colors using downscaling
    labeled_array = fast_color_to_label(image_array, colors)

    # Expand the labels until they touch
    expanded_labels = fast_expand_labels(labeled_array)

    # Initialize the expanded image array for visualization
    expanded_image_array = np.zeros_like(image_array)

    # Convert colors to numpy array for easy indexing
    colors_array = np.array(colors)

    # Apply the colors to the expanded regions
    mask = expanded_labels > 0
    expanded_image_array[mask] = colors_array[expanded_labels[mask] - 1]

    # Display the expanded objects together (commented out for script use)
    # plt.imshow(expanded_rgb_image)
    # plt.title('Expanded Bacteria Until Touch (RGB)')
    # plt.axis('off')
    # plt.show()

    # Identify and report neighboring objects by ID
    id_map = {idx + 1: bacteria[idx]['id'] for idx in range(len(bacteria))}
    neighbors = {}
    for region in regionprops(expanded_labels):
        label_id = region.label
        min_row, min_col, max_row, max_col = region.bbox
        mask = expanded_labels[min_row:max_row, min_col:max_col] == label_id
        expanded_mask = binary_dilation(mask)
        neighboring_labels = np.unique(expanded_labels[min_row:max_row, min_col:max_col][expanded_mask])
        neighboring_labels = neighboring_labels[neighboring_labels != 0]
        neighboring_labels = neighboring_labels[neighboring_labels != label_id]
        neighbors[id_map[label_id]] = [id_map[n] for n in neighboring_labels]

    # Print the neighboring objects by ID
    for bacterium_id, neighbor_list in neighbors.items():
        for neighbor_id in neighbor_list:
            rows.append((time + 1, bacterium_id, neighbor_id))

    return rows


def neighbor_finders(pickle_folder):

    rows = []

    for time, fname in enumerate(glob.glob(pickle_folder + '/*.pickle')):

        bacteria = []

        # Load the data from the pickle file
        data = pickle.load(open(fname, 'rb'))
        cs = data['cellStates']

        x_vals = []
        y_vals = []

        # Extract relevant data from the cell states
        for it in cs:
            bacteria.append(
                {'id': cs[it].id, 'center': np.array(cs[it].pos[:2]), 'direction': np.array(cs[it].dir[:2]),
                 'length': cs[it].length / 0.0144, 'width': cs[it].radius / 0.0144,
                 'endpoint1': cs[it].ends[0][:2] / 0.0144,
                 'endpoint2': cs[it].ends[1][:2] / 0.0144}
            )
            x_vals.extend([cs[it].ends[0][0], cs[it].ends[1][0]])
            y_vals.extend([cs[it].ends[0][1], cs[it].ends[1][1]])

        rows = find_neighbors(bacteria, time, rows)

    # Create a dataframe from the list of tuples
    df = pd.DataFrame(rows, columns=['Image Number', 'First Object id', 'Second Object id'])
    return df
