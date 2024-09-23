import os
import cv2  # conda install -c conda-forge opencv
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def argnotnan(arr):
    non_nan_indices = np.where(~np.isnan(arr))[0]
    non_nan_values = arr[non_nan_indices]
    return non_nan_values


def set_spines_visible(ax=None, color="k", ls="-"):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linestyle(ls)

def generate_video_with_text(vid_params, img_fp, text_params, bbox=None):
    """ Generate a video with bbox cut-out if provided """
    video = cv2.VideoWriter(**vid_params)
    
    if bbox is not None:
        xmin, xmax, ymin, ymax = bbox
    else:
        xmin, xmax, ymin, ymax = 0, vid_params['frameSize'][0], 0, vid_params['frameSize'][1]
    
    for nth, fp in enumerate(img_fp):
        frame = cv2.imread(fp, 1)
        if frame is None:
            print(f"Warning: Frame {fp} could not be read.")
            continue

        frame = frame[ymin:ymax, xmin:xmax]
        cv2.putText(frame, f"frame_{str(nth)}", **text_params)
        video.write(frame)
    
    video.release()
    cv2.destroyAllWindows()
    

def plot_scalebar(ax, mpp, width_microns, position=(10, 10), color="k", linewidth=2, fontsize=9):
    """
    Plots a scalebar on a Matplotlib Axes object.

    Parameters:
    ax (matplotlib.axes.Axes): The Axes object to plot the scalebar on.
    mpp (float): Micron per pixel value.
    width_microns (float): Desired scalebar width in microns.
    position (tuple): (x, y) position for the scalebar.
    color (str): Color of the scalebar. Default is white.
    linewidth (int): Line width of the scalebar. Default is 2.
    """
    width_pixels = width_microns * mpp
    scalebar = Line2D(
        [position[0], position[0] + width_pixels], 
        [position[1], position[1]], 
        color=color, 
        linewidth=linewidth
    )
    ax.add_line(scalebar)
    
    # Add text label for the scalebar
    if fontsize:
        ax.text(
            position[0], position[1] - 10, 
            f'{width_microns} µm', 
            color=color, 
            verticalalignment='top', 
            horizontalalignment='left',
            fontsize=fontsize
        )
    

def create_scalarmappable(colormap, data, vminmax=None):
    """
    Creates a ScalarMappable and corresponding colors for given data.

    Parameters:
    colormap (str or Colormap): Name of the colormap to use or a Colormap object.
    data (array-like): Data to be mapped to colors.
    vminmax (tuple, optional): Tuple specifying (vmin, vmax) for normalization. If None, use min and max of data.

    Returns:
    colors (array-like): Colors corresponding to the data values.
    smap (matplotlib.cm.ScalarMappable): ScalarMappable object.
    """
    if vminmax is None:
        vmin, vmax = np.min(data), np.max(data)
    else:
        vmin, vmax = vminmax

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    if isinstance(colormap, str):
        cmap = plt.get_cmap(colormap)
    else:
        cmap = colormap
        
    colors = cmap(norm(data))
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    return colors, smap



def calculate_average_image(images):
    image_stack = np.stack(images, axis=-1)
    average_image = np.mean(image_stack, axis=-1).astype(np.float32)
    return average_image


def subtract_average_image(images, average_image):
    variation_images = []
    for image in images:
        variation_image = (image.astype(np.float32) - average_image).astype(np.float32)
        variation_images.append(variation_image)
    return variation_images


def translate_images_to_mean_keypoint(images, keypoints):
    """
    Translates a list of grayscale images so that the keypoints align to their mean position.
    
    Parameters:
    images (list of numpy.ndarray): List of grayscale images.
    keypoints (list of tuple): List of (x, y) tuples representing keypoints in the images.
    
    Returns:
    list of numpy.ndarray: List of translated images.
    """
    # Calculate the mean x and y coordinates
    keypoints_array = np.array(keypoints)
    mean_x, mean_y = np.mean(keypoints_array, axis=0)
    
    translated_images = []
    
    for image, (x, y) in zip(images, keypoints):
        # Calculate translation values
        dx = int(mean_x) - int(x)
        dy = int(mean_y) - int(y)
        
        # Define the translation matrix
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        
        # Translate the image
        translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
        translated_images.append(translated_image)
    
    return translated_images



def extract_segments_aligned(signal, troughs, before=90, after=90):
    # Calculate the length of the resulting array
    num_segments = len(troughs)
    segment_length = before + after
    # Create an array filled with NaN values
    aligned_segments = np.full((num_segments, segment_length), np.nan)

    # Paste windowed signals around the troughs into the array
    for i, trough in enumerate(troughs):
        start_idx = max(0, trough - before)
        end_idx = min(len(signal), trough + after)
        segment = signal[start_idx:end_idx]
        aligned_segments[i, (before - (trough - start_idx)):(before + (end_idx - trough))] = segment

    return aligned_segments


def autocorr_with_interpolation(x):

    # Interpolate NaN values
    valid_indices = np.arange(len(x))[~np.isnan(x)]
    interp_func = interp1d(valid_indices, x[~np.isnan(x)], kind='linear', fill_value='extrapolate')
    x_interpolated = interp_func(np.arange(len(x)))
    
    # Calculate autocorrelation of interpolated signal
    autocorr_result = np.correlate(x_interpolated, x_interpolated, mode='full')
    autocorr_result /= np.max(autocorr_result)  # Normalize autocorrelation
    
    return autocorr_result


def get_contours_from_segmasks(segmentation_mask_filepaths):
    contour_list = []

    for nth in range(len(segmentation_mask_filepaths)):
        m = segmentation_mask_filepaths[nth]
        im = cv2.imread(m)
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(nth, contours)
        
        if contours:
            contour_list.append(contours[-1])
        else:
            contour_list.append(tuple())

    return contour_list


def get_contour_stats(contour_list, min_area=40, min_contour_points=10):
    """_summary_

    Args:
        contour_list (_type_): _description_
        min_area (int, optional): _description_. Defaults to 40.
        min_contour_points (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    
    stats_dict = {
        "centroid_x": [],
        "centroid_y": [],
        "area": [],
        "min_area": [],
        "extent": [],
        "perimeter": [],
        "aspect_ratio": [],
        "orientation": [],
        #"centroid_dist": []
    }

    for nth, cnt in enumerate(contour_list):
        #print(nth)
        if np.squeeze(cnt).size <= min_contour_points:  # require at least n=5 x and n=5 y points for contour
            for key in stats_dict:
                stats_dict[key].append(np.nan)
            continue

        M = cv2.moments(cnt)

        # Centroids
        centroid_x = int(M['m10'] / M['m00'])
        centroid_y = int(M['m01'] / M['m00'])
        stats_dict["centroid_x"].append(centroid_x)
        stats_dict["centroid_y"].append(centroid_y)

        # Area
        area = cv2.contourArea(cnt)
        stats_dict["area"].append(area)

        # Perimeter
        perimeter = cv2.arcLength(cnt, True)
        stats_dict["perimeter"].append(perimeter)

        # Bounding Rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        stats_dict["min_area"].append(rect_area)

        # Extent
        y_indices, x_indices = np.squeeze(cnt)[:, 0], np.squeeze(cnt)[:, 1]
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        extent = max(y_max - y_min, x_max - x_min)
        stats_dict["extent"].append(extent)

        # Aspect Ratio
        aspect_ratio = float(w) / h
        stats_dict["aspect_ratio"].append(aspect_ratio)

        # Orientation
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        except cv2.error:
            angle = np.nan
        stats_dict["orientation"].append(angle)

    # Convert dictionary to DataFrame
    df = pd.DataFrame(stats_dict)

    # Replace with NaNs for all measures and time
    if (df["area"] < min_area).any():
        df[df["area"] < min_area] = np.nan

    # Calculate centroid distance
    df["centroid_dist"] = np.sqrt((df["centroid_x"].diff() ** 2) + (df["centroid_y"].diff() ** 2))

    return df


def rotate_masks_and_contours(centers, rot_angles, cmask_filepaths):
    """_summary_

    Args:
        centers (_type_): _description_
        rot_angles (_type_): _description_
        cmask_filepaths (_type_): _description_
        img_filepaths (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    rot_masks = []
    rot_contours = []
    
    for center, rot_angle, cmask in zip(centers, rot_angles, cmask_filepaths):
        if not np.isnan(rot_angle):
            # ensure integer values
            center = (int(center[0]), int(center[1]))
            
            # Rot mat
            rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rot_angle, scale=1)
            cmask = cv2.cvtColor(cv2.imread(cmask, 1), cv2.COLOR_BGR2GRAY)

            # Rot
            rot_mask = cv2.warpAffine(src=cmask, M=rotate_matrix, dsize=(cmask.shape[1], cmask.shape[0]))
            _, thresh = cv2.threshold(rot_mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Return
            rot_masks.append(rot_mask)
            rot_contours.append(contours[-1])
            
    return rot_masks, rot_contours
    

def rotate_masks_and_images(centers, rot_angles, cmask_filepaths, img_filepaths):
    """_summary_

    Args:
        centers (_type_): _description_
        rot_angles (_type_): _description_
        cmask_filepaths (_type_): _description_
        img_filepaths (_type_): _description_

    Returns:
        _type_: _description_
    """
    rot_masks = []
    rot_images = []

    for center, rot_angle, cmask, img_filepath in zip(centers, rot_angles, cmask_filepaths, img_filepaths):
        if not np.isnan(rot_angle):
            # Rot mat
            rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rot_angle, scale=1)
            cmask = cv2.cvtColor(cv2.imread(cmask, 1), cv2.COLOR_BGR2GRAY)

            # Rot mask
            rot_mask = cv2.warpAffine(src=cmask, M=rotate_matrix, dsize=(cmask.shape[1], cmask.shape[0]))

            # Read and rot image
            img = cv2.cvtColor(cv2.imread(img_filepath, 1), cv2.COLOR_BGR2RGB)
            rot_img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(img.shape[1], img.shape[0]))

            # Append
            rot_masks.append(rot_mask)
            rot_images.append(rot_img)

    return rot_masks, rot_images


def find_top_bottom_points_from_contours(contours):
    """_summary_

    Args:
        contours (_type_): _description_

    Returns:
        _type_: _description_
    """
    top_bottom_points = []

    for contour in contours:
        if len(contour) > 0:
            # Find the point with the minimum and maximum y-coordinate
            topmost_point = tuple(contour[contour[:, :, 1].argmin()][0])
            bottommost_point = tuple(contour[contour[:, :, 1].argmax()][0])
            distance = np.linalg.norm(np.array(topmost_point) - np.array(bottommost_point))

            top_bottom_points.append({"top_x": topmost_point[0], "top_y": topmost_point[1],
                                       "bottom_x": bottommost_point[0], "bottom_y": bottommost_point[1],
                                       "distance": distance})
        else:
            # If the contour is empty, append None for topmost and bottommost points
            top_bottom_points.append({"top_x": None, "top_y": None, "bottom_x": None, "bottom_y": None, "distance": None})
            
    return top_bottom_points


def transform_points_to_original_space(df, centers, rot_angles):
    " transform top and bottom points back to original space "
    inverse_rotated_points = []

    for i, row in df.iterrows():
        top_point = (row['top_x'], row['top_y'])
        bottom_point = (row['bottom_x'], row['bottom_y'])
        if np.isnan(centers[i]).any():
            continue
        else:
            center = centers[i]
            center = (int(center[0]), int(center[1]))
            rot_angle = rot_angles[i]

            # Obtain the inverse rotation matrix
            inverse_rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-rot_angle, scale=1)
            inverse_rotate_matrix = cv2.invertAffineTransform(inverse_rotate_matrix)

            # Apply the inverse rotation matrix to the rotated top and bottom points
            homogeneous_top_point = np.array([top_point[0], top_point[1], 1])
            inverse_rotated_top_point = np.dot(inverse_rotate_matrix, homogeneous_top_point)

            homogeneous_bottom_point = np.array([bottom_point[0], bottom_point[1], 1])
            inverse_rotated_bottom_point = np.dot(inverse_rotate_matrix, homogeneous_bottom_point)

            # Append the inverse rotated points to the list
            inverse_rotated_points.append({
                'top_x': inverse_rotated_top_point[0],
                'top_y': inverse_rotated_top_point[1],
                'bottom_x': inverse_rotated_bottom_point[0],
                'bottom_y': inverse_rotated_bottom_point[1]
            })

    # Convert the list of points to a DataFrame
    inverse_rotated_df = pd.DataFrame(inverse_rotated_points)

    return inverse_rotated_df


def get_video_properties(vid_path: str, vid_filename: str):
    """ get useful video properties """

    vid_filepath = f"{vid_path}{vid_filename}"
    video = cv2.VideoCapture(vid_filepath)

    # Properties
    fps = int(video.get(cv2.CAP_PROP_FPS))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video.release()
    
    return fps, num_frames, width, height


def extract_frames_from_video(vid_path: str, vid_filename: str, frames_filepaths: str, zfill_=5, every_nth_frame=1):
    """ extract image frames from video """
    
    os.makedirs(frames_filepaths, exist_ok=True)
    vid_filepath = f"{vid_path}/{vid_filename}"
    
    video = cv2.VideoCapture(vid_filepath)
    frame_counter = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_counter % every_nth_frame == 0:
            cv2.imwrite(f"{frames_filepaths}/frame_{str(frame_counter).zfill(zfill_)}.jpg", frame)  
        frame_counter += 1
    video.release()



def export_selected_frames(video_path, image_save_dir, selected_frames, fill_value=5):
    """_summary_

    Args:
        video_path (_type_): _description_
        image_save_dir (_type_): _description_
        selected_frames (_type_): _description_
        fill_value (int, optional): _description_. Defaults to 5.
    """
    
    cap = cv2.VideoCapture(video_path)

    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    for frame_index in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if ret:
            filename = f"frame_{str(frame_index).zfill(fill_value)}.jpg"
            cv2.imwrite(os.path.join(image_save_dir, filename), frame)
        else:
            print(f"Error reading frame at index {frame_index}.")

    cap.release()
    

def normalize_contours_by_centers(contours, centers):
    """_summary_

    Args:
        contours (_type_): _description_
        centers (_type_): _description_

    Returns:
        _type_: _description_
    """
    normalized_contours = []

    for contour, center in zip(contours, centers):
        if np.isnan(center).any():
            continue
        else:
            center = (int(center[0]), int(center[1]))
            center_x, center_y = center
            normalized_contour = np.zeros_like(contour)

            # Normalize contour points by subtracting center coordinates
            for i in range(len(contour)):
                normalized_contour[i][0][0] = contour[i][0][0] - center_x
                normalized_contour[i][0][1] = contour[i][0][1] - center_y

            normalized_contours.append(normalized_contour)

    return normalized_contours



def shift_contours(contours, shift_x, shift_y):
    """_summary_

    Args:
        contours (_type_): _description_
        shift_x (_type_): _description_
        shift_y (_type_): _description_

    Returns:
        _type_: _description_
    """
    shifted_contours = []

    for contour in contours:
        shifted_contour = contour + np.array([[[shift_x, shift_y]]])
        shifted_contours.append(shifted_contour)

    return shifted_contours


def get_trajectory_stats(df_trajectories):
    
    """
    Obtain some basic statistics of trajectories to use for
    rejection and selection of trajectories for further processing
    """
    
    ctraj = df_trajectories
    particles = ctraj.particle.unique()

    print("number of particles = number of trajectories:", len(particles))

    stats = dict()
    frames_present = []

    for nth, prt in enumerate(particles):
        sub = ctraj[ctraj.particle == prt]
        
        if not sub.empty:
            x, y = sub.x, sub.y
            dx, dy = x.diff(), y.diff()
            
            stats[int(prt)] = {
                "particle" : prt,
                "x_avg" : x.mean(),
                "y_avg" : y.mean(),
                "x_std" : x.std(),
                "y_std" : y.std(),
                "travel_dist" : sum(np.sqrt(dx[1:] ** 2 + dy[1:] ** 2)),
                "frames_present" : sub["frame"].to_list(), 
                "first_frame" : sub["frame"].min(),
                "last_frame" : sub["frame"].max(),
                "nframes" : len(sub["frame"].unique()),
                "ep_mean" : sub["ep"].mean(),
                "ep_median" : sub["ep"].median(),
                "displacement_first_last" : np.sqrt((x.to_list()[-1] - x.to_list()[0]) ** 2 + (y.to_list()[-1] - y.to_list()[0]) ** 2)
        }
            frames_present.extend(sub["frame"].to_numpy())

    frames_present = np.array(frames_present)
    stats = pd.DataFrame.from_dict(stats, orient="index")
    
    return stats, frames_present
