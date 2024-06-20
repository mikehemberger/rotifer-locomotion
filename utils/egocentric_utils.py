# egocentric utils
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def plot_colorline(x, y, colors, ax=None, **kwargs):
    """
    Plot a line with colored segments.
    
    Parameters:
    x : list or array
        X coordinates of the data points.
    y : list or array
        Y coordinates of the data points.
    colors : list of tuples
        List of color tuples (e.g., (r, g, b) or (r, g, b, a)) for each segment.
    **kwargs : keyword arguments
        Additional arguments to pass to the plt.plot function (e.g., lw, ms, etc.).
    """
    if len(x) != len(y) or len(x) != len(colors):
        raise ValueError("The lengths of x, y, and colors must be the same.")
    
    if ax is None:
        ax = plt.gca()
            
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], color=colors[i], **kwargs)
    

def create_square_bbox(c_y, c_x, max_extent, image_shape):
    half_extent = max_extent // 2
    y_min = max(0, int(c_y - half_extent))
    y_max = min(image_shape[0], int(c_y + half_extent))
    x_min = max(0, int(c_x - half_extent))
    x_max = min(image_shape[1], int(c_x + half_extent))
    return y_min, y_max, x_min, x_max


def process_images_with_extents(image_paths, centers_of_mass, max_extent, target_size, output_paths=None, grayscale=False):
    results = []

    for idx, (image_path, (c_y, c_x)) in enumerate(zip(image_paths, centers_of_mass)):
        # Skip processing if center of mass is NaN
        if np.isnan(c_y) or np.isnan(c_x):
            results.append(None)  # Append None or some indicator of skipping
            continue
        
        if grayscale:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.cvtColor(cv2.imread(image_path, 1), cv2.COLOR_BGR2RGB)#image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        # Check if the image was successfully read
        if image is None:
            raise FileNotFoundError(f"The file at {image_path} was not found or could not be read.")
        
        # Create the square bounding box
        y_min, y_max, x_min, x_max = create_square_bbox(c_y, c_x, max_extent, image.shape[:2])

        # Crop the image using the bounding box
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Resize the cropped image to the target size
        resized_image = cv2.resize(cropped_image, target_size)
        
        if not grayscale:
            # Convert back to RGB for saving
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
        results.append(resized_image)

        if output_paths:
            output_path = output_paths[idx]
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            success = cv2.imwrite(output_path, resized_image)
            if not success:
                raise IOError(f"Could not save the image to {output_path}")

    return results


def process_images_with_masks(image_paths, mask_paths, centers_of_mass, max_extent, target_size, output_paths=None):
    results = []

    for idx, ((image_path, mask_path), (c_y, c_x)) in enumerate(zip(zip(image_paths, mask_paths), centers_of_mass)):
        # Skip processing if center of mass is NaN
        if np.isnan(c_y) or np.isnan(c_x):
            results.append(None)  # Append None or some indicator of skipping
            continue
        
        # Read the image in RGB
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        # Check if the image was successfully read
        if image is None:
            raise FileNotFoundError(f"The file at {image_path} was not found or could not be read.")
        
        # Read the segmentation mask in grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if the mask was successfully read
        if mask is None:
            raise FileNotFoundError(f"The mask file at {mask_path} was not found or could not be read.")
        
        # Create the square bounding box
        y_min, y_max, x_min, x_max = create_square_bbox(c_y, c_x, max_extent, image.shape[:2])

        # Crop the image and mask using the bounding box
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        # Apply the segmentation mask to the image
        masked_image = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)

        # Change the background to white (where mask == 0)
        white_background = np.ones_like(cropped_image) * 128 # 255
        final_image = np.where(cropped_mask[..., None] == 0, white_background, masked_image)

        # Resize the final image to the target size
        resized_image = cv2.resize(final_image, target_size)
        results.append(resized_image)

        # Save the processed image after resizing
        if output_paths:
            output_path = output_paths[idx]
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            success = cv2.imwrite(output_path, resized_image)
            if not success:
                raise IOError(f"Could not save the image to {output_path}")

    return results