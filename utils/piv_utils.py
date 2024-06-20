import os
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter

# reject function on dataframes
# save traj and params function

# save_traj_and_parameters(False)>loadin
# save_features_and_parameters(False)>loadin


def plot_bbox(bbox, c="w", ax=None):
    """ plots a bounding box """
    if ax is None:
        ax = plt.gca()
    
    ax.plot([bbox[0], bbox[1], bbox[1], bbox[0], bbox[0]],
            [bbox[2], bbox[2], bbox[3], bbox[3], bbox[2]], "--", color=c, lw=1)


# def construct_binned_velocity_maps(positions, velocities, image_height, image_width, bin_size):
#     """
#     Constructs binned velocity maps for the x and y components.

#     Parameters:
#     - positions: numpy array of shape (N, 2) containing x and y coordinates of particles.
#     - velocities: numpy array of shape (N, 2) containing x and y velocity components of particles.
#     - image_height: height of the image.
#     - image_width: width of the image.
#     - bin_size: size of the bins.

#     Returns:
#     - vx_map: 2D numpy array representing the binned x velocity components.
#     - vy_map: 2D numpy array representing the binned y velocity components.
#     - x_edges: bin edges along the x-axis.
#     - y_edges: bin edges along the y-axis.
#     """
#     x, y = positions[:, 0], positions[:, 1]
#     vx, vy = velocities[:, 0], velocities[:, 1]

#     x_bins = np.arange(0, image_width + bin_size, bin_size)
#     y_bins = np.arange(0, image_height + bin_size, bin_size)

#     vx_map, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=vx)
#     vy_map, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=vy)

#     # Count the number of particles in each bin to average the velocities
#     count_map, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])

#     vx_map = vx_map / np.maximum(count_map, 1)  # Avoid division by zero
#     vy_map = vy_map / np.maximum(count_map, 1)  # Avoid division by zero

#     return vx_map, vy_map, x_edges, y_edges


def calculate_vorticity_binned(vx_map, vy_map, x_edges, y_edges):
    """
    Calculates the vorticity from binned velocity maps.

    Parameters:
    - vx_map: 2D numpy array representing the binned x velocity components.
    - vy_map: 2D numpy array representing the binned y velocity components.
    - x_edges: bin edges along the x-axis.
    - y_edges: bin edges along the y-axis.

    Returns:
    - vorticity: 2D numpy array representing the vorticity.
    """
    dx = (x_edges[1] - x_edges[0])
    dy = (y_edges[1] - y_edges[0])

    dvdx = np.gradient(vy_map, axis=1) / dx
    dudx = np.gradient(vx_map, axis=0) / dy

    vorticity = dvdx - dudx

    return vorticity


def reindex_calculate_displacement_and_stack(df, full_frames):
    particles = df['particle'].unique()
    displacements = []

    for particle in particles:
        particle_df = df[df['particle'] == particle].set_index('frame')
        particle_df = particle_df.reindex(full_frames, fill_value=np.nan)
        
        # Interpolate missing values
        particle_df['x'] = particle_df['x'].interpolate(method='linear')
        particle_df['y'] = particle_df['y'].interpolate(method='linear')
        
        # Calculate displacement
        dx = particle_df['x'].diff()
        dy = particle_df['y'].diff()
        displacement = np.sqrt(dx**2 + dy**2)
        
        # Preserve NaNs where data is missing
        displacement[particle_df['x'].isna() | particle_df['y'].isna()] = np.nan
        displacements.append(displacement.values)
    
    displacement_array = np.array(displacements)
    
    return displacement_array



def interpolate_2d_map(data_map, x_edges, y_edges, scale_factor=2, method='linear', smooth=False, sigma=1):
    """
    Interpolates a 2D map to a higher resolution and optionally applies Gaussian smoothing.

    Parameters:
    - data_map: 2D numpy array representing the map to be interpolated.
    - x_edges: bin edges along the x-axis.
    - y_edges: bin edges along the y-axis.
    - scale_factor: factor by which to scale the number of bins.
    - method: interpolation method to use ('linear', 'nearest').
    - smooth: whether to apply Gaussian smoothing.
    - sigma: standard deviation for Gaussian kernel if smoothing is applied.

    Returns:
    - interpolated_map: 2D numpy array representing the interpolated map.
    - new_x_edges: new bin edges along the x-axis.
    - new_y_edges: new bin edges along the y-axis.
    """
    # Calculate the new dimensions
    new_shape = (data_map.shape[0] * scale_factor, data_map.shape[1] * scale_factor)

    # Interpolate the map to the new shape
    if method == 'linear':
        interpolated_map = zoom(data_map, scale_factor, order=1)
    elif method == 'nearest':
        interpolated_map = zoom(data_map, scale_factor, order=0)
    else:
        raise ValueError("Method not supported. Use 'linear' or 'nearest'.")

    # Apply Gaussian smoothing if requested
    if smooth:
        interpolated_map = gaussian_filter(interpolated_map, sigma=sigma)

    # Generate new edges based on the original edges and scale factor
    new_x_edges = np.linspace(x_edges[0], x_edges[-1], new_shape[1] + 1)
    new_y_edges = np.linspace(y_edges[0], y_edges[-1], new_shape[0] + 1)

    return interpolated_map, new_x_edges, new_y_edges



def construct_velocity_map(particle_positions, velocities, image_height, image_width, bin_size, threshold_count=20):
    """
    Constructs a 2D velocity map from particle positions and their associated velocities.

    Parameters:
    - particle_positions: a numpy array of shape (N, 2) representing particle positions.
    - velocities: a numpy array of shape (N,) representing velocities at each particle position.
    - image_height: height of the image (H).
    - image_width: width of the image (W).
    - bin_size: size of the bins for the histogram (assumed to be the same for both dimensions).
    - threshold_count: minimum number of particles required in a bin to consider its velocity.

    Returns:
    - velocity_map: 2D numpy array representing the average velocity map.
    - x_edges: bin edges along the x-axis.
    - y_edges: bin edges along the y-axis.
    """
    num_bins_x = image_width // bin_size
    num_bins_y = image_height // bin_size

    x_coords, y_coords = particle_positions[:, 0], particle_positions[:, 1]

    # Create a 2D histogram for counting particles in each bin
    counts, x_edges, y_edges = np.histogram2d(
        x_coords, y_coords, bins=[num_bins_x, num_bins_y],
        range=[[0, image_width], [0, image_height]]
    )
    
    # Create a 2D histogram for summing velocities in each bin
    velocity_sum, _, _ = np.histogram2d(
        x_coords, y_coords, bins=[num_bins_x, num_bins_y],
        range=[[0, image_width], [0, image_height]], weights=velocities
    )
    
    # Compute the average velocity by dividing the summed velocities by the counts
    with np.errstate(divide='ignore', invalid='ignore'):
        velocity_map = np.true_divide(velocity_sum, counts)
        velocity_map[~np.isfinite(velocity_map)] = 0  # Replace NaN and inf with 0
    
    # Apply threshold to filter out bins with fewer particles than threshold_count
    velocity_map[counts < threshold_count] = 0
    
    return velocity_map, x_edges, y_edges



def construct_occupancy_map(particle_positions, image_height, image_width, bin_size):
    num_bins_x = image_width // bin_size
    num_bins_y = image_height // bin_size
    x_coords, y_coords = particle_positions[:, 0], particle_positions[:, 1]
    occupancy_map, x_edges, y_edges = np.histogram2d(
        x_coords, y_coords, bins=[num_bins_x, num_bins_y],
        range=[[0, image_width], [0, image_height]]
    )
    return occupancy_map, x_edges, y_edges


def smooth_velocity_field(vx, vy, sigma):
    """Smooth the velocity field using a Gaussian filter."""
    vx_smoothed = gaussian_filter(vx, sigma=sigma)
    vy_smoothed = gaussian_filter(vy, sigma=sigma)
    return vx_smoothed, vy_smoothed

def resample_velocity_field(vx, vy, x_bins, y_bins, resample_factor):
    """Resample the velocity field and bins to a coarser grid."""
    vx_resampled = zoom(vx, resample_factor)
    vy_resampled = zoom(vy, resample_factor)
    x_bins_resampled = zoom(x_bins, resample_factor)
    y_bins_resampled = zoom(y_bins, resample_factor)
    return vx_resampled, vy_resampled, x_bins_resampled, y_bins_resampled

def calculate_vorticity(vx, vy, dx, dy):
    """Calculate the vorticity from the velocity field."""
    dvx_dy, dvx_dx = np.gradient(vx, dy, dx)
    dvy_dy, dvy_dx = np.gradient(vy, dy, dx)
    vorticity = dvx_dy - dvy_dx
    return vorticity


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
