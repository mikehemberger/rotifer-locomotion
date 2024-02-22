import cv2  # conda install -c conda-forge opencv
import numpy as np
import pandas as pd

def get_contours_from_segmasks(segmentation_mask_filepaths):
    contour_list = []

    for nth in range(len(segmentation_mask_filepaths)):
        m = segmentation_mask_filepaths[nth]
        im = cv2.imread(m)
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_list.append(contours[-1])

    return contour_list


def get_contour_stats(contour_list):
    areas = []   
    aspect_ratios = []
    orients = []
    min_rect_area = []
    extents = []
    perimeters = []
    centroids_x, centroids_y = [], []
    #diff_centroids = []

    for cnt in contour_list:
        M = cv2.moments(cnt)
        # Centroids
        centroids_x.append(int(M['m10']/M['m00']))
        centroids_y.append(int(M['m01']/M['m00']))

        # Area
        area = cv2.contourArea(cnt)
        areas.append(area)
        # perimeter
        perimeters.append(cv2.arcLength(cnt, True))
        # extent
        x,y,w,h = cv2.boundingRect(cnt)
        rect_area = w * h
        #extent = float(area) / rect_area
        #extents.append(extent)
        # aspect ratio
        #aspect_ratios.append(float(w)/h)
        # Orients
        (x,y), (MA,ma), angle = cv2.fitEllipse(cnt)
        orients.append(angle)
        # rotated bounding box
        (x,y), (MA,ma), angle = cv2.minAreaRect(cnt)
        min_rect_area.append(w * h)
        # extent modified
        extents.append(float(area) / (w * h))
        aspect_ratios.append(float(w)/h)
    
    # Dataframe
    #x_frames = np.linspace(0, len(contour_list), len(contour_list)).astype("int")
    df = pd.DataFrame(centroids_x, columns=["centroid_x"])
    
    # Stats
    #df["centroid_x"] = centroids_x
    df["centroid_y"] = centroids_y
    df["area"] = areas
    df["min_area"] = min_rect_area
    df["extent"] = extents
    df["perimeter"] = perimeters
    df["aspect_ratio"] = aspect_ratios
    df["orientation"] = orients

    # CLEAN AREA
    # Replace with NaNs for all measures and time
    if (df["area"] < 5).any():
        remove_idx = df[df["area"] < 5].index.values[0]
        df.iloc[remove_idx, :] = np.nan
    
    df["centroid_dist"] = np.sqrt((df["centroid_x"].diff() ** 2) + (df["centroid_y"].diff() ** 2))

    return df