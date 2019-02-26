from skimage import io, color, measure, draw, img_as_bool
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import cv2
from matplotlib.patches import Ellipse, Circle, Rectangle
from scipy.linalg.basic import LinAlgError

from circle_fitting import *


def process_image(image, image_name):
    N = image['scores'].size
    result = []
    for i in range(N):
        Nroi = image['rois'][i].size
        s = list(image["rois"][i])
        narrow = np.zeros((1, 1))

        # Check the shape compared to the input image
        if s[2]-s[0] == 1:
            s[2] + =1
        if s[2] > image['masks'].shape[0]:
            s[2] -= 1
            s[0] -= 1
        if s[3]-s[1] == 1:
            s[3] += 1
        if s[3] > image['masks'].shape[1]:
            s[3] -= 1
            s[1] -= 1
        aspect = (s[3]-s[1])/(s[2]-s[0])
        athreshold = 3

        # If it's not close to a circle, ignore it
        if aspect > athreshold or aspect < 1/athreshold:
            continue

        shp = (s[3]-s[1], s[2]-s[0])
        area_BB = shp[0]*shp[1]
        res = [0.5*(s[3]+s[1]),
               0.5*(s[0]+s[2]),
               shp[0],
               shp[1],
               image['scores'][i],
               aspect]

        # Fit the centroid 'mass' weighted circle
        try:
            cx, cy, cd = centroid(image['masks'][..., i], s)
            area_centroid = np.pi*(cd/2)**2
            res.extend([cx, cy, cd, area_centroid / area_BB])
        except FitError as e:
            res.extend([None]*4)

        # Fit the least square error circle
        try:
            cx, cy, cd = proc_circle(image['masks'][..., i], s)
            area_circle = np.pi*(cd/2)**2
            res.extend([cx, cy, cd, area_centroid / area_BB])
        except FitError as e:
            res.extend([None]*4)

        # Fit the least square error direct fit ellipse
        try:
            cx, cy, cdx, cdy, cphi = proc_ellipse(image['masks'][..., i], s)
            area_circle = np.pi * (cdx / 2) * (cdy / 2)
            res.extend([cx, cy, cdx, cdy, cphi, area_centroid / area_BB])
        except FitError as e:
            res.extend([None]*6)

        result.append(res)

    columns = ['bbCx',  # Bounding box centre X
               'bbCy',   # Bounding box centre Y
               'bbNx',  # Bounding box width
               'bbNy',  # Bounding box height
               "Score",  # certainty of identification
               "Aspect",  # aspect ratio of the box
               'x_centroid',  # x location of the mass centroid
               'y_centroid',  # y location of the mass centroid
               'diameter_centroid',
               'fraction_centroid',  # centroid area over BBox area
               'x_circle',  # LMSE circle X
               'y_circle',  # LMSE circle Y
               'diameter_circle',  # LMSE circle diameter
               'fraction_circle',  # LMSE circle / BBox area
               'x_ellipse',  # direct ellipse X
               'y_ellipse',  # direct ellipse X
               'width_ellipse',  # direct ellipse width
               'height_ellipse',  # direct ellipse height
               'angle_ellipse',  # direct ellipse angle from horizontal
               'fraction_ellipse'  # ellipse / BBox area
               ]
    df = pd.DataFrame(result, columns=columns)
    df["image_name"] = image_name
    return df