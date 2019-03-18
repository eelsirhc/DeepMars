import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import cv2
from matplotlib.patches import Ellipse, Circle, Rectangle
from scipy.linalg.basic import LinAlgError
from  . import template_match_target
from . import fiterror
from . import circle_fitting as cf
import pandas as pd

minrad_ = template_match_target.minrad_  # 5
maxrad_ = template_match_target.maxrad_  # 40
longlat_thresh2_ = template_match_target.longlat_thresh2_  # 1.8
rad_thresh_ = template_match_target.rad_thresh_  # 1.0
template_thresh_ = template_match_target.template_thresh_  # 0.5
target_thresh_ = template_match_target.target_thresh_  # 0.1

def template_match_t(target, image_name=None,minrad=minrad_, maxrad=maxrad_,
                     longlat_thresh2=longlat_thresh2_, rad_thresh=rad_thresh_,
                     template_thresh=template_thresh_,
                     target_thresh=target_thresh_):

    image_name = image_name or "unknown"
    df = process_image(target, image_name)
    df = df[df[df["radius_circle"] > minrad] & df[df["radius_circle"] < maxrad]]
    return df

def process_image(image, image_name):
    N = image['scores'].size
    result = []
    for i in range(N):
        
        Nroi = image['rois'][i].size
        s = list(image["rois"][i])
        narrow = np.zeros((1, 1))

        # Check the shape compared to the input image
        if s[2]-s[0] == 1:
            s[2] += 1
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
        res = [i,N,0.5*(s[3]+s[1]),
               0.5*(s[0]+s[2]),
               shp[0],
               shp[1],
               image['scores'][i],
               aspect]

        # Fit the centroid 'mass' weighted circle

#        try:
#            cx, cy, cd = cf.centroid(image['masks'][..., i], s)
#            area_centroid = np.pi*(cd/2)**2
#            res.extend([cx, cy, cd, area_centroid / area_BB])
#        except fiterror.FitError as e:
#            res.extend([None]*4)

        # Fit the least square error circle
        try:
            s=image['masks'].shape[0] if image['masks'].shape[0] > image['masks'].shape[1] else image['masks'].shape[1]
            cx, cy, cd = cf.proc_circle(image['masks'][..., i], s)
            if (cd > s):
                raise fiterror.FitError("Size too big to be fit")
            area_circle = np.pi*(cd/2)**2
            res.extend([cx, cy, cd/2, area_circle / area_BB])
        except fiterror.FitError as e:
            res.extend([None]*4)

        # Fit the least square error direct fit ellipse
        try:
            raise fiterror.FitError("bob")
            cx, cy, cdx, cdy, cphi = cf.proc_ellipse(image['masks'][..., i], s)
            area_circle = np.pi * (cdx / 2) * (cdy / 2)
            res.extend([cx, cy, cdx, cdy, cphi, area_circle / area_BB])
        except fiterror.FitError as e:
            res.extend([None]*6)

        result.append(res)

    columns = ['mask','nmax','bbCx',  # Bounding box centre X
               'bbCy',   # Bounding box centre Y
               'bbNx',  # Bounding box width
               'bbNy',  # Bounding box height
               "Score",  # certainty of identification
               "Aspect",  # aspect ratio of the box
  #             'x_centroid',  # x location of the mass centroid
  #             'y_centroid',  # y location of the mass centroid
  #             'diameter_centroid',
  #             'fraction_centroid',  # centroid area over BBox area
               'x_circle',  # LMSE circle X
               'y_circle',  # LMSE circle Y
               'radius_circle',  # LMSE circle diameter
               'fraction_circle',  # LMSE circle / BBox area
               'x_ellipse',  # direct ellipse X
               'y_ellipse',  # direct ellipse X
               'width_ellipse',  # direct ellipse width
               'height_ellipse',  # direct ellipse height
               'angle_ellipse',  # direct ellipse angle from horizontal
               'fraction_ellipse'  # ellipse / BBox area
               ]
    df = pd.DataFrame(result, columns=columns, dtype=float)
    df["image_name"] = float(image_name)
    return df


    
def rcnn_template_match_t2c(target, csv_coords, name="unknown",minrad=minrad_, maxrad=maxrad_,
                       longlat_thresh2=longlat_thresh2_,
                       rad_thresh=rad_thresh_, template_thresh=template_thresh_,
                       target_thresh=target_thresh_, rmv_oor_csvs=0):

    templ_coords = process_image(target,name)
    
    templ_csv = templ_coords[["x_circle","y_circle","radius_circle"]].dropna().values
    
    data = template_match_target.template_match_c(templ_csv, target, csv_coords, minrad=minrad, maxrad=maxrad,
                            longlat_thresh2=longlat_thresh2,
                            rad_thresh=rad_thresh, template_thresh=template_thresh,
                            target_thresh=target_thresh, rmv_oor_csvs=0)
    
    return [templ_coords]+ list(data[1:])
