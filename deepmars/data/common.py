from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from PIL import Image
import cartopy.crs as ccrs
import cartopy.img_transform as cimg
import collections
import cv2
import h5py
import deepmars.utils.transform as trf
import tifffile
import os

def MarsDEM(filename=None):
    """Reads the Mars DEM from a large tiff file.

    Paramters
    ----------
    filename : str, optional
        path of the DEM file, default to environment variable DM_MarsDEM

    Returns
    --------
    dem : numpy.ndarray
        The image as a numpy array.
    """

    if filename is None:
        filename = os.getenv("DM_MarsDEM")
    name,ext = os.path.splitext(filename)
    if ext==".tif":
        dem = tifffile.imread(filename)
    elif ext==".png":
        dem = imageio.imread(filename)
    #print(dem.shape,filename)
    return dem

def ReadRobbinsCraters(filename=None):
    """Reads the crater database TSV

    Parameters
    -----------
    filename : str, optional
        path of the TSV file, defaults to the value found in the environment variable DM_CraterTable
    
    Returns
    --------
    craters : pandas.Dataframe
        A dataframe of the crater table
    """

    if filename is None:
        filename = os.getenv("DM_CraterTable")
        #print(filename)
#    craters = pd.read_table(filename,sep='\t',engine='python',index_col=False)
    craters = pd.read_csv(filename,index_col=False)
    keep_columns = ["LATITUDE_CIRCLE_IMAGE",
                    "LONGITUDE_CIRCLE_IMAGE",
                    "DIAM_CIRCLE_IMAGE",
    #                "DIAM_CIRCLE_SD_IMAGE",
    #                "DEPTH_RIM_TOPOG",
    #                "DEPTH_RIM_SD_TOPOG",
    #                "DEPTH_SURFACE_TOPOG",
    #                "DEPTH_SURFACE_SD_TOPOG",
    #                "DEPTH_FLOOR_TOPOG",
    #                "DEPTH_FLOOR_SD_TOPOG",
                    ]

    craters = craters[keep_columns]
    craters.columns = ["Lat","Long","Diameter (km)",
#                       "SD Diameter (km)",
#                       "DEPTH_RIM_TOPOG",
#                       "DEPTH_RIM_SD_TOPOG",
#                       "DEPTH_SURFACE_TOPOG",
#                       "DEPTH_SURFACE_SD_TOPOG",
#                       "DEPTH_FLOOR_TOPOG",
#                       "DEPTH_FLOOR_SD_TOPOG",
                       ]
    return craters

def regrid_shape_aspect(regrid_shape, target_extent):
    """Helper function copied from cartopy.img_transform for resizing an image
    without changing its aspect ratio.

    Parameters
    ----------
    regrid_shape : int or float
        Target length of the shorter axis (in units of pixels).
    target_extent : some
        Width and height of the target image (generally not in units of
        pixels).

    Returns
    -------
    regrid_shape : tuple
        Width and height of the target image in pixels.
    """
    if not isinstance(regrid_shape, collections.Sequence):
        target_size = int(regrid_shape)
        x_range, y_range = np.diff(target_extent)[::2]
        desired_aspect = x_range / y_range
        if x_range >= y_range:
            regrid_shape = (target_size * desired_aspect, target_size)
        else:
            regrid_shape = (target_size, target_size / desired_aspect)
    return regrid_shape


def WarpImage(img, iproj, iextent, oproj, oextent,
              origin="upper", rgcoeff=1.2):
    """Warps images with cartopy.img_transform.warp_array, then plots them with
    imshow.  Based on cartopy.mpl.geoaxes.imshow.

    Parameters
    ----------
    img : numpy.ndarray
        Image as a 2D array.
    iproj : cartopy.crs.Projection instance
        Input coordinate system.
    iextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of input.
    oproj : cartopy.crs.Projection instance
        Output coordinate system.
    oextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of output.
    origin : "lower" or "upper", optional
        Based on imshow convention for displaying image y-axis.  "upper" means
        [0,0] is in the upper-left corner of the image; "lower" means it's in
        the bottom-left.
    rgcoeff : float, optional
        Fractional size increase of transformed image height.  Generically set
        to 1.2 to prevent loss of fidelity during transform (though some of it
        is inevitably lost due to warping).
    """

    if iproj == oproj:
        raise Warning("Input and output transforms are identical!"
                      "Returing input!")
        return img

    if origin == 'upper':
        # Regridding operation implicitly assumes origin of image is
        # 'lower', so adjust for that here.
        img = img[::-1]

    # rgcoeff is padding when we rescale the image later.
    regrid_shape = rgcoeff * min(img.shape)
    regrid_shape = regrid_shape_aspect(regrid_shape,
                                       oextent)

    # cimg.warp_array uses cimg.mesh_projection, which cannot handle any
    # zeros being used in iextent.  Create iextent_nozeros to fix.
    iextent_nozeros = np.array(iextent)
    iextent_nozeros[iextent_nozeros == 0] = 1e-8
    iextent_nozeros = list(iextent_nozeros)

    imgout, extent = cimg.warp_array(img,
                                     source_proj=iproj,
                                     source_extent=iextent_nozeros,
                                     target_proj=oproj,
                                     target_res=regrid_shape,
                                     target_extent=oextent,
                                     mask_extrapolated=True)

    if origin == 'upper':
        # Transform back.
        imgout = imgout[::-1]

    return imgout


def WarpImagePad(img, iproj, iextent, oproj, oextent, origin="upper",
                 rgcoeff=1.2, fillbg="black"):
    """Wrapper for WarpImage that adds padding to warped image to make it the
    same size as the original.

    Parameters
    ----------
    img : numpy.ndarray
        Image as a 2D array.
    iproj : cartopy.crs.Projection instance
        Input coordinate system.
    iextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of input.
    oproj : cartopy.crs.Projection instance
        Output coordinate system.
    oextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of output.
    origin : "lower" or "upper", optional
        Based on imshow convention for displaying image y-axis.  "upper" means
        [0,0] is in the upper-left corner of the image; "lower" means it's in
        the bottom-left.
    rgcoeff : float, optional
        Fractional size increase of transformed image height.  Generically set
        to 1.2 to prevent loss of fidelity during transform (though some of it
        is inevitably lost due to warping).
    fillbg : 'black' or 'white', optional.
        Fills padding with either black (0) or white (255) values.  Default is
        black.

    Returns
    -------
    imgo : PIL.Image.Image
        Warped image with padding
    imgw.size : tuple
        Width, height of picture without padding
    offset : tuple
        Pixel width of (left, top)-side padding
    """
    # Based off of <https://stackoverflow.com/questions/2563822/
    # how-do-you-composite-an-image-onto-another-image-with-pil-in-python>

    if type(img) == Image.Image:
        img = np.asanyarray(img)

    # Check that we haven't been given a corrupted image.
    assert img.sum() > 0, "Image input to WarpImagePad is blank!"

    # Set background colour
    if fillbg == "white":
        bgval = 255
    else:
        bgval = 0

    # Warp image.
    imgw = WarpImage(img, iproj, iextent, oproj, oextent,
                     origin=origin, rgcoeff=rgcoeff)

    # Remove mask, turn image into Image.Image.
    imgw = np.ma.filled(imgw, fill_value=bgval)
    imgw = Image.fromarray(imgw, mode="L")

    # Resize to height of original, maintaining aspect ratio.  Note
    # img.shape = height, width, and imgw.size and imgo.size = width, height.
    imgw_loh = imgw.size[0] / imgw.size[1]

    # If imgw is stretched horizontally compared to img.
    if imgw_loh > (img.shape[1] / img.shape[0]):
        imgw = imgw.resize([img.shape[0],
                            int(np.round(img.shape[0] / imgw_loh))],
                           resample=Image.NEAREST)
    # If imgw is stretched vertically.
    else:
        imgw = imgw.resize([int(np.round(imgw_loh * img.shape[0])),
                            img.shape[0]], resample=Image.NEAREST)

    # Make background image and paste two together.
    imgo = Image.new('L', (img.shape[1], img.shape[0]), (bgval))
    offset = ((imgo.size[0] - imgw.size[0]) // 2,
              (imgo.size[1] - imgw.size[1]) // 2)
    imgo.paste(imgw, offset)

    return imgo, imgw.size, offset


def WarpCraterLoc(craters, geoproj, oproj, oextent, imgdim, llbd=None,
                  origin="upper"):
    """Wrapper for WarpImage that adds padding to warped image to make it the
    same size as the original.

    Parameters
    ----------
    craters : pandas.DataFrame
        Crater info
    geoproj : cartopy.crs.Geodetic instance
        Input lat/long coordinate system
    oproj : cartopy.crs.Projection instance
        Output coordinate system
    oextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max)
        of output
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels
    llbd : list-like
        Long/lat limits (long_min, long_max,
        lat_min, lat_max) of image
    origin : "lower" or "upper"
        Based on imshow convention for displaying image y-axis.
        "upper" means that [0,0] is upper-left corner of image;
        "lower" means it is bottom-left.

    Returns
    -------
    ctr_wrp : pandas.DataFrame
        DataFrame that includes pixel x, y positions
    """

    # Get subset of craters within llbd limits
    if llbd is None:
        ctr_wrp = craters
    else:
        ctr_xlim = ((craters["Long"] >= llbd[0]) &
                    (craters["Long"] <= llbd[1]))
        ctr_ylim = ((craters["Lat"] >= llbd[2]) &
                    (craters["Lat"] <= llbd[3]))
        ctr_wrp = craters.loc[ctr_xlim & ctr_ylim, :].copy()

    # Get output projection coords.
    # [:,:2] becaus we don't need elevation data
    # If statement is in case ctr_wrp has nothing in it
    if ctr_wrp.shape[0]:
        ilong = ctr_wrp["Long"].as_matrix()
        ilat = ctr_wrp["Lat"].as_matrix()
        res = oproj.transform_points(x=ilong, y=ilat,
                                     src_crs=geoproj)[:, :2]

        # Get output
        ctr_wrp["x"], ctr_wrp["y"] = trf.coord2pix(res[:, 0], res[:, 1],
                                                   oextent, imgdim,
                                                   origin=origin)
    else:
        ctr_wrp["x"] = []
        ctr_wrp["y"] = []

    return ctr_wrp

############# Warp Plate Carree to Orthographic ###############

def PlateCarree_to_Orthographic(img, llbd, craters, iglobe=None,
                                ctr_sub=False, arad=1737.4, origin="upper",
                                rgcoeff=1.2, slivercut=0.):
    """Transform Plate Carree image and associated csv file into Orthographic.

    Parameters
    ----------
    img : PIL.Image.image or str
        File or filename.
    llbd : list-like
        Long/lat limits (long_min, long_max, lat_min, lat_max) of image.
    craters : pandas.DataFrame
        Craters catalogue.
    iglobe : cartopy.crs.Geodetic instance
        Globe for images.  If False, defaults to spherical Moon.
    ctr_sub : bool, optional
        If `True`, assumes craters dataframe includes only craters within
        image. If `False` (default_, llbd used to cut craters from outside
        image out of (copy of) dataframe.
    arad : float
        World radius in km.  Default is Moon (1737.4 km).
    origin : "lower" or "upper", optional
        Based on imshow convention for displaying image y-axis.  "upper"
        (default) means that [0,0] is upper-left corner of image; "lower" means
        it is bottom-left.
    rgcoeff : float, optional
        Fractional size increase of transformed image height.  By default set
        to 1.2 to prevent loss of fidelity during transform (though warping can
        be so extreme that this might be meaningless).
    slivercut : float from 0 to 1, optional
        If transformed image aspect ratio is too narrow (and would lead to a
        lot of padding, return null images).

    Returns
    -------
    imgo : PIL.Image.image
        Transformed, padded image in PIL.Image format.
    ctr_xy : pandas.DataFrame
        Craters with transformed x, y pixel positions and pixel radii.
    distortion_coefficient : float
        Ratio between the central heights of the transformed image and original
        image.
    centrallonglat_xy : pandas.DataFrame
        xy position of the central longitude and latitude.
    """

    # If user doesn't provide Moon globe properties.
    if not iglobe:
        iglobe = ccrs.Globe(semimajor_axis=arad*1000.,
                            semiminor_axis=arad*1000., ellipse=None)

    # Set up Geodetic (long/lat), Plate Carree (usually long/lat, but not when
    # globe != WGS84) and Orthographic projections.
    geoproj = ccrs.Geodetic(globe=iglobe)
    iproj = ccrs.PlateCarree(globe=iglobe)
    oproj = ccrs.Orthographic(central_longitude=np.mean(llbd[:2]),
                              central_latitude=np.mean(llbd[2:]),
                              globe=iglobe)

    # Create and transform coordinates of image corners and edge midpoints.
    # Due to Plate Carree and Orthographic's symmetries, max/min x/y values of
    # these 9 points represent extrema of the transformed image.
    xll = np.array([llbd[0], np.mean(llbd[:2]), llbd[1]])
    yll = np.array([llbd[2], np.mean(llbd[2:]), llbd[3]])
    xll, yll = np.meshgrid(xll, yll)
    xll = xll.ravel()
    yll = yll.ravel()

    # [:,:2] because we don't need elevation data.
    res = iproj.transform_points(x=xll, y=yll, src_crs=geoproj)[:, :2]
    iextent = [min(res[:, 0]), max(res[:, 0]), min(res[:, 1]), max(res[:, 1])]

    res = oproj.transform_points(x=xll, y=yll, src_crs=geoproj)[:, :2]
    oextent = [min(res[:, 0]), max(res[:, 0]), min(res[:, 1]), max(res[:, 1])]

    # Sanity check for narrow images; done before the most expensive part of
    # the function.
    oaspect = (oextent[1] - oextent[0]) / (oextent[3] - oextent[2])
    if oaspect < slivercut:
        return [None, None, None, None]

    if type(img) != Image.Image:
        img = Image.open(img).convert("L")

    # Warp image.
    imgo, imgwshp, offset = WarpImagePad(img, iproj, iextent, oproj, oextent,
                                         origin=origin, rgcoeff=rgcoeff,
                                         fillbg="black")

    # Convert crater x, y position.
    if ctr_sub:
        llbd_in = None
    else:
        llbd_in = llbd
    ctr_xy = WarpCraterLoc(craters, geoproj, oproj, oextent, imgwshp,
                           llbd=llbd_in, origin=origin)
    # Shift crater x, y positions by offset (origin doesn't matter for y-shift,
    # since padding is symmetric).
    ctr_xy.loc[:, "x"] += offset[0]
    ctr_xy.loc[:, "y"] += offset[1]

    # Pixel scale for orthographic determined (for images small enough that
    # tan(x) approximately equals x + 1/3x^3 + ...) by l = R_moon*theta,
    # where theta is the latitude extent of the centre of the image.  Because
    # projection transform doesn't guarantee central vertical axis will keep
    # its pixel resolution, we need to calculate the conversion coefficient
    #   C = (res[7,1]- res[1,1])/(oextent[3] - oextent[2]).
    #   C0*pix height/C = theta
    # Where theta is the latitude extent and C0 is the theta per pixel
    # conversion for the Plate Carree image).  Thus
    #   l_ctr = R_moon*C0*pix_ctr/C.
    distortion_coefficient = ((res[7, 1] - res[1, 1]) /
                              (oextent[3] - oextent[2]))
    if distortion_coefficient < 0.7:
        raise ValueError("Distortion Coefficient cannot be"
                         " {0:.2f}!".format(distortion_coefficient))
    pixperkm = trf.km2pix(imgo.size[1], llbd[3] - llbd[2],
                          dc=distortion_coefficient, a=arad)
    ctr_xy["Diameter (pix)"] = ctr_xy["Diameter (km)"] * pixperkm

    # Determine x, y position of central lat/long.
    centrallonglat = pd.DataFrame({"Long": [xll[4]], "Lat": [yll[4]]})
    centrallonglat_xy = WarpCraterLoc(centrallonglat, geoproj, oproj, oextent,
                                      imgwshp, llbd=llbd_in, origin=origin)

    # Shift central long/lat
    centrallonglat_xy.loc[:, "x"] += offset[0]
    centrallonglat_xy.loc[:, "y"] += offset[1]

    return [imgo, ctr_xy, distortion_coefficient, centrallonglat_xy]

############# Create target dataset (and helper functions) #############

def circlemaker(r=10.):
    """
    Creates circle mask of radius r.
    """
    # Based on <https://stackoverflow.com/questions/10031580/
    # how-to-write-simple-geometric-shapes-into-numpy-arrays>

    # Mask grid extent (+1 to ensure we capture radius).
    rhext = int(r) + 1

    xx, yy = np.mgrid[-rhext:rhext + 1, -rhext:rhext + 1]
    circle = (xx**2 + yy**2) <= r**2

    return circle.astype(float)


def ringmaker(r=10., dr=1):
    """
    Creates ring of radius r and thickness dr.

    Parameters
    ----------
    r : float
        Ring radius
    dr : int
        Ring thickness (cv2.circle requires int)
    """
    # See <http://docs.opencv.org/2.4/modules/core/doc/
    # drawing_functions.html#circle>, supplemented by
    # <http://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html>
    # and <https://github.com/opencv/opencv/blob/
    # 05b15943d6a42c99e5f921b7dbaa8323f3c042c6/modules/imgproc/
    # src/drawing.cpp>.

    # mask grid extent (dr/2 +1 to ensure we capture ring width
    # and radius); same philosophy as above
    rhext = int(np.ceil(r + dr / 2.)) + 1

    # cv2.circle requires integer radius
    mask = np.zeros([2 * rhext + 1, 2 * rhext + 1], np.uint8)

    # Generate ring
    ring = cv2.circle(mask, (rhext, rhext), int(np.round(r)), 1, thickness=dr)

    return ring.astype(float)


def get_merge_indices(cen, imglen, ks_h, ker_shp):
    """Helper function that returns indices for merging stencil with base
    image, including edge case handling.  x and y are identical, so code is
    axis-neutral.

    Assumes INTEGER values for all inputs!
    """

    left = cen - ks_h
    right = cen + ks_h + 1

    # Handle edge cases.  If left side of stencil is beyond the left end of
    # the image, for example, crop stencil and shift image index to lefthand
    # side.
    if left < 0:
        img_l = 0
        g_l = -left
    else:
        img_l = left
        g_l = 0
    if right > imglen:
        img_r = imglen
        g_r = ker_shp - (right - imglen)
    else:
        img_r = right
        g_r = ker_shp

    return [img_l, img_r, g_l, g_r]


def make_mask(craters, img, binary=True, rings=False, ringwidth=1,
              truncate=True):
    """Makes crater mask binary image (does not yet consider crater overlap).

    Parameters
    ----------
    craters : pandas.DataFrame
        Craters catalogue that includes pixel x and y columns.
    img : numpy.ndarray
        Original image; assumes colour channel is last axis (tf standard).
    binary : bool, optional
        If True, returns a binary image of crater masks.
    rings : bool, optional
        If True, mask uses hollow rings rather than filled circles.
    ringwidth : int, optional
        If rings is True, ringwidth sets the width (dr) of the ring.
    truncate : bool
        If True, truncate mask where image truncates.

    Returns
    -------
    mask : numpy.ndarray
        Target mask image.
    """

    # Load blank density map
    imgshape = img.shape[:2]
    mask = np.zeros(imgshape)
    cx = craters["x"].values.astype('int')
    cy = craters["y"].values.astype('int')
    radius = craters["Diameter (pix)"].values / 2.

    for i in range(craters.shape[0]):
        if rings:
            kernel = ringmaker(r=radius[i], dr=ringwidth)
        else:
            kernel = circlemaker(r=radius[i])
        # "Dummy values" so we can use get_merge_indices
        kernel_support = kernel.shape[0]
        ks_half = kernel_support // 2

        # Calculate indices on image where kernel should be added
        [imxl, imxr, gxl, gxr] = get_merge_indices(cx[i], imgshape[1],
                                                   ks_half, kernel_support)
        [imyl, imyr, gyl, gyr] = get_merge_indices(cy[i], imgshape[0],
                                                   ks_half, kernel_support)

        # Add kernel to image
        mask[imyl:imyr, imxl:imxr] += kernel[gyl:gyr, gxl:gxr]

    if binary:
        mask = (mask > 0).astype(float)

    if truncate:
        if img.ndim == 3:
            mask[img[:, :, 0] == 0] = 0
        else:
            mask[img == 0] = 0

    return mask

############# Create dataset (and helper functions) #############

def AddPlateCarree_XY(craters, imgdim, cdim=[-180., 180., -90., 90.], 
                      origin="upper"):
    """Adds x and y pixel locations to craters dataframe.

    Parameters
    ----------
    craters : pandas.DataFrame
        Crater info
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels
    cdim : list-like, optional
        Coordinate limits (x_min, x_max, y_min, y_max) of image.  Default is
        [-180., 180., -90., 90.].
    origin : "upper" or "lower", optional
        Based on imshow convention for displaying image y-axis.
        "upper" means that [0,0] is upper-left corner of image;
        "lower" means it is bottom-left.
    """
    x, y = trf.coord2pix(craters["Long"].as_matrix(),
                         craters["Lat"].as_matrix(),
                         cdim, imgdim, origin=origin)
    craters["x"] = x
    craters["y"] = y


def ResampleCraters(craters, llbd, imgheight, arad=1737.4, minpix=0):
    """Crops crater file, and removes craters smaller than some minimum value.

    Parameters
    ----------
    craters : pandas.DataFrame
        Crater dataframe.
    llbd : list-like
        Long/lat limits (long_min, long_max, lat_min, lat_max) of image.
    imgheight : int
        Pixel height of image.
    arad : float, optional
        World radius in km.  Defaults to Moon radius (1737.4 km).
    minpix : int, optional
        Minimium crater pixel size to be included in output.  Default is 0
        (equvalent to no cutoff).

    Returns
    -------
    ctr_sub : pandas.DataFrame
        Cropped and filtered dataframe.
    """

    # Get subset of craters within llbd limits.
    ctr_xlim = (craters["Long"] >= llbd[0]) & (craters["Long"] <= llbd[1])
    ctr_ylim = (craters["Lat"] >= llbd[2]) & (craters["Lat"] <= llbd[3])
    ctr_sub = craters.loc[ctr_xlim & ctr_ylim, :].copy()

    if minpix > 0:
        # Obtain pixel per km conversion factor.  Use latitude because Plate
        # Carree doesn't distort along this axis.
        pixperkm = trf.km2pix(imgheight, llbd[3] - llbd[2], dc=1., a=arad)
        minkm = minpix / pixperkm

        # Remove craters smaller than pixel limit.
        ctr_sub = ctr_sub[ctr_sub["Diameter (km)"] >= minkm]

    ctr_sub.reset_index(inplace=True, drop=True)

    return ctr_sub


def InitialImageCut(img, cdim, newcdim):
    """Crops image, so that the crop output can be used in GenDataset.

    Parameters
    ----------
    img : PIL.Image.Image
        Image
    cdim : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of image.
    newcdim : list-like
        Crop boundaries (x_min, x_max, y_min, y_max).  There is
        currently NO CHECK that newcdim is within cdim!

    Returns
    -------
    img : PIL.Image.Image
        Cropped image
    """
    x, y = trf.coord2pix(np.array(newcdim[:2]), np.array(newcdim[2:]), cdim,
                         img.shape, origin="upper")

    # y is backward since origin is upper!
    #box = [x[0], y[1], x[1], y[0]]
    #img = img.crop(box)
    #img.load()
    #print(x,y)
    img = img[int(x[0]):int(x[1]),int(y[1]):int(y[0])]
    #print(img.shape)
    return img
