from . import fiterror
from skimage import io, color, measure, draw, img_as_bool
import numpy as np
from scipy import optimize

def cost_circle(params, image):
    """Cost function for circle finding."""
    x0, y0, r = params
    coords = draw.circle_perimeter(int(y0), int(x0), int(r), shape=image.shape)
    template = np.zeros_like(image)
    template[coords] = 1
    return -np.sum(template == image)


def iou_circle(x0, y0, r, image):
    """intersection over union calculation for circle finding."""
    coords = draw.circle_perimeter(int(y0), int(x0), int(r), shape=image.shape)
    pr = np.zeros_like(image)
    pr[coords] = 1
    overlap = image*pr
    union = image+pr
    IOU = overlap.sum()/float(union.sum())
    return IOU


def calc_R(xc, yc, x, y):
    """Calculate the distance of each data points from the center (xc, yc) """
    return np.clip(np.sqrt((x-xc)**2 + (y-yc)**2),1,256)


def f_2b(c, x, y):
    """Calculate the algebraic distance between the 2D points
       and the mean circle centered at c=(xc, yc). """
    Ri = calc_R(*c, x, y)
    return Ri - Ri.mean()


def Df_2b(c, x, y):
    """ Jacobian of f_2b

    The axis corresponding to derivatives must
    be coherent with the col_deriv option of leastsq."""
    xc, yc = c
    df2b_dc = np.empty((len(c), x.size))
    Ri = calc_R(xc, yc, x, y)
    df2b_dc[0] = (xc - x)/Ri                   # dR/dxc
    df2b_dc[1] = (yc - y)/Ri                   # dR/dyc
    df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

    return df2b_dc


def circle_fit(data):
    """Fit a circle to X,Y data.

    Returns: 
        xc, yc, r : x location, y location, radius
        """
    x, y = data
    center_estimate = np.mean(x), np.mean(y)
    center_2b, ier = optimize.leastsq(f_2b, center_estimate,
                                      args=(x, y), Dfun=Df_2b,
                                      col_deriv=True)
    xc_2b, yc_2b = center_2b
    Ri_2b = calc_R(*center_2b, x, y)
    R_2b = Ri_2b.mean()
    residu_2b = sum((Ri_2b - R_2b)**2)
    return (xc_2b, yc_2b), R_2b

def centroid(fm, roi):
    try:
        narrow = fm[roi[0]:roi[2], roi[1]:roi[3]]
        regions = measure.regionprops(narrow.astype(int), coordinates='rc')
        bubble = regions[0]
        (y0, x0), r = bubble.centroid, bubble.major_axis_length / 2.
        x0, y0, r = optimize.fmin_bfgs(cost_circle, (x0, y0, r),
                                       args=(fm,), disp=False)
        res = [y0+roi[1], x0+roi[0], r*2]
        return res
    except Exception as e:
        raise
        raise fiterror.FitError(e)


def proc_circle(fm, roi):
    """Process an image stencil of a circle into a fitted circle."""

    try:
        data = np.where(fm)
        if (fm.max() == 0 or len(data) < 1):
            raise fiterror.FitError("No data to fit in this mask")
        center, radius = circle_fit(data)
        return [center[1],center[0],radius*2]
    except Exception as e:
        raise fiterror.FitError(e)
