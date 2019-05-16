import numpy as np
import pandas as pd
import deepmars.utils.transform as trf

#########################
def add_unique_craters(craters, craters_unique, thresh_longlat2, thresh_rad, return_indices=False):
    """Generates unique crater distribution by filtering out duplicates.

    Parameters
    ----------
    craters : array
        Crater tuples from a single image in the form (long, lat, radius).
    craters_unique : array
        Master array of unique crater tuples in thefo rm (long, lat, radius)
    thresh_longlat2 : float.
        Hyperparameter that controls the minimum squared longitude/latitude
        difference between craters to be considered unique entries.
    thresh_rad : float
        Hyperparaeter that controls the minimum squared radius difference
        between craters to be considered unique entries.

    Returns
    -------
    craters_unique : array
        Modified master array of unique crater tuples with new crater entries.
    """
    k2d = 180. / (np.pi * 3389.0)       # km to deg
    indices=[]

    for j in range(len(craters)):
        add_crater = False
        if len(craters_unique) == 0:
               add_crater = True
        else:
            Long, Lat, Rad = craters_unique.T
            lo, la, r = craters[j].T
            la_m = (la + Lat) / 2.
            minr = np.minimum(r, Rad)       # be liberal when filtering dupes
            # duplicate filtering criteria
            dL = (((Long - lo) / (minr * k2d / np.cos(np.pi * la_m / 180.)))**2
                  + ((Lat - la) / (minr * k2d))**2)
            dR = np.abs(Rad - r) / minr
            index = (dR < thresh_rad) & (dL < thresh_longlat2)
            
            if len(np.where(index == True)[0]) == 0:
                add_crater = True
            
        if add_crater:
            indices.append(j)
            if len(craters_unique):
                craters_unique = np.vstack((craters_unique, craters[j]))
            else:
                craters_unique = np.array(craters[j])

    if return_indices:
        return craters_unique, indices
    return craters_unique

#########################
def estimate_longlatdiamkm(dim, llbd, distcoeff, coords, ind=None):
    """First-order estimation of long/lat, and radius (km) from
    (Orthographic) x/y position and radius (pix).

    For images transformed from ~6000 pixel crops of the 30,000 pixel
    LROC-Kaguya DEM, this results in < ~0.4 degree latitude, <~0.2
    longitude offsets (~2% and ~1% of the image, respectively) and ~2% error in
    radius. Larger images thus may require an exact inverse transform,
    depending on the accuracy demanded by the user.

    Parameters
    ----------
    dim : tuple or list
        (width, height) of input images.
    llbd : tuple or list
        Long/lat limits (long_min, long_max, lat_min, lat_max) of image.
    distcoeff : float
        Ratio between the central heights of the transformed image and original
        image.
    coords : numpy.ndarray
        Array of crater x coordinates, y coordinates, and pixel radii.

    Returns
    -------
    craters_longlatdiamkm : numpy.ndarray
        Array of crater longitude, latitude and radii in km.
    """
    # Expand coords.
    if ind is None:
        long_pix, lat_pix, radii_pix = coords.T
    else:
        raise("Not implemented")
    # Determine radius (km).
    km_per_pix = 1. / trf.km2pix(dim[1], llbd[3] - llbd[2], dc=distcoeff)
    radii_km = radii_pix * km_per_pix

    # Determine long/lat.
    deg_per_pix = km_per_pix * 180. / (np.pi * 3389.0)
    long_central = 0.5 * (llbd[0] + llbd[1])
    lat_central = 0.5 * (llbd[2] + llbd[3])

    # Iterative method for determining latitude.
    lat_deg_firstest = lat_central - deg_per_pix * (lat_pix - dim[1] / 2.)
    latdiff = abs(lat_central - lat_deg_firstest)
    # Protect against latdiff = 0 situation.

    latdiff[latdiff < 1e-7] = 1e-7
    lat_deg = lat_central - (deg_per_pix * (lat_pix - dim[1] / 2.) *
                             (np.pi * latdiff / 180.) /
                             np.sin(np.pi * latdiff / 180.))
    # Determine longitude using determined latitude.
    long_deg = long_central + (deg_per_pix * (long_pix - dim[0] / 2.) /
                               np.cos(np.pi * lat_deg / 180.))

    # Return combined long/lat/radius array.
    return np.column_stack((long_deg, lat_deg, radii_km))
