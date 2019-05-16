#!/usr/bin/env python
"""Unique Crater Distribution Functions

Functions for extracting craters from model target predictions and filtering
out duplicates.
"""
import click
import logging
import keras.models as km
import deepmars.features.template_match_target as tmt
import deepmars.utils as utils
import numpy as np
import h5py
import sys
import os
import deepmars.utils.processing as proc
import deepmars.utils.transform as trf
import time
import pandas as pd
from joblib import Parallel, delayed

def load_model(model=None):
    utils.load_env()
    if model is None:
        model = utils.getenv("DM_KERAS_MODEL")
    if isinstance(model,str):
        model = km.load_model(model)
    return model


#########################
def get_model_preds(CP):
    """Reads in or generates model predictions.

    Parameters
    ----------
    CP : dict
        Containins directory locations for loading data and storing
        predictions.

    Returns
    -------
    craters : h5py
        Model predictions.
    """
    logger = logging.getLogger(__name__)

    n_imgs, dtype = CP['n_imgs'], CP['datatype']
    logger.info("Reading %s" % CP['dir_data'])

    data = h5py.File(CP['dir_data'], 'r')
    if n_imgs < 0:
        n_imgs = data['input_images'].shape[0]

    Data = {
        dtype: [data['input_images'][:n_imgs].astype('float32'),
                data['target_masks'][:n_imgs].astype('float32')]
    }
    data.close()
    proc.preprocess(Data)

    model = load_model(CP['dir_model'])
    logger.info("Making prediction on %d images" % n_imgs)
    preds = model.predict(Data[dtype][0])
    logger.info("Finished prediction on %d images" % n_imgs)
    # save
    h5f = h5py.File(CP['dir_preds'], 'w')
    h5f.create_dataset(dtype, data=preds, compression='gzip', compression_opts=9)
    print("Successfully generated and saved model predictions.")
    return preds

#########################
def add_unique_craters(craters, craters_unique, thresh_longlat2, thresh_rad):
    """Generates unique crater distribution by filtering out duplicates.

    Parameters
    ----------
    craters : array
        Crater tuples from a single image in the form (long, lat, radius).
    craters_unique : array
        Master array of unique crater tuples in the form (long, lat, radius)
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
    Long, Lat, Rad = craters_unique.T
    for j in range(len(craters)):
        lo, la, r = craters[j].T
        la_m = (la + Lat) / 2.
        minr = np.minimum(r, Rad)       # be liberal when filtering dupes
        
        # duplicate filtering criteria
        dL = (((Long - lo) / (minr * k2d / np.cos(np.pi * la_m / 180.)))**2
              + ((Lat - la) / (minr * k2d))**2)
        dR = np.abs(Rad - r) / minr
        index = (dR < thresh_rad) & (dL < thresh_longlat2)
        
        if len(np.where(index == True)[0]) == 0:
            craters_unique = np.vstack((craters_unique, craters[j]))
    return craters_unique

#########################
def estimate_longlatdiamkm(dim, llbd, distcoeff, coords):
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
    long_pix, lat_pix, radii_pix = coords.T

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

def match_template(pred, craters,i, index, dim, withmatches=False):
    img = proc.get_id(i+index)
    found=False
    valid=False
    diam = 'Diameter (pix)'
    if withmatches:
        N_match, N_csv, N_detect, maxr, err_lo, err_la, err_r, frac_dupes = -1,-1,-1,-1,-1,-1,-1,-1

        if img in craters:
            csv = craters[img]
            found=True
        if found:
            minrad, maxrad = 3,50
            cutrad = 0.8
            csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
            csv = csv[(csv['x'] + cutrad * csv[diam] / 2 <= dim[0])]
            csv = csv[(csv['y'] + cutrad * csv[diam] / 2 <= dim[1])]
            csv = csv[(csv['x'] - cutrad * csv[diam] / 2 > 0)]
            csv = csv[(csv['y'] - cutrad * csv[diam] / 2 > 0)]
            if len(csv) >= 3:
                valid = True
                csv = np.asarray((csv['x'], csv['y'], csv[diam] / 2)).T


    if valid:
        coords, N_match, N_csv, N_detect, maxr, err_lo, err_la, err_r, frac_dupes = tmt.template_match_t2c(pred,csv)
        df2 = pd.DataFrame(np.array([N_match, N_csv, N_detect, maxr, err_lo, err_la, err_r, frac_dupes])[None,:],
                           columns=["N_match", "N_csv", "N_detect", "maxr", "err_lo", "err_la", "err_r", "frac_dupes"],index=[img])
    else:
        coords = tmt.template_match_t(pred)
        df2=None
    return [coords,df2]

def extract_unique_craters(CP, craters_unique,index=0, start=0,stop=-1, withmatches=False):
    """Top level function that extracts craters from model predictions,
    converts craters from pixel to real (degree, km) coordinates, and filters
    out duplicate detections across images.

    Parameters
    ----------
    CP : dict
        Crater Parameters needed to run the code.
    craters_unique : array
        Empty master array of unique crater tuples in the form 
        (long, lat, radius).

    Returns
    -------
    craters_unique : array
        Filled master array of unique crater tuples.
    """
    logger = logging.getLogger(__name__)
    # Load/generate model preds
    try:
        preds = h5py.File(CP['dir_preds'], 'r')[CP['datatype']]
        logger.info("Loaded model predictions successfully")
    except:
        logger.info("Couldnt load model predictions {}, generating".format(CP["dir_preds"]))
        preds = get_model_preds(CP)

    # need for long/lat bounds
    P = h5py.File(CP['dir_data'], 'r')


    llbd, pbd, distcoeff = ('longlat_bounds', 'pix_bounds',
                            'pix_distortion_coefficient')
    #r_moon = 1737.4
    dim = (float(CP['dim']), float(CP['dim']))

    N_matches_tot = 0
    if start < 0:
        start = 0
    if stop < 0:
        stop = P['input_images'].shape[0]

    start = np.clip(start,0,P['input_images'].shape[0]-1)
    stop =  np.clip(stop,1,P['input_images'].shape[0])
    craters_h5 = pd.HDFStore(CP['dir_craters'],'w')


    csvs=[]
    if withmatches:
        craters = pd.HDFStore(CP['dir_input_craters'], 'r')
        matches = []#pd.DataFrame(columns=["N_match", "N_csv", "N_detect", "maxr", "err_lo", "err_la", "err_r", "frac_dupes"])

    full_craters = dict()
    if withmatches:
        for i in range(start,stop):
            img = proc.get_id(i+index)
            if img in craters:
                full_craters[img]=craters[img]

    
    res = Parallel(n_jobs=int(utils.getenv("DM_NCPU")), verbose=5)(delayed(match_template)(preds[i],full_craters, i,index,dim,withmatches=withmatches) for i in range(start,stop))

    for i in range(start,stop):
        coords, df2 = res[i]
        if withmatches:
            matches.append(df2)
        img = proc.get_id(i+index)
        # convert, add to master dist
        if len(coords) > 0:
            new_craters_unique = estimate_longlatdiamkm(
                dim, P[llbd][img], P[distcoeff][img][0], coords)
            N_matches_tot += len(coords)

            # Only add unique (non-duplicate) craters
            if len(craters_unique) > 0:
                craters_unique = add_unique_craters(new_craters_unique,
                                                    craters_unique,
                                                    CP['llt2'], CP['rt'])
            else:
                craters_unique = np.concatenate((craters_unique,
                                                 new_craters_unique))
            
            data = np.hstack([new_craters_unique*np.array([1,1,2])[None,:],coords*np.array([1,1,2])[None,:]])
            df = pd.DataFrame(data,columns=['Long','Lat','Diameter (km)','x','y','Diameter (pix)'])
            craters_h5[img] = df[['Lat','Long','Diameter (km)','x','y','Diameter (pix)']]
            craters_h5.flush()
            
    logger.info("Saving to %s with %d unique craters" % (CP['dir_result'],len(craters_unique)))
    np.save(CP['dir_result'], craters_unique)
    alldata = craters_unique*np.array([1,1,2])[None,:]
    df = pd.DataFrame(alldata,columns=['Long','Lat','Diameter (km)'])
    craters_h5["all"] = df[['Lat','Long','Diameter (km)']]
    if withmatches:
        craters_h5["matches"] = pd.concat(matches)
        craters.close()
    craters_h5.flush()
    craters_h5.close()
    
    return craters_unique



@click.group()
def predict():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    utils.load_env()
    pass

@predict.command()
@click.argument('index',type=int,nargs=-1)
@click.option('--prefix',default="test")
@click.option('--output_prefix',default=None)
@click.option("--model",default=None)
def cnn_prediction(index, prefix, output_prefix, model):
    """ Make predictions.
    """
    logger = logging.getLogger(__name__)
    logger.info('making predictions.')
    start_time = time.time()

    if model is None:
        model = os.path.join(utils.getenv("DM_ROOTDIR"),'data/models/model_keras2.h5')
        model = load_model(model)

    for ivalue in index:
        if ivalue is None:
            indexstr = ""
        else:
            indexstr = "_{:05d}".format(ivalue)

        if output_prefix is None:
            output_prefix = prefix
        # Crater Parameters
        CP = dict(dim=256,datatype=prefix,
                  n_imgs=-1,
                  dir_model = model,
                  dir_data = os.path.join(utils.getenv("DM_ROOTDIR"),'data/processed/%s_images%s.hdf5' % (prefix,indexstr)),
                  dir_preds = os.path.join(utils.getenv("DM_ROOTDIR"),'data/predictions/%s_preds%s.hdf5' % (output_prefix, indexstr)))

        get_model_preds(CP)

    elapsed_time = time.time() - start_time
    logger.info("Time elapsed: {0:.1f} min".format(elapsed_time / 60.))
    
@predict.command()
@click.argument('llt2',type=float)
@click.argument('rt',type=float)
@click.option('--index',type=int, default=None)
@click.option('--prefix',default="test")
@click.option('--start',default=-1)
@click.option('--stop',default=-1)
@click.option('--matches',is_flag=True,default=False)
@click.option("--model",default=None)
def make_prediction(llt2, rt, index, prefix, start, stop, matches, model):
    """ Make predictions.
    """
    logger = logging.getLogger(__name__)
    logger.info('making predictions.')
    start_time = time.time()
    if index is None:
        indexstr = ""
    else:
        indexstr = "_{:05d}".format(index)

    # Crater Parameters
    CP = {}
    # Image width/height, assuming square images.
    CP['dim'] = 256
    # Data type - train, dev, test
    CP['datatype'] = prefix
    # Number of images to extract craters from
    CP['n_imgs'] = -1 # all of them
    # Hyperparameters
    CP['llt2'] = llt2    # D_{L,L} from Silburt et. al (2017)
    CP['rt'] = rt     # D_{R} from Silburt et. al (2017)
    # Location of model to generate predictions (if they don't exist yet)
    if model is None:
        model = os.path.join(utils.getenv("DM_ROOTDIR"),'data/models/model_keras2.h5')
    CP['dir_model'] =  model
#    # Location of where hdf5 data images are stored
#    CP['dir_data'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/processed/%s_images.hdf5' % CP['datatype'])
#    # Location of where model predictions are/will be stored
#    CP['dir_preds'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/processed/%s_preds_n%d.hdf5' % (CP['datatype'], CP['n_imgs']))
#    # Location of where final unique crater distribution will be stored
#    CP['dir_result'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/processed/%s_craterdist.npy' % (CP['datatype']))
#
    # Location of where hdf5 data images are stored
    CP['dir_data'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/processed/processed.sys/%s_images%s.hdf5' % (prefix,indexstr))
    # Location of where model predictions are/will be stored
    CP['dir_preds'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/predictions/mars_trained/%s_preds%s.hdf5' % (CP['datatype'], indexstr))
    # Location of where final unique crater distribution will be stored
    CP['dir_result'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/predictions/mars_trained/%s_craterdist%s.npy' % (CP['datatype'], indexstr))
    # Location of hdf file containing craters found
    CP['dir_craters'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/predictions/mars_trained/%s_craterdist%s.hdf5' % (CP['datatype'], indexstr))
    # Location of hdf file containing craters found
    CP['dir_input_craters'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/processed/processed.sys/%s_craters%s.hdf5' % (prefix, indexstr))
    
    craters_unique = np.empty([0, 3])
    craters_unique = extract_unique_craters(CP, craters_unique,index=index, start=start, stop=stop, withmatches=matches)


    elapsed_time = time.time() - start_time
    logger.info("Time elapsed: {0:.1f} min".format(elapsed_time / 60.))

if __name__ == '__main__':
    predict()

