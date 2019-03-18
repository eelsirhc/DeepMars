import rcnn
import click
from dotenv import find_dotenv, load_dotenv
import logging
from pathlib import Path
import os
import time
import numpy as np
import deepmars.features.template_match_target as tmt
import deepmars.features.rcnn_features as rcnnf
import deepmars.utils.processing as proc
import deepmars.utils as utils
import h5py
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm, trange
from deepmars.models.common import estimate_longlatdiamkm, add_unique_craters
import sys

@click.group()
def predict():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    import sys
    sys.path.append(os.getenv("DM_ROOTDIR"))
    pass


def load_model(path, CP):
    model = rcnn.modellib.MaskRCNN(mode="inference",
                                   config=CP["config"],
                                   model_dir=CP["save_dir"])
    model.load_weights(path, by_name=True)
    return model


def get_model_preds(CP, index=0):
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
    # proc.preprocess(Data)

    model = load_model(CP['dir_model'], CP)
    logger.info("Making prediction on %d  images" % n_imgs)
    bs=CP["config"].IMAGES_PER_GPU
    ds = (bs,Data[dtype][0].shape[1],Data[dtype][0].shape[2],1)
    #    preds = dict((ival,model.detect(d.reshape(ds))[0]) for ival,d in enumerate(tqdm(Data[dtype][0])))
    preds = dict()
    for istart in trange(0,len(Data[dtype][0]),bs):
        p = model.detect(Data[dtype][0][istart:istart+bs].reshape(ds))
        for ival, d in enumerate(p):
            preds[istart+ival] = d
    dataset = rcnn.CraterDataset()
#    dataset_test.load_shapes(test_indices)
#    dataset_test.prepare()

    logger.info("Finished prediction on %d images" % n_imgs)
   
    # save
    h5f = h5py.File(CP['dir_preds'], 'w')
    keys = ['rois','class_ids','scores','masks']
    for ival, data in preds.items():
        for k in keys:
            h5f.create_dataset(dtype+"/"+str(ival)+"/"+k, data=data[k], compression='gzip', compression_opts=9)
#    h5f.create_dataset(dtype, data=preds, compression='gzip', compression_opts=9)
    print("Successfully generated and saved model predictions.")
    return preds

def match_rcnn(pred, craters, i, index, dim, withmatches=False):
    img = proc.get_id(i+index)
    found=False
    valid=False
    diam = 'Diameter (pix)'
    csv=[]
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
        coords, N_match, N_csv, N_detect, maxr, err_lo, err_la, err_r, frac_dupes = rcnnf.rcnn_template_match_t2c(pred,csv,name=i)
        df2 = pd.DataFrame(np.array([N_match, N_csv, N_detect, maxr, err_lo, err_la, err_r, frac_dupes])[None,:],
                           columns=["N_match", "N_csv", "N_detect", "maxr", "err_lo", "err_la", "err_r", "frac_dupes"],index=[img])
    else:
        coords = rcnnf.process_image(pred,i)
        df2=None
    return [coords,df2]

def rcnn_extract_unique_craters(CP, craters_unique, index=0, start=0,stop=-1, withmatches=False):
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
        print("Loaded model predictions successfully")
    except Exception as e:
        raise
        logger.info("Couldnt load model predictions {}, generating".format(CP["dir_preds"]))
#        preds = get_model_preds(CP)    

    #copied from predict_model
        
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

    #end copy
    picklable = dict()

    for k in trange(start,stop):
        picklable[str(k)] = dict((kk,vv[:]) for kk,vv in preds[str(k)].items())

    res = Parallel(n_jobs=1#int(utils.getenv("DM_NCPU"))
                   , verbose=8, batch_size=5)(delayed(match_rcnn)(picklable[str(i)],full_craters, i,index,dim,withmatches=withmatches) for i in range(start,stop))

    res = dict([(k,res[k]) for k in range(start,stop)])
#    res = dict([(k,match_rcnn(preds[str(k)],full_craters, k,index,dim,withmatches=withmatches)) for k in trange(start,stop)])

#
    for i in range(start, stop):
        data,df2 = res[i]
        data["image_name"] = np.float32(index + data["image_name"])
        img = proc.get_id(i+index)
        if img not in P[llbd]:
            # no image probably
            if len(data["x_circle"].dropna()):
                print("problem image: {}".format(img))
            continue
        if withmatches:
            matches.append(df2)

        # convert, add to master dist
        dc = pd.DataFrame(data.dropna(how='any',subset=["x_circle","y_circle","radius_circle"]))
        if len(dc) > 0:
            #circles
            new_craters_unique = estimate_longlatdiamkm(
                dim, P[llbd][img], P[distcoeff][img][0], dc[["x_circle","y_circle","radius_circle"]].values).T

            dc['Long'] = new_craters_unique[0]
            dc['Lat'] = new_craters_unique[1]
            dc['radius_km'] = new_craters_unique[2]
            dc["Diameter (km)"]  = dc["radius_km"]*2
            dc["Diameter (pix)"] = dc["radius_circle"]*2
            if craters_unique is None:
                craters_unique = pd.DataFrame([], columns=dc.columns)

            N_matches_tot += len(dc)
            ind = ["Long","Lat","radius_km"]

            # Only add unique (non-duplicate) craters
            
            if len(dc):
                cu,indices = add_unique_craters(dc[ind].values,
                                                craters_unique[ind].values,
                                                CP['llt2'], CP['rt'], return_indices=True)
                craters_unique = craters_unique.append(dc.iloc[indices])
            lab = ["Lat","Long","Diameter (km)","x_circle","y_circle","Diameter (pix)"]+list(craters_unique.columns)
            labels = []
            for l in lab:
                if l not in labels:
                    labels.append(l)
            craters_unique["image_name"] = craters_unique["image_name"].astype(int)
            dc["image_name"] = dc["image_name"].astype(int)
            craters_h5[img] = dc[labels]
            craters_h5.flush()
            
    logger.info("Saving to %s with %d unique craters" % (CP['dir_result'],len(craters_unique)))
    np.save(CP['dir_result'], craters_unique)
    alldata = craters_unique#*np.array([1,1,2])[None,:]
#    df = pd.DataFrame(alldata,columns=['Long','Lat','Diameter (km)'])
    craters_h5["all"] = alldata#df[['Lat','Long','Diameter (km)']]
    if withmatches:
        craters_h5["matches"] = pd.concat(matches)
        craters.close()
    craters_h5.flush()
    craters_h5.close()

    return craters_unique


    
    return None


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
    #if model is None:
    #    model = os.path.join(utils.getenv("DM_ROOTDIR"),'data/models/model_keras2.h5')
    CP['dir_model'] = model
    CP['save_dir'] = 'models'

    # Location of where hdf5 data images are stored
    CP['dir_data'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/processed/%s_images%s.hdf5' % (prefix,indexstr))
    # Location of where model predictions are/will be stored
    CP['dir_preds'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/predictions/rcnn/%s_preds%s.hdf5' % (CP['datatype'], indexstr))
    # Location of where final unique crater distribution will be stored
    CP['dir_result'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/predictions/rcnn/%s_craterdist%s.npy' % (CP['datatype'], indexstr))
    # Location of hdf file containing craters found
    CP['dir_craters'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/predictions/rcnn/%s_craterdist%s.hdf5' % (CP['datatype'], indexstr))
    # Location of hdf file containing craters found
    CP['dir_input_craters'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/processed/%s_craters%s.hdf5' % (prefix, indexstr))
    
    craters_unique = None

    class CraterConfig(rcnn.Config):
        # Give the configuration a recognizable name
        NAME = "craters"
        # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
        # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # Number of classes (including background)
        NUM_CLASSES = 1 + 1  # background + 1 shapes
        # Use small images for faster training. Set the limits of the small side
        # the large side, and that determines the image shape.
        IMAGE_MIN_DIM = CP["dim"]
        IMAGE_MAX_DIM = CP["dim"]
        # Use smaller anchors because our image and objects are small
        RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
        # Reduce training ROIs per image because the images are small and have
        # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
        TRAIN_ROIS_PER_IMAGE = 128
        # Use a small epoch since the data is simple
        STEPS_PER_EPOCH = 1
        # use small validation steps since the epoch is small
        VALIDATION_STEPS = 1

        IMAGE_CHANNEL_COUNT = 1
        MEAN_PIXEL = np.array([128.])

    class InferenceConfig(CraterConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 4

    config = InferenceConfig()
    CP["config"] = config
    craters_unique = rcnn_extract_unique_craters(CP, craters_unique,
                                                 index=index, start=start,
                                                 stop=stop,
                                                 withmatches=matches)


    elapsed_time = time.time() - start_time
    logger.info("Time elapsed: {0:.1f} min".format(elapsed_time / 60.))

if __name__ == '__main__':
    predict()
