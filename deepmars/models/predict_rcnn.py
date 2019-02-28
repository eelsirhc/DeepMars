import rcnn
import click
from dotenv import find_dotenv, load_dotenv
import logging
from pathlib import Path
import os
import numpy as np

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
                                   config=CP["dir_model"],
                                   model_dir=CP["save_dir"])
    model.load_weights(CP["model"], by_name=True)
    return model


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
    # proc.preprocess(Data)

    model = load_model(CP['dir_model'], CP)
    logger.info("Making prediction on %d images" % n_imgs)
    preds = model.predict(Data[dtype][0])

    dataset = rcnn.CraterDataset()
    dataset_test.load_shapes(test_indices)
    dataset_test.prepare()

    logger.info("Finished prediction on %d images" % n_imgs)
   
    # save
    #h5f = h5py.File(CP['dir_preds'], 'w')
    #h5f.create_dataset(dtype, data=preds, compression='gzip', compression_opts=9)
    print("Successfully generated and saved model predictions.")
    return preds


def extract_unique_craters(CP, craters_unique, index=0, start=0,stop=-1, withmatches=False):
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
        raise #preds = h5py.File(CP['dir_preds'], 'r')[CP['datatype']]
        logger.info("Loaded model predictions successfully")
    except:
        logger.info("Couldnt load model predictions {}, generating".format(CP["dir_preds"]))
        preds = get_model_preds(CP)
    

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
    CP['dir_data'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/processed/processed.sys/%s_images%s.hdf5' % (prefix,indexstr))
    # Location of where model predictions are/will be stored
    CP['dir_preds'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/predictions/rcnn/%s_preds%s.hdf5' % (CP['datatype'], indexstr))
    # Location of where final unique crater distribution will be stored
    CP['dir_result'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/predictions/rcnn/%s_craterdist%s.npy' % (CP['datatype'], indexstr))
    # Location of hdf file containing craters found
    CP['dir_craters'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/predictions/rcnn/%s_craterdist%s.hdf5' % (CP['datatype'], indexstr))
    # Location of hdf file containing craters found
    CP['dir_input_craters'] = os.path.join(utils.getenv("DM_ROOTDIR"),'data/processed/processed.sys/%s_craters%s.hdf5' % (prefix, indexstr))
    
    craters_unique = np.empty([0, 3])

    class CraterConfig(rcnn.Config):
        # Give the configuration a recognizable name
        NAME = "craters"
        # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
        # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
        GPU_COUNT = 1
        IMAGES_PER_GPU = MP["bs"]
        # Number of classes (including background)
        NUM_CLASSES = 1 + 1  # background + 1 shapes
        # Use small images for faster training. Set the limits of the small side
        # the large side, and that determines the image shape.
        IMAGE_MIN_DIM = MP["dim"]
        IMAGE_MAX_DIM = MP["dim"]
        # Use smaller anchors because our image and objects are small
        RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
        # Reduce training ROIs per image because the images are small and have
        # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
        TRAIN_ROIS_PER_IMAGE = 128
        # Use a small epoch since the data is simple
        STEPS_PER_EPOCH = MP["n_train"] // MP["bs"]
        # use small validation steps since the epoch is small
        VALIDATION_STEPS = MP["n_dev"] // MP["bs"]

        IMAGE_CHANNEL_COUNT = 1
        MEAN_PIXEL = np.array([128.])

    class InferenceConfig(CraterConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

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