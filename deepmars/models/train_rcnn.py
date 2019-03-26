import rcnn
import click
from dotenv import find_dotenv, load_dotenv
import logging
from pathlib import Path
import os
import numpy as np

minrad_ = 5
maxrad_ = 40
longlat_thresh2_ = 1.8
rad_thresh_ = 1.0
template_thresh_ = 0.5
target_thresh_ = 0.1


@click.group()
def dl():
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


def train_rcnn(config, MP):

    # training dataset
    dataset_train = rcnn.CraterDataset()
    dataset_train.load_craters(MP["train_indices"])
    dataset_train.prepare()

    # Validation dataset
    dataset_val = rcnn.CraterDataset()
    dataset_val.load_craters(MP["dev_indices"])
    dataset_val.prepare()

    COCO_MODEL_PATH = os.path.join(MP["save_dir"], "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
            rcnn.utils.download_trained_weights(COCO_MODEL_PATH)
            
    # ## Create Model
    # Create model in training mode
    model = rcnn.modellib.MaskRCNN(mode="training",
                              config=config,
                              model_dir=MP["save_dir"])


    from imgaug import augmenters as iaa
    augmentation = iaa.Sequential([
        iaa.SomeOf((0,2), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Affine(rotate=90),
                       iaa.Affine(rotate=180),
                       iaa.Affine(rotate=270)])
        ]),
        iaa.Sometimes(0.5,iaa.OneOf([iaa.GaussianBlur((0.0, 3.0)),
                                     iaa.AverageBlur((2, 5)),
                                     iaa.GammaContrast((0.5,2.0))
                   ]))
    ])

#    
    # Which weights to start with?
    init_with = None#"last"#"#models/craters20190227T1444/mask_rcnn_craters_0004.h5"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        print(model.find_last())
        model.load_weights(model.find_last(), by_name=True)
    else:
        pass
#        print(init_with)
#        model.load_weights(init_path, by_name=True)

    print("full training")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                augmentation=augmentation,
                layers='all')

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/10,
                epochs=20,
                augmentation=augmentation,
                layers='all')
#    # Train in two stages:
#    # 1. Only the heads. Here we're freezing all the
#    # backbone layers and training only the randomly
#    # initialized layers (i.e. the ones that we didn't
#    # use pre-trained weights from MS COCO). To train
#    # only the head layers, pass `layers='heads'` to
#    # the `train()` function.
#    #
#    # 2. Fine-tune all layers. For this simple example
#    # it's not necessary, but we're including it to
#    # show the process. Simply pass `layers="all` to
#    # train all layers.
#
#
# Train the head branches
#    # Passing layers="heads" freezes all layers except the head
#    # layers. You can also pass a regular expression to select
#    # which layers to train by name pattern.
#    model.train(dataset_train, dataset_val, 
#                learning_rate=config.LEARNING_RATE, 
#                epochs=4, 
#                layers='heads')
#
#    # Fine tune all layers
#    # Passing layers="all" trains all layers. You can also 
#    # pass a regular expression to select which layers to
#    # train by name pattern.
#    model.train(dataset_train, dataset_val, 
#                learning_rate=config.LEARNING_RATE / 10,
#                epochs=16, 
#                layers="all")

    # Save weights
    model.keras_model.save_weights(MP["final_save_name"])




@dl.command()
@click.option("--model", default=None)
def train_model(model):
    """Run Convolutional Neural Network Training

    Execute the training of MaskRCNN on
    images of Mars and binary ring targets.
    """

    # Model Parameters
    MP = {}

    # Directory of train/dev/test image and crater hdf5 files.
    MP['dir'] = os.path.join(os.getenv("DM_ROOTDIR"),'data/processed/')

    # Image width/height, assuming square images.
    MP['dim'] = 256

    # Batch size: smaller values = less memory but less accurate gradient estimate
    MP['bs'] = 8

    # Number of training epochs.
    MP['epochs'] = 50

    # Number of train/valid/test samples, needs to be a multiple of batch size.

    MP['train_indices'] = list(np.arange(162000,206000,2000)) #list(np.arange(162000, 208000, 2000))
    MP['dev_indices']   = list(np.arange(161000,206000,4000)) #list(np.arange(161000, 206000, 4000))
    MP['test_indices']  = list(np.arange(163000,206000,2000)) #list(np.arange(163000, 206000, 4000))

    MP['n_train'] = len(MP["train_indices"])*1000
    MP['n_dev'] = len(MP["dev_indices"])*1000
    MP['n_test'] = len(MP["test_indices"])*1000
    print(MP["n_train"], MP["n_dev"], MP["n_test"])

    # Save model (binary flag) and directory.
    MP['save_models'] = 1
    MP["calculate_custom_loss"] = False
    MP['save_dir'] = 'models'
    MP['final_save_name'] = 'model_rcnn.h5'

    # initial model
    MP["model"] = model

       
    # Model Parameters (to potentially iterate over, keep in lists).
    #df = pd.read_csv("runs.csv")
    #for na,ty in [("filter_length", int),
    #                    ("lr", float),
    #                    ("n_filters", int),
    #                    ("init", str),
    #                    ("lambda", float),
    #                    ("dropout", float)]:
    #    MP[na] = df[na].astype(ty).values
    
    #MP['N_runs'] = len(MP['lambda'])                # Number of runs
    #MP['filter_length'] = [3]       # Filter length
#    MP['lr'] = [0.0001]             # Learning rate
#    MP['n_filters'] = [112]         # Number of filters
#    MP['init'] = ['he_normal']      # Weight initialization
#    MP['lambda'] = [1e-6]           # Weight regularization
#    MP['dropout'] = [0.15]          # Dropout fraction
    
    # Iterating over parameters example.
    #    MP['N_runs'] = 2
    #    MP['lambda']=[1e-4,1e-4]
    print(MP)
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
        BACKBONE = "resnet50"        
    config = CraterConfig()
    train_rcnn(config,MP)



if __name__ == '__main__':
    dl()
