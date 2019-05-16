#!/usr/bin/env python
"""Convolutional Neural Network Training Functions

Functions for building and training a (UNET) Convolutional Neural Network on
images of the Moon and binary ring targets.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import h5py

from keras.models import Model
from keras.layers.core import Dropout, Reshape
from keras.regularizers import l2

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard
from keras import backend as K
K.set_image_dim_ordering('tf')

import deepmars.features.template_match_target as tmt
import deepmars.utils.processing as proc

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
from joblib import Parallel, delayed
from tqdm import tqdm, trange

# Check Keras version - code will switch API if needed.
from keras import __version__ as keras_version
k2 = True if keras_version[0] == '2' else False

# If Keras is v2.x.x, create Keras 1-syntax wrappers.
if not k2:
    from keras.models import load_model
    from keras.layers import merge, Input
    from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                            UpSampling2D)

else:
    from keras.models import load_model
    from keras.layers import Concatenate, Input
    from keras.layers.convolutional import (Conv2D, MaxPooling2D,
                                            UpSampling2D)

    def merge(layers, mode=None, concat_axis=None):
        """Wrapper for Keras 2's Concatenate class (`mode` is discarded)."""
        return Concatenate(axis=concat_axis)(list(layers))

    def Convolution2D(n_filters, FL, FLredundant, activation=None,
                      init=None, W_regularizer=None, border_mode=None):
        """Wrapper for Keras 2's Conv2D class."""
        return Conv2D(n_filters, FL, activation=activation,
                      kernel_initializer=init,
                      kernel_regularizer=W_regularizer,
                      padding=border_mode)

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


########################
def get_param_i(param, i):
    """Gets correct parameter for iteration i.

    Parameters
    ----------
    param : list
        List of model hyperparameters to be iterated over.
    i : integer
        Hyperparameter iteration.

    Returns
    -------
    Correct hyperparameter for iteration i.
    """
    if len(param) > i:
        return param[i]
    else:
        return param[0]

########################
def custom_image_generator(data, target, batch_size=32):
    """Custom image generator that manipulates image/target pairs to prevent
    overfitting in the Convolutional Neural Network.

    Parameters
    ----------
    data : array
        Input images.
    target : array
        Target images.
    batch_size : int, optional
        Batch size for image manipulation.

    Yields
    ------
    Manipulated images and targets.
        
    """
    D, L, W = data.shape[0],data[0].shape[0], data[0].shape[1]
    while True:
        shuffle_index = np.arange(D)
        np.random.shuffle(shuffle_index) #only shuffle once each loop through the data
        for i in np.arange(0, len(data), batch_size):
#            print("a",i,batch_size, len(data))
            index = shuffle_index[i:i+batch_size]
            d, t = data[index].copy(), target[index].copy()

            # Random color inversion
            # for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
            #     d[j][d[j] > 0.] = 1. - d[j][d[j] > 0.]

            # Horizontal/vertical flips
            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
#                print(d.shape, t.shape, batch_size)
                d[j], t[j] = np.fliplr(d[j]), np.fliplr(t[j])      # left/right
            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
#                print(d.shape, t.shape, batch_size)
                d[j], t[j] = np.flipud(d[j]), np.flipud(t[j])      # up/down

            # Random up/down & left/right pixel shifts, 90 degree rotations
            npix = 15
            h = np.random.randint(-npix, npix + 1, batch_size)    # Horizontal shift
            v = np.random.randint(-npix, npix + 1, batch_size)    # Vertical shift
            r = np.random.randint(0, 4, batch_size)               # 90 degree rotations
            for j in range(batch_size):
                d[j] = np.pad(d[j], ((npix, npix), (npix, npix), (0, 0)),
                              mode='constant')[npix + h[j]:L + h[j] + npix,
                                               npix + v[j]:W + v[j] + npix, :]
                t[j] = np.pad(t[j], (npix,), mode='constant')[npix + h[j]:L + h[j] + npix, 
                                                              npix + v[j]:W + v[j] + npix]
                d[j], t[j] = np.rot90(d[j], r[j]), np.rot90(t[j], r[j])
            yield (d, t)

def t2c(pred, csv,i,
        minrad=minrad_, maxrad=maxrad_, longlat_thresh2=longlat_thresh2_,
        rad_thresh=rad_thresh_, template_thresh=template_thresh_, target_thresh=target_thresh_) :
#    print(minrad, maxrad, longlat_thresh2, rad_thresh, template_thresh, target_thresh)
    return np.hstack([i,tmt.template_match_t2c(pred, csv, 
                                               minrad=minrad, maxrad=maxrad, longlat_thresh2=longlat_thresh2,
                                               rad_thresh=rad_thresh, template_thresh=template_thresh, target_thresh=target_thresh)])

def diagnostic(res,beta):
    counter,N_match, N_csv, N_detect, mrad, err_lo, err_la, err_r, frac_duplicates = np.array(res).T
       
    w=np.where(N_match==0)
    #if len(w[0]):
    #    for i in w[0]:
    #        print("skipping iteration %d,N_csv=%d,N_detect=%d,N_match=%d" %
    #              (counter[i], N_csv[i], N_detect[i], N_match[i]))

    w=np.where(N_match>0)
    counter,N_match, N_csv, N_detect, mrad, err_lo, err_la, errr_, frac_dupes =\
        counter[w],N_match[w], N_csv[w], N_detect[w], mrad[w], err_lo[w], err_la[w], err_r[w], frac_duplicates[w]
    
    precision = N_match/(N_match + (N_detect - N_match))
    recall = N_match/N_csv
    fscore = (1 + beta**2) * (recall * precision) / (precision * beta**2 + recall)
    diff = N_detect - N_match
    frac_new = diff / (N_detect + diff)
    frac_new2 = diff / (N_csv + diff)
    frac_duplicates = frac_dupes
    
    return dict(precision=precision,
                recall=recall,
                fscore=fscore,
                frac_new=frac_new,
                frac_new2=frac_new2,
                err_lo=err_lo,
                err_la=err_la,
                err_r=err_r,
                frac_duplicates=frac_duplicates,
                maxrad=mrad,
                counter=counter, N_match=N_match, N_csv=N_csv)


def get_metrics(data, craters_images, dim, model, name,beta=1,offset=0,
                minrad=minrad_, maxrad=maxrad_,
                longlat_thresh2=longlat_thresh2_,
                rad_thresh=rad_thresh_, template_thresh=template_thresh_,
                target_thresh=target_thresh_, rmv_oor_csvs=0):
    """Function that prints pertinent metrics at the end of each epoch. 

    Parameters
    ----------
    data : hdf5
        Input images.
    craters : hdf5
        Pandas arrays of human-counted crater data. 
    dim : int
        Dimension of input images (assumes square).
    model : keras model object
        Keras model
    beta : int, optional
        Beta value when calculating F-beta score. Defaults to 1.
    """
    X, Y = data[0], data[1]
    craters, images = craters_images
    # Get csvs of human-counted craters
    csvs = []
#    minrad, maxrad = 3, 50
    cutrad, n_csvs = 0.8, len(X)
    diam = 'Diameter (pix)'
    #print(n_csvs,list(craters.keys()))
    for i in range(len(X)):
        imname = images[i]#        name = "img_{0:05d}".format(i)
        found = False
        for crat in craters:
            if imname in crat:
                csv = crat[imname]
                found=True
        if not found:
            csvs.append([-2])
#            print("not found: ",imname)
            continue
#        print("found: ",imname)
        # remove small/large/half craters
        csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
        csv = csv[(csv['x'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['y'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['x'] - cutrad * csv[diam] / 2 > 0)]
        csv = csv[(csv['y'] - cutrad * csv[diam] / 2 > 0)]
        if len(csv) < 3:    # Exclude csvs with few craters
            csvs.append([-1])
        else:
            csv_coords = np.asarray((csv['x'], csv['y'], csv[diam] / 2)).T
            csvs.append(csv_coords)

    # Calculate custom metrics
    print("csvs: {}".format(len(csvs)))
    print("")
    print("*********Custom Loss*********")
    recall, precision, fscore = [], [], []
    frac_new, frac_new2, mrad = [], [], []
    err_lo, err_la, err_r = [], [], []
    frac_duplicates = []

    if isinstance(model, Model):
        preds = None
#        print(X[6].min(),X[6].max(),X.dtype,np.percentile(X[6],99))
        preds = model.predict(X, verbose=1)
        # save
        h5f = h5py.File("predictions.hdf5", 'w')
        h5f.create_dataset(name, data=preds)
        print("Successfully generated and saved model predictions.")
    else:
        preds = model
    #print(csvs)
    countme = [i for i in range(n_csvs) if len(csvs[i])>=3]
    print("Processing {} fields".format(len(countme)))

#    kwargs = dict(minrad=minrad, maxrad=maxrad, longlat_thresh2=longlat_thresh2,
#                  rad_thresh=rad_thresh, template_thresh=template_thresh, target_thresh=target_thresh)
    res = Parallel(n_jobs=24, verbose=5)(delayed(t2c)(preds[i], csvs[i],i, 
                                                      minrad=minrad, maxrad=maxrad, longlat_thresh2=longlat_thresh2,
                                                      rad_thresh=rad_thresh, template_thresh=template_thresh, target_thresh=target_thresh) 
                                         for i in range(n_csvs) if len(csvs[i])>=3)

    if len(res)==0:
        print("No valid results: ", res)
        return None
    diag = diagnostic(res,beta)
    print(len(diag["recall"]))
    #print("binary XE score = %f" % model.evaluate(X, Y))
    if len(diag["recall"]) > 3:
        for fname,data in [("N_match/N_csv (recall)",diag["recall"]),
                         ("N_match/(N_match + (N_detect-N_match)) (precision)",diag["precision"]),
                         ("F_{} score".format(beta), diag["fscore"]),
                         ("(N_detect - N_match)/N_detect (fraction of craters that are new)", diag["frac_new"]),
                         ("(N_detect - N_match)/N_csv (fraction of craters that are new, 2)", diag["frac_new2"])]:
            print("mean and std of %s = %f, %f" %
                  (fname,np.mean(data), np.std(data)))
        for fname,data in [("fractional longitude diff",diag["err_lo"]),
                          ("fractional latitude diff",diag["err_la"]),
                          ("fractional radius diff", diag["err_r"]),
                         ]:
            print("median and IQR %s = %f, 25:%f, 75:%f" %
             (fname, np.median(data), np.percentile(data, 25), np.percentile(data, 75)))

        print("""mean and std of maximum detected pixel radius in an image =
             %f, %f""" % (np.mean(diag["maxrad"]), np.std(diag["maxrad"])))
        print("""absolute maximum detected pixel radius over all images =
              %f""" % np.max(diag["maxrad"]))
        print("")
        return diag

########################
def build_model(dim, learn_rate, lmbda, drop, FL, init, n_filters):
    """Function that builds the (UNET) convolutional neural network. 

    Parameters
    ----------
    dim : int
        Dimension of input images (assumes square).
    learn_rate : float
        Learning rate.
    lmbda : float
        Convolution2D regularization parameter. 
    drop : float
        Dropout fraction.
    FL : int
        Filter length.
    init : string
        Weight initialization type.
    n_filters : int
        Number of filters in each layer.

    Returns
    -------
    model : keras model object
        Constructed Keras model.
    """
    print('Making UNET model...')
    img_input = Input(batch_shape=(None, dim, dim, 1))

    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(img_input)
    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a1)
    a1P = MaxPooling2D((2, 2), strides=(2, 2))(a1)

    a2 = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a1P)
    a2 = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a2)
    a2P = MaxPooling2D((2, 2), strides=(2, 2))(a2)

    a3 = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a2P)
    a3 = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a3)
    a3P = MaxPooling2D((2, 2), strides=(2, 2),)(a3)

    u = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(a3P)
    u = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2, 2))(u)
    u = merge((a3, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2, 2))(u)
    u = merge((a2, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2, 2))(u)
    u = merge((a1, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    # Final output
    final_activation = 'sigmoid'
    u = Convolution2D(1, 1, 1, activation=final_activation, init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Reshape((dim, dim))(u)
    if k2:
        model = Model(inputs=img_input, outputs=u)
    else:
        model = Model(input=img_input, output=u)

    optimizer = Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    print(model.summary())

    return model

########################
def test_model(Data, Craters, MP, i_MP):
# Static params
    dim, nb_epoch, bs = MP['dim'], MP['epochs'], MP['bs']

    # Iterating params
    FL = get_param_i(MP['filter_length'], i_MP)
    learn_rate = get_param_i(MP['lr'], i_MP)
    n_filters = get_param_i(MP['n_filters'], i_MP)
    init = get_param_i(MP['init'], i_MP)
    lmbda = get_param_i(MP['lambda'], i_MP)
    drop = get_param_i(MP['dropout'], i_MP)

    model = load_model(MP["model"])
    get_metrics(Data[MP["test_dataset"]], Craters[MP["test_dataset"]], dim, model,MP["test_dataset"])


def train_and_test_model(Data, Craters, MP, i_MP):
    """Function that trains, tests and saves the model, printing out metrics
    after each model. 

    Parameters
    ----------
    Data : dict
        Inputs and Target Moon data.
    Craters : dict
        Human-counted crater data.
    MP : dict
        Contains all relevant parameters.
    i_MP : int
        Iteration number (when iterating over hypers).
    """
    # Static params
    dim, nb_epoch, bs = MP['dim'], MP['epochs'], MP['bs']

    # Iterating params
    FL = get_param_i(MP['filter_length'], i_MP)
    learn_rate = get_param_i(MP['lr'], i_MP)
    n_filters = get_param_i(MP['n_filters'], i_MP)
    init = get_param_i(MP['init'], i_MP)
    lmbda = get_param_i(MP['lambda'], i_MP)
    drop = get_param_i(MP['dropout'], i_MP)

    # Build model
    if MP["model"] is not None:
        model = load_model(MP["model"])
    else:
        model = build_model(dim, learn_rate, lmbda, drop, FL, init, n_filters)

    # Main loop
    n_samples = MP['n_train']
    for nb in range(nb_epoch):
        if k2:
            model.fit_generator(
                custom_image_generator(Data['train'][0], Data['train'][1],
                                       batch_size=bs),
                steps_per_epoch=n_samples/bs, epochs=1, verbose=1,
                # validation_data=(Data['dev'][0],Data['dev'][1]), #no gen
                validation_data=custom_image_generator(Data['dev'][0],
                                                       Data['dev'][1],
                                                       batch_size=bs),
                validation_steps=MP['n_dev']/bs,
                callbacks=[
                    TensorBoard(log_dir="./logs",
                                histogram_freq=0,
                                batch_size=bs,
                                write_graph=True,
                                write_grads=False,
                                write_images=False,
                                embeddings_freq=0,
                                embeddings_layer_names=None,
                                embeddings_metadata=None,
                                embeddings_data=None,
                                update_freq='epoch'),
                    EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
        else:
            model.fit_generator(
                custom_image_generator(Data['train'][0], Data['train'][1],
                                       batch_size=bs),
                samples_per_epoch=n_samples, nb_epoch=1, verbose=1,
                # validation_data=(Data['dev'][0],Data['dev'][1]), #no gen
                validation_data=custom_image_generator(Data['dev'][0],
                                                       Data['dev'][1],
                                                       batch_size=bs),
                nb_val_samples=n_samples,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
        model_save_name = os.path.join(MP["save_dir"],"model_{}_{}_{}_{}_{}_{}_{}.hdf5".format(learn_rate,n_filters,init,lmbda,drop,nb,nb_epoch))
    
        if MP['save_models']:
            model.save(model_save_name)
        if MP["calculate_custom_loss"]:
            get_metrics(Data['dev'], Craters['dev'], dim, model,"dev")


    if MP["save_models"] == 1:
        model.save(os.path.join(MP["save_dir"],MP["final_save_name"]))

    print("###################################")
    print("##########END_OF_RUN_INFO##########")
    print("""learning_rate=%e, batch_size=%d, filter_length=%e, n_epoch=%d
          n_train=%d, img_dimensions=%d, init=%s, n_filters=%d, lambda=%e
          dropout=%f""" % (learn_rate, bs, FL, nb_epoch, MP['n_train'],
                           MP['dim'], init, n_filters, lmbda, drop))
    if MP["calculate_custom_loss"]:
        get_metrics(Data['test'], Craters['test'], dim, model,"test")
    print("###################################")
    print("###################################")

########################
def get_models(MP,prefix="sys"):
    """Top-level function that loads data files and calls train_and_test_model.

    Parameters
    ----------
    MP : dict
        Model Parameters.
    """
    dir = MP['dir']
    n_train, n_dev, n_test = MP['n_train'], MP['n_dev'], MP['n_test']

    # Load data
    def load_files(numbers,test, this_dataset,prefix="sys"):
        res0 = []
        res1 = []
        files = []
        craters = []
        images = []
        npic = 0
        if not test or (test and this_dataset):
            for n in tqdm(numbers):
                files.append(h5py.File(os.path.join(dir,"{0}_images_{1:05d}.hdf5".format(prefix, n)),'r'))
                images.extend( ["img_{0:05d}".format(a) for a in np.arange(n,n+1000)])
                res0.append(files[-1]["input_images"][:].astype('float32'))
                npic = npic + len(res0[-1])
                res1.append(files[-1]["target_masks"][:].astype('float32'))
                files[-1].close()
                craters.append(pd.HDFStore(os.path.join(dir,"{0}_craters_{1:05d}.hdf5".format(prefix,n)),'r'))
            res0 = np.vstack(res0)
            res1 = np.vstack(res1)
        return files, res0, res1,npic,craters, images

    train_files, train0,train1, Ntrain,train_craters,train_images = load_files(MP["train_indices"], MP["test"],MP["test_dataset"]=="train",prefix=prefix)
    print(Ntrain,n_train)

    dev_files, dev0,dev1, Ndev,dev_craters,dev_images = load_files(MP["dev_indices"],MP["test"],MP["test_dataset"]=="dev",prefix=prefix)
    print(Ndev,n_dev)

    test_files, test0,test1, Ntest,test_craters,test_images = load_files(MP["test_indices"], MP["test"], MP["test_dataset"]=="test",prefix=prefix)
    print(Ntest,n_test)

    Data = {
        "train":[train0,train1],
        "dev":[dev0,dev1],
        "test":[test0[:n_test],test1[:n_test]]
        }

    # Rescale, normalize, add extra dim
    proc.preprocess(Data)

    # Load ground-truth craters
    Craters = { 
        'train': [train_craters,train_images],
        'dev': [dev_craters,dev_images],
        'test': [test_craters, test_images]
    }

    # Iterate over parameters
    if MP["test"]:
        test_model(Data, Craters, MP, 0)
        return
    else:
        for i in range(MP['N_runs']):
            train_and_test_model(Data, Craters, MP, i)


@dl.command()
@click.option("--test",is_flag=True,default=False)
@click.option("--test_dataset", default="dev")
@click.option("--model", default=None)
@click.option("--prefix", default="sys")
def train_model(test,test_dataset,model,prefix):
    """Run Convolutional Neural Network Training
    
    Execute the training of a (UNET) Convolutional Neural Network on
    images of the Moon and binary ring targets.
    """
    
    # Model Parameters
    MP = {}
    
    # Directory of train/dev/test image and crater hdf5 files.
    MP['dir'] = os.path.join(os.getenv("DM_ROOTDIR"),'data/processed/')
    
    # Image width/height, assuming square images.
    MP['dim'] = 256
    
    # Batch size: smaller values = less memory but less accurate gradient estimate
    MP['bs'] = 10
    
    # Number of training epochs.
    MP['epochs'] = 30
    
    # Number of train/valid/test samples, needs to be a multiple of batch size.

    MP['train_indices'] = list(np.arange(0,20000,1000))
    MP['dev_indices']   = list(np.arange(20000,25000,1000))
    MP['test_indices']  = list(np.arange(25000,26000,1000))
    #    MP['test_indices']  = 90000#list(np.arange(10000,184000,8000))
                                 
    MP['n_train'] = len(MP["train_indices"])*1000
    MP['n_dev'] = len(MP["dev_indices"])*1000
    MP['n_test'] = len(MP["test_indices"])*1000
    print(MP["n_train"],MP["n_dev"],MP["n_test"])
    
    # Save model (binary flag) and directory.
    MP['save_models'] = 1
    MP["calculate_custom_loss"] = False
    MP['save_dir'] = 'models'
    MP['final_save_name'] = 'model.h5'

    #initial model
    MP["model"]=model

    #testing only
    MP["test"] = test
    MP["test_dataset"] = test_dataset
    
    # Model Parameters (to potentially iterate over, keep in lists).
    df = pd.read_csv("runs.csv")
    for na,ty in [("filter_length",int),
                        ("lr",float),
                        ("n_filters",int),
                        ("init",str),
                        ("lambda",float),
                        ("dropout",float)]:
        MP[na] = df[na].astype(ty).values
    
    MP['N_runs'] = len(MP['lambda'])                # Number of runs
    MP['filter_length'] = [3]       # Filter length
#    MP['lr'] = [0.0001]             # Learning rate
#    MP['n_filters'] = [112]         # Number of filters
#    MP['init'] = ['he_normal']      # Weight initialization
#    MP['lambda'] = [1e-6]           # Weight regularization
#    MP['dropout'] = [0.15]          # Dropout fraction
    
    # Iterating over parameters example.
    #    MP['N_runs'] = 2
    #    MP['lambda']=[1e-4,1e-4]
    print(MP)
    get_models(MP,prefix=prefix)


if __name__ == '__main__':
    dl()
