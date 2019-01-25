# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
import numpy as np
import tifffile
import time
import deepmars.utils.transform as trf
import cartopy.crs as ccrs
import cartopy.img_transform as cimg
import h5py
from scipy.ndimage import zoom
from skimage.transform import resize
from skimage import img_as_int, img_as_uint, img_as_ubyte, exposure
from PIL import Image
import imageio
import collections
from deepmars.data.common import *

@click.group()
def data():
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

def GenDataset(img, craters, loc, rawlen_range=[512, 1024],
               rawlen_dist='log', ilen=256, cdim=[-180., 180., -90., 90.],
               arad=3371., minpix=0, tglen=256, binary=True, rings=True,
               ringwidth=1, truncate=True, amt=100, istart=0, seed=None,
               verbose=False,sample=False, systematic=False):

    # just in case we ever make this user-selectable...
    origin = "upper"
    print(img.shape)
    AddPlateCarree_XY(craters, img.shape, cdim=cdim, origin=origin)

    iglobe = ccrs.Globe(semimajor_axis=arad*1000., semiminor_axis=arad*1000.,
                        ellipse=None)

    # Zero-padding for hdf5 keys.
    zeropad = 5#int(np.log10(amt)) + 1
    def lower(x,y):
        if x is None:
            return y
        else:
            return x if x < y else y
        
    def upper(x,y):
        if x is None:
            return y
        else:
            return x if x > y else y
    
    def check4(x,y,mapping=None):
        if mapping is None:
            mapping=[lower,upper,lower,upper]
        return [func(a,b) for func,a,b in zip(mapping,x,y)]

    latlim=[None,None,None,None]
    pixlim=[None,None,None,None]

    def stepping(amt,istart):
        overlap=0.2
        import itertools

        def yy(res, edge):
            step = int(res*edge)
            for ix,iy in itertools.product(np.arange(0,img.shape[0],step),
                                           np.arange(0,img.shape[1],step)):
                if ix+res > img.shape[0]:
                    ix=img.shape[0]-res
                if iy+res > img.shape[1]:
                    iy=img.shape[1]-res
                yield (ix,iy)
        res = rawlen_range[0]
        flag=True
        edge = 1 - overlap
        counter=0
        i=0
        mystart=0
        total_counter=0
        while flag:
            for counter,vals in enumerate(yy(res,edge)):
                total_counter+=1
                if total_counter > istart and total_counter <= istart+amt:
                    #logger.debug("rawlen %d" % res)
                    yield (i,vals[0],vals[1],res)
                    i=i+1
                else:
                    pass #throw away
                if total_counter >= istart+amt:
                    return
            if res>rawlen_range[1]:
                flag=False
            else:
                res=res*2
        

    [i,xc,yc,rawlen] = list(stepping(1,loc))[0]
    print(i,xc,yc,rawlen)
    i,xc,yc,rawlen = [0,44226, 46683, 1024]
    print(i,xc,yc,rawlen)


    if True:

        img_number = "img_{i:0{zp}d}".format(i=istart + i, zp=zeropad)

        box = np.array([xc, yc, xc + rawlen, yc + rawlen], dtype='int32')
        pixlim = check4(pixlim,box,[lower,lower,upper,upper])
        print("box=", box)
        im = img[box[0]:box[2],box[1]:box[3]]
        im = img_as_uint(exposure.rescale_intensity(im.astype(np.int32),out_range=(0,2**16-1)))
        print("imshape", img.shape, im.shape)
        # Obtain long/lat bounds for coordinate transform.
        ix = box[::2]
        iy = box[1::2]

        llong, llat = trf.pix2coord(ix, iy, cdim, img.shape,
                                    origin=origin)
        llbd = np.r_[llong, llat[::-1]]


        print("llbd: ", llbd, box)
        print(im.min(),im.max(), im.dtype, im.shape)
        im = resize(im,(ilen, ilen))
        im = img_as_ubyte(im)#-im.min())
        im = Image.fromarray(im.T,'L')#Image.open(img).convert("L")
#        print(im.min(),im.max(), im.shape)

        # Remove all craters that are too small to be seen in image.
        ctr_sub = ResampleCraters(craters, llbd, im.size[0], arad=arad,
                                  minpix=minpix)
        print("craters: ",ctr_sub['Long'].min(),ctr_sub['Long'].max(),ctr_sub['Lat'].min(),ctr_sub['Lat'].max())
        # Convert Plate Carree to Orthographic.
        [imgo_arr, ctr_xy, distortion_coefficient, clonglat_xy] = (
            PlateCarree_to_Orthographic(
                im, llbd, ctr_sub, iglobe=iglobe, ctr_sub=True,
                arad=arad, origin=origin, rgcoeff=1.2, slivercut=0.2))
        print("craters: ",ctr_sub['x'].min(),ctr_sub['x'].max(),ctr_sub['y'].min(),ctr_sub['y'].max())
        if imgo_arr is None:
            print("Discarding narrow image: {} {} {} {}".format(*llbd))

        assert np.asanyarray(imgo_arr).sum() > 0, ("Sum of imgo is zero!  There likely was "
                                    "an error in projecting the cropped "
                                    "image.")

        # Make target mask.  Used Image.BILINEAR resampling because
        # Image.NEAREST creates artifacts.  Try Image.LANZCOS if BILINEAR still
        # leaves artifacts).
        #tgt = resize(imgo_arr,(tglen, tglen))
        tgt = np.asanyarray(imgo_arr.resize((tglen,tglen),resample=Image.BILINEAR))

        import deepmars.data.mask as dm
        mask = dm.make_mask(ctr_xy, tgt, binary=binary, rings=rings,
                         ringwidth=ringwidth, truncate=False)#truncate)

        # Output everything to file.
        return np.asanyarray(imgo_arr), mask, np.asanyarray(im)


def make_dataset(filename, loc, source_cdim=[-180,180,-90,90], sub_cdim=[-180,180,-90,90], rawlen_range=(256,8192),mola=None):#input_filepath, output_filepath):

    ilen = 256
    tglen = 256
    minpix = 3.
    R_km = 3371.0
    truncate = True
    ringwidth = 1
    verbose = True
    craters = ReadRobbinsCraters()
    sample=False
    systematic=True
    if mola is None:
        img = MarsDEM(filename).T
    else:
        img = mola

    return img, GenDataset(img, craters, loc, rawlen_range=rawlen_range,
                    ilen=ilen, cdim=sub_cdim,
                    arad=R_km, minpix=minpix, tglen=tglen, binary=True,
                    rings=True, ringwidth=ringwidth, truncate=truncate,
                            amt=1, istart=loc, verbose=verbose,sample=sample,systematic=systematic)


if __name__=="__main__":
    load_dotenv(find_dotenv())
    print(make_dataset("./data/external/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif",175100))
