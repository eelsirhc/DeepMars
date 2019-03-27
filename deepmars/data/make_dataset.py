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


def GenDataset(img, craters, outhead, rawlen_range=[512, 1024],
               rawlen_dist='log', ilen=256, cdim=[-180., 180., -90., 90.],
               arad=3371., minpix=0, tglen=256, binary=True, rings=True,
               ringwidth=1, truncate=True, amt=100, istart=0, seed=None,
               verbose=False,sample=False, systematic=False):
    """Generates random dataset from a global DEM and crater catalogue.

    The function randomly samples small images from a global digital elevation
    map (DEM) that uses a Plate Carree projection, and converts the small
    images to Orthographic projection.  Pixel coordinates and radii of craters
    from the catalogue that fall within each image are placed in a
    corresponding Pandas dataframe.  Images and dataframes are saved to disk in
    hdf5 format.

    Parameters
    ----------
    img : PIL.Image.Image
        Source image.
    craters : pandas.DataFrame
        Crater catalogue .csv.
    outhead : str
        Filepath and file prefix of the image and crater table hdf5 files.
    rawlen_range : list-like, optional
        Lower and upper bounds of raw image widths, in pixels, to crop from
        source.  To always crop the same sized image, set lower bound to the
        same value as the upper.  Default is [300, 4000].
    rawlen_dist : 'uniform' or 'log'
        Distribution from which to randomly sample image widths.  'uniform' is
        uniform sampling, and 'log' is loguniform sampling.
    ilen : int, optional
        Input image width, in pixels.  Cropped images will be downsampled to
        this size.  Default is 256.
    cdim : list-like, optional
        Coordinate limits (x_min, x_max, y_min, y_max) of image.  Default is
        LRO-Kaguya's [-180., 180., -60., 60.].
    arad : float. optional
        World radius in km.  Defaults to Moon radius (1737.4 km).
    minpix : int, optional
        Minimum crater diameter in pixels to be included in crater list.
        Useful when the smallest craters in the catalogue are smaller than 1
        pixel in diameter.
    tglen : int, optional
        Target image width, in pixels.
    binary : bool, optional
        If True, returns a binary image of crater masks.
    rings : bool, optional
        If True, mask uses hollow rings rather than filled circles.
    ringwidth : int, optional
        If rings is True, ringwidth sets the width (dr) of the ring.
    truncate : bool
        If True, truncate mask where image truncates.
    amt : int, optional
        Number of images to produce.  100 by default.
    istart : int
        Output file starting number, when creating datasets spanning multiple
        files.
    seed : int or None
        np.random.seed input (for testing purposes).
    verbose : bool
        If True, prints out number of image being generated.
    """


    logger = logging.getLogger(__name__)
    logger.info('making data set')
    # just in case we ever make this user-selectable...
    origin = "upper"

    # Seed random number generator.
    if seed is None:
        seed = os.getenv("DM_SEED")
    
    if seed is not None:
        seed=int(seed)
        logger.info("Seed: %d" % seed)

    np.random.seed(seed)

    # Get craters.
    if not sample:
        AddPlateCarree_XY(craters, img.shape, cdim=cdim, origin=origin)

    iglobe = ccrs.Globe(semimajor_axis=arad*1000., semiminor_axis=arad*1000.,
                        ellipse=None)

    # Create random sampler (either uniform or loguniform).
    if rawlen_dist == 'log':
        rawlen_min = np.log10(rawlen_range[0])
        rawlen_max = np.log10(rawlen_range[1])

        def random_sampler():
            return int(10**np.random.uniform(rawlen_min, rawlen_max))
    else:

        def random_sampler():
            return np.random.randint(rawlen_range[0], rawlen_range[1] + 1)

    # Initialize output hdf5s.
    imgs_h5 = h5py.File(outhead + '_images_{:05d}.hdf5'.format(istart), 'w')
    imgs_h5_inputs = imgs_h5.create_dataset("input_images", (amt, ilen, ilen),
                                            dtype='uint8', compression="gzip", compression_opts=9)
    imgs_h5_inputs.attrs['definition'] = "Input image dataset."
    imgs_h5_tgts = imgs_h5.create_dataset("target_masks", (amt, tglen, tglen),
                                          dtype='float32', compression="gzip", compression_opts=9)
    imgs_h5_tgts.attrs['definition'] = "Target mask dataset."
    imgs_h5_llbd = imgs_h5.create_group("longlat_bounds")
    imgs_h5_llbd.attrs['definition'] = ("(long min, long max, lat min, "
                                        "lat max) of the cropped image.")
    imgs_h5_box = imgs_h5.create_group("pix_bounds")
    imgs_h5_box.attrs['definition'] = ("Pixel bounds of the Global DEM region"
                                       " that was cropped for the image.")
    imgs_h5_dc = imgs_h5.create_group("pix_distortion_coefficient")
    imgs_h5_dc.attrs['definition'] = ("Distortion coefficient due to "
                                      "projection transformation.")
    imgs_h5_cll = imgs_h5.create_group("cll_xy")
    imgs_h5_cll.attrs['definition'] = ("(x, y) pixel coordinates of the "
                                       "central long / lat.")
    craters_h5 = pd.HDFStore(outhead + '_craters_{:05d}.hdf5'.format(istart), 'w')


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

    def randomized(amt,istart):
        for i in range(amt):
            # Determine image size to crop.
            rawlen = random_sampler()
            logger.debug("rawlen %d" % rawlen)
            if rawlen > img.shape[1]:
                rawlen = img.shape[1]
                yc=0
                xc = np.random.randint(0, img.shape[0] - rawlen)
                logger.debug("using all of the Y domain")
            else:
                xc = np.random.randint(0, img.shape[0] - rawlen)
                yc = np.random.randint(0, img.shape[1] - rawlen)
            yield (i,xc,yc,rawlen)

#def uniform_yield(start,amt,dims,res_low,res_high,overlap=0.2):
    def stepping(amt,istart):
        overlap=0.25
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
                    logger.debug("rawlen %d" % res)
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

    if systematic:
        iterator = stepping
    else:
        iterator = randomized

    for i,xc,yc,rawlen in iterator(amt,istart):
        #print("==== ",i,xc,yc,rawlen)
        # Current image number.
        img_number = "img_{i:0{zp}d}".format(i=istart + i, zp=zeropad)
        if verbose:
            logger.info("Generating {0}".format(img_number))


        box = np.array([xc, yc, xc + rawlen, yc + rawlen], dtype='int32')
        logger.debug("{} {}".format(xc,yc))
        logger.debug("{}".format(img.shape))
        pixlim = check4(pixlim,box,[lower,lower,upper,upper])
        
        # Load necessary because crop may be a lazy operation; im.load() should
        # copy it.  See <http://pillow.readthedocs.io/en/3.1.x/
        # reference/Image.html>.
        #im = img.crop(box)
        #im.load()
        #print(box, img.shape)
        im = img[box[0]:box[2],box[1]:box[3]]
        im = img_as_uint(exposure.rescale_intensity(im.astype(np.int32),out_range=(0,2**16-1)))
        # Obtain long/lat bounds for coordinate transform.
        ix = box[::2]
        iy = box[1::2]

        llong, llat = trf.pix2coord(ix, iy, cdim, img.shape,
                                    origin=origin)
        llbd = np.r_[llong, llat[::-1]]


        logger.info("Limits: {} {} {} {}".format(*llbd))
        if sample:
            #save the limits here.
            sds_box = imgs_h5_box.create_dataset(img_number, (4,), dtype='int32')
            sds_box[...] = box
            sds_llbd = imgs_h5_llbd.create_dataset(img_number, (4,), dtype='float')
            sds_llbd[...] = llbd
            continue
        # Downsample image.
#        print(im.dtype)
#        print(im[im.shape[0]//2])
        
        print(im.min(),im.max(), im.dtype)
        im = resize(im,(ilen, ilen))
        im = img_as_ubyte(im)#-im.min()
        im = Image.fromarray(im.T,'L')#Image.open(img).convert("L")        

#        print(im[im.shape[0]//2])
#        print(im.dtype)
        # Remove all craters that are too small to be seen in image.
        ctr_sub = ResampleCraters(craters, llbd, im.size[0], arad=arad,
                                  minpix=minpix)

#        im =  Image.fromarray(im.T, mode="L")
#        #print(llong, llat,im.size)

        # Convert Plate Carree to Orthographic.
        [imgo_arr, ctr_xy, distortion_coefficient, clonglat_xy] = (
            PlateCarree_to_Orthographic(
                im, llbd, ctr_sub, iglobe=iglobe, ctr_sub=True,
                arad=arad, origin=origin, rgcoeff=1.2, slivercut=0.2))

        if imgo_arr is None:
            logger.warning("Discarding narrow image: {} {} {} {}".format(*llbd))
            continue
#        imgo_arr = np.asanyarray(imgo)
        #print(imgo_arr, imgo_arr.sum())
        assert np.asanyarray(imgo_arr).sum() > 0, ("Sum of imgo is zero!  There likely was "
                                    "an error in projecting the cropped "
                                    "image.")

        # Make target mask.  Used Image.BILINEAR resampling because
        # Image.NEAREST creates artifacts.  Try Image.LANZCOS if BILINEAR still
        # leaves artifacts).
#        tgt = resize(imgo_arr,(tglen, tglen))
        tgt = np.asanyarray(imgo_arr.resize((tglen,tglen),resample=Image.BILINEAR))

        import deepmars.data.mask as dm
        mask = dm.make_mask(ctr_xy, tgt, binary=binary, rings=rings,
                         ringwidth=ringwidth, truncate=truncate)

        # Output everything to file.
        imgs_h5_inputs[i, ...] = imgo_arr
        imgs_h5_tgts[i, ...] = mask
        #save the limits here.
        sds_box = imgs_h5_box.create_dataset(img_number, (4,), dtype='int32')
        sds_box[...] = box
        sds_llbd = imgs_h5_llbd.create_dataset(img_number, (4,), dtype='float')
        sds_llbd[...] = llbd
        sds_dc = imgs_h5_dc.create_dataset(img_number, (1,), dtype='float')
        sds_dc[...] = np.array([distortion_coefficient])
        sds_cll = imgs_h5_cll.create_dataset(img_number, (2,), dtype='float')
        sds_cll[...] = clonglat_xy.loc[:, ['x', 'y']].values.ravel()

        craters_h5[img_number] = ctr_xy

        imgs_h5.flush()
        craters_h5.flush()
    
    imgs_h5.close()
    craters_h5.close()


@data.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
@click.option("--filename",default=None,type=str)
@click.option("--istart",default=0)
@click.option("--amt",default=3000)
@click.option("--sample",is_flag=True,default=False)
@click.option("--systematic",is_flag=True,default=False)
@click.option("--prefix",default="test")
@click.option("--source_cdim", default= (-180., 180., -90., 90.), nargs=4,type=float)
@click.option("--sub_cdim", default= (-180., 180., -90., 90.), nargs=4,type=float)
@click.option("--rawlen_range", default=(256,8192), nargs=2,type=int)
def make_dataset(filename, istart, amt,sample,systematic,prefix, source_cdim, sub_cdim, rawlen_range):#input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    start_time = time.time()

    # Use MPI4py?  Set this to False if it's not supposed by the system.
    use_mpi4py = False
    
        # Output filepath and file header.  Eg. if outhead = "./input_data/train",
    # files will have extension "./out/train_inputs.hdf5" and
    # "./out/train_targets.hdf5"
    outhead = os.path.join(os.getenv("DM_ROOTDIR"),"data/processed/{}".format(prefix))
    
    # Number of images to make (if using MPI4py, number of image per thread to
    # make).
#    amt = 3000

    
    # Range of image widths, in pixels, to crop from source image (input images
    # will be scaled down to ilen). For Orthogonal projection, larger images are
    # distorted at their edges, so there is some trade-off between ensuring images
    # have minimal distortion, and including the largest craters in the image.
    
    # Distribution to sample from rawlen_range - "uniform" for uniform, and "log"
    # for loguniform.
    rawlen_dist = 'log'
    
    # Size of input images.
    ilen = 256
    
    # Size of target images.
    tglen = 256
    
    # [Min long, max long, min lat, max lat] dimensions of source image.
#    if source_cdim is None:
#        source_cdim =
    
    # [Min long, max long, min lat, max lat] dimensions of the region of the source
    # to use when randomly cropping.  Used to distinguish training from test sets.
#    if sub_cdim is None:
#        sub_cdim = [-180., 180., -90., 90.]
    
    # Minimum pixel diameter of craters to include in in the target.
    minpix = 3.
    
    # Radius of the world in km (1737.4 for Moon).
    R_km = 3371.0
    
    ### Target mask arguments. ###
    
    # If True, truncate mask where image has padding.
    truncate = True
    
    # If rings = True, thickness of ring in pixels.
    ringwidth = 1
    
    # If True, script prints out the image it's currently working on.
    verbose = True

    if sample:
        craters=None
        logger.info("no craters loaded")
    else:
        craters = ReadRobbinsCraters()
        logger.info("found {} craters in the database".format(len(craters)))
#
    logger.info("Mars Reading image :{}".format(filename))
    img = MarsTHEMIS(filename)[::8,::8].T
#    img = MarsDEM(filename).T
    logger.info("Mars DEM resolution {} by {}".format(img.shape[0],img.shape[1]))

    if use_mpi4py:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        #print("Thread {0} of {1}".format(rank, size))
        istart = rank * amt
        sub_cdim = [-180., 180., -90., 90.]
        if rank < 2:
            msize=1
            orank = rank
        else:
            msize = 2**(np.floor(np.log2(rank))).astype(int)
            orank = 2**(np.floor(np.log2(rank))).astype(int)
        rk = rank
        rank = rank - orank
        result = [i for i in range(1,1+msize//1) if size%i==0]
        xv = result[len(result)//2]
        yv = msize//xv
        dx = (source_cdim[1]-source_cdim[0])/xv
        dy = (source_cdim[3]-source_cdim[2])/yv
        #print(xv,yv,dx,dy)
        overlap=0.1
        sub_cdim = np.array([-180+dx*(rank%xv)-overlap*dx,
                             -180+dx*(1+rank%xv)+overlap*dy,
                             -90+dy*(rank//xv)-overlap*dx,
                             -90+dy*(1+rank//xv)+overlap*dy]).astype(int)
        sub_cdim = list(np.hstack([np.clip(sub_cdim[:2],-180,180),np.clip(sub_cdim[2:],-90,90)]).astype(int))

        print(",".join([str(x) for x in np.hstack([xv,yv,rk,sub_cdim])]))
    else:
        pass

    # Sample subset of image.  Co-opt igen.ResampleCraters to remove all
    # craters beyond cdim (either sub or source).
    if sub_cdim != source_cdim:
        img = InitialImageCut(img, source_cdim, sub_cdim)
        logger.info("Subsampled DEM resolution {} by {}".format(img.shape[0],img.shape[1]))
        logger.info("Covering Long {} to {}, lat {} to {}".format(sub_cdim[0],sub_cdim[1],sub_cdim[2],sub_cdim[3]))

    # This always works, since sub_cdim < source_cdim.
    if not sample:
        craters = ResampleCraters(craters, sub_cdim, None, arad=R_km)
    GenDataset(img, craters, outhead, rawlen_range=rawlen_range,
                    rawlen_dist=rawlen_dist, ilen=ilen, cdim=sub_cdim,
                    arad=R_km, minpix=minpix, tglen=tglen, binary=True,
                    rings=True, ringwidth=ringwidth, truncate=truncate,
                    amt=amt, istart=istart, verbose=verbose,sample=sample,systematic=systematic)

    elapsed_time = time.time() - start_time
    logger.info("Time elapsed: {0:.1f} min".format(elapsed_time / 60.))

if __name__ == '__main__':
    data()
