from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import h5py
import pandas as pd
import numpy as np

class CraterConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "craters"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 4000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 1000


class CraterDataset(utils.Dataset):
    """Load the crater dataset from a pregenerated HDF5 file."""
    def load_image(self, image_id):
        pass


    def load_mask(self, image_id):
        pass


    def image_reference(self, image_id):
        pass


    def load_shapes(self, indices):
        self.index = indices
        self.directory = "/disks/work/lee/DL/dmrcnn/data/processed/"

        self.image_file = []
        self.crater_file = []
        self.gen_imgs = []
        self.gen_craters = []
        self.count = []
        counter=0
        for jval, index in enumerate(indices):
            image_file = self.directory + "sys_images_{:05d}.hdf5".format(index) 
            crater_file =  self.directory + "sys_craters_{:05d}.hdf5".format(index)
            gen_imgs =  h5py.File(image_file, 'r')
            gen_craters = pd.HDFStore(crater_file, 'r')
            self.gen_imgs.append(gen_imgs)
            self.add_class("shapes",1,"crater")
            count = len(gen_imgs["input_images"])
            self.count.append(count)
            from tqdm import tqdm, trange
            gi = gen_imgs["input_images"][:]
            for i in trange(count):
                #loop over images and craters, grab crater locations
                image_name="img_{:05d}".format(i+index)
                if image_name not in gen_craters:
                    continue
                image=None
                image = np.ones([256,256, 3], dtype=np.uint8)
                im = gi[i]
                image[:, :, 0] = im
                image[:, :, 1] = im
                image[:, :, 2] = im
                
                craters = gen_craters[image_name]
                craters["Radius (pix)"] = craters["Diameter (pix)"]/2
                craters = craters[craters["Diameter (pix)"] > 6]
                craters = craters[["x","y","Radius (pix)"]].values.astype(int)
                if len(craters) == 0:
                    continue
                self.add_image("shapes", image_id=counter, path=jval,
                               width=256, height=256,
                               craters=craters, image=image, mask=None, class_ids=None,
                               image_name=image_name,
                               shapes=["crater"]*len(craters), image_index=i)

#            gen_craters.close()
#            gen_imgs.close()
                
    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        if info["image"] is None:
            image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
            im = self.gen_imgs[info["path"]]["input_images"][info["image_index"]]
            image[:,:,0] = im
            image[:,:,1] = im
            image[:,:,2] = im
            self.image_info[image_id]["image"] = image
        else:
            image = self.image_info[image_id]["image"]
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["craters"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        if info["mask"] is None:
            craters = info['craters']
            count = len(craters)
            mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
            for i, dims in enumerate(craters):
                mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                "circle", dims, 1)
            # Handle occlusions
            occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
            for i in range(count-2, -1, -1):
                mask[:, :, i] = mask[:, :, i] * occlusion
                occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
            # Map class names to class IDs.
            class_ids = np.array([self.class_names.index(s) for s in info["shapes"]])
            info["mask"] = mask.astype(np.bool)
            info["class_ids"] = class_ids.astype(np.int32)
        else:
            mask, class_ids = info["mask"], info["class_ids"]
        return mask, class_ids

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == "circle":
            cv2.circle(image, (x, y), s, color, 2)
        return image

