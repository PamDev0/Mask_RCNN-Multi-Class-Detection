from mrcnn.config import Config
from mrcnn import model as modellib, utils
from matplotlib import pyplot as plt
from PIL import *
from mrcnn.visualize import display_images
from mrcnn import visualize
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
import streamlit as st
from glob import glob
import skimage.draw
import numpy as np
import datetime
import random
import json
import leaf
import sys
import io
import os

ROOT_DIR = os.path.abspath("../../../")
sys.path.append(ROOT_DIR)  # To find local version of the library


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        # print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path, plugin='pil', pilmode="RGB")
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        #file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        #skimage.io.imsave("/content/sample_data" + file_name, splash)
    #print("Saved to ", "/content/drive/MyDrive/MundodasPlantas/masks_rcnn" + file_name)

def inference(image, weights):
    
    class CustomConfig(Config):
        """Configuration for training on the leaf dataset.
        Derives from the base Config class and overrides some values.
        """
        # Give the configuration a recognizable name
        NAME = "custom"

        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        IMAGES_PER_GPU = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 2  # Background + (background, rust
        
        # Number of training steps per epoch
        STEPS_PER_EPOCH = 100

        # Skip detections with < 70% confidence
        DETECTION_MIN_CONFIDENCE = 0.7

        USE_MINI_MASK = True

        LEARNING_RATE = 0.002

    class InferenceConfig(CustomConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    weights_folder = os.path.split(os.path.abspath(weights))[0]

    model = modellib.MaskRCNN(mode="inference",
                              model_dir="weights_folder" + '/', config=config)

    weights_path = weights

    # Load weights
    print("Loading weights ", weights_path)
    
    model.load_weights(weights_path, by_name=True)

    image_path = image
    model_path = weights_path
        
    assert image_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        # print("Running on {}".format(image_path))
        # Read image

        image = skimage.io.imread(image_path, plugin='pil', pilmode="RGB")

        # Detect objects
        loaded_model = model.load_weights(model_path, by_name=True)
                
        r = model.detect([image], verbose=1)[0]
        # Predict 
        p = r
        class_names = ['BG', 'rust', 'background']

        predict = visualize.display_instances(image, p['rois'], p['masks'], p['class_ids'], 
                            class_names, p['scores'])

        splash = color_splash(image, r['masks'])
        splashed = display_images([splash], cols=1)
        
        st.image([image, predict, splashed])

        return splash

def remove_bg_from_image(splash, thresh_slider):
     ## (1) Read
    gray = cv2.cvtColor(splash, cv2.COLOR_BGR2GRAY)

    ## (2) Threshold
    th, threshed = cv2.threshold(gray, thresh_slider, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    ## (3) Find the min-area contour
    _cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(_cnts, key=cv2.contourArea)
    for cnt in cnts:
        if cv2.contourArea(cnt) > 100:
            break

    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    #gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)

    fill_color = [255, 255, 255] # any BGR color value to fill with
    mask_value = 255          # 1 channel white (can be any non-zero uint8 value)

    # our stencil - some `mask_value` contours on black (zeros) background, 
    # the image has same height and width as `img`, but only 1 color channel
    stencil  = np.zeros(gray.shape[:-1]).astype(np.uint8)
    cv2.fillPoly(stencil, cnts, mask_value)

    sel = stencil != mask_value # select everything that is not mask_value
    gray[sel] = fill_color

    ## (4) Create mask and do bitwise-op
    mask = np.zeros(gray.shape[:-1],np.uint8)
    cv2.drawContours(mask, [cnt],-1, 255, -1)
    dst = cv2.bitwise_and(splash, splash, mask=mask)

    dst[mask == 0] = (255, 255, 255)

    no_bg = display_images([dst], cols=1)
    st.image(no_bg)



