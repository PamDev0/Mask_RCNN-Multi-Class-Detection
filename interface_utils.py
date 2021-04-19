from mrcnn import model as modellib, utils
from matplotlib import pyplot as plt
from mrcnn.config import Config
from PIL import *
from mrcnn import visualize
from glob import glob
import skimage.draw
import numpy as np
import datetime
import random
import json
import leaf
import sys
import tk
import io
import os

ROOT_DIR = os.path.abspath("../../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

def get_img_data(f, maxsize=(850, 500), first=False):
    """Generate image data using PIL
    """
    img = Image.open(f)
    img.thumbnail(maxsize)
    if first:                   
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    img = ImageTk.PhotoImage(img)
    return img

