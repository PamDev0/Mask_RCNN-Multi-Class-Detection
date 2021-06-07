# Ref: https://towardsdatascience.com/image-classification-of-uploaded-files-using-streamlits-killer-new-feature-7dd6aa35fe0

# streamlit run app.py
from mrcnn.visualize import display_images
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from interface_utils import *
from mrcnn import visualize 
import streamlit as st
import numpy as np
import time
import leaf
import cv2
import os

st.title("Upload + Segmentation")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpeg", "jpg", "tiff", "bmp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

left_column, right_column = st.beta_columns(2)

# Add a slider to the sidebar:
thresh_slider = st.sidebar.slider(
    'Select a value for threshold',
    0, 255
)

pressed = left_column.button('Predict!')
if pressed:
<<<<<<< HEAD
    st.write("Predicting...")
    start = time.time()

    weights = ''

    splash_image = inference(uploaded_file, weights)

    end = time.time()
    total_time_prediction = int((end - start))
    print("Tempo decorrido da exeução da predição: " + str(total_time_prediction) + " segundos.")
    
pressed_2 = right_column.button('Remove background!')
if pressed_2:
    st.write("Removing background from predicted image...")
    no_bg = remove_bg_from_image(splash_image)
    st.image(no_bg)

# botão save
# https://www.reddit.com/r/PySimpleGUI/comments/eewt6t/saving_a_graph/
# plot original + predicted
=======
    st.write("Classifying...")
    class InferenceConfig(leaf.CustomConfig):

        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()

    weights = ".../Mask_RCNN-Multi-Class-Detection/custom20210415T0038_mask_rcnn_custom_0029.h5"
    weights_path = ".../Mask_RCNN-Multi-Class-Detection/"

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=weights_path)
    model_path = weights
    model.load_weights(model_path, by_name=True)
    class_names = ['BG', 'rust', 'background']
    test_image = skimage.io.imread(filename)
    print(filename)
    predictions = model.detect([test_image], verbose=1) # We are replicating the same image to fill up the batch_size
    print(predictions)
    p = predictions[0]
    visualization = visualize.display_instances(test_image, p['rois'], p['masks'], p['class_ids'], 
                                                        class_names, p['scores'])
    print("IHA")

    st.image(visualization)

        # botão save
        # https://www.reddit.com/r/PySimpleGUI/comments/eewt6t/saving_a_graph/
        # plot original + predicted
        # https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
>>>>>>> 59a5ee8e988b5fa673e1c6a84632a55a96d7c69e

expander = st.beta_expander("FAQ")
expander.write("Trabalho de conclusão de curso. Engenharia Elétrica - 2021.1")

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
<<<<<<< HEAD
st.button("Recomeçar")
=======
st.button("Re-run")
>>>>>>> 59a5ee8e988b5fa673e1c6a84632a55a96d7c69e
