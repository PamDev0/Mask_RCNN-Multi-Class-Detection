# Ref: https://towardsdatascience.com/image-classification-of-uploaded-files-using-streamlits-killer-new-feature-7dd6aa35fe0

# streamlit run app.py
from mrcnn.visualize import display_images
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from interface_utils import *
from mrcnn import visualize 
from mrcnn.visualize import display_images
import streamlit as st
import numpy as np
import time
import leaf
import cv2
import os
from PIL import Image
import io
from keras.preprocessing import image

st.title("Upload + Segmentation")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpeg", "jpg", "tiff", "bmp"])
if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.content))
    #img_array = np.array(image)
    st.write(img_array)
    #st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Image Uploaded :) \nReady to predict!")

left_column, right_column = st.beta_columns(2)

# Add a slider to the sidebar:
thresh_slider = st.sidebar.slider(
    'Select a value for threshold',
    0, 255
)

pressed = left_column.button('Predict!')
if pressed:
    st.write("Predicting...")
    start = time.time()

    weights = '/content/mask_rcnn_custom_0033.h5' # colab path

    splash_image = inference(image, weights)

    end = time.time()
    total_time_prediction = int((end - start))
    st.write("Tempo decorrido da exeução da predição: " + str(total_time_prediction) + " segundos.")
    
pressed_2 = right_column.button('Remove background!')
if pressed_2:
    st.write("Removing background from predicted image...")
    no_bg = remove_bg_from_image(splash_image)
    st.image(no_bg)
    
expander = st.beta_expander("FAQ")
expander.write("Trabalho de conclusão de curso. Engenharia Elétrica - 2021.1")

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Recomeçar")
