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

st.title("Segmentação e remoção de fundo")
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["png", "jpeg", "jpg", "tiff", "bmp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagem original.', use_column_width=True)
    save_uploadedfile(image)
    st.write("Upload concluído com sucesso :)")

left_column, right_column = st.beta_columns(2)

pressed = left_column.button('Segmentar!')
if pressed:
    st.write("Segmentação em andamento...")
    start = time.time()
    
    weights = '/content/mask_rcnn_custom_0033.h5' # colab path
    
    splash_image = inference('/content/out.jpg', weights)
    st.image('/content/predicted.jpg', caption='Imagem Segmentada.', use_column_width=True)
    # other image
    st.image('/content/splashed.jpg', caption='Ferrugem em evidência.', use_column_width=True)

    end = time.time()
    total_time_prediction = int((end - start))
    st.write("Tempo decorrido da execução da predição: " + str(total_time_prediction) + " segundos.")
    
pressed_2 = right_column.button('Remover o fundo!')
if pressed_2:
    st.write("Removendo fundo da imagem utilizando a API do remove.bg ...")
    no_bg = remove_bg_from_image('/content/splashed.jpg')
    st.image('/content/splashed.jpg_no_bg.png')
    st.write('Processo finalizado.')
expander = st.beta_expander("Sobre")
expander.write("Para baixar a imagem, basta clicar com o lado direito do mouse encima da imagem e selecionar a opção de salvar.\nTrabalho de conclusão de curso. Engenharia Elétrica - 2021.1")

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Recomeçar")
