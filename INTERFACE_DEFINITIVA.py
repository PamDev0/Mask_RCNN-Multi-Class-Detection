from interface_utils import *
import PySimpleGUI as sg 
import leaf
import os

sg.ChangeLookAndFeel('LightGreen')      
sg.SetOptions(element_padding=(0, 0))  

# ------ Menu Definition ------ #      
menu_def = [['Arquivo', ['Abrir', 'Salvar', 'Sair'  ]],
            ['Ajuda', 'Sobre...'], ]  

# ------ GUI Defintion ------ #   
filename = sg.popup_get_file('Escolha o arquivo para análise:', no_window=False)      
image_elem = sg.Image(data=get_img_data(filename, first=True))
filename_display_elem = sg.Text(filename, size=(80, 3))

layout = [[sg.Menu(menu_def, )],
          [filename_display_elem],
          [image_elem],
          [sg.Button('Analisar!', key='ANALISE')]]

window = sg.Window("Análise de ferrugem", layout, default_element_size=(12, 1), auto_size_text=False, auto_size_buttons=False,      
                    default_button_element_size=(12, 1))

# ------ Loop & Process button menu choices ------ #      
while True:      
    event, values = window.read()      
    if event == sg.WIN_CLOSED or event == 'Sair':      
        break      
    # ------ Process menu choices ------ #      
    if event == 'Sobre...':      
        sg.popup('Sobre esse programa', 'Version 1.0')      
    elif event == 'Abrir':
        filename = sg.popup_get_file('Escolha o arquivo para análise:', no_window=False)      
        image_elem.update(data=get_img_data(filename, first=True))
        filename_display_elem.update(filename)

        # ------ Analyze image with trained model ------ #

    elif event == 'ANALISE':
        class InferenceConfig(leaf.CustomConfig):

            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
        config.display()

        weights = "/home/pam/Desktop/TCC_GITHUB/models/Mask_RCNN-Multi-Class-Detection/custom20210415T0038_mask_rcnn_custom_0029.h5"
        weights_path = os.path.basename(weights)

        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=weights_path)
        model_path = weights
        model.load_weights(model_path, by_name=True)
        class_names = ['BG', 'leaf', 'rust', 'background']
        test_image = skimage.io.imread(filename)
        predictions = model.detect([test_image], verbose=1) # We are replicating the same image to fill up the batch_size
        p = predictions[0]
        visualization = visualize.display_instances(test_image, p['rois'], p['masks'], p['class_ids'], 
                                                        class_names, p['scores'])

        image_elem.update(data=get_img_data(visualization, first=True))

        # botão save
        # https://www.reddit.com/r/PySimpleGUI/comments/eewt6t/saving_a_graph/
        # plot original + predicted
        # https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/