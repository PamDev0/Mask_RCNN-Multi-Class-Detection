# https://stackoverflow.com/questions/5904969/how-to-print-a-dictionarys-key

# lista os arquivos que tem a segmentação presente no json
import json
import os

with open('/home/pam/Desktop/streamlit_tcc/Mask_RCNN-Multi-Class-Detection/Leaf_OLD/train/via_region_data.json') as json_file:
    data = json.load(json_file)

    for key, value in data.items() :
        a = open("dataset.txt", "w")
        for key in data:
            f = key
            print(f)
            a.write(str(f) + os.linesep)