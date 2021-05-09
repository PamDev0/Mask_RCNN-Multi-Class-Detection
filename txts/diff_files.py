# https://www.kite.com/python/answers/how-to-write-a-list-to-a-file-in-python
# cria um arquivo com a diferença entre dois arquivos

import re
import os 

my_file = open("/home/pam/Desktop/streamlit_tcc/dataset.txt", "r")

content = my_file.read()
line = re.sub('\n', ',', content)
content_list_dataset = line.split(",")
my_file.close()

my_file = open("/home/pam/Desktop/streamlit_tcc/files_train.txt", "r")

content = my_file.read()
line = re.sub('\n', ',', content)
content_list_train = line.split(",")
my_file.close()

diff = set(content_list_train) - set(content_list_dataset)

textfile = open("files_to_segment.txt", "w")

for element in diff:

    textfile.write(element + "\n")

textfile.close()