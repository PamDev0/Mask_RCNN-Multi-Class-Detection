# https://stackoverflow.com/questions/12199120/python-list-all-the-file-names-in-a-directory-and-its-subdirectories-and-then-p
# cria arquivo com o nome de todos os arquivos em um diretório

import os

a = open("files_train.txt", "w")
for path, subdirs, files in os.walk('/home/pam/Desktop/streamlit_tcc/bracol/rust/'):
   for filename in files:
     f = os.path.join(path, filename)
     a.write(str(filename) + os.linesep) 