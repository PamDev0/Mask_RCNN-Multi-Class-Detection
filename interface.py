import PySimpleGUI as sg
import leaf

# Set color scheme theme
sg.theme('Dark Teal 6')

def ok_and_close():
    if not folder:
        answer = sg.PopupOKCancel('Você deseja fechar a aplicação?\nAperte Ok para fechar / Aperte Cancel para voltar')
        if answer == "Cancel":
            folder = sg.popup_get_folder('Escolha a pasta contendo a(s) imagem(ns) para análise:', default_path='')
        else:
            raise SystemExit()

folder = sg.popup_get_folder('Selecione a pasta contendo os arquivos para análise:', default_path='')
ok_and_close()

# PIL supported image types
img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")

# get list of files in folder
flist0 = os.listdir(folder)

# create sub list of image files (no sub folders, no wrong file types)
fnames = [f for f in flist0 if os.path.isfile(
    os.path.join(folder, f)) and f.lower().endswith(img_types)]

num_files = len(fnames)                # number of iamges found
if num_files == 0:
    sg.popup('No files in folder')
    raise SystemExit()

del flist0                             # no longer needed

