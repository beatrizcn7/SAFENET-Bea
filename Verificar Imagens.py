from PIL import Image
import os

# Caminho da pasta principal onde estão as pastas numeradas
root_dir = 'Fotos'

# Percorre todas as subpastas e arquivos de forma recursiva
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        # Verifica se o arquivo é uma imagem (usando a extensão)
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(subdir, file)
            try:
                img = Image.open(image_path)
                img.verify()  # Verifica se a imagem está OK
            except (IOError, SyntaxError) as e:
                print(f'Arquivo corrompido: {image_path}')
