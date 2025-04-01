# Beatriz Neves, 13 de setembro de 2024
# Input = Fotos
# Output = Fotos Redimensionadas
# Objetivo = Redimensionar as fotos para 224x224


import os
import tensorflow as tf

# Função para redimensionar e normalizar a imagem
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Decodifica a imagem
    img = tf.image.resize(img, [224, 224])  # Redimensiona para 224x224
    return img

# Função para salvar a imagem redimensionada
def salvar_imagem(img, save_path):
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)  # Converte para formato de imagem
    img_encoded = tf.image.encode_jpeg(img)  # Codifica de volta para JPEG
    tf.io.write_file(save_path, img_encoded)  # Salva a imagem

# Caminho da pasta original com as pastas dos identificadores de ponte
root_dir = 'Fotos'  # Substituir pelo caminho correto da pasta com as fotos
output_dir = 'Fotos Redimensionadas'  # Onde serão salvas as novas fotos

# Cria a pasta de destino se não existir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Percorre as pastas com os identificadores de ponte, ordenando por identificador
for subdir in sorted(os.listdir(root_dir), key=lambda x: int(x) if x.isdigit() else float('inf')):
    ponte_dir = os.path.join(root_dir, subdir)

    # Verifica se o subdiretório é uma pasta com identificador numérico
    if not subdir.isdigit():
        continue  # Pula se não for uma pasta de identificador válido

    ponte_id = subdir  # O nome da pasta é o ID da ponte

    # Cria a nova pasta com o identificador da ponte dentro de "Fotos Redimensionadas"
    ponte_output_dir = os.path.join(output_dir, ponte_id)
    if not os.path.exists(ponte_output_dir):
        os.makedirs(ponte_output_dir)

    # Processa e salva cada imagem dentro da pasta
    for file in os.listdir(ponte_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')):  # Verifica se é uma imagem
            image_path = os.path.join(ponte_dir, file)
            img = preprocess_image(image_path)

            # Cria o caminho para salvar a imagem redimensionada
            save_path = os.path.join(ponte_output_dir, file)
            salvar_imagem(img, save_path)

            print(f'Imagem {file} processada e salva em {save_path}')
