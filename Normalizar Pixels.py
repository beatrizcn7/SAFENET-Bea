import os
import tensorflow as tf

# Função para normalizar a imagem (valores de pixel entre 0 e 1)
def normalize_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Decodifica a imagem
    img = tf.image.convert_image_dtype(img, tf.float32)  # Converte para float32 e normaliza (0 a 1)
    return img

# Função para salvar a imagem normalizada
def salvar_imagem_normalizada(img, save_path):
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)  # Converte de volta para formato de imagem uint8
    img_encoded = tf.image.encode_jpeg(img)  # Codifica de volta para JPEG
    tf.io.write_file(save_path, img_encoded)  # Salva a imagem

# Caminho da pasta original com as fotos redimensionadas
root_dir = 'Fotos Redimensionadas'  # Substituir pelo caminho correto da pasta com as fotos redimensionadas
output_dir = 'Fotos Normalizadas'  # Onde serão salvas as imagens normalizadas

# Cria a pasta de destino se não existir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Percorre as pastas com os identificadores de ponte
for subdir, _, files in os.walk(root_dir):
    ponte_id = os.path.basename(subdir)  # O nome da pasta é o ID da ponte
    if not ponte_id.isdigit():
        continue  # Pula se não for uma pasta de identificador válido

    # Cria a nova pasta com o identificador da ponte dentro de "Fotos Normalizadas"
    ponte_output_dir = os.path.join(output_dir, ponte_id)
    if not os.path.exists(ponte_output_dir):
        os.makedirs(ponte_output_dir)

    # Processa e salva cada imagem
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):  # Verifica se é uma imagem
            image_path = os.path.join(subdir, file)
            img = normalize_image(image_path)

            # Cria o caminho para salvar a imagem normalizada
            save_path = os.path.join(ponte_output_dir, file)
            salvar_imagem_normalizada(img, save_path)

            print(f'Imagem {file} normalizada e salva em {save_path}')
