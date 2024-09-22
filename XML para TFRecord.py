import os
import shutil
import tensorflow as tf
import xml.etree.ElementTree as ET
from PIL import Image
import io


def create_tf_example(image_path, xml_path):
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()
    image = Image.open(io.BytesIO(encoded_image_data))
    width, height = image.size

    tree = ET.parse(xml_path)
    root = tree.getroot()

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # Extraindo os atributos adicionais
    structure_type = root.find('.//structure/type').text
    material = root.find('.//structure/material').text
    year_range = root.find('.//structure/year_range').text

    # Mapeia o nome da classe para um índice específico (ajuste conforme necessário)
    class_mapping = {
        'Tabuleiro simples/apoiado': 1,  # Exemplo de mapeamento, ajuste conforme necessário
        # Adicione mais mapeamentos aqui
    }

    class_name = structure_type
    classes_text.append(class_name.encode('utf8'))
    classes.append(class_mapping.get(class_name, 0))  # Usa 0 como valor padrão se a classe não estiver mapeada

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[os.path.basename(image_path).encode('utf8')])),
        'image/source_id': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[os.path.basename(image_path).encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/structure/type': tf.train.Feature(bytes_list=tf.train.BytesList(value=[structure_type.encode('utf8')])),
        'image/structure/material': tf.train.Feature(bytes_list=tf.train.BytesList(value=[material.encode('utf8')])),
        'image/structure/year_range': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[year_range.encode('utf8')])),
    }))
    return tf_example


def create_tf_record(output_path, image_path, xml_path):
    writer = tf.io.TFRecordWriter(output_path)
    tf_example = create_tf_example(image_path, xml_path)
    writer.write(tf_example.SerializeToString())
    writer.close()


def organize_images_and_tf_records(input_dir, output_dir):
    # Cria a pasta de saída, se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Listar e ordenar os arquivos XML e imagens
    files = [f for f in os.listdir(input_dir) if f.endswith('.xml') or f.endswith('.jpg')]
    files.sort()  # Ordenar por nome de arquivo

    # Processar cada arquivo
    for file in files:
        if file.endswith('.xml'):
            base_name = os.path.splitext(file)[0]
            image_file = base_name + '.jpg'
            xml_path = os.path.join(input_dir, file)
            image_path = os.path.join(input_dir, image_file)

            if os.path.exists(image_path):
                # Copia a imagem para a nova pasta
                shutil.copy(image_path, os.path.join(output_dir, image_file))

                # Cria o arquivo TFRecord
                tfrecord_path = os.path.join(output_dir, base_name + '.tfrecord')
                create_tf_record(tfrecord_path, image_path, xml_path)
            else:
                print(f"Imagem correspondente não encontrada: {image_path}")


# Defina os diretórios
input_dir = 'Pasta Final'
output_dir = 'Pasta Final TFRecord'

# Execute a função
organize_images_and_tf_records(input_dir, output_dir)
