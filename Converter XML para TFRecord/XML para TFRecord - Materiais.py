import os
import shutil
import tensorflow as tf
import xml.etree.ElementTree as ET
from PIL import Image
import io

# Função para mapear o material para a classe correta
def map_material_to_class(material):
    if "Alvenaria" in material:
        return 0
    elif "Aço" in material:
        return 1
    elif "Betão Armado" in material:
        return 2
    else:
        return -1  # Classe desconhecida

# Exemplo de como adicionar debug para verificar o conteúdo extraído
def create_tf_example(image_path, xml_path):
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()
    image = Image.open(io.BytesIO(encoded_image_data))
    width, height = image.size

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Verifique se o elemento 'structure' e 'material' estão sendo encontrados corretamente
    structure_element = root.find('.//structure')  # Busca a tag <structure>
    material_element = structure_element.find('material')  # Busca a tag <material> dentro de <structure>

    if material_element is not None:
        material = material_element.text
        print(f"Material encontrado: {material}")  # Debug
    else:
        print(f"Elemento 'material' não encontrado no arquivo: {xml_path}")
        material = "Desconhecido"  # ou outra ação de sua escolha

    # Mapear o material para a classe
    label = map_material_to_class(material)
    print(f"Classe atribuída: {label}")  # Debug

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[os.path.basename(image_path).encode('utf8')])),
        'image/source_id': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[os.path.basename(image_path).encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'image/structure/material': tf.train.Feature(bytes_list=tf.train.BytesList(value=[material.encode('utf8')])),
    }))

    return tf_example, label, material


# Função para criar o arquivo TFRecord
def create_tf_record(output_path, image_path, xml_path):
    writer = tf.io.TFRecordWriter(output_path)
    tf_example, label, material = create_tf_example(image_path, xml_path)
    writer.write(tf_example.SerializeToString())
    writer.close()
    return label, material

# Função principal para organizar os arquivos e criar os TFRecords
def organize_images_and_tf_records(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Listar e ordenar os arquivos XML e imagens
    files = [f for f in os.listdir(input_dir) if f.endswith('.xml') or f.endswith('.jpg')]

    # Ordenar os arquivos por nome (garante ordem crescente)
    files.sort(key=lambda x: int(x.split('_')[0]))  # Supondo que os arquivos têm um número no nome, como '1_1.jpg'

    label_counts = {0: 0, 1: 0, 2: 0}

    # Processar cada arquivo
    for file in files:
        if file.endswith('.xml'):
            base_name = os.path.splitext(file)[0]
            image_file = base_name + '.jpg'
            xml_path = os.path.join(input_dir, file)
            image_path = os.path.join(input_dir, image_file)

            if os.path.exists(image_path):
                # Copiar apenas a imagem para a pasta de saída
                shutil.copy(image_path, os.path.join(output_dir, image_file))

                # Cria o arquivo TFRecord e conta o label
                tfrecord_path = os.path.join(output_dir, base_name + '.tfrecord')
                label, material = create_tf_record(tfrecord_path, image_path, xml_path)

                # Contagem de labels
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    print(f"Material desconhecido: {material}")

    # Escreve a contagem das labels no arquivo de texto
    output_file = 'Material/Classes - Material.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Contagem de classes:\n")
        for label, count in label_counts.items():
            f.write(f"Classe {label}: {count} ficheiros\n")

    print(f"Conteúdo com labels e contagem foi salvo no arquivo: {output_file}")

# Defina os diretórios
input_dir = 'Pasta Final'
output_dir = 'Pasta Final TFRecord - Material'

# Execute a função
organize_images_and_tf_records(input_dir, output_dir)
