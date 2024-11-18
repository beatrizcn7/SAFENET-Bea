# -*- coding: utf-8 -*-
import os
import shutil
import tensorflow as tf
import xml.etree.ElementTree as ET
from PIL import Image
import io


# Função para mapear o material e o intervalo de anos para a classe correta
def map_to_class(material, year_interval):
    if "Alvenaria" in material:
        return 0
    elif "Aço" in material:
        if year_interval == "Antes de 1983":
            return 1
        elif year_interval == "Depois de 1983":
            return 2
    elif "Betão Armado" in material:
        if year_interval == "Antes de 1983":
            return 3
        elif year_interval == "Depois de 1983":
            return 4
    return -1  # Classe desconhecida


# Função para criar um exemplo TF
def create_tf_example(image_path, xml_path):
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()
    image = Image.open(io.BytesIO(encoded_image_data))
    width, height = image.size

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extrair material e intervalo de anos
    structure_element = root.find('.//structure')
    material_element = structure_element.find('material')
    year_element = structure_element.find('year_range')

    material = material_element.text if material_element is not None else "Desconhecido"
    year_interval = year_element.text if year_element is not None else "Desconhecido"

    print(f"Material encontrado: {material}, Intervalo de Anos encontrado: {year_interval}")

    # Mapear para a classe correta
    label = map_to_class(material, year_interval)
    if label == -1:
        print(f"Erro: Material ou intervalo de anos desconhecido: {material}, {year_interval}")

    # Criar exemplo TF
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
        'image/structure/year_interval': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[year_interval.encode('utf8')])),
    }))

    return tf_example, label, material, year_interval


# Função para criar o arquivo TFRecord
def create_tf_record(output_path, image_path, xml_path):
    writer = tf.io.TFRecordWriter(output_path)
    tf_example, label, material, year_interval = create_tf_example(image_path, xml_path)
    writer.write(tf_example.SerializeToString())
    writer.close()
    return label, material, year_interval


# Função principal para organizar os arquivos e criar os TFRecords
def organize_images_and_tf_records(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith('.xml') or f.endswith('.jpg')]
    files.sort(key=lambda x: int(x.split('_')[0]))

    label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for file in files:
        if file.endswith('.xml'):
            base_name = os.path.splitext(file)[0]
            image_file = base_name + '.jpg'
            xml_path = os.path.join(input_dir, file)
            image_path = os.path.join(input_dir, image_file)

            if os.path.exists(image_path):
                # Copiar imagem para a pasta de saída
                shutil.copy(image_path, os.path.join(output_dir, image_file))

                # Criar TFRecord
                tfrecord_path = os.path.join(output_dir, base_name + '.tfrecord')
                label, material, year_interval = create_tf_record(tfrecord_path, image_path, xml_path)

                # Atualizar contagem de classes
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    print(f"Material ou intervalo de anos desconhecido: {material}, {year_interval}")

    # Salvar contagem de classes em um arquivo de texto
    output_file = 'Material + Ano/Classes - Material + Ano.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Contagem de classes:\n")
        for label, count in label_counts.items():
            f.write(f"Classe {label}: {count} ficheiros\n")

    print(f"Conteúdo com labels e contagem foi salvo no arquivo: {output_file}")


# Definir diretórios de entrada e saída
input_dir = 'Pasta Final'
output_dir = 'Pasta Final TFRecord - Material + Ano'

# Executar a função principal
organize_images_and_tf_records(input_dir, output_dir)
