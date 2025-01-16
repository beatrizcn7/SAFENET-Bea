# -*- coding: utf-8 -*-
import os
import shutil
import tensorflow as tf
import xml.etree.ElementTree as ET
from PIL import Image
import io

# Função para mapear as propriedades para as classes
def map_to_class(structure, material, year_interval):
    if "Alvenaria" in material:
        return 0  # Classe 0: Material - Alvenaria
    elif "Aço" in material:
        return 1  # Classe 1: Material - Aço
    elif "Betão Armado" in material:
        if "Arco" in structure:
            return 2  # Classe 2: Material - Betão Armado; Estrutura - Arco
        elif "Pórtico" in structure:
            return 3  # Classe 3: Material - Betão Armado; Estrutura - Pórtico
        elif "Tabuleiro simples/apoiado" in structure:
            return 4  # Classe 4: Material - Betão Armado; Estrutura - Tabuleiro simples/apoiado
        elif "Vãos Multiplos" in structure:
            return 5  # Classe 5: Material - Betão Armado; Estrutura - Vãos Múltiplos
    return -1  # Classe -1: Não classificado

# Função para criar um exemplo TF
def create_tf_example(image_path, xml_path):
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()
    image = Image.open(io.BytesIO(encoded_image_data))
    width, height = image.size

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extrair estrutura, material e intervalo de anos
    structure_element = root.find('.//structure')
    material_element = structure_element.find('material')
    year_element = structure_element.find('year_range')
    type_element = structure_element.find('type')

    structure = type_element.text if type_element is not None else "Desconhecido"
    material = material_element.text if material_element is not None else "Desconhecido"
    year_interval = year_element.text if year_element is not None else "Desconhecido"

    # Mapear para a classe correta
    label = map_to_class(structure, material, year_interval)

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
        'image/structure/year_interval': tf.train.Feature(bytes_list=tf.train.BytesList(value=[year_interval.encode('utf8')])),
    }))

    return tf_example, label

# Função para criar o arquivo TFRecord
def create_tf_record(output_path, image_path, xml_path):
    writer = tf.io.TFRecordWriter(output_path)
    tf_example, label = create_tf_example(image_path, xml_path)
    writer.write(tf_example.SerializeToString())
    writer.close()
    return label

# Função principal para organizar os arquivos e criar os TFRecords
def organize_images_and_tf_records(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith('.xml') or f.endswith('.jpg')]
    files.sort(key=lambda x: int(x.split('_')[0]))

    label_counts = {i: 0 for i in range(6)}
    deleted_files_count = 0

    for file in files:
        if file.endswith('.xml'):
            base_name = os.path.splitext(file)[0]
            image_file = base_name + '.jpg'
            xml_path = os.path.join(input_dir, file)
            image_path = os.path.join(input_dir, image_file)

            if os.path.exists(image_path):
                # Criar TFRecord e verificar a classe
                tfrecord_path = os.path.join(output_dir, base_name + '.tfrecord')
                label = create_tf_record(tfrecord_path, image_path, xml_path)

                # Atualizar contagem de classes ou deletar arquivos não classificados
                if label != -1:
                    label_counts[label] += 1
                    shutil.copy(image_path, os.path.join(output_dir, image_file))
                else:
                    # Arquivo não classificado, deletar imagem e XML
                    os.remove(image_path)
                    os.remove(xml_path)
                    deleted_files_count += 1

    # Salvar contagem de classes em um arquivo de texto
    output_file = 'Material + Estrutura/Classes - Material + Estrutura .txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Contagem de classes:\n")
        for label, count in label_counts.items():
            f.write(f"Classe {label}: {count} ficheiros\n")

    print(f"Conteúdo com labels e contagem foi salvo no arquivo: {output_file}")
    print(f"Número de ficheiros eliminados: {deleted_files_count}")

# Definir diretórios de entrada e saída
input_dir = 'Pasta Final'
output_dir = 'Pasta Final TFRecord - Material + Estrutura'

# Executar a função principal
organize_images_and_tf_records(input_dir, output_dir)
