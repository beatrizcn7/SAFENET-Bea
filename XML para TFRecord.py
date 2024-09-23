import os
import shutil
import tensorflow as tf
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image
import io


def load_combinations(comb_excel):
    df_combinacoes = pd.read_excel(comb_excel)
    combinacao_para_label = {}
    for index, row in df_combinacoes.iterrows():
        combinacao = (row['Tipo de Estrutura'], row['Intervalo de Anos'], row['Material'])
        combinacao_para_label[combinacao] = index  # Mapeia a combinação para um label numérico
    return combinacao_para_label


def create_tf_example(image_path, xml_path, combinacao_para_label):
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()
    image = Image.open(io.BytesIO(encoded_image_data))
    width, height = image.size

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extraindo os atributos adicionais
    structure_type = root.find('.//structure/type').text
    material = root.find('.//structure/material').text
    year_range = root.find('.//structure/year_range').text

    # Criar a combinação
    combinacao = (structure_type, year_range, material)
    label = combinacao_para_label.get(combinacao, -1)  # -1 se não encontrar a combinação

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
        'image/structure/type': tf.train.Feature(bytes_list=tf.train.BytesList(value=[structure_type.encode('utf8')])),
        'image/structure/material': tf.train.Feature(bytes_list=tf.train.BytesList(value=[material.encode('utf8')])),
        'image/structure/year_range': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[year_range.encode('utf8')])),
    }))

    return tf_example, label, (structure_type, year_range, material)


def create_tf_record(output_path, image_path, xml_path, combinacao_para_label):
    writer = tf.io.TFRecordWriter(output_path)
    tf_example, label, combinacao = create_tf_example(image_path, xml_path, combinacao_para_label)
    writer.write(tf_example.SerializeToString())
    writer.close()
    return label, combinacao


def organize_images_and_tf_records(input_dir, output_dir, comb_excel):
    combinacao_para_label = load_combinations(comb_excel)

    # Cria a pasta de saída, se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Listar e ordenar os arquivos XML e imagens
    files = [f for f in os.listdir(input_dir) if f.endswith('.xml') or f.endswith('.jpg')]
    files.sort()  # Ordenar por nome de arquivo

    # Contagem de labels
    label_counts = {}
    combinations_label_map = {}

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

                # Cria o arquivo TFRecord e conta o label
                tfrecord_path = os.path.join(output_dir, base_name + '.tfrecord')
                label, combinacao = create_tf_record(tfrecord_path, image_path, xml_path, combinacao_para_label)

                # Atualiza a contagem do label
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

                # Mapeia a combinação para o label para o arquivo de texto
                combinations_label_map[label] = combinacao
            else:
                print(f"Imagem correspondente não encontrada: {image_path}")

    # Escreve as combinações associadas às labels em um arquivo de texto
    output_file = 'Labels.txt'  # Define o caminho desejado fora da pasta TFRecord
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Contagem de labels:\n")
        for label in sorted(label_counts.keys()):  # Ordenar as labels
            combinacao = combinations_label_map.get(label, ("-", "-", "-"))
            f.write(f"Label {label}: {combinacao} - {label_counts[label]}\n")

    print(f"Conteúdo com labels e contagem foi salvo no arquivo: {output_file}")


# Defina os diretórios e o caminho do Excel
input_dir = 'Pasta Final'
output_dir = 'Pasta Final TFRecord'
comb_excel = 'Combinações/Combinações JSON 2.xlsx'

# Execute a função
organize_images_and_tf_records(input_dir, output_dir, comb_excel)
