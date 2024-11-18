# -*- coding: utf-8 -*-
import tensorflow as tf
from PIL import Image
import io
import matplotlib.pyplot as plt


# Função para ler e exibir o conteúdo de um arquivo TFRecord
def read_tfrecord(tfrecord_path):
    # Criar um iterador para o arquivo TFRecord
    raw_dataset = tf.data.TFRecordDataset([tfrecord_path])

    # Definir as características que o TFRecord contém
    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
        'image/structure/material': tf.io.FixedLenFeature([], tf.string),
    }

    # Função para parsear cada exemplo
    def _parse_function(proto):
        # Parse do exemplo com base no 'feature_description'
        parsed_features = tf.io.parse_single_example(proto, feature_description)

        # Decodificar a imagem
        image = tf.image.decode_jpeg(parsed_features['image/encoded'])
        image = tf.image.resize(image, [parsed_features['image/height'], parsed_features['image/width']])

        # Obter as informações
        filename = parsed_features['image/filename']
        label = parsed_features['image/object/class/label']
        material = parsed_features['image/structure/material']

        return image, filename, label, material

    # Aplicar a função de parsing
    parsed_dataset = raw_dataset.map(_parse_function)

    # Iterar sobre o dataset e exibir a primeira imagem
    for image, filename, label, material in parsed_dataset:
        print(f"Arquivo: {filename.numpy().decode('utf-8')}")
        print(f"Label: {label.numpy()}")
        print(f"Material: {material.numpy().decode('utf-8')}")

        # Exibir a imagem
        plt.imshow(image.numpy())
        plt.axis('off')
        plt.show()
        break  # Exibir apenas a primeira imagem (remover isso para ver todas)


# Caminho do arquivo TFRecord
tfrecord_path = 'Pasta Final TFRecord - Material/1_1.tfrecord'

# Ler e exibir o conteúdo do arquivo TFRecord
read_tfrecord(tfrecord_path)
