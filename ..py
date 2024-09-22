import os
import tensorflow as tf

def parse_tf_example(tf_example):
    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/structure/type': tf.io.FixedLenFeature([], tf.string),
        'image/structure/material': tf.io.FixedLenFeature([], tf.string),
        'image/structure/year_range': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(tf_example, feature_description)
    return parsed_features

def inspect_tfrecord(tfrecord_file):
    print(f"Inspecionando: {tfrecord_file}")
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    for tf_example in dataset.take(1):  # Apenas pegar o primeiro exemplo para inspecionar
        parsed_features = parse_tf_example(tf_example)
        print(f"Filename: {parsed_features['image/filename'].numpy().decode('utf8')}")
        print(f"Type: {parsed_features.get('image/structure/type', 'N/A').numpy().decode('utf8')}")
        print(f"Material: {parsed_features.get('image/structure/material', 'N/A').numpy().decode('utf8')}")
        print(f"Year Range: {parsed_features.get('image/structure/year_range', 'N/A').numpy().decode('utf8')}")

        # Converter SparseTensor em denso para acessar valores
        labels = tf.sparse.to_dense(parsed_features['image/object/class/label']).numpy()
        classes_text = tf.sparse.to_dense(parsed_features['image/object/class/text']).numpy()

        print(f"Labels: {labels}")
        print(f"Classes Text: {[text.decode('utf8') for text in classes_text]}")
        print("")

def list_and_inspect_tfrecords(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.tfrecord'):
            file_path = os.path.join(directory, filename)
            inspect_tfrecord(file_path)

# Defina o diretório dos arquivos TFRecord
tfrecords_dir = 'Pasta Final TFRecord'

# Execute a função
list_and_inspect_tfrecords(tfrecords_dir)
