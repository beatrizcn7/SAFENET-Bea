from collections import Counter
import tensorflow as tf


# Função para contar as labels no dataset de treino
def count_labels_in_tfrecords(file_paths):
    label_counter = Counter()

    raw_dataset = tf.data.TFRecordDataset(file_paths)

    # Função para parsear os TFRecords
    def parse_tfrecord(example_proto):
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
        }
        return tf.io.parse_single_example(example_proto, feature_description)

    # Iterar sobre o dataset e contar as labels
    for record in raw_dataset.map(parse_tfrecord):
        label = record['image/object/class/label'].numpy()
        label_counter[int(label)] += 1

    return label_counter


# Caminho para os TFRecords de treino
train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Treino/*.tfrecord')

# Contar as labels no conjunto de treino
train_label_counts = count_labels_in_tfrecords(train_tfrecords)

# Ordenar e exibir as contagens de labels por ordem crescente
sorted_label_counts = sorted(train_label_counts.items(), key=lambda x: x[1])

print("Contagem de labels no conjunto de treino (ordem crescente):")
for label, count in sorted_label_counts:
    print(f"Label {label}: {count}")
