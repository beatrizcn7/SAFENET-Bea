import os
import tensorflow as tf

# Função para contar arquivos de cada label em uma pasta
def count_labels_in_folder(folder_path):
    label_count = {}

    # Carregar todos os arquivos TFRecord na pasta
    tfrecord_files = tf.io.gfile.glob(os.path.join(folder_path, '*.tfrecord'))

    for tfrecord_file in tfrecord_files:
        # Ler o arquivo TFRecord
        raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

        # Contar cada label
        for raw_record in raw_dataset:
            example = parse_tfrecord(raw_record)
            label = example['image/object/class/label'].numpy()

            # Incrementar a contagem do label
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

    return label_count

# Função para analisar cada exemplo no TFRecord
def parse_tfrecord(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example, feature_description)

# Caminho das pastas
base_path = 'Pasta Final TFRecord - Material'
folders = ['Treino', 'Validação', 'Teste']

# Dicionário para armazenar contagens de labels
all_label_counts = {}

# Contar os labels em cada pasta
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    label_counts = count_labels_in_folder(folder_path)
    all_label_counts[folder] = label_counts

# Escrever resultados em um arquivo de texto
with open('Material/Classes nas Pastas - Material.txt', 'w') as f:
    for folder, counts in all_label_counts.items():
        f.write(f'Pasta: {folder}\n')
        for label in sorted(counts.keys()):  # Ordenar as labels em ordem crescente
            count = counts[label]
            f.write(f'Classe {label}: {count}\n')
        f.write('\n')  # Linha em branco entre pastas

print("Concluído!.")
