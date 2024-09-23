import tensorflow as tf
import os

# Definir a altura e largura que as imagens devem ter (dimensões do modelo ResNet)
IMAGE_SIZE = 224


# Função para parsear o conteúdo do TFRecord
def _parse_function(proto):
    # Definir a estrutura do TFRecord que queremos extrair
    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
    }

    # Extrair as features do TFRecord
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # Decodificar a imagem JPEG
    image = tf.io.decode_jpeg(parsed_features['image/encoded'], channels=3)

    # Redimensionar a imagem para o tamanho adequado (por exemplo, 224x224)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # Normalizar a imagem (valores entre 0 e 1)
    image = image / 255.0

    # Extrair o label da classe (associado à combinação de atributos)
    label = tf.cast(parsed_features['image/object/class/label'], tf.int32)

    return image, label


# Função para carregar e pré-processar os dados dos arquivos TFRecord
def load_dataset(tfrecord_dir, batch_size=32, shuffle=True):
    # Listar todos os arquivos TFRecord no diretório
    tfrecord_files = [os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith('.tfrecord')]

    # Cria um dataset a partir dos arquivos TFRecord
    dataset = tf.data.TFRecordDataset(tfrecord_files)

    # Aplica a função de parsing a cada exemplo
    dataset = dataset.map(_parse_function)

    # Embaralha os dados se necessário
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    # Divide os dados em batches
    dataset = dataset.batch(batch_size)

    # Repetição automática durante o treinamento
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


# Especificar os diretórios para os datasets
train_dir = 'Pasta Final TFRecord/Treino'  # Substitua pelo caminho correto
val_dir = 'Pasta Final TFRecord/Validação'  # Substitua pelo caminho correto
test_dir = 'Pasta Final TFRecord/Teste'  # Substitua pelo caminho correto

# Carrega os datasets de treino, validação e teste
train_dataset = load_dataset(train_dir, batch_size=32)
val_dataset = load_dataset(val_dir, batch_size=32, shuffle=False)
test_dataset = load_dataset(test_dir, batch_size=32, shuffle=False)

# Verificação: Mostra o shape do primeiro batch de treino
for images, labels in train_dataset.take(1):
    print("Batch de treino - Imagens:", images.shape)
    print("Batch de treino - Labels:", labels)
