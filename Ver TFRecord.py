import tensorflow as tf


def _parse_function(proto):
    # Definição do schema do TFRecord
    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/class/label': tf.io.FixedLenFeature([1], tf.int64)
    }

    # Parse dos dados do TFRecord
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # Obtendo os dados da imagem e o formato
    image_data = parsed_features['image/encoded']
    image_format = parsed_features['image/format']

    # Checar o formato da imagem e garantir que é JPEG
    # Usamos tf.strings para comparar os valores de forma eficiente no gráfico
    image_format = tf.strings.regex_full_match(image_format, b'jpeg')
    image_format = tf.cast(image_format, tf.bool)

    # Levanta um erro se o formato não for JPEG
    if not image_format:
        raise ValueError(f"Formato da imagem não é JPEG, mas sim: {image_format.numpy()}")

    # Decodificar a imagem
    image = tf.image.decode_jpeg(image_data, channels=3)

    return image, parsed_features['image/class/label']


def read_tfrecord(tfrecord_path):
    # Ler o arquivo TFRecord
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Mapear a função de parsing para o dataset
    parsed_dataset = raw_dataset.map(_parse_function)

    # Iterar sobre o dataset
    for image, label in parsed_dataset:
        print(f"Imagem: {image.shape}, Label: {label.numpy()}")



# Caminho do arquivo TFRecord
tfrecord_path = 'Pasta Final TFRecord - Material + Ano + Estrutura/Treino/1021_2.tfrecord'

# Chamar a função para ler o TFRecord
read_tfrecord(tfrecord_path)
