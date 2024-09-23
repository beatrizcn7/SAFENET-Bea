import tensorflow as tf

def check_tfrecord_structure(file_path):
    raw_dataset = tf.data.TFRecordDataset(file_path)

    for raw_record in raw_dataset.take(5):  # Mostra os primeiros 5 registros
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)

# Chame a função passando o caminho do seu arquivo TFRecord
check_tfrecord_structure('Pasta Final TFRecord/Teste/4_2.tfrecord')
