import os
import tensorflow as tf

# Caminho para a pasta que contém os arquivos TFRecord
pasta_tfrecords = 'Pasta Final TFRecord'


# Função para ler e exibir o conteúdo dos arquivos TFRecord
def ler_tfrecords(pasta_tfrecords):
    for file_name in os.listdir(pasta_tfrecords):
        file_path = os.path.join(pasta_tfrecords, file_name)
        if file_name.endswith(".tfrecord"):
            print(f"Arquivo: {file_name}")
            try:
                raw_dataset = tf.data.TFRecordDataset(file_path)
                for raw_record in raw_dataset:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())

                    # Filtrando e exibindo apenas os campos desejados
                    filtered_example = {
                        "image/height": example.features.feature["image/height"].int64_list.value,
                        "image/width": example.features.feature["image/width"].int64_list.value,
                        "image/filename": example.features.feature["image/filename"].bytes_list.value,
                        "image/source_id": example.features.feature["image/source_id"].bytes_list.value,
                        "image/format": example.features.feature["image/format"].bytes_list.value,
                        "image/object/class/label": example.features.feature["image/object/class/label"].int64_list.value,
                        "image/structure/type": [s.decode('utf-8') for s in example.features.feature["image/structure/type"].bytes_list.value],
                        "image/structure/material": [s.decode('utf-8') for s in example.features.feature["image/structure/material"].bytes_list.value],
                        "image/structure/year_range": [s.decode('utf-8') for s in example.features.feature["image/structure/year_range"].bytes_list.value],
                    }

                    # Exibe o conteúdo filtrado do TFRecord no terminal
                    print(filtered_example)
            except Exception as e:
                print(f"Erro ao processar {file_name}: {e}")
            print("\n" + "=" * 50 + "\n")


# Executar a função
ler_tfrecords(pasta_tfrecords)
