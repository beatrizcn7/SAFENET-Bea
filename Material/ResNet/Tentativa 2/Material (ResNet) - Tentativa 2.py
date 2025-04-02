# Beatriz Neves, 11 de novembro de 2024
# Input = Pasta Final TFRecord - Material (Treino, Teste e Validação)
# Output = Impressão da loss e accuracy no terminal assim como o tempo
# Objetivo = Treinar um modelo ResNet50 com a informação input


# ----------------- Bibliotecas ---------------
# Importar a principal biblioteca de Machine Learning e Deep Learning.
# Criar, treinar e implementar modelos de redes neurais.
import tensorflow as tf
# Importar a API de alto nível que facilita a construção e treino de redes neurais dentro do TensorFlow
from tensorflow.keras import layers, models
# Importar o modelo pré-treinado ResNet50
from tensorflow.keras.applications import ResNet50
# Importar a biblioteca para contar o tempo
import time


# --------------------- Modelo ------------------
# Instanciar um modelo base com pesos pré-treinados
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Criar um novo modelo na parte superior.
inputs = layers.Input(shape=(224, 224, 3))  # Definir a entrada do modelo com tamanho 224x224 e 3 canais de cor
x = base_model(inputs, training=False)  # Passar a entrada pela base do modelo (sem treino das camadas base)
x = layers.GlobalAveragePooling2D()(x)  # Aplicar uma camada de pooling global para reduzir a dimensionalidade

# Criar uma camada densa com 3 classes (saída da classificação) e função de ativação softmax
outputs = layers.Dense(3, activation='softmax')(x)

# Criar o modelo final
model = models.Model(inputs, outputs)  # Definir o modelo com as entradas e saídas especificadas

# Descongelar algumas camadas para fine-tuning (ajustar a rede para os dados)
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Compilar o modelo com o otimizador, função de perda e métricas
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Função para carregar e processar os dados dos ficheiros TFRecord
def parse_tfrecord(example):
    # Definir o formato dos dados no TFRecord
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example, feature_description)

# Função para carregar o dataset a partir de vários ficheiros TFRecord
def load_dataset(file_paths):
    # Criar um dataset a partir de múltiplos arquivos TFRecord
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    return raw_dataset.map(parse_tfrecord).map(
        lambda x: (
            tf.image.resize(tf.image.decode_jpeg(x['image/encoded'], channels=3), [224, 224]),
            x['image/object/class/label']
        )
    )

# Pastas dos ficheiros TFRecords (Treino e Teste)
train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material/Treino/*.tfrecord')
val_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material/Validação/*.tfrecord')

# Carregar os ficheiros TFRecords
train_dataset = load_dataset(train_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(val_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Início da contagem de tempo
start_time = time.time()

# Treinar o modelo
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Carregar o conjunto de teste
test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material/Teste/*.tfrecord')
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_dataset)

end_time = time.time()  # Fim
total_time = (end_time - start_time) / 60  # Converter para minutos

# Imprimir o tempo total
print(f"Tempo total: {total_time:.2f} minutos")

# Imprime os resultados da perda e da exatidão dos dados teste
print(f'Teste Loss: {test_loss}')
print(f'Teste Accuracy: {test_accuracy}')