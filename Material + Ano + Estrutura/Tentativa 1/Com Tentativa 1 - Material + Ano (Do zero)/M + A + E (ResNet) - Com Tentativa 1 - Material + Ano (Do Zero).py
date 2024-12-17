# ----------------- Bibliotecas ---------------
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import time

# --------------------- Modelo ------------------
# Carregar o modelo salvo para a propriedade "Material"
model = tf.keras.models.load_model('Tentativa 1 - Material + Ano (do zero).h5')

model.pop()  # Remover a última camada de saída (5 classes)
model.add(layers.Dense(11, activation='softmax'))  #  11 classes

# Congelar as camadas do modelo base, menos a última camada adicionada
for layer in model.layers[:-1]:
    layer.trainable = False

# Compilar o modelo novamente com a nova configuração
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Função para carregar TFRecords
def parse_tfrecord(example):
    # Definir o formato do TFRecord
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example, feature_description)

def load_dataset(file_paths):
    # Criar dataset a partir dos ficheiros TFRecord
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    return raw_dataset.map(parse_tfrecord).map(
        lambda x: (
            tf.image.resize(tf.image.decode_jpeg(x['image/encoded'], channels=3), [224, 224]),
            x['image/object/class/label']
        )
    )

# Pastas dos ficheiros TFRecords (Treino, Validação e Teste)
train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Ano/Treino/*.tfrecord')
val_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Ano/Validação/*.tfrecord')
test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Ano/Teste/*.tfrecord')

# Carregar os datasets
train_dataset = load_dataset(train_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(val_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Início da contagem de tempo
start_time = time.time()

# Treinar o modelo com os novos dados (Material e Ano)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_dataset)

# Fim da contagem de tempo
end_time = time.time()
total_time = (end_time - start_time) / 60

# Imprimir o tempo total
print(f"Tempo total: {total_time:.2f} minutos")

# Imprime os resultados da perda e da exatidão dos dados teste
print(f'Teste Loss: {test_loss}')
print(f'Teste Accuracy: {test_accuracy}')