# ----------------- Bibliotecas ---------------
import tensorflow as tf # Principal biblioteca de Machine Learnig e Deep Learning. Cria, treina e implementa modelos de resdes neurais.
from tensorflow.keras import layers, models # API de alto nível que facilita a construção e treino de redes neurais dentro do TensorFlow
from tensorflow.keras.applications import ResNet50 # Modelo pré-treinado utilizado

# --------------------- Modelo ------------------
# Cria uma instância do modelo ResNet50 com pesos pré-treinados do ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Aumentar os dados de treino com rotações, refleção, zoom aleatoriamente + COMENTÁRIO
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"), # verticalmente ou horizontalmente
    layers.RandomRotation(0.2), # até 20%
    layers.RandomZoom(0.2), # até 10%
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
], name="data_augmentation")

# Criar um novo modelo + COMENTÁRIO
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)), # Definir a forma de entrada
    data_augmentation,  # Adicionar a camada de aumento de dados
    base_model,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(43, activation='softmax')  # As labels vão da 0 à 42 (43 combinações possíveis)
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Resumir o modelo
# model.summary()

# Função para carregar TFRecords
def parse_tfrecord(example):
    # Defina o formato do TFRecord
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example, feature_description)

def load_dataset(file_paths):
    # Cria um dataset a partir de múltiplos ficheiros TFRecord
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    return raw_dataset.map(parse_tfrecord).map(
        lambda x: (
            tf.image.resize(tf.image.decode_jpeg(x['image/encoded'], channels=3), [224, 224]),
            x['image/object/class/label']
        )
    )

# Pastas dos ficheiros TFRecords (Treino e Teste)
train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Treino/*.tfrecord')
val_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Validação/*.tfrecord')

# Carregar os datasets
train_dataset = load_dataset(train_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(val_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# COMENTÁRIO
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Treinar o modelo + COMENTÁRIO
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping]
)

# COMENTÁRIO
for layer in base_model.layers[-30:]:  # Ajuste o número de camadas conforme necessário
    layer.trainable = True

# COMENTÁRIO
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# COMENTÁRIO
fine_tune_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping]
)

# Carregar o conjunto de teste
test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Teste/*.tfrecord')
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_dataset)

# Imprime os resultados da perda e da exatidão dos dados teste
print(f'Teste Loss: {test_loss}')
print(f'Teste Accuracy: {test_accuracy}')