import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

# Carregar o modelo ResNet50 com pesos pré-treinados do ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Criar uma camada de aumento de dados
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),  # Aumentar a rotação
    layers.RandomZoom(0.3),  # Aumentar o zoom
    layers.RandomContrast(0.3),  # Aumentar o contraste
    layers.RandomTranslation(0.2, 0.2),  # Adicionar translação
], name="data_augmentation")

# Criar um novo modelo com camadas adicionais
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,  # Adicionar a camada de aumento de dados
    base_model,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),  # Camada densa intermediária
    layers.Dropout(0.5),  # Regularização para evitar overfitting
    layers.Dense(43, activation='softmax')  # 43 combinações possíveis (labels de 0 a 42)
])

# Compilar o modelo (primeira fase)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Função para carregar TFRecords
def parse_tfrecord(example_proto):
    # Definir o formato do TFRecord
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def load_dataset(file_paths):
    # Criar um dataset a partir de múltiplos arquivos TFRecord
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    return raw_dataset.map(parse_tfrecord).map(
        lambda x: (
            tf.image.resize(tf.image.decode_jpeg(x['image/encoded'], channels=3), [224, 224]),
            x['image/object/class/label']
        )
    )

# Diretórios dos TFRecords
train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Treino/*.tfrecord')
val_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Validação/*.tfrecord')

# Carregar os datasets
train_dataset = load_dataset(train_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(val_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Definir os pesos de classe ajustados peso da classe = (total)/((número de exemplos da classe)*(número de classes))
class_weights = {
    0: 33.5,     # 94 exemplos
    1: 312.2,    # 10 exemplos
    2: 208.2,    # 15 exemplos
    3: 223.0,    # 14 exemplos
    4: 184.2,    # 17 exemplos
    5: 6244.0,   # 1 exemplo
    6: 28.9,     # 108 exemplos
    7: 447.4,    # 7 exemplos
    8: 6244.0,   # 1 exemplo
    9: 390.3,    # 8 exemplos
    10: 624.4,   # 5 exemplos
    11: 195.3,   # 16 exemplos
    12: 60.0,    # 52 exemplos
    13: 312.2,   # 10 exemplos
    14: 240.2,   # 13 exemplos
    15: 1041.3,  # 3 exemplos
    16: 6244.0,  # 1 exemplo
    17: 624.4,   # 5 exemplos
    18: 6244.0,  # 1 exemplo
    20: 3.96,    # 789 exemplos
    21: 32.4,    # 96 exemplos
    22: 390.3,   # 8 exemplos
    23: 780.5,   # 4 exemplos
    24: 65.0,    # 48 exemplos
    25: 72.7,    # 43 exemplos
    26: 71.0,    # 44 exemplos
    27: 14.4,    # 217 exemplos
    28: 284.7,   # 11 exemplos
    29: 1041.3,  # 3 exemplos
    30: 284.7,   # 11 exemplos
    31: 520.4,   # 6 exemplos
    32: 94.7,    # 33 exemplos
    33: 1.0,     # 3122 exemplos
    34: 15.7,    # 198 exemplos
    35: 120.3,   # 26 exemplos
    36: 31.0,    # 101 exemplos
    38: 208.2,   # 15 exemplos
    39: 780.5,   # 4 exemplos
    40: 780.5,   # 4 exemplos
    41: 1561.0,  # 2 exemplos
    42: 6244.0   # 1 exemplo
}

# Treinar o modelo (primeira fase)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights,  # Adicionar pesos de classe aqui
    callbacks=[early_stopping]
)

# Descongelar as últimas camadas da ResNet
for layer in base_model.layers[-50:]:  # Ajuste o número de camadas conforme necessário
    layer.trainable = True

# Compilar novamente o modelo com uma taxa de aprendizado menor
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Continuar o treinamento (segunda fase com fine-tuning)
fine_tune_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights,  # Incluir novamente os pesos de classe
    callbacks=[early_stopping]
)

# Avaliar o modelo
# Carregar o conjunto de teste
test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Teste/*.tfrecord')
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
