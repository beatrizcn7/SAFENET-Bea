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
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.3),
    layers.RandomTranslation(0.2, 0.2),
], name="data_augmentation")


# Função para carregar TFRecords
def parse_tfrecord(example_proto):
    # Definir o formato do TFRecord
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(parsed_example['image/encoded'], channels=3)
    image = tf.image.resize(image, [224, 224])
    return image, parsed_example['image/object/class/label']


def load_dataset(file_paths, augment=False):
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(parse_tfrecord)

    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

    return dataset


# Diretórios dos TFRecords
train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Treino/*.tfrecord')
val_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Validação/*.tfrecord')

# Carregar os datasets
train_dataset = load_dataset(train_tfrecords, augment=True).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(val_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Criar um novo modelo com camadas adicionais
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    base_model,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(43, activation='softmax')
])

# Compilar o modelo antes de treinar
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Definir os pesos de classe ajustados
class_weights = {
    0: 33.5,
    1: 312.2,
    2: 208.2,
    3: 223.0,
    4: 184.2,
    5: 6244.0,
    6: 28.9,
    7: 447.4,
    8: 6244.0,
    9: 390.3,
    10: 624.4,
    11: 195.3,
    12: 60.0,
    13: 312.2,
    14: 240.2,
    15: 1041.3,
    16: 6244.0,
    17: 624.4,
    18: 6244.0,
    20: 3.96,
    21: 32.4,
    22: 390.3,
    23: 780.5,
    24: 65.0,
    25: 72.7,
    26: 71.0,
    27: 14.4,
    28: 284.7,
    29: 1041.3,
    30: 284.7,
    31: 520.4,
    32: 94.7,
    33: 1.0,
    34: 15.7,
    35: 120.3,
    36: 31.0,
    38: 208.2,
    39: 780.5,
    40: 780.5,
    41: 1561.0,
    42: 6244.0
}

# Treinar o modelo (primeira fase)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Descongelar as últimas camadas da ResNet
for layer in base_model.layers[-50:]:
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
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Avaliar o modelo no conjunto de teste
test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Teste/*.tfrecord')
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

test_loss, test_accuracy = model.evaluate(test_dataset)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
