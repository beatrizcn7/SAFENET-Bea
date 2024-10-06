import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import pandas as pd
import matplotlib.pyplot as plt

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

# Criar um novo modelo com camadas adicionais
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    base_model,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(43, activation='softmax')  # 43 combinações possíveis
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

# Definir os pesos de classe ajustados
class_weights = {
    0: 33.5, 1: 312.2, 2: 208.2, 3: 223.0, 4: 184.2, 5: 6244.0, 6: 28.9, 7: 447.4, 8: 6244.0, 9: 390.3,
    10: 624.4, 11: 195.3, 12: 60.0, 13: 312.2, 14: 240.2, 15: 1041.3, 16: 6244.0, 17: 624.4, 18: 6244.0, 20: 3.96,
    21: 32.4, 22: 390.3, 23: 780.5, 24: 65.0, 25: 72.7, 26: 71.0, 27: 14.4, 28: 284.7, 29: 1041.3, 30: 284.7,
    31: 520.4, 32: 94.7, 33: 1.0, 34: 15.7, 35: 120.3, 36: 31.0, 38: 208.2, 39: 780.5, 40: 780.5, 41: 1561.0, 42: 6244.0
}

# Treinar o modelo (primeira fase)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Armazenar as métricas em uma lista
history_data = []

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Coletar as métricas de treinamento e validação
for epoch in range(len(history.history['loss'])):
    history_data.append({
        'epoch': epoch + 1,
        'loss': history.history['loss'][epoch],
        'accuracy': history.history['accuracy'][epoch],
        'val_loss': history.history['val_loss'][epoch],
        'val_accuracy': history.history['val_accuracy'][epoch]
    })

# Criar um DataFrame do pandas e salvar em um arquivo Excel
results_df = pd.DataFrame(history_data)
results_df.to_excel('Resultados 2.xlsx', index=False)

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

# Função para criar os gráficos de Loss e Accuracy
def plot_metrics(history, title='Loss e Accuracy'):
    # Obter as métricas de treino e validação
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Melhor epoch baseado na menor loss de validação
    best_epoch = val_loss.index(min(val_loss)) + 1

    # Criar subplots para Loss e Accuracy
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Loss
    ax[0].plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
    ax[0].plot(range(1, len(val_loss) + 1), val_loss, label='Val Loss')
    ax[0].set_title('Loss over Epochs')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
    ax[0].legend()

    # Plot Accuracy
    ax[1].plot(range(1, len(train_acc) + 1), train_acc, label='Train Accuracy')
    ax[1].plot(range(1, len(val_acc) + 1), val_acc, label='Val Accuracy')
    ax[1].set_title('Accuracy over Epochs')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
    ax[1].legend()

    # Ajustar layout e salvar a figura
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig('Loss e Accuracy 2.png')
    plt.show()

# Chamar a função para plotar os gráficos usando os dados da primeira fase do treinamento
plot_metrics(history)

# Avaliar o modelo
test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Teste/*.tfrecord')
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

