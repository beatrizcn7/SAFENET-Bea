import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import numpy as np
import pandas as pd  # Para salvar os resultados em Excel
import time  # Biblioteca para contar o tempo
import matplotlib.pyplot as plt  # Biblioteca para plotar gráficos

# Carregar o modelo ResNet50 com pesos pré-treinados do ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Camada de aumento de dados
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.3),
    layers.RandomTranslation(0.2, 0.2),
], name="data_augmentation")

# Construir o modelo completo
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    base_model,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(43, activation='softmax')  # 43 classes
])

# Compilar o modelo (primeira fase)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Função para carregar TFRecords
def parse_tfrecord(example_proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def load_dataset(file_paths):
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    return raw_dataset.map(parse_tfrecord).map(
        lambda x: (
            tf.image.resize(tf.image.decode_jpeg(x['image/encoded'], channels=3), [224, 224]),
            x['image/object/class/label']
        )
    )

# Contar os exemplos por classe e garantir que todas as classes estejam representadas
def compute_class_weights(dataset, num_classes=43):
    class_counts = {}
    total_count = 0

    for _, label in dataset:
        label = int(label.numpy())  # Converter o rótulo para um inteiro
        class_counts[label] = class_counts.get(label, 0) + 1
        total_count += 1

    class_weights = {
        label: total_count / (count * num_classes) if count > 0 else 0
        for label, count in class_counts.items()
    }

    # Garantir que todas as classes estejam representadas de 0 até num_classes - 1
    for label in range(num_classes):
        if label not in class_weights:
            class_weights[label] = 0

    return class_weights

# Diretórios dos TFRecords
train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Treino/*.tfrecord')
val_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Validação/*.tfrecord')

# Carregar os datasets
train_dataset = load_dataset(train_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(val_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Calcular os pesos de classe automaticamente
class_weights = compute_class_weights(load_dataset(train_tfrecords).batch(1), num_classes=43)

# Treinar o modelo (primeira fase)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Início da contagem de tempo
start_time = time.time()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Criar DataFrame para salvar resultados da primeira fase
results_df = pd.DataFrame({
    'epoch': range(1, len(history.history['loss']) + 1),
    'loss': history.history['loss'],
    'accuracy': history.history['accuracy'],
    'val_loss': history.history['val_loss'],
    'val_accuracy': history.history['val_accuracy']
})

# Descongelar as últimas camadas da ResNet
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Compilar novamente com uma taxa de aprendizado menor
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

# Atualizar DataFrame com resultados da segunda fase
fine_tune_df = pd.DataFrame({
    'epoch': range(len(results_df) + 1, len(results_df) + len(fine_tune_history.history['loss']) + 1),
    'loss': fine_tune_history.history['loss'],
    'accuracy': fine_tune_history.history['accuracy'],
    'val_loss': fine_tune_history.history['val_loss'],
    'val_accuracy': fine_tune_history.history['val_accuracy']
})

# Concatenar resultados da primeira e segunda fase
results_df = pd.concat([results_df, fine_tune_df], ignore_index=True)

# Salvar o DataFrame em um arquivo Excel
results_df.to_excel("Resultados 2.xlsx", index=False)

# Avaliar o modelo
test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Teste/*.tfrecord')
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

test_loss, test_accuracy = model.evaluate(test_dataset)

end_time = time.time()  # Fim
total_time = (end_time - start_time) / 60  # Converter para minutos
# Imprimir o tempo total
print(f"Tempo total: {total_time:.2f} minutos")

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Criar gráficos para Loss e Accuracy
epochs_range = range(1, len(history.history['loss']) + 1)

plt.figure(figsize=(14, 6))

# Gráfico de Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Treino')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validação')
plt.title('Accuracy ao longo das épocas')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Gráfico de Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Treino')
plt.plot(epochs_range, history.history['val_loss'], label='Validação')
plt.title('Loss ao longo das épocas')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# Salvar os gráficos
plt.suptitle('Loss e Accuracy')
plt.savefig('Loss e Accuracy 2.png')
plt.show()
