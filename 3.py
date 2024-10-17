# ----------------- Bibliotecas ---------------
import tensorflow as tf # Principal biblioteca de Machine Learnig e Deep Learning. Cria, treina e implementa modelos de resdes neurais.
from tensorflow.keras import layers, models # API de alto nível que facilita a construção e treino de redes neurais dentro do TensorFlow
from tensorflow.keras.applications import ResNet50 # Modelo pré-treinado utilizado
import pandas as pd # Serva para manipulação e análise de dados
import matplotlib.pyplot as plt # Para criar gráficos de diferentes tipos
import seaborn as sns # Para criar gráficos mais elaborados
from sklearn.metrics import confusion_matrix # Usada para calcular a matriz de confusão
import numpy as np # Útil para operações matemáticas

# --------------------- Modelo ------------------
# Cria uma instância do modelo ResNet50 com pesos pré-treinados do ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Aumentar os dados de treino com rotações, refleção, zoom, contraste e brilho aleatoriamente
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"), # verticalmente ou horizontalmente
    layers.RandomRotation(0.2), # até 20%
    layers.RandomZoom(0.2), # até 20%
    layers.RandomContrast(0.2), # até 20%
    layers.RandomBrightness(0.2), # até 20%
], name="data_augmentation")

# Criar um novo modelo
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)), # Definir a forma de entrada
    data_augmentation,  # Adicionar a camada de aumento de dados
    base_model,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'), # Camada conectada com 1024 neurónios e a função ativação relu
    layers.Dropout(0.5), # dropout com uma taxa de 50%
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

# Pastas dos ficheiros TFRecords (Treino e Validação)
train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Treino/*.tfrecord')
val_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Validação/*.tfrecord')

# Carregar os datasets
train_dataset = load_dataset(train_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(val_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Interrompe o treino do modelo de forma antecipada se o desempenho no conjunto de validação não melhorar
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# COMENTÁRIO
history_metrics = {
    "epoch": [],
    "loss": [],
    "accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

# Treinar o modelo
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping] # Pode parar o treino se o desempenho nos dados de validação não melhorar
)

# COMENTÁRIO
for epoch in range(len(history.history['loss'])):
    history_metrics['epoch'].append(epoch + 1)
    history_metrics['loss'].append(history.history['loss'][epoch])
    history_metrics['accuracy'].append(history.history['accuracy'][epoch])
    history_metrics['val_loss'].append(history.history['val_loss'][epoch])
    history_metrics['val_accuracy'].append(history.history['val_accuracy'][epoch])

# Ativa o treino das últimas 30 camadas
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Compilar o modelo usando o otimizador Adam com uma taxa de aprendizagem
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Inicia o treino do modelo por 10 épocas e implementa um callback
fine_tune_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping]
)

# COMENTÁRIO
for epoch in range(len(fine_tune_history.history['loss'])):
    history_metrics['epoch'].append(epoch + 11)  # Continuar a numeração das épocas
    history_metrics['loss'].append(fine_tune_history.history['loss'][epoch])
    history_metrics['accuracy'].append(fine_tune_history.history['accuracy'][epoch])
    history_metrics['val_loss'].append(fine_tune_history.history['val_loss'][epoch])
    history_metrics['val_accuracy'].append(fine_tune_history.history['val_accuracy'][epoch])

# Carregar o conjunto de teste
test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Teste/*.tfrecord')
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_dataset)

# Imprime os resultados da perda e da exatidão dos dados teste
print(f'Teste Loss: {test_loss}')
print(f'Teste Accuracy: {test_accuracy}')

# ------------- Representação de dados ----------
# Guardar o dataframe de resultados num arquivo Excel
df_metrics = pd.DataFrame(history_metrics)
df_metrics.to_excel('Resultados.xlsx', index=False)

print('Excel criado')

plt.figure(figsize=(12, 5))

# Identificar a melhor epoch em Accuracy
best_val_accuracy = max(history_metrics['val_accuracy'])
best_epoch_accuracy = history_metrics['epoch'][history_metrics['val_accuracy'].index(best_val_accuracy)]

# Identificar a melhor epoch em Loss
best_val_loss = min(history_metrics['val_loss'])
best_epoch_loss = history_metrics['epoch'][history_metrics['val_loss'].index(best_val_loss)]

# Gráfico 1: Accuracy ao longo das épocas
plt.subplot(1, 2, 2)
plt.plot(history_metrics['epoch'], history_metrics['accuracy'], label='Accuracy de Treino')
plt.plot(history_metrics['epoch'], history_metrics['val_accuracy'], label='Accuracy de Validação')
plt.axvline(x=best_epoch_accuracy, color='r', linestyle='--', label=f'Melhor Epoch {best_epoch_accuracy}')
plt.title('Accuracy ao longo dos Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Gráfico 2: Loss ao longo das épocas
plt.subplot(1, 2, 1)
plt.plot(history_metrics['epoch'], history_metrics['loss'], label='Loss de Treino')
plt.plot(history_metrics['epoch'], history_metrics['val_loss'], label='Loss de Validação')
plt.axvline(x=best_epoch_loss, color='r', linestyle='--', label=f'Melhor Epoch {best_epoch_loss}')
plt.title('Loss ao longo dos Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# Ajustar layout e mostrar os gráficos
plt.tight_layout()
plt.savefig('Loss e Accuracy.png')  # Guardar os gráficos numa imagem
plt.show()

print('Gráficos de Accuracy e Loss criados')

y_true = [] # armazena as labels verdadeiras de cada batch dos dados teste
y_pred = [] # armazena as previsões feitas pelo modelo

# Recolher as labels verdadeiras e as previsões
for images, labels in test_dataset:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Calcular a matriz de confusão
cm = confusion_matrix(y_true, y_pred, labels=range(43))

# Salvar a matriz de confusão em Excel
df_cm = pd.DataFrame(cm, index=range(43), columns=range(43))
df_cm.to_excel("Matrix de Confusao.xlsx", index_label="True Labels")

# Gráfico da matriz de confusão
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(43), yticklabels=range(43))
plt.title('Confusion Matrix')
plt.xlabel('Previsto Label')
plt.ylabel('Verdadeiro Label')
plt.savefig('Matriz de Confusão.png')
plt.show()

print('Matriz de confusão criada')