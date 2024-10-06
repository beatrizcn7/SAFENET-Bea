import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns  # Para uma melhor visualização da matriz de confusão

# Carregar o modelo ResNet50 com pesos pré-treinados do ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Criar uma camada de aumento de dados
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# Criar um novo modelo
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),  # Definir a forma de entrada
    data_augmentation,  # Adicione a camada de aumento de dados
    base_model,
    layers.Flatten(),
    layers.Dense(43, activation='softmax')  # As labels vão da 0 à 42 (43 combinações possíveis)
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Função para carregar TFRecords
def parse_tfrecord(example_proto):
    # Defina o formato do TFRecord
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def load_dataset(file_paths):
    # Cria um dataset a partir de múltiplos arquivos TFRecord
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

# Treinar o modelo
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Avaliar o modelo
# Carregar o conjunto de teste
test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Teste/*.tfrecord')
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_dataset)

# Imprime os resultados da perda e da exatidão do modelo
print(f'Teste Loss: {test_loss}')
print(f'Teste Accuracy: {test_accuracy}')

# Obter as previsões no conjunto de teste
y_true = []
y_pred = []

for images, labels in test_dataset:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Calcular a matriz de confusão
cm = confusion_matrix(y_true, y_pred, labels=range(43))

# Plotar o gráfico da matriz de confusão
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(43), yticklabels=range(43))
plt.title('Confusion Matrix')
plt.xlabel('Previsto Label')
plt.ylabel('Verdadeiro Label')
plt.savefig('Matriz de Confusão.png')  # Salvar a matriz de confusão como imagem
plt.show()

print('Matriz de confusão criada')

# Inicializar listas para armazenar as métricas
accuracies, precisions, recalls, f1_scores = [], [], [], []

# Cálculo das métricas para cada classe
for i in range(len(cm)):
    TP = cm[i, i]
    FP = np.sum(cm[:, i]) - TP
    FN = np.sum(cm[i, :]) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    # Exatidão
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracies.append(accuracy)

    # Precisão
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    precisions.append(precision)

    # Recuperação
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    recalls.append(recall)

    # F1 Score
    f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    f1_scores.append(f1)

# Criar um DataFrame com as métricas
metrics_df = pd.DataFrame({
    'Label': range(43),
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1 Score': f1_scores
})

# Salvar o DataFrame das métricas em um arquivo Excel
metrics_df.to_excel('Métricas.xlsx', index=False)

print('Métricas calculadas')

# Criar um DataFrame com os resultados de cada época
results_df = pd.DataFrame({
    'Epoch': range(1, len(history.history['accuracy']) + 1),
    'Loss': history.history['loss'],
    'Accuracy': history.history['accuracy'],
    'Val Loss': history.history['val_loss'],
    'Val Accuracy': history.history['val_accuracy']
})

# Salvar o DataFrame de resultados em um arquivo Excel
results_df.to_excel('Resultados.xlsx', index=False)

print('Excel criado')

# Plotar os gráficos de Accuracy e Loss
epochs = range(1, len(history.history['accuracy']) + 1)

# Identificar a melhor epoch com base na validação (para marcar no gráfico)
best_epoch_acc = epochs[history.history['val_accuracy'].index(max(history.history['val_accuracy']))]
best_epoch_loss = epochs[history.history['val_loss'].index(min(history.history['val_loss']))]

plt.figure(figsize=(12, 5))

# Gráfico 1: Accuracy ao longo das épocas
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], label='Accuracy de Treino', marker='o')
plt.plot(epochs, history.history['val_accuracy'], label='Accuracy de Validação', marker='o')
plt.axvline(x=best_epoch_acc, color='r', linestyle='--', label=f'Melhor Epoch {best_epoch_acc}')
plt.title('Accuracy ao longo dos Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Gráfico 2: Loss ao longo das épocas
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], label='Loss de Treino', marker='o')
plt.plot(epochs, history.history['val_loss'], label='Loss de Validação', marker='o')
plt.axvline(x=best_epoch_loss, color='r', linestyle='--', label=f'Melhor Epoch {best_epoch_loss}')
plt.title('Loss ao longo dos Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Ajustar layout e mostrar os gráficos
plt.tight_layout()
plt.savefig('Loss e Accuracy.png')  # Salvar os gráficos em uma imagem
plt.show()

print('Gráficos de Accuracy e Loss criados')

# Plotar gráficos para cada métrica individual e salvar em imagens separadas
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
file_names = ['Accuracy.png', 'Precision.png', 'Recall.png', 'F1 Score.png']

for metric, file_name in zip(metrics, file_names):
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['Label'], metrics_df[metric], marker='o')
    plt.title(f'{metric} por Label')
    plt.xlabel('Label')
    plt.ylabel(metric)
    plt.xticks(metrics_df['Label'])  # Mostrar todas as classes no eixo x
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)  # Salvar gráfico como imagem
    plt.show()

print('Gráficos individuais criados.')
