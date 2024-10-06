import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Carregar o modelo ResNet50 com pesos pré-treinados do ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Criar uma camada de aumento de dados
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
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

# Treinar o modelo (primeira fase)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Criar listas para armazenar as métricas
history_metrics = {
    "epoch": [],
    "loss": [],
    "accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

# Treinar o modelo e armazenar as métricas
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping]
)

# Armazenar as métricas após cada época
for epoch in range(len(history.history['loss'])):
    history_metrics['epoch'].append(epoch + 1)
    history_metrics['loss'].append(history.history['loss'][epoch])
    history_metrics['accuracy'].append(history.history['accuracy'][epoch])
    history_metrics['val_loss'].append(history.history['val_loss'][epoch])
    history_metrics['val_accuracy'].append(history.history['val_accuracy'][epoch])

# Descongelar as últimas camadas da ResNet
for layer in base_model.layers[-30:]:  # Ajuste o número de camadas conforme necessário
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
    callbacks=[early_stopping]
)

# Armazenar as métricas após o fine-tuning
for epoch in range(len(fine_tune_history.history['loss'])):
    history_metrics['epoch'].append(epoch + 11)  # Continuar a numeração das épocas
    history_metrics['loss'].append(fine_tune_history.history['loss'][epoch])
    history_metrics['accuracy'].append(fine_tune_history.history['accuracy'][epoch])
    history_metrics['val_loss'].append(fine_tune_history.history['val_loss'][epoch])
    history_metrics['val_accuracy'].append(fine_tune_history.history['val_accuracy'][epoch])

# Avaliar o modelo
# Carregar o conjunto de teste
test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Teste/*.tfrecord')
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Criar um DataFrame a partir das métricas
df_metrics = pd.DataFrame(history_metrics)

# Salvar o DataFrame em um arquivo Excel
df_metrics.to_excel("Resultados 5.xlsx", index=False)

print("Resultados salvos em Excel.")

# Criar gráficos de Loss e Accuracy
plt.figure(figsize=(12, 6))

# Gráfico de Loss
plt.subplot(1, 2, 1)
plt.plot(history_metrics['epoch'], history_metrics['loss'], label='Loss Treino')
plt.plot(history_metrics['epoch'], history_metrics['val_loss'], label='Loss Validação')
plt.title('Loss ao longo das Épocas')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

# Gráfico de Accuracy
plt.subplot(1, 2, 2)
plt.plot(history_metrics['epoch'], history_metrics['accuracy'], label='Accuracy Treino')
plt.plot(history_metrics['epoch'], history_metrics['val_accuracy'], label='Accuracy Validação')
plt.title('Accuracy ao longo das Épocas')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()

# Salvar os gráficos em um arquivo
plt.tight_layout()
plt.savefig('Loss_Accuracy 5.png')
plt.close()

print("Gráficos de Loss e Accuracy salvos.")

# Realizar previsões no conjunto de teste
y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(tf.argmax(predictions, axis=1).numpy())

# Calcular a matriz de confusão
cm = confusion_matrix(y_true, y_pred)

# Calcular as métricas
TP = cm.diagonal()  # Verdadeiros positivos
FP = cm.sum(axis=0) - TP  # Falsos positivos
FN = cm.sum(axis=1) - TP  # Falsos negativos
TN = cm.sum() - (FP + FN + TP)  # Verdadeiros negativos

# Substituir divisões inválidas por zero
precision = np.divide(TP, (TP + FP), out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
recall = np.divide(TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
f1 = np.divide(2 * (precision * recall), (precision + recall), out=np.zeros_like(precision, dtype=float), where=(precision + recall) != 0)

# Calcular a exatidão global
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Verificar se as métricas possuem o mesmo tamanho antes de criar o DataFrame
min_length = min(len(accuracy), len(precision), len(recall), len(f1))

# Criar um DataFrame para as métricas
df_metrics_final = pd.DataFrame({
    'Classe': range(min_length),  # Número de classes
    'Exatidão': accuracy[:min_length],
    'Precisão': precision[:min_length],
    'Recuperação': recall[:min_length],
    'F1': f1[:min_length]
})

# Salvar o DataFrame em um arquivo Excel
df_metrics_final.to_excel("Métricas 5.xlsx", index=False)

print("Métricas salvas em Excel.")

# Criar gráficos para as quatro métricas: Accuracy, F1 Score, Precision, Recall
metrics = {
    "Accuracy": accuracy,
    "F1 Score": f1,
    "Precision": precision,
    "Recall": recall
}

# Verificação e substituição de valores NaN por 0
for key in metrics:
    metrics[key] = np.nan_to_num(metrics[key])

# Função para plotar e salvar os gráficos
def plot_and_save(metric_name, values):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(values)), values)
    plt.title(f'{metric_name} por Classe')
    plt.xlabel('Classe')
    plt.ylabel(metric_name)
    plt.savefig(f'{metric_name} 5.png')
    plt.close()
    print(f"Gráfico de {metric_name} salvo.")

# Plotar e salvar gráficos de cada métrica
for metric_name, values in metrics.items():
    plot_and_save(metric_name, values)

# Criar gráfico da matriz de confusão
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(min_length), yticklabels=range(min_length))
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')

# Salvar a matriz de confusão como imagem
plt.savefig('Matriz de Confusão 5.png')
plt.close()

print("Matriz de confusão salva como 'Matriz de Confusão 4.png'.")