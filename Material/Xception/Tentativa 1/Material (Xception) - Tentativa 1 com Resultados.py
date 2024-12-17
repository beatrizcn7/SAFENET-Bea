# ----------------- Bibliotecas ---------------
import tensorflow as tf # Principal biblioteca de Machine Learnig e Deep Learning. Cria, treina e implementa modelos de resdes neurais.
from tensorflow.keras import layers, models # API de alto nível que facilita a construção e treino de redes neurais dentro do TensorFlow
from tensorflow.keras.applications import Xception # Modelo pré-treinado utilizado
import pandas as pd # Serva para manipulação e análise de dados
import matplotlib.pyplot as plt # Para criar gráficos de diferentes tipos
import seaborn as sns # Para criar gráficos mais elaborados
from sklearn.metrics import confusion_matrix, roc_curve, auc # Usada para calcular a matriz de confusão, e a curva ROC e AUC
import numpy as np # Útil para operações matemáticas
import time  # Biblioteca para contar o tempo
from sklearn.preprocessing import label_binarize # Transforma um array de classes numa matriz binária

# --------------------- Modelo ------------------
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model.summary()

def parse_tfrecord(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    image = tf.image.decode_jpeg(parsed_example['image/encoded'], channels=3)
    image = tf.image.resize(image, [224, 224])
    label = parsed_example['image/object/class/label']
    return image, label

def load_dataset(file_paths):
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    dataset = raw_dataset.map(parse_tfrecord)
    return dataset

train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material/Treino/*.tfrecord')
val_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material/Validação/*.tfrecord')

train_dataset = load_dataset(train_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(val_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

start_time = time.time()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Guardar o modelo treinado
model.save('Tentativa 1 - Material.h5')
print('Guardado o modelo!')

test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material/Teste/*.tfrecord')
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

test_loss, test_accuracy = model.evaluate(test_dataset)

end_time = time.time()  # Fim
total_time = (end_time - start_time) / 60  # Converter para minutos
# Imprimir o tempo total
print(f"Tempo total: {total_time:.2f} minutos")

# Imprime os resultados da perda e da exatidão dos dados teste
print(f'Teste Loss: {test_loss}')
print(f'Teste Accuracy: {test_accuracy}')

# ------------- Representação de dados ----------
history_dict = history.history  # Obtém os dados do histórico de treino

data = {
    'Epoch': list(range(1, len(history_dict['loss']) + 1)),  # Número da época
    'Loss': history_dict['loss'],
    'Accuracy': history_dict['accuracy'],
    'Val_Loss': history_dict.get('val_loss', ['N/A'] * len(history_dict['loss'])),
    'Val_Accuracy': history_dict.get('val_accuracy', ['N/A'] * len(history_dict['loss']))
}

# Cria um DataFrame com os dados
df = pd.DataFrame(data)

# Exporta para um ficheiro Excel
df.to_excel('Ao longo das Épocas.xlsx', index=False)

print("Excel done")

# Determina o melhor epoch
best_epoch_acc = np.argmax(history_dict['val_accuracy']) + 1
best_epoch_loss = np.argmin(history_dict['val_loss']) + 1

# Criação da figura
plt.figure(figsize=(12, 6))

# Gráfico da Accuracy
plt.subplot(1, 2, 1)
plt.plot(data['Epoch'], data['Accuracy'], label='Treino', marker='o')
plt.plot(data['Epoch'], data['Val_Accuracy'], label='Validação', marker='o')
plt.axvline(best_epoch_acc, color='r', linestyle='--', label=f'Melhor Epoch ({best_epoch_acc})')
plt.title('Accuracy ao longo das Épocas')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Gráfico da Loss
plt.subplot(1, 2, 2)
plt.plot(data['Epoch'], data['Loss'], label='Treino', marker='o')
plt.plot(data['Epoch'], data['Val_Loss'], label='Validação', marker='o')
plt.axvline(best_epoch_loss, color='r', linestyle='--', label=f'Melhor Epoch ({best_epoch_loss})')
plt.title('Loss ao longo das Épocas')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Guardar a figura
plt.tight_layout()
plt.savefig('Loss e Accuracy.png')
print("Loss e Accuracy done")

y_true = [] # armazena as classess verdadeiras de cada batch dos dados teste
y_pred = [] # armazena as previsões feitas pelo modelo
y_pred_probs = [] # armazena as probabilidades previstas para cada classe

# Recolher as classes verdadeiras e as previsões
for images, labels in test_dataset:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
    y_pred_probs.extend(preds)

# Colocar binário as classes
y_true_binarized = label_binarize(y_true, classes=range(3))


# Calcular a matriz de confusão
cm = confusion_matrix(y_true, y_pred, labels=range(3))

# Inicializar as listas para armazenar as métricas (exatidão, precisão, recuperação e F1)
accuracies, precisions, recalls, f1_scores = [], [], [], []

# Calcular as métricas para cada label
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

    # F1
    f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    f1_scores.append(f1)

# Criar um dataframe com as métricas
metrics_df = pd.DataFrame({
    'Classe': range(3),
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1 Score': f1_scores
})

# Salvar o dataframe das métricas num arquivo Excel
metrics_df.to_excel('Métricas.xlsx', index=False)

print('Métricas calculadas')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
file_names = ['Accuracy.png', 'Precision.png', 'Recall.png', 'F1 Score.png']

for metric, file_name in zip(metrics, file_names):
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['Classe'], metrics_df[metric], marker='o')
    plt.title(f'{metric} por Classe')
    plt.xlabel('Classe')
    plt.ylabel(metric)
    plt.xticks(metrics_df['Classe'])  # Mostrar todas as labels no eixo x
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)  # Guardar gráfico como imagem
    plt.show()

print('Gráficos individuais criados.')

# Fazer o gráfico da matriz de confusão
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(3), yticklabels=range(3))
plt.title('Matriz de Confusão')
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Verdadeira')
plt.savefig('Matriz de Confusão.png')  # Guardar a matriz de confusão como imagem
plt.show()

print('Matriz de confusão criada')

# Criar um dataframe a partir da matriz de confusão
cm_df = pd.DataFrame(cm, index=[f'Verdadeiro {i}' for i in range(3)], columns=[f'Previsto {i}' for i in range(3)])

# Salvar o dataframe num arquivo Excel
cm_df.to_excel('Matriz de Confusão.xlsx', index=True)

print('Matriz de confusão guardado no Excel')

def plot_roc_for_three_classes(classes, file_name):
    plt.figure(figsize=(20, 5))  # Ajuste para 3 colunas e 1 linha

    for idx, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, cls], np.array(y_pred_probs)[:, cls])
        roc_auc = auc(fpr, tpr)

        plt.subplot(1, 5, idx + 1)
        plt.plot(fpr, tpr, color='blue', label=f'Classe {cls} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Falso Positivo')
        plt.ylabel('Verdadeiro Positivo')
        plt.title(f'ROC Classe {cls}')
        plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

# Gerar o gráfico para as 3 classes
plot_roc_for_three_classes(range(3), 'Material/Xception/Tentativa 1/ROC.png')

print('Gráficos ROC e AUC feitos.')
