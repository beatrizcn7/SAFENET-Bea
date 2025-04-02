# Beatriz Neves, 18 de dezembro de 2024
# Input = Pasta Final TFRecord - Material + Ano (Treino, Teste e Valida��o)
# Output = Impress�o da loss e accuracy no terminal assim como o tempo, e os diversos ficheiros
# (Accuracy, F1 Score, Loss e Accuracy ao longo das �pocas, Matriz de Confus�o, M�tricas, Precision, Recall, ROC)
# Objetivo = Obter os resultados da mesma tentativa


import tensorflow as tf # Principal biblioteca de Machine Learnig e Deep Learning. Cria, treina e implementa modelos de resdes neurais.
from tensorflow.keras import layers, models # API de alto n�vel que facilita a constru��o e treino de redes neurais dentro do TensorFlow
from tensorflow.keras.applications import Xception # Modelo pr�-treinado utilizado
import pandas as pd # Serva para manipula��o e an�lise de dados
import matplotlib.pyplot as plt # Para criar gr�ficos de diferentes tipos
import seaborn as sns # Para criar gr�ficos mais elaborados
from sklearn.metrics import confusion_matrix, roc_curve, auc # Usada para calcular a matriz de confus�o, e a curva ROC e AUC
import numpy as np # �til para opera��es matem�ticas
import time  # Biblioteca para contar o tempo
from sklearn.preprocessing import label_binarize # Transforma um array de classes numa matriz bin�ria

# --------------------- Modelo ------------------
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(5, activation='softmax')
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

train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Ano/Treino/*.tfrecord')
val_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Ano/Valida��o/*.tfrecord')

train_dataset = load_dataset(train_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(val_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

start_time = time.time()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Guardar o modelo treinado
# model.save('Tentativa 1 - Material + Ano (Do zero).h5')
# print('Guardado o modelo!')

test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Ano/Teste/*.tfrecord')
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

test_loss, test_accuracy = model.evaluate(test_dataset)

end_time = time.time()  # Fim
total_time = (end_time - start_time) / 60  # Converter para minutos
# Imprimir o tempo total
print(f"Tempo total: {total_time:.2f} minutos")

# Imprime os resultados da perda e da exatid�o dos dados teste
print(f'Teste Loss: {test_loss}')
print(f'Teste Accuracy: {test_accuracy}')

# ------------- Representa��o de dados ----------
history_dict = history.history  # Obt�m os dados do hist�rico de treino

data = {
    'Epoch': list(range(1, len(history_dict['loss']) + 1)),  # N�mero da �poca
    'Loss': history_dict['loss'],
    'Accuracy': history_dict['accuracy'],
    'Val_Loss': history_dict.get('val_loss', ['N/A'] * len(history_dict['loss'])),
    'Val_Accuracy': history_dict.get('val_accuracy', ['N/A'] * len(history_dict['loss']))
}

# Cria um DataFrame com os dados
df = pd.DataFrame(data)

# Exporta para um ficheiro Excel
df.to_excel('Ao longo das �pocas.xlsx', index=False)

print("Excel done")

# Determina o melhor epoch
best_epoch_acc = np.argmax(history_dict['val_accuracy']) + 1
best_epoch_loss = np.argmin(history_dict['val_loss']) + 1

# Cria��o da figura
plt.figure(figsize=(12, 6))

# Gr�fico da Accuracy
plt.subplot(1, 2, 1)
plt.plot(data['Epoch'], data['Accuracy'], label='Treino', marker='o')
plt.plot(data['Epoch'], data['Val_Accuracy'], label='Valida��o', marker='o')
plt.axvline(best_epoch_acc, color='r', linestyle='--', label=f'Melhor Epoch ({best_epoch_acc})')
plt.title('Accuracy ao longo das �pocas')
plt.xlabel('�pocas')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Gr�fico da Loss
plt.subplot(1, 2, 2)
plt.plot(data['Epoch'], data['Loss'], label='Treino', marker='o')
plt.plot(data['Epoch'], data['Val_Loss'], label='Valida��o', marker='o')
plt.axvline(best_epoch_loss, color='r', linestyle='--', label=f'Melhor Epoch ({best_epoch_loss})')
plt.title('Loss ao longo das �pocas')
plt.xlabel('�pocas')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Guardar a figura
plt.tight_layout()
plt.savefig('Loss e Accuracy.png')
print("Loss e Accuracy done")

y_true = [] # armazena as classess verdadeiras de cada batch dos dados teste
y_pred = [] # armazena as previs�es feitas pelo modelo
y_pred_probs = [] # armazena as probabilidades previstas para cada classe

# Recolher as classes verdadeiras e as previs�es
for images, labels in test_dataset:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
    y_pred_probs.extend(preds)

# Colocar bin�rio as classes
y_true_binarized = label_binarize(y_true, classes=range(5))

# Calcular a matriz de confus�o
cm = confusion_matrix(y_true, y_pred, labels=range(3))

# Inicializar as listas para armazenar as m�tricas (exatid�o, precis�o, recupera��o e F1)
accuracies, precisions, recalls, f1_scores = [], [], [], []

# Calcular as m�tricas para cada label
for i in range(len(cm)):
    TP = cm[i, i]
    FP = np.sum(cm[:, i]) - TP
    FN = np.sum(cm[i, :]) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    # Exatid�o
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracies.append(accuracy)

    # Precis�o
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    precisions.append(precision)

    # Recupera��o
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    recalls.append(recall)

    # F1
    f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    f1_scores.append(f1)

# Criar um dataframe com as m�tricas
metrics_df = pd.DataFrame({
    'Classe': range(5),
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1 Score': f1_scores
})

# Salvar o dataframe das m�tricas num arquivo Excel
metrics_df.to_excel('M�tricas.xlsx', index=False)

print('M�tricas calculadas')

print('M�tricas calculadas')

# Fazer os gr�ficos para cada m�trica individual e guardar separadamente
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
    plt.savefig(file_name)  # Guardar gr�fico como imagem
    plt.show()

print('Gr�ficos individuais criados.')

# Fazer o gr�fico da matriz de confus�o
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(5), yticklabels=range(5))
plt.title('Matriz de Confus�o')
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Verdadeira')
plt.savefig('Matriz de Confus�o.png')  # Guardar a matriz de confus�o como imagem
plt.show()

print('Matriz de confus�o criada')

# Criar um dataframe a partir da matriz de confus�o
cm_df = pd.DataFrame(cm, index=[f'Verdadeiro {i}' for i in range(5)], columns=[f'Previsto {i}' for i in range(5)])

# Salvar o dataframe num arquivo Excel
cm_df.to_excel('Matriz de Confus�o.xlsx', index=True)

print('Matriz de confus�o guardado no Excel')

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

# Gerar o gr�fico para as 3 classes
plot_roc_for_three_classes(range(5), 'ROC.png')

print('Gr�ficos ROC e AUC feitos.')

print('Gr�ficos ROC e AUC feitos.')

