# ----------------- Bibliotecas ---------------
import tensorflow as tf # Principal biblioteca de Machine Learnig e Deep Learning. Cria, treina e implementa modelos de resdes neurais.
from tensorflow.keras import layers, models # API de alto n�vel que facilita a constru��o e treino de redes neurais dentro do TensorFlow
from tensorflow.keras.applications import ResNet50 # Modelo pr�-treinado utilizado
import pandas as pd # Serve para manipula��o e an�lise de dados
import matplotlib.pyplot as plt # Para criar gr�ficos de diferentes tipos
import seaborn as sns # Para criar gr�ficos mais elaborados
from sklearn.metrics import confusion_matrix, roc_curve, auc # Usada para calcular a matriz de confus�o, e a curva ROC e AUC
import numpy as np # �til para opera��es matem�ticas
import time  # Biblioteca para contar o tempo
from sklearn.preprocessing import label_binarize # Transforma um array de classes numa matriz bin�ria


# --------------------- Modelo ------------------
# Carregar o modelo salvo para a propriedade "Material"
model = tf.keras.models.load_model('Tentativa 1 - Material + Ano (aprendido com Tentativa 1 - Material).h5')

model.pop()  # Remover a �ltima camada de sa�da (5 classes)
model.add(layers.Dense(11, activation='softmax'))  #  11 classes

# Congelar as camadas do modelo base, menos a �ltima camada adicionada
for layer in model.layers[:-20]:
    layer.trainable = False

# Compilar o modelo novamente com a nova configura��o
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fun��o para carregar TFRecords
def parse_tfrecord(example):
    # Definir o formato do TFRecord
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example, feature_description)

def load_dataset(file_paths):
    # Criar dataset a partir dos ficheiros TFRecord
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    return raw_dataset.map(parse_tfrecord).map(
        lambda x: (
            tf.image.resize(tf.image.decode_jpeg(x['image/encoded'], channels=3), [224, 224]),
            x['image/object/class/label']
        )
    )

# Pastas dos ficheiros TFRecords (Treino, Valida��o e Teste)
train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Ano + Estrutura/Treino/*.tfrecord')
val_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Ano + Estrutura/Valida��o/*.tfrecord')

# Carregar os datasets
train_dataset = load_dataset(train_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(val_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# In�cio da contagem de tempo
start_time = time.time()

# Treinar o modelo com os novos dados (Material e Ano)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Guardar o modelo treinado
model.save('Tentativa 1- Material + Ano + Estrutura (Tentativa 1 - Material + Ano (aprendido com Tentativa 1 - Material)).h5')
print('Guardado o modelo!')

test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Ano + Estrutura/Teste/*.tfrecord')
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_dataset)

# Fim da contagem de tempo
end_time = time.time()
total_time = (end_time - start_time) / 60

# Imprimir o tempo total
print(f"Tempo total: {total_time:.2f} minutos")

# Imprime os resultados da perda e da exatid�o dos dados teste
print(f'Teste Loss: {test_loss}')
print(f'Teste Accuracy: {test_accuracy}')

# ----------------- Representa��o dos dados -----------------

# Criar um dataframe com os resultados de cada �poca
results_df = pd.DataFrame({
    'Epoch': range(1, len(history.history['accuracy']) + 1),
    'Loss': history.history['loss'],
    'Accuracy': history.history['accuracy'],
    'Val Loss': history.history['val_loss'],
    'Val Accuracy': history.history['val_accuracy']
})

# Guardar o dataframe de resultados num arquivo Excel
results_df.to_excel('Ao longo das �pocas.xlsx', index=False)

print('Excel criado')

# Fazer os gr�ficos de accuracy e loss
epochs = range(1, len(history.history['accuracy']) + 1)

# Identificar a melhor epoch
best_epoch_acc = epochs[history.history['val_accuracy'].index(max(history.history['val_accuracy']))]
best_epoch_loss = epochs[history.history['val_loss'].index(min(history.history['val_loss']))]

plt.figure(figsize=(12, 5))

# Gr�fico 1: Accuracy ao longo das �pocas
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], label='Accuracy de Treino', marker='o')
plt.plot(epochs, history.history['val_accuracy'], label='Accuracy de Valida��o', marker='o')
plt.axvline(x=best_epoch_acc, color='r', linestyle='--', label=f'Melhor Epoch {best_epoch_acc}')
plt.title('Accuracy ao longo dos Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Gr�fico 2: Loss ao longo das �pocas
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], label='Loss de Treino', marker='o')
plt.plot(epochs, history.history['val_loss'], label='Loss de Valida��o', marker='o')
plt.axvline(x=best_epoch_loss, color='r', linestyle='--', label=f'Melhor Epoch {best_epoch_loss}')
plt.title('Loss ao longo dos Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Ajustar layout e mostrar os gr�ficos
plt.tight_layout()
plt.savefig('Loss e Accuracy.png')  # Guardar os gr�ficos numa imagem
plt.show()

print('Gr�ficos de Accuracy e Loss criados')

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
y_true_binarized = label_binarize(y_true, classes=range(11))


# Calcular a matriz de confus�o
cm = confusion_matrix(y_true, y_pred, labels=range(11))

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
    'Classe': range(11),
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1 Score': f1_scores
})

# Salvar o dataframe das m�tricas num arquivo Excel
metrics_df.to_excel('M�tricas.xlsx', index=False)

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
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(11), yticklabels=range(11))
plt.title('Matriz de Confus�o')
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Verdadeira')
plt.savefig('Matriz de Confus�o.png')  # Guardar a matriz de confus�o como imagem
plt.show()

print('Matriz de confus�o criada')

# Criar um dataframe a partir da matriz de confus�o
cm_df = pd.DataFrame(cm, index=[f'Verdadeiro {i}' for i in range(11)], columns=[f'Previsto {i}' for i in range(11)])

# Salvar o dataframe num arquivo Excel
cm_df.to_excel('Matriz de Confus�o.xlsx', index=True)

print('Matriz de confus�o guardado no Excel')

# Fun��o para fazer o gr�fico de ROC para 11 classes
def plot_roc_for_eleven_classes(classes, file_name):
    plt.figure(figsize=(20, 10))  # Ajustar tamanho para 2 linhas

    for idx, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, cls], np.array(y_pred_probs)[:, cls])
        roc_auc = auc(fpr, tpr)

        # Organizar em 2 linhas: 6 gr�ficos na primeira linha, 5 na segunda
        plt.subplot(2, 6, idx + 1)  # Configurar subplot para 2 linhas e 6 colunas (�ltima posi��o ficar� vazia)
        plt.plot(fpr, tpr, color='blue', label=f'Classe {cls} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Falso Positivo')
        plt.ylabel('Verdadeiro Positivo')
        plt.title(f'ROC Classe {cls}')
        plt.legend(loc="lower right")

        # Ajustar o layout para evitar sobreposi��o
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

# Gerar o gr�fico para as 11 classes
plot_roc_for_eleven_classes(range(11), 'ROC.png')

print('Gr�ficos ROC e AUC feitos.')