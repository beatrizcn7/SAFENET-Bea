import os
import random
import shutil

# Diretório que contém os arquivos TFRecord
tfrecord_dir = 'Pasta Final TFRecord - Material + Ano + Estrutura'
output_train_dir = 'Pasta Final TFRecord - Material + Ano + Estrutura/Treino'
output_val_dir = 'Pasta Final TFRecord - Material + Ano + Estrutura/Validação'
output_test_dir = 'Pasta Final TFRecord - Material + Ano + Estrutura/Teste'

# Criar pastas para treino, validação e teste se não existirem
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# Listar todos os arquivos TFRecord na pasta
tfrecord_files = [f for f in os.listdir(tfrecord_dir) if f.endswith('.tfrecord')]

# Embaralhar a lista de arquivos
random.shuffle(tfrecord_files)

# Definir proporções
train_split = 0.8
val_split = 0.1
test_split = 0.1

# Determinar quantos arquivos irão para cada conjunto
total_files = len(tfrecord_files)
train_size = int(total_files * train_split)
val_size = int(total_files * val_split)
test_size = total_files - train_size - val_size  # O restante vai para o teste

# Separar os arquivos
train_files = tfrecord_files[:train_size]
val_files = tfrecord_files[train_size:train_size + val_size]
test_files = tfrecord_files[train_size + val_size:]

# Função para mover arquivos para as pastas correspondentes
def move_files(file_list, output_dir):
    for file in file_list:
        src_path = os.path.join(tfrecord_dir, file)
        dst_path = os.path.join(output_dir, file)
        shutil.move(src_path, dst_path)

# Mover arquivos para as pastas de treino, validação e teste
move_files(train_files, output_train_dir)
move_files(val_files, output_val_dir)
move_files(test_files, output_test_dir)

print(f"Total de ficheiros: {total_files}")
print(f"Treino: {len(train_files)} ficheiros")
print(f"Validação: {len(val_files)} ficheiros")
print(f"Teste: {len(test_files)} ficheiros")
