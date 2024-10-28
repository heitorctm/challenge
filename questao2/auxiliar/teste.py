import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
import json
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine, cityblock, euclidean
from .redes import redes

def calcular_similaridade(embedding1, embedding2, metrica):
    if metrica == "cosine":
        distancia = cosine(embedding1, embedding2)
        similaridade = np.round((1 - distancia) * 100, 2)
    elif metrica == "euclidean":
        distancia = euclidean(embedding1, embedding2)
        similaridade = np.round(1 / (1 + distancia) * 100, 2)
    elif metrica == "manhattan":
        distancia = cityblock(embedding1, embedding2)
        similaridade = np.round(1 / (1 + distancia) * 100, 2)
    elif metrica == "cosine_sklearn":
        similaridade = np.round(cosine_similarity([embedding1], [embedding2])[0][0] * 100, 2)
    else:
        raise ValueError(f"similaridade '{metrica}' nao suportada.")
    return similaridade

def carregar_modelo_de_embedding(caminho_modelo):
    modelo_completo = tf.keras.models.load_model(caminho_modelo, compile=False)
    embedding_model = modelo_completo.get_layer('Embedding')
    return embedding_model

def carregar_imagem(caminho_imagem, rede, tamanho=None):
    if rede in redes:
        _, preprocess_input, tamanho_rede = redes[rede]
        tamanho = tamanho or tamanho_rede
    else:
        raise ValueError(f'Rede {rede} nao suportada.')
    
    img = image.load_img(caminho_imagem, target_size=tamanho)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

def gerar_embedding(modelo, imagem):
    imagem = tf.expand_dims(imagem, axis=0)
    embedding = modelo.predict(imagem)[0]
    return embedding

def carregar_banco_de_dados(caminho_csv):
    banco_de_dados = pd.read_csv(caminho_csv)
    banco_de_dados['embedding'] = banco_de_dados['embedding'].apply(lambda x: np.array(json.loads(x)))
    return banco_de_dados

def encontrar_imagem_mais_semelhante_no_banco(embedding_referencia, banco_de_dados):
    banco_embeddings = np.vstack(banco_de_dados['embedding'].values)
    
    distancias = [euclidean(embedding_referencia, emb) for emb in banco_embeddings]
    
    indice_mais_similar = np.argmin(distancias)
    similaridade_maxima = (1 / (1 + distancias[indice_mais_similar])) * 100  
    
    caminho_mais_similar = banco_de_dados.loc[indice_mais_similar, 'id']
    label_mais_similar = banco_de_dados.loc[indice_mais_similar, 'label']
    
    return caminho_mais_similar, label_mais_similar, similaridade_maxima


def plotar_imagens_com_similaridade(imagem_referencia_path, imagem_encontrada_path, similaridade):
    imagem_referencia = cv2.imread(imagem_referencia_path)
    imagem_referencia = cv2.cvtColor(imagem_referencia, cv2.COLOR_BGR2RGB)

    imagem_encontrada = cv2.imread(imagem_encontrada_path)
    imagem_encontrada = cv2.cvtColor(imagem_encontrada, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(imagem_referencia)
    plt.title('Imagem de Referência')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(imagem_encontrada)
    plt.title(f'Imagem Mais Semelhante\nSimilaridade: {similaridade:.2f}%')
    plt.axis('off')
    
    plt.suptitle(f'Similaridade: {similaridade:.2f}%')
    plt.show()


def gerar_embeddings_dataset(base_dir, rede, modelo):
    image_paths = [] 
    labels = []
    embeddings = []

    # Criar dataset
    all_image_paths = []
    all_labels = []
    for person_folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, person_folder)
        if os.path.isdir(folder_path):
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                all_image_paths.append(image_path)
                all_labels.append(person_folder)
    
    def load_and_preprocess_image(path, label):
        path = path.numpy().decode('utf-8')
        img = carregar_imagem(path, rede=rede)
        return img, label

    image_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_labels))
    image_ds = image_ds.map(lambda path, label: tf.py_function(load_and_preprocess_image, [path, label], [tf.float32, tf.string]), num_parallel_calls=tf.data.AUTOTUNE)
    image_ds = image_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    # Gerar embeddings em batch
    for batch_images, batch_labels in image_ds:
        batch_embeddings = modelo.predict(batch_images)
        embeddings.extend([json.dumps(embedding.tolist()) for embedding in batch_embeddings])
        labels.extend(batch_labels.numpy().tolist())
        image_paths.extend(batch_labels.numpy().tolist())
    
    # Criar DataFrame
    banco_de_dados = pd.DataFrame({
        'id': all_image_paths,
        'label': labels,
        'embedding': embeddings
    })

    return banco_de_dados

def comparar_e_plotar_imagens(caminho_imagem1, caminho_imagem2, modelo, rede, metrica="euclidean"):
    img1 = carregar_imagem(caminho_imagem1, rede=rede)
    img2 = carregar_imagem(caminho_imagem2, rede=rede)
    
    embedding1 = gerar_embedding(modelo, img1)
    embedding2 = gerar_embedding(modelo, img2)
    
    similaridade = calcular_similaridade(embedding1, embedding2, metrica=metrica)
    
    plotar_imagens_com_similaridade(caminho_imagem1, caminho_imagem2, similaridade)

