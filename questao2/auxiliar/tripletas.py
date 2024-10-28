import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from scipy.spatial.distance import cosine, cityblock

def gerar_tripleta_aleatoria(dataset):
    grupos_por_label = dataset.groupby('label')
    labels = dataset['label'].unique()

    while True:
        label_ancora = random.choice(labels)
        imagens_positivas = grupos_por_label.get_group(label_ancora)['id'].tolist()
        
        if len(imagens_positivas) < 2:
            continue

        ancora = random.choice(imagens_positivas)
        positivo = random.choice([img for img in imagens_positivas if img != ancora])

        label_negativo = random.choice([l for l in labels if l != label_ancora])
        imagens_negativas = grupos_por_label.get_group(label_negativo)['id'].tolist()
        
        if len(imagens_negativas) == 0:
            continue 

        negativo = random.choice(imagens_negativas)
        
        return ancora, positivo, negativo

def gerar_embeddings(modelo, ancora_img, positivo_img, negativo_img):
    distances = modelo.predict({"anchor": ancora_img, "positive": positivo_img, "negative": negativo_img})
    ap_distance = distances[:, 0]
    an_distance = distances[:, 1]
    return ap_distance, an_distance

def calcular_similaridade(ap_distance, an_distance, metrica="euclidean"):

    if metrica == "euclidean":
        positive_similarity = np.round((1 / (1 + ap_distance)) * 100, 2)
        negative_similarity = np.round((1 / (1 + an_distance)) * 100, 2)
    elif metrica == "manhattan":
        positive_similarity = np.round((1 / (1 + cityblock(ap_distance, np.zeros_like(ap_distance)))) * 100, 2)
        negative_similarity = np.round((1 / (1 + cityblock(an_distance, np.zeros_like(an_distance)))) * 100, 2)
    else:
        raise ValueError(f"Métrica '{metrica}' não é suportada.")
    
    return positive_similarity, negative_similarity

def mostrar_tripleta(tripleta_selecionada, positive_similarity, negative_similarity):
    ancora_img = cv2.imread(tripleta_selecionada[0])
    positivo_img = cv2.imread(tripleta_selecionada[1])
    negativo_img = cv2.imread(tripleta_selecionada[2])

    ancora_nome = os.path.basename(os.path.dirname(tripleta_selecionada[0]))
    positivo_nome = os.path.basename(os.path.dirname(tripleta_selecionada[1]))
    negativo_nome = os.path.basename(os.path.dirname(tripleta_selecionada[2]))

    imagens_concatenadas = np.hstack((ancora_img, positivo_img, negativo_img))

    plt.figure(figsize=(6, 3))
    plt.imshow(cv2.cvtColor(imagens_concatenadas, cv2.COLOR_BGR2RGB))
    plt.title(f'ancora: {ancora_nome}\npositivo: {positivo_nome}, {positive_similarity}%\n negativo: {negativo_nome}, {negative_similarity}%')
    plt.axis('off')
    plt.show()

def teste_tripleta_aleatoria(tripleta, modelo,metrica):
    ancora_img = cv2.imread(tripleta[0])
    positivo_img = cv2.imread(tripleta[1])
    negativo_img = cv2.imread(tripleta[2])
    
    # Redimensiona as imagens e adiciona a dimensão de batch
    ancora_img = cv2.resize(ancora_img, (112, 112))
    positivo_img = cv2.resize(positivo_img, (112, 112))
    negativo_img = cv2.resize(negativo_img, (112, 112))

    # Adiciona a dimensão de batch para que o modelo possa processar as imagens
    ancora_img = np.expand_dims(ancora_img, axis=0)
    positivo_img = np.expand_dims(positivo_img, axis=0)
    negativo_img = np.expand_dims(negativo_img, axis=0)
    
    ap_distance, an_distance = gerar_embeddings(modelo, ancora_img, positivo_img, negativo_img)
    positive_similarity, negative_similarity = calcular_similaridade(ap_distance, an_distance,metrica)
    
    mostrar_tripleta(tripleta, positive_similarity, negative_similarity)

