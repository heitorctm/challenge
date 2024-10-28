import os
import numpy as np
import pandas as pd
import tensorflow as tf
from .tripletas import gerar_tripleta_aleatoria, gerar_embeddings, calcular_similaridade


def avaliar_modelo_tripleta(
        dataset,
        rede,
        id_rodada,
        batch_size,
        imagenet,
        data_aug,
        nome_otimizador,
        nome_loss,
        lr,
        attention,
        camada_pooling,
        regularizacao_tipo,
        regularizacao_valor,
        denses,
        dropouts,
        embedding_size,
        path_modelos_salvos,
        path_log_teste,
        obs=""):
    """
    Avalia modelos salvos utilizando tripletas do dataset e salva os resultados em um arquivo CSV.
    """
    resultados_lista = []

    for nome_modelo in os.listdir(path_modelos_salvos):
        if nome_modelo.startswith(f'{rede}-{id_rodada}'):
            caminho_completo = os.path.join(path_modelos_salvos, f'{nome_modelo}')
            modelo_teste = tf.keras.models.load_model(caminho_completo, compile=False)
            print(nome_modelo)
            media_similaridade_pos, media_similaridade_neg = avaliar_tripletas(
                dataset=dataset,
                modelo=modelo_teste
            )

            densas_preenchidas = denses + [None] * (5 - len(denses))
            dropouts_preenchidos = dropouts + [None] * (5 - len(dropouts))

            chaves = [
                'arquivo', 'rede', 'imagenet', 'dataaug', 'otimizador', 'lossname', 'lr', 
                'attention', 'batchsize', 'pooling', 'regularizacao_tipo', 
                'regularizacao_valor', 'embedding_size'
            ] + \
            [f'dense_{i+1}' for i in range(5)] + [f'dropout_{i+1}' for i in range(5)] + \
            ['media_similaridade_pos', 'media_similaridade_neg', 'obs']
            
            valores = [
                nome_modelo, rede, imagenet, data_aug, nome_otimizador, nome_loss, lr, 
                attention, batch_size, camada_pooling, regularizacao_tipo, 
                regularizacao_valor, embedding_size
            ] + densas_preenchidas + dropouts_preenchidos + \
            [media_similaridade_pos, media_similaridade_neg, obs]

            resultado_dict = dict(zip(chaves, valores))
            resultados_lista.append(resultado_dict)
    
    salvar_resultados_tripleta(resultados_lista, path_log_teste)

def salvar_resultados_tripleta(resultados_lista, path_log_teste):
    """
    Salva os resultados da avaliação das tripletas em um arquivo CSV.
    """
    log_teste_df = pd.DataFrame(resultados_lista)
    cabecalho = not os.path.isfile(f'{path_log_teste}/log_tripleta_teste.csv')
    log_teste_df.to_csv(f'{path_log_teste}/log_tripleta_teste.csv', mode='a', header=cabecalho, index=False)

def avaliar_tripletas(dataset, modelo):
    """
    Avalia as tripletas geradas do dataset usando o modelo fornecido e calcula similaridade.
    """
    similaridades_pos = []
    similaridades_neg = []

    for batch_index, batch in enumerate(dataset):
        try:
            ancora_img, positivo_img, negativo_img = batch[0]
            print(f"Batch {batch_index} - Formato das imagens: ancora {ancora_img.shape}, positivo {positivo_img.shape}, negativo {negativo_img.shape}")  # Debug print
            ap_distance, an_distance = gerar_embeddings(modelo, ancora_img, positivo_img, negativo_img)
            positive_similarity, negative_similarity = calcular_similaridade(ap_distance, an_distance)
            
            if isinstance(positive_similarity, (list, np.ndarray)) and isinstance(negative_similarity, (list, np.ndarray)):
                positive_similarity = np.mean(positive_similarity)
                negative_similarity = np.mean(negative_similarity)
            
            similaridades_pos.append(positive_similarity)
            similaridades_neg.append(negative_similarity)
        except Exception as e:
            print(f"Erro ao processar o batch {batch_index}: {e}")  

    media_similaridade_pos = np.mean(similaridades_pos) if similaridades_pos else 0
    media_similaridade_neg = np.mean(similaridades_neg) if similaridades_neg else 0
    
    return media_similaridade_pos, media_similaridade_neg
