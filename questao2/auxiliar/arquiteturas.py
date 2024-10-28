import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout
from tensorflow.keras import regularizers
from .attention_blocks import cbam_block, squeeze_excite_block
from .redes import redes
from .data_aug import data_augmentation

def criar_modelo_embedding(base_model, attention, pooling, denses, dropouts, regularizacao_tipo=None, regularizacao_valor=0.001, embedding_size=128):
    """
    Cria o modelo de embedding, que será usado no modelo siamês.
    """
    x = base_model.output
    if attention == 'se':
        x = squeeze_excite_block(x)
    elif attention == 'cbam':
        x = cbam_block(x)

    if pooling == 'global_max':
        x = GlobalMaxPooling2D(name='global_max_pool')(x)
    elif pooling == 'global_avg':
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)

    if regularizacao_tipo == 'l1':
        regularizacao = regularizers.l1(regularizacao_valor)
    elif regularizacao_tipo == 'l2':
        regularizacao = regularizers.l2(regularizacao_valor)
    elif regularizacao_tipo == 'l1_l2':
        regularizacao = regularizers.l1_l2(l1=regularizacao_valor, l2=regularizacao_valor)
    else:
        regularizacao = None

    for i, (camadas, dropout_rate) in enumerate(zip(denses, dropouts)):
        if camadas > 0:
            x = Dense(camadas, activation='relu', kernel_regularizer=regularizacao)(x)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)

    embedding = Dense(embedding_size, activation='linear')(x)
    return Model(inputs=base_model.input, outputs=embedding, name="Embedding")

def criar_modelo_siamese(rede, weights, otimizador, embedding_model, margin=2):
    """
    Cria o modelo siamês completo, utilizando o modelo de embedding.
    """
    if rede in redes:
        _, _, img_tamanho = redes[rede]
    else:
        raise ValueError(f'Não existe a rede especificada: {rede}')

    anchor_input = Input(name="anchor", shape=(img_tamanho[0], img_tamanho[1], 3))
    positive_input = Input(name="positive", shape=(img_tamanho[0], img_tamanho[1], 3))
    negative_input = Input(name="negative", shape=(img_tamanho[0], img_tamanho[1], 3))

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    distances = euclidean_distance([anchor_embedding, positive_embedding, negative_embedding])

    siamese_net = Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
    siamese_net.compile(optimizer=otimizador, loss=triplet_loss(margin))
    
    return siamese_net

def criar_dataset(imagens_paths, labels, batch_size, rede, shuffle=False, repeat=True, data_aug=False):

    _, preprocess_input, img_tamanho = redes[rede]
    dataset = tf.data.Dataset.from_tensor_slices((imagens_paths, labels))

    def map_func(imagem_path, label):
        imagem = processar_imagem(imagem_path, preprocess_input, img_tamanho)
        return imagem, label

    dataset = dataset.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()

    if data_aug:
        dataset = dataset.map(lambda imagem, label: (data_augmentation(imagem), label), num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.map(lambda imagens, labels: gerar_tripletas_dinamicamente(imagens, labels), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def configurar_pipeline(imagens_paths_treino, labels_treino, imagens_paths_val, labels_val,
                        imagens_paths_teste, labels_teste, rede, weights, otimizador, attention,
                        denses, dropouts, pooling, regularizacao_tipo=None,
                        regularizacao_valor=0.001, margin=2, embedding_size=128,
                        batch_size=32, shuffle=True, data_aug=False):

    if rede in redes:
        RedeClasse, _, _ = redes[rede]
        base_model = RedeClasse(weights=weights, include_top=False)
        base_model.trainable = False
    else:
        raise ValueError(f'Não existe a rede especificada: {rede}')

    embedding_model = criar_modelo_embedding(
        base_model, attention, pooling, denses, dropouts, 
        regularizacao_tipo, regularizacao_valor, embedding_size
    )

    # Criar o modelo siamês completo
    modelo_siamese = criar_modelo_siamese(rede, weights, otimizador, embedding_model, margin)

    # Criar datasets de treino, validação e teste
    dataset_treino = criar_dataset(imagens_paths_treino, labels_treino, batch_size, rede, shuffle, repeat=True, data_aug=data_aug)
    dataset_validacao = criar_dataset(imagens_paths_val, labels_val, batch_size, rede, repeat=True)
    dataset_teste = criar_dataset(imagens_paths_teste, labels_teste, batch_size, rede, repeat=False)

    return modelo_siamese, dataset_treino, dataset_validacao, dataset_teste

def euclidean_distance(vectors):
    anchor, positive, negative = vectors
    ap_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1, keepdims=True) 
    an_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1, keepdims=True)  
    return tf.concat([ap_distance, an_distance], axis=1) 

def triplet_loss(margin):
    def loss(_, y_pred):
        ap_distance, an_distance = y_pred[:, 0], y_pred[:, 1]
        return tf.maximum(ap_distance - an_distance + margin, 0.0)
    return loss

def processar_imagem(nome_do_arquivo, preprocess_input, img_tamanho):
    nome_da_imagem = tf.io.read_file(nome_do_arquivo)
    imagem_decodificada = tf.image.decode_jpeg(nome_da_imagem, channels=3)
    imagem_redimensionada = tf.image.resize(imagem_decodificada, img_tamanho)
    imagem_normalizada = preprocess_input(imagem_redimensionada)
    return imagem_normalizada


def gerar_tripletas_dinamicamente(imagens, labels):
    ancora_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    positivo_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    negativo_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    def cond(i, ancora_list, positivo_list, negativo_list):
        return i < tf.shape(labels)[0] // 3

    def body(i, ancora_list, positivo_list, negativo_list):
        ancora = imagens[i]
        label_ancora = labels[i]

        positivos = tf.boolean_mask(imagens, tf.equal(labels, label_ancora))
        negativos = tf.boolean_mask(imagens, tf.not_equal(labels, label_ancora))

        positivo = tf.cond(tf.shape(positivos)[0] > 0, 
                           lambda: positivos[tf.random.uniform([], maxval=tf.shape(positivos)[0], dtype=tf.int32)], 
                           lambda: ancora)

        # Selecionar o negativo mais difícil (mais próximo da ancora)
        distancias_negativos = tf.reduce_sum(tf.square(negativos - ancora), axis=[1, 2, 3])
        indice_negativo_dificil = tf.argmin(distancias_negativos)
        negativo_dificil = tf.cond(tf.shape(negativos)[0] > 0, 
                                   lambda: negativos[indice_negativo_dificil], 
                                   lambda: ancora)

        ancora_list = ancora_list.write(i, ancora)
        positivo_list = positivo_list.write(i, positivo)
        negativo_list = negativo_list.write(i, negativo_dificil)

        return i + 1, ancora_list, positivo_list, negativo_list

    i = tf.constant(0)
    _, ancora_list, positivo_list, negativo_list = tf.while_loop(cond, body, [i, ancora_list, positivo_list, negativo_list])

    ancora_imgs = ancora_list.stack()
    positivo_imgs = positivo_list.stack()
    negativo_imgs = negativo_list.stack()

    return (ancora_imgs, positivo_imgs, negativo_imgs), tf.zeros(tf.shape(ancora_imgs)[0])
