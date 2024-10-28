import tensorflow as tf
from .redes import redes
from .data_aug import data_augmentation

def criar_dataset(imagens_paths, labels, batch_size, rede, modelo_embedding, shuffle=False, repeat=True, data_aug=False):
    _, preprocess_input, img_tamanho = redes[rede]
    
    dataset = tf.data.Dataset.from_tensor_slices((imagens_paths, labels))

    # Função para processar a imagem
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

    dataset = dataset.map(lambda imagens, labels: gerar_tripletas_dinamicamente(imagens, labels, modelo_embedding), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def processar_imagem(nome_do_arquivo, preprocess_input, img_tamanho):
    nome_da_imagem = tf.io.read_file(nome_do_arquivo)
    imagem_decodificada = tf.image.decode_jpeg(nome_da_imagem, channels=3)
    imagem_redimensionada = tf.image.resize(imagem_decodificada, img_tamanho)
    imagem_normalizada = preprocess_input(imagem_redimensionada)
    return imagem_normalizada

def gerar_tripletas_dinamicamente(imagens, labels, modelo_embedding):
    ancora_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    positivo_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    negativo_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    embeddings = modelo_embedding(imagens, training=False)

    def cond(i, ancora_list, positivo_list, negativo_list):
        return i < tf.shape(labels)[0]

    def body(i, ancora_list, positivo_list, negativo_list):
        ancora = embeddings[i]
        label_ancora = labels[i]

        positivos = embeddings[tf.equal(labels, label_ancora)]
        negativos = embeddings[tf.not_equal(labels, label_ancora)]

        positivo = tf.cond(tf.size(positivos) > 0, lambda: positivos[tf.random.uniform([], maxval=tf.shape(positivos)[0], dtype=tf.int32)], lambda: ancora)

        distancias_negativos = tf.reduce_sum(tf.square(negativos - ancora), axis=1)

        indice_negativo_dificil = tf.argmin(distancias_negativos)
        negativo_dificil = tf.cond(tf.size(negativos) > 0, lambda: negativos[indice_negativo_dificil], lambda: ancora)

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
