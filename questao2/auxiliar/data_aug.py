from tensorflow import keras
from tensorflow.keras import layers

'''
aplica data augmentation pra aumentar a variabilidade dos dados de entrada durante o treinamento
os augmentations sao aplicados de forma aleatoria a cada imagem do dataset.
documentacao pra mais detalhes:

https://keras.io/api/layers/preprocessing_layers/

transformacoes aplicadas:
    - RandomRotation: rotaciona a imagem aleatoriamente dentro do fator especificado
    - RandomTranslation: translada a imagem aleatoriamente para cima/baixo e esquerda/direita com fator de altura/largura especificado
    - RandomZoom: aplica zoom aleatorio na imagem (para dentro e fora) com fator de altura/largura especificado
    - RandomFlip: inverte a imagem horizontalmente de forma aleatoria

outras:
    - RandomContrast: altera o contraste da imagem de forma aleatoria
    - RandomBrightness: ajusta o brilho da imagem aleatoriamente dentro de um intervalo especificado
    - RandomCrop: corta uma parte da imagem aleatoriamente
    - RandomWidth: ajusta a largura da imagem de forma aleatoria dentro de um fator especificado
    - RandomHeight: ajusta a altura da imagem de forma aleatoria dentro de um fator especificado
    - RandomShear: aplica uma deformacao de cisalhamento aleatoria na imagem
    - RandomRescale: reescala a imagem para alterar seu brilho ou contraste
    - GaussianNoise: adiciona ruido gaussiano para aumentar a robustez a ruidos nas imagens
'''

data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(factor=0.2, fill_mode="nearest"),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="nearest"),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2, fill_mode="nearest"),
        layers.RandomFlip(mode="horizontal"),
        layers.RandomContrast(factor=0.2),
        layers.RandomBrightness(factor=0.2),
        layers.GaussianNoise(stddev=0.05)
        #layers.RandomWidth(factor=0.1, fill_mode="nearest"),
        #layers.RandomHeight(factor=0.1, fill_mode="nearest"),
        #layers.Rescaling(scale=1.0/255)
    ]
)
