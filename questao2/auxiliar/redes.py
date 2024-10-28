from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception_v3
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inception_resnet_v2
from keras.applications.efficientnet_v2 import preprocess_input as preprocess_efficientnet_v2s
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet50v2
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as preprocess_vgg19
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, ResNet152V2, ResNet50V2, EfficientNetV2L, EfficientNetV2S, EfficientNetV2M, EfficientNetV2B3

''' 
dicionario com as redes, a funcao de pre processamento e o tamanho da imagem que é recomendado na documentação do TF
esse tamanho de imagem pode ser modificado, é apenas a recomendacao da documentacao

https://keras.io/api/applications/
'''
padrao = (112,112)

redes = {
    'InceptionV3': (InceptionV3, preprocess_inception_v3, padrao),  # 299
    'InceptionResNetV2': (InceptionResNetV2, preprocess_inception_resnet_v2, padrao),  # 299
    'EfficientNetV2L': (EfficientNetV2L, preprocess_efficientnet_v2s, padrao),  # 480
    'EfficientNetV2S': (EfficientNetV2S, preprocess_efficientnet_v2s, padrao),  # 384
    'ResNet50V2': (ResNet50V2, preprocess_resnet50v2, padrao),  # 224
    'ResNet152V2': (ResNet152V2, preprocess_resnet50v2, padrao),  # 224
    'EfficientNetV2M': (EfficientNetV2M, preprocess_efficientnet_v2s, padrao),  # 480
    'EfficientNetV2B3': (EfficientNetV2B3, preprocess_efficientnet_v2s, padrao),  # 300
    'VGG19': (VGG19, preprocess_vgg19, padrao)  # 224
}