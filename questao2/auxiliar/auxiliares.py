import tensorflow as tf
import pandas as pd

def configurar_gpu():
    tf_gpu = tf.config.experimental.list_physical_devices('GPU')
    if tf_gpu:
        try:
            tf.config.experimental.set_memory_growth(tf_gpu[0], True)
            gpu_name = tf_gpu[0].name
            print(f"Dispositivo GPU: {gpu_name}")
            
            from tensorflow.python.client import device_lib
            local_device_protos = device_lib.list_local_devices()
            for device in local_device_protos:
                if device.device_type == 'GPU':
                    print(f"GPU: {device.physical_device_desc}")
        except RuntimeError as e:
            print(e)
    else:
        print('Nenhuma GPU encontrada')

def proximo_id(caminho_csv_ids):
    df = pd.read_csv(caminho_csv_ids)
    if df.empty:
        id_rodada = 1
    else:
        id_rodada = df['id_rodada'].max() + 1
    
    novo_id = pd.DataFrame({'id_rodada': [id_rodada]})
    novo_id.to_csv(caminho_csv_ids, mode='a', header=False, index=False)
    
    return id_rodada

