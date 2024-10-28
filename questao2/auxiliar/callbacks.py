from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

def callbacks(path_modelos_salvos, path_csv_log, rede, id_rodada, check_best, early_stop, log, reduce_lr, reduce_lr_epocas, fator_reduce_lr, early_stop_epocas):
    '''
    Callbacks para o treinamento do modelo, incluindo salvamento dos melhores modelos,
    early stopping, logging de CSV e ajuste da taxa de aprendizado.


            :param path_modelos_salvos: diretorio onde os modelos salvos serao armazenados
            :param path_csv_log: diretorio onde o log em CSV sera salvo
            :param rede: nome da rede utilizada no modelo
            :param id_rodada: identificador da rodada atual de treinamento
            :param check_best: flag para salvar o melhor modelo com base na melhor acuracia de validacao
            :param early_stop: flag para usar o EarlyStopping (para interromper o treinamento se nao houver melhora)
            :param log: flag para salvar logs de treinamento em um arquivo CSV
            :param reduce_lr: flag para reduzir a taxa de aprendizado quando a perda de validacao nao melhorar
            :param reduce_lr_epocas: numero de epocas sem melhora para acionar a reducao da taxa de aprendizado
            :param fator_reduce_lr: fator pelo qual a taxa de aprendizado sera reduzida
            :param early_stop_epocas: numero de epocas sem melhora para acionar o EarlyStopping
            :return lista de callbacks configurados de acordo com as flags fornecidas
    '''
    callbacks = []
    print("callbacks adicionados:") 

    if check_best:
        print('check best loss')
        checkpoint_best_loss = ModelCheckpoint(
            monitor='val_loss',
            filepath=f"{path_modelos_salvos}/{rede}-{id_rodada}_epoca-{{epoch:02d}}.h5",
            save_weights_only=False,
            save_best_only=True,
            verbose=1,
        )
        callbacks.append(checkpoint_best_loss)

    if early_stop:
        print('early stop')
        early_stopping = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=early_stop_epocas,
            verbose=1,
            min_delta=0.03
        )
        callbacks.append(early_stopping)

    if log:
        print('log')
        csv_log = CSVLogger(
            path_csv_log,
            append=True
        )
        callbacks.append(csv_log)

    if reduce_lr:
        print('reduce lr')
        reduce_learning_rate = ReduceLROnPlateau(
            monitor='val_loss',
            factor=fator_reduce_lr,
            min_delta=0.2,
            patience=reduce_lr_epocas,
            min_lr=0.00000001,
            verbose=1
        )
        callbacks.append(reduce_learning_rate)

    return callbacks
