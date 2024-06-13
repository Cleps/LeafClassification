import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from util.evaluation import Evaluation
from util.plots import Plots
from util.preprocessing import Preprocessing
from models.pavicnet_script import PavicNetMC
import keras

# INSTRuçõES
# PARA O BRACOL, colocar o dataset Dataset_Bracol_A.rar na pasta data/ (esse dataset tem todas as folhas, 
# dentro dele tem a pasta sympton)



def main(model='pavicnet'):


 # Verificar dispositivos disponíveis
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"GPUs disponíveis: {physical_devices}")
    else:
        print("Nenhuma GPU encontrada. Verifique a instalação do TensorFlow e os drivers da GPU.")

    # Configurar logs detalhados de dispositivos
    # tf.debugging.set_log_device_placement(True)

    # bracol --> Dataset_Bracol_A.rar

# PRE PROCESSING DATA
    preprocessing = Preprocessing(delete_images=True, dataset='jmuben') # pre processar o dataset selecionando o

    X, y = preprocessing.create_mydataset()
  
    # print(f"Shape of X: {X.shape}")
    # print(f"Shape of y: {y.shape}")

    plots = Plots()
    

    if(preprocessing.getdataset() == 'bracol'):
        plots.save_bracol_images()
    if(preprocessing.getdataset() == 'jmuben'):
        plots.save_jmuben_images()


# SPLITING DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20, shuffle=True) # random_state=20,
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=20, shuffle=True) # random_state=20,
    del X
    del y

    plots.save_train_images(X_train,X_test,X_val)


# DEFINING MODEL
    if (model == 'pavicnet'):
        pavicnet = PavicNetMC()
        model = pavicnet.def_pavic_model()
    # model.summary()



# TRAINING MODEL
    N_LABELS = 5
    EPOCHS = 300
    LR = 0.0001
    batch_size = 8 # 32
    # Compile the model

    # early stop
    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        mode="auto",
        restore_best_weights=True,
        start_from_epoch=0,
    )

    print('------------  TRAINING MODEL -------------------')
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    #loss='categorical_crossentropy', # categorical cross
    loss='binary_crossentropy', # categorical cross
    metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=EPOCHS,
                callbacks = [callback],
                validation_data=(X_test, y_test))

    print('-------------  MODEL TRAINED ------------------')

# SAVE MODEL
    # modelName = 'InceptionresV1JMUBEN'  # nome do modelo
    # model.save(f'./output/model_{modelName}.hdf5')
    # print("Saved model to disk")

# SAVE HISTORY CURVES
    plots.plot_curves(history, 'TrainingCurves')

    preds = model.predict(X_val)
    preds = np.array(preds> 0.2) #limiar

    eval = Evaluation()
    print('\n\n\n---------------- EVALUATION -------------------------------\n\n\n')
    eval.evaluate(y_val, preds)

if __name__ == "__main__":
    main(sys.argv[1:])