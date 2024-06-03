import tensorflow as tf




class PavicNetMC:
    def __init__(self):
        pass
        
    def residual_block(self, inputs):
        residual = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1)(inputs)
        residual = tf.keras.layers.BatchNormalization()(residual)
        residual = tf.keras.layers.ReLU()(residual)
        residual = tf.keras.layers.MaxPool2D(2)(residual)

        residual = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1)(residual)
        residual = tf.keras.layers.BatchNormalization()(residual)
        residual = tf.keras.layers.ReLU()(residual)
        residual = tf.keras.layers.MaxPool2D(2)(residual)

        residual = tf.keras.layers.Flatten()(residual) # ------------------------------------------------------------ COMENTA AQUI

        residual = tf.keras.layers.Dense(128, activation='relu')(residual)

        return residual
            
    def def_pavic_model(self):
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(4)(x)

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(2)(x)

        residual = self.residual_block(x)
        #residual = residual_block(residual)

        #x = tf.keras.layers.Flatten()(residual)
        x = tf.keras.layers.Dense(64, activation='relu')(residual)

        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(32, activation='relu')(x)

        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(16, activation='relu')(x)

        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(8, activation='relu')(x)

        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(5, activation='softmax')(x)

        #x = tf.keras.layers.add([x, residual]) #------------- bloco residual nas camadas densas
        model = tf.keras.Model(inputs=inputs, outputs=x)
        PavicNet_MC = model
        return PavicNet_MC

    ### -------------------------------

    # Compilando o modelo PAVICNET-MC
    # model_all_0 = Model_3()