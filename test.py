


import tensorflow as tf
print(tf.__version__)
tf.config.list_physical_devices('GPU')
print("GPU Available: ", tf.test.is_gpu_available())
