 
import os
import cv2
import numpy as np
import tensorflow as tf
import patoolib



class Preprocessing:
    def __init__(self, path='data', dataset='bracol'):

        if(dataset == 'bracol'):
            if not os.path.exists("./data/dataimages/"):
                os.makedirs("./data/dataimages/")
            
            if not os.path.exists("./data/dataimages/symptom"):
                patoolib.extract_archive(path + './Dataset_Bracol_A.rar', outdir='./data/dataimages/')
                print(' --Dataset extracted')
            else:
                print('-- Dataset already extracted')

            path = './data/dataimages/symptom'




        self.IMG_SIZE = 224 # Specify height and width of image to match the input format of the model
        self.CHANNELS = 3    
        

        self.cerscospora = path + '/cercospora'
        self.healthy = path + '/health'
        self.leaf_rust = path + '/rust'
        self.miner = path + '/miner'
        self.phoma = path + '/phoma'
        self.folders = [self.healthy, self.leaf_rust, self.miner, self.cerscospora, self.phoma]
        
    

        
    def create_mydataset(self):
        if not os.path.exists("./dataset"):
            os.makedirs("./dataset")
        
        X_train, y_train = [], []
        for filename in os.listdir(self.healthy):
            out = './dataset/{}.png'.format(filename.split('.')[0])
            image = cv2.imread(self.healthy+'/'+filename)
            cv2.imwrite(out, image)
            img = tf.keras.utils.load_img(out, target_size = (224, 224, 1),color_mode="rgb")
            img = tf.keras.utils.img_to_array(img)
            img= img/255.
            X_train.append(img)
            y_train.append([1, 0, 0, 0, 0])
        for filename in os.listdir(self.leaf_rust):
            out = './dataset/{}.png'.format(filename.split('.')[0])
            image = cv2.imread(self.leaf_rust+'/'+filename)
            cv2.imwrite(out, image)
            img = tf.keras.utils.load_img(out, target_size = (224, 224, 1),color_mode="rgb")
            img = tf.keras.utils.img_to_array(img)
            img= img/255.
            X_train.append(img)
            y_train.append([0, 1, 0, 0, 0])
        for filename in os.listdir(self.miner):
            out = './dataset/{}.png'.format(filename.split('.')[0])
            image = cv2.imread(self.miner+'/'+filename)
            cv2.imwrite(out, image)
            img = tf.keras.utils.load_img(out, target_size = (224, 224, 1),color_mode="rgb")
            img = tf.keras.utils.img_to_array(img)
            img= img/255.
            X_train.append(img)
            y_train.append([0, 0, 1, 0, 0])
        for filename in os.listdir(self.cerscospora):
            out = './dataset/{}.png'.format(filename.split('.')[0])
            image = cv2.imread(self.cerscospora+'/'+filename)
            cv2.imwrite(out, image)
            img = tf.keras.utils.load_img(out, target_size = (224, 224, 1),color_mode="rgb")
            img = tf.keras.utils.img_to_array(img)
            img= img/255.
            X_train.append(img)
            y_train.append([0, 0, 0, 1, 0])
        for filename in os.listdir(self.phoma):
            out = './dataset/{}.png'.format(filename.split('.')[0])
            image = cv2.imread(self.phoma+'/'+filename)
            cv2.imwrite(out, image)
            img = tf.keras.utils.load_img(out, target_size = (224, 224, 1),color_mode="rgb")
            img = tf.keras.utils.img_to_array(img)
            img= img/255.
            X_train.append(img)
            y_train.append([0, 0, 0, 0, 1])

        X = np.array(X_train)
        y = np.array(y_train)
        return X, y


