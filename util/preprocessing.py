import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
import patoolib

class Preprocessing:
    def __init__(self, delete_images=False, path='data', dataset='bracol'):
        self.dataset = dataset

        if delete_images:
            self.apagar_arquivos_na_pasta('./data/dataimages/')
            self.apagar_arquivos_na_pasta('./dataset')
            self.apagar_arquivos_na_pasta('./output/')
            print('-- old images deleted')

        if dataset == 'bracol':
            if not os.path.exists("./data/dataimages/"):
                os.makedirs("./data/dataimages/")
            
            if not os.path.exists("./data/dataimages/symptom"):
                patoolib.extract_archive(path + './Dataset_Bracol_A.rar', outdir='./data/dataimages/')
                print(' --Dataset extracted')
            else:
                print('-- Dataset already extracted')

            path = './data/dataimages/symptom'

            self.cerscospora = path + '/cercospora'
            self.healthy = path + '/health'
            self.leaf_rust = path + '/rust'
            self.miner = path + '/miner'
            self.phoma = path + '/phoma'

        if dataset == 'jmuben':
            if not os.path.exists("./data/dataimages/"):
                os.makedirs("./data/dataimages/")
            
            if not os.path.exists("./data/dataimages/dataset_original_aumentado"):
                patoolib.extract_archive(path + './dataset_jmuben.rar', outdir='./data/dataimages/')
                print('-- Dataset extracted')
            else:
                print('-- Dataset already extracted')

            path = './data/dataimages/dataset_original_aumentado'

            self.cerscospora = path + '/Cerscospora'
            self.healthy = path + '/Healthy'
            self.leaf_rust = path + '/Leaf rust'
            self.miner = path + '/Miner'
            self.phoma = path + '/Phoma'

        self.IMG_SIZE = 224 # Specify height and width of image to match the input format of the model
        self.CHANNELS = 3    

        self.folders = [self.healthy, self.leaf_rust, self.miner, self.cerscospora, self.phoma]

    def getdataset(self):
        return self.dataset

    def create_mydataset(self):
        if not os.path.exists("./dataset"):
            os.makedirs("./dataset")
        print('creating X and y data...')
        X_train, y_train = [], []
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for filename in os.listdir(self.healthy):
            image = cv2.imread(self.healthy + '/' + filename)
            image = transform(image)
            X_train.append(image)
            y_train.append(0)  # class index for healthy
        for filename in os.listdir(self.leaf_rust):
            image = cv2.imread(self.leaf_rust + '/' + filename)
            image = transform(image)
            X_train.append(image)
            y_train.append(1)  # class index for leaf rust
        for filename in os.listdir(self.miner):
            image = cv2.imread(self.miner + '/' + filename)
            image = transform(image)
            X_train.append(image)
            y_train.append(2)  # class index for miner
        for filename in os.listdir(self.cerscospora):
            image = cv2.imread(self.cerscospora + '/' + filename)
            image = transform(image)
            X_train.append(image)
            y_train.append(3)  # class index for cercospora
        for filename in os.listdir(self.phoma):
            image = cv2.imread(self.phoma + '/' + filename)
            image = transform(image)
            X_train.append(image)
            y_train.append(4)  # class index for phoma

        X = torch.stack(X_train)
        y = torch.tensor(y_train)
        print('returning X and y data')
        return X, y

    def apagar_arquivos_na_pasta(self, caminho_pasta):
        for arquivo in os.listdir(caminho_pasta):
            caminho_arquivo = os.path.join(caminho_pasta, arquivo)
            if os.path.isfile(caminho_arquivo) or os.path.islink(caminho_arquivo):
                os.remove(caminho_arquivo)
            elif os.path.isdir(caminho_arquivo):
                shutil.rmtree(caminho_arquivo)
