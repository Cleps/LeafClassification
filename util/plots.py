
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

class Plots:
    def __init__(self):
        pass


    def save_train_images(self, X_train, X_test, X_val):
        if not os.path.exists("./output/imagesGraph"):
            os.makedirs("./output/imagesGraph")
        out = './output/imagesGraph/'

        labels = [X_train.shape[0], X_test.shape[0], X_val.shape[0]]

        index = ['Train Set', 'Test Set', 'Validation Set']

        df = pd.DataFrame({'Quantidade': labels}, index=index)
        ax = df.plot.bar(rot=0, color={'Quantidade': "red"}, width = 0.2)
        for container in ax.containers:
            ax.bar_label(container)
        plt.ylabel("Dados", fontsize=15)
        out = './output/imagesGraph/'
        plt.savefig(out+'trainimages')
        print('Train images graph saved in: '+out)

    def save_bracol_images(self):

        if not os.path.exists("./output/imagesGraph"):
            os.makedirs("./output/imagesGraph")
        out = './output/imagesGraph/'

        data_folder = './data/dataimages/symptom'
        healthy2 = len(os.listdir(data_folder +'/health'))
        phome2 = len(os.listdir(data_folder +'/phoma'))
        miner2 = len(os.listdir(data_folder +'/miner'))
        rust2 =  len(os.listdir(data_folder +'/rust'))
        cercospora2 = len(os.listdir(data_folder +'/cercospora'))
        total = healthy2 + phome2 + miner2 + rust2 + cercospora2

        labels = [healthy2, phome2, miner2 ,rust2 ,cercospora2]

        index = ['Healthy', 'Phome', 'Miner', 'Rust', 'Cercospora']

        print('Total Images =', total)

        df = pd.DataFrame({'Quantidade': labels}, index=index)
        fig, ax = plt.subplots(figsize=(7, 6))  # Ajuste os valores de largura e altura conforme necessário
        ax = df.plot.bar(rot=0, color={'Quantidade': "green"}, ax=ax)

        # Define manualmente os limites do eixo x
        ax.set_xlim(-0.6, len(index) - 0.2)
        plt.xticks(range(len(index)), index)

        ax.set_ylim(0, max(labels) + 300)

        for container in ax.containers:
            ax.bar_label(container)

        plt.ylabel("Dados", fontsize=15)
        plt.savefig(out+'bracolimages')
        print('BRACOL images graph saved in: '+out)


    def save_jmuben_images(self):
        if not os.path.exists("./output/imagesGraph"):
            os.makedirs("./output/imagesGraph")
        out = './output/imagesGraph/'

        data_folder = './data/dataimages/dataset_original_aumentado'
        healthy2 = len(os.listdir(data_folder +'/Healthy'))
        phome2 = len(os.listdir(data_folder +'/Phoma'))
        miner2 = len(os.listdir(data_folder +'/Miner'))
        rust2 =  len(os.listdir(data_folder +'/Leaf rust'))
        cercospora2 = len(os.listdir(data_folder +'/Cerscospora'))
        total = healthy2 + phome2 + miner2 + rust2 + cercospora2

        labels = [healthy2, phome2, miner2 ,rust2 ,cercospora2]

        index = ['Healthy', 'Phome', 'Miner', 'Rust', 'Cercospora']

        print('Total Images =', total)

        df = pd.DataFrame({'Quantidade': labels}, index=index)

        fig, ax = plt.subplots(figsize=(7, 6) , dpi = 150)  # Ajuste os valores de largura e altura conforme necessário
        ax = df.plot.bar(rot=0, color={'Quantidade': "green"}, ax=ax)

        # Define manualmente os limites do eixo x
        ax.set_xlim(-0.6, len(index) - 0.2)
        plt.xticks(range(len(index)), index, fontsize=16)

        ax.set_ylim(0, max(labels) + 300)

        for container in ax.containers:
            ax.bar_label(container)
        plt.ylabel("Dados", fontsize=16)
        plt.savefig(out+'jmubenimages')
        print('JMUBEN images graph saved in: '+out)



    # Func de plotar resultados

    def plot_curves(self, history, title):
        fig_1 = plt.figure(figsize=(5, 3))
        epochs = range(1, len(history.history['loss'])+1)
        plt.plot(epochs, history.history['loss'], label="Train loss")
        plt.plot(epochs, history.history['val_loss'], label="Test Loss")
        plt.title("Training Loss Curve") #: +title)
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        # plt.legend();
        plt.savefig('./output/loss_'+title+'.png')
        #plt.close(fig_1)

        fig_2 = plt.figure(figsize=(5, 3))
        epochs = range(1, len(history.history['accuracy'])+1)
        plt.plot(epochs, history.history['accuracy'], label="Train Acc")
        plt.plot(epochs, history.history['val_accuracy'], label="Test Acc")
        plt.title("Training Accuracy Curve") #: +title)
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        # plt.legend();
        plt.savefig('./output/acc_'+title+'.png')
        #plt.close(fig_2)