
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

class Plots:
    def __init__(self):
        pass

    def plot_bracol_images(self):

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
        plt.savefig(out)
        print('BRACOL images graph saved in: '+out)


    def plot_jmuben_images(self):
        if not os.path.exists("./output/imagesGraph"):
            os.makedirs("./output/imagesGraph")
        out = './output/imagesGraph/'

        data_folder = '/data/dataimages/dataset_original_aumentado'
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
        plt.show(out)
        print('JMUBEN images graph saved in: '+out)