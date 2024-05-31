import os
from util.preprocessing import Preprocessing

def main():


    # bracol --> Dataset_Bracol_A.rar

    preprocessing = Preprocessing(dataset='bracol') # pre processar o dataset selecionando o

    X, y = preprocessing.create_mydataset()

  
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")



if __name__ == "__main__":
    main()