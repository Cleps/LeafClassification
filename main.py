import os
from util.preprocessing import Preprocessing



# INSTRuçõES
# PARA O BRACOL, colocar o dataset Dataset_Bracol_A.rar na pasta data/ (esse dataset tem todas as folhas, dentro dele tem a pasta sympton)



def main():


    # bracol --> Dataset_Bracol_A.rar

    preprocessing = Preprocessing(dataset='bracol') # pre processar o dataset selecionando o

    X, y = preprocessing.create_mydataset()

  
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")



if __name__ == "__main__":
    main()