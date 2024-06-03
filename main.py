import os

from sklearn.model_selection import train_test_split
from util.plots import Plots
from util.preprocessing import Preprocessing



# INSTRuçõES
# PARA O BRACOL, colocar o dataset Dataset_Bracol_A.rar na pasta data/ (esse dataset tem todas as folhas, 
# dentro dele tem a pasta sympton)



def main():


    # bracol --> Dataset_Bracol_A.rar

    preprocessing = Preprocessing(dataset='bracol') # pre processar o dataset selecionando o

    X, y = preprocessing.create_mydataset()
  
    # print(f"Shape of X: {X.shape}")
    # print(f"Shape of y: {y.shape}")

    plots = Plots()
    
    plots.save_bracol_images()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20, shuffle=True) # random_state=20,
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=20, shuffle=True) # random_state=20,
    del X
    del y

    plots.save_train_images(X_train,X_test,X_val)




if __name__ == "__main__":
    main()