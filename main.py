from util.preprocessing import Preprocessing

def main():

    preprocessing = Preprocessing()

    X, y = preprocessing.create_mydataset()

  
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

  
if __name__ == "__main__":
    main()
