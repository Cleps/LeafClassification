import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from util.evaluation import Evaluation
from util.plots import Plots
from util.preprocessing import Preprocessing

def main():
    # Verificar se há GPU disponível
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        print(f"GPU disponível: {torch.cuda.get_device_name(0)}")
    else:
        print("Nenhuma GPU encontrada. Verifique a instalação do PyTorch e os drivers da GPU.")

    # PRE PROCESSING DATA
    preprocessing = Preprocessing(dataset='bracol')
    X, y = preprocessing.create_mydataset()
  
    plots = Plots()
    plots.save_bracol_images()

    # SPLITING DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=20, shuffle=True)
    
    plots.save_train_images(X_train, X_test, X_val)

    # Load ResNet50
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)  # 5 classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # TRAINING MODEL
    EPOCHS = 300
    batch_size = 8

    print('------------  TRAINING MODEL -------------------')
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i+batch_size].to(device)
            labels = y_train[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/total}, Accuracy: {100 * correct/total}")

    print('-------------  MODEL TRAINED ------------------')

    # SAVE MODEL
    model_name = 'ResNet50_bracol.pth'
    torch.save(model.state_dict(), model_name)
    print("Model saved to disk")

    # Evaluation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val.to(device))
        _, val_preds = torch.max(val_outputs, 1)
    
    eval = Evaluation()
    print('\n\n\n---------------- EVALUATION -------------------------------\n\n\n')
    eval.evaluate(y_val.cpu().numpy(), val_preds.cpu().numpy())

if __name__ == "__main__":
    main()
