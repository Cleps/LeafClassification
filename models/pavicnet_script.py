import torch
import torch.nn as nn

class PavicNetMC(nn.Module):
    def __init__(self):
        super(PavicNetMC, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(4)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2)
        
        self.residual_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.residual_bn1 = nn.BatchNorm2d(64)
        self.residual_maxpool1 = nn.MaxPool2d(2)
        
        self.residual_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.residual_bn2 = nn.BatchNorm2d(128)
        self.residual_maxpool2 = nn.MaxPool2d(2)
        
        # Assumindo que a entrada tem tamanho [batch_size, 128, 6, 6] após o maxpool
        self.fc1 = nn.Linear(128 * 13 * 13, 128)  # Ajuste conforme o tamanho da saída do maxpool
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 5)

        self.dropout = nn.Dropout(0.2)

    def residual_block(self, x):
        residual = self.residual_conv1(x)
        residual = self.residual_bn1(residual)
        residual = self.relu(residual)
        residual = self.residual_maxpool1(residual)
        
        residual = self.residual_conv2(residual)
        residual = self.residual_bn2(residual)
        residual = self.relu(residual)
        residual = self.residual_maxpool2(residual)
        
        residual = residual.reshape(residual.size(0), -1)  # Flatten
        
        residual = self.fc1(residual)
        
        return residual

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        residual = self.residual_block(x)

        x = self.fc2(residual)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc6(x)
        
        return x
