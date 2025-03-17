import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim


class HistogramDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.class_to_idx = {}

        # 클래스 폴더 가져오기
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for class_name in classes:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for variant in os.listdir(class_path):
                variant_path = os.path.join(class_path, variant)
                if not os.path.isdir(variant_path):
                    continue

                for file in os.listdir(variant_path):
                    if file.endswith(".txt"):
                        file_path = os.path.join(variant_path, file)
                        histogram = np.loadtxt(file_path, delimiter=",")
                        self.data.append(histogram)
                        self.labels.append(self.class_to_idx[class_name])

    
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

class CNN_Model(nn.Module):
    def __init__(self, input_size = 128, num_class = 43):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=32, kernel_size=3,stride=1)
        self.conv2 = nn.Conv1d(in_channels=32,out_channels=64, kernel_size=3,stride=1)
        self.conv3 = nn.Conv1d(in_channels=64,out_channels=128, kernel_size=3,stride=1)
        self.conv4 = nn.Conv1d(in_channels=128,out_channels=256, kernel_size=3,stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1536, 256)
        self.fc2 = nn.Linear(256, num_class) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = self.relu(self.conv4(x))
        x = self.pool(x)

        x = x.view(x.shape[0], -1)  
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

def train(model, train_loader, optimizer, criterion, device, num_epochs):
    model.train()  

    for epoch in range(num_epochs):
        total_loss, total_correct = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)  

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = total_correct / len(train_loader.dataset)
        avg_loss = total_loss / len(train_loader)  

        print(f"Epoch [{epoch+1}/{num_epochs}] | avg.Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = total_correct / len(val_loader.dataset)
    return total_loss / len(val_loader), val_acc


def test(model, test_loader, device):
    model.eval()
    total_correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)

            total_correct += (outputs.argmax(1) == labels).sum().item()

    test_acc = total_correct / len(test_loader.dataset)
    return test_acc


batch_size = 32

train_dataset = HistogramDataset(r"C:\Users\lee73\OneDrive\Desktop\Save_Histogram\dataset\train")
val_dataset = HistogramDataset(r"C:\Users\lee73\OneDrive\Desktop\Save_Histogram\dataset\validation")
test_dataset = HistogramDataset(r"C:\Users\lee73\OneDrive\Desktop\Save_Histogram\dataset\test")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 데이터 로드 확인
data_iter = iter(train_loader)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN_Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epochs= 20

train(model, train_loader, optimizer, criterion, device, num_epochs)






