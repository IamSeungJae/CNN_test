import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import random

# 랜덤 시드 고정
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 설정 파일 로드
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

config = load_config("config.json")

class HistogramDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.class_to_idx = {}

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
    def __init__(self, input_size=128, num_class=43):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1536, 256)
        self.fc2 = nn.Linear(256, num_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
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

def train(model, train_loader, optimizer, criterion, device):
    model.train()
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
    return total_correct / len(train_loader.dataset), total_loss / len(train_loader)

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
    return total_loss / len(val_loader), total_correct / len(val_loader.dataset)

def test(model, test_loader, device):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            total_correct += (outputs.argmax(1) == labels).sum().item()
    return total_correct / len(test_loader.dataset)


start = time.time()

# 데이터 로드
train_dataset = HistogramDataset(config["train_data_path"])
val_dataset = HistogramDataset(config["val_data_path"])
test_dataset = HistogramDataset(config["test_data_path"])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# 모델 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN_Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
criterion = nn.CrossEntropyLoss()

# TensorBoard 조소 설정 및 폴더 생성
log_base_dir = r"C:\Users\lee73\OneDrive\Desktop\lab\tensorboard"
i = 1
while os.path.exists(os.path.join(log_base_dir, f"test{i}")):
    i += 1
log_dir = os.path.join(log_base_dir, f"test{i}")
writer = SummaryWriter(log_dir=log_dir)

# 학습 루프 (train & validation)
for epoch in range(config["num_epochs"]):
    train_acc, train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    writer.add_scalar("Loss/Train", train_loss, epoch + 1)
    writer.add_scalar("Accuracy/Train", train_acc, epoch + 1)
    writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
    writer.add_scalar("Accuracy/Validation", val_acc, epoch + 1)

    print(f"Epoch [{epoch+1}/{config['num_epochs']}]")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")

# 테스트
test_acc = test(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.4f}")

writer.close()

# 결과 파일 저장
base_filename = os.path.join(log_dir, "test")
file_extension = ".txt"

i = 1
output_file = f"{base_filename}{file_extension}"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("========== Results ==========\n")
    f.write(f"Final Train Loss: {train_loss:.4f}\n")
    f.write(f"Final Train Accuracy: {train_acc:.4f}\n")
    f.write(f"Final Validation Loss: {val_loss:.4f}\n")
    f.write(f"Final Validation Accuracy: {val_acc:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")

    f.write("========== Parameter ==========\n")
    for key, value in config.items():
        f.write(f"{key}: {value}\n")

print(f"Results saved to {output_file}")


end = time.time()
print(f'tensorboard 실행 코드: tensorboard --logdir="{log_dir}"')
print(f"소요 시간: {end - start:.5f} sec")
