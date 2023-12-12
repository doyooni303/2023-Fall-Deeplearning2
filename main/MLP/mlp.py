import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 데이터 로딩 및 전처리
def load_data(train_path, valid_path, test_path):
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values.reshape(-1, 1)
    X_valid = valid_df.iloc[:, :-1].values
    y_valid = valid_df.iloc[:, -1].values.reshape(-1, 1)
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values.reshape(-1, 1)
    
    # 데이터를 PyTorch 텐서로 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    return X_train_tensor, y_train_tensor, X_valid_tensor, y_valid_tensor, X_test_tensor, y_test_tensor

# MLP 모델 클래스
class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        
        x = self.fc4(x)
        return x

# 모델 훈련 함수
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 스케줄러 스텝
        scheduler.step()
        
        # Validate the model
        model.eval()
        with torch.no_grad():
            valid_loss = sum(criterion(model(inputs), targets) for inputs, targets in valid_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}, Valid Loss: {valid_loss/len(valid_loader)}')
    
    print("Training complete")

# 데이터 파일 경로
train_path = './train_False_df.csv'
valid_path = './valid_False_df.csv' 
test_path = './test_df.csv'

# 데이터 로드
X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(train_path, valid_path, test_path)

# DataLoader 설정
batch_size = 128
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

# 모델 초기화
input_size = X_train.shape[1]
model = MLPRegressor(input_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# 모델 훈련
train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=300)