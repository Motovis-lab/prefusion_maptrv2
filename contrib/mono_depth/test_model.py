import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. 创建随机数据集
class RandomDataset(Dataset):
    def __init__(self, num_samples, input_size, num_classes):
        self.X = torch.randn(num_samples, input_size)  # 随机特征
        # 随机生成标签 (0 到 num_classes-1)
        self.y = torch.randint(0, num_classes, (num_samples,))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. 定义模型
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, criterion):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.criterion = criterion
        self.dummy_loss = torch.tensor([0.],dtype=torch.float32, requires_grad=True).cuda()
    
    def forward(self, x, labels, idx):
        if idx % 3==2:
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            
            return self.criterion(x, labels)
        else:
            return self.dummy_loss

# 3. 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for idx, (inputs, labels) in enumerate(train_loader):
            # 前向传播
            loss = model(inputs,labels, idx)      
            print(loss)      
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 计算平均训练损失
        train_loss = train_loss / len(train_loader)
        
        
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        

# 4. 主函数
def main():
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置超参数
    input_size = 10
    hidden_size = 20
    output_size = 3  # 类别数
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 32
    
    # 创建随机数据集
    train_dataset = RandomDataset(num_samples=1000, input_size=input_size, num_classes=output_size)
    val_dataset = RandomDataset(num_samples=200, input_size=input_size, num_classes=output_size)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 创建模型
    model = SimpleModel(input_size, hidden_size, output_size, criterion)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

if __name__ == '__main__':
    main()
