import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from torch_nn_model import RNN_Classifier,LSTM_Classifier,GRU_Classifier
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

# 超参数组合配置（可自由扩展）
param_grid = {
    'lr': [1e-3],  # 不同学习率 , 1e-3, 1e-4
    'batch_size': [64],  # 不同批次大小
    'epochs': [200],  # 不同训练轮数 #30,
}

# 设备配置
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 加载数据（固定数据加载部分）
olivetti_faces = fetch_olivetti_faces(
    data_home="/Users/circleLee/Develop/workspace_py/nlp/week04/data/olivetti_lfw_home/")
x_train, x_test, y_train, y_test = train_test_split(
    olivetti_faces.images, olivetti_faces.target, test_size=0.2
)
print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# 声明训练过程的tensorboard
writer = SummaryWriter()


# 定义训练函数（接受超参数）
def train_model(lr, batch_size, epochs):
    # 创建 DataLoader
    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(x_test).float(),
        torch.from_numpy(y_test).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型和优化器
    model = GRU_Classifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # 记录训练过程
    loss_history = []
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            img = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(img)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if epoch % 10 == 0:
                writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)
        # 验证阶段
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (img, clzz) in enumerate(test_loader):
                img = img.to(DEVICE)
                clzz = clzz.to(DEVICE)
                outputs = model(img)
                _, predicted = outputs.max(1)
                total += clzz.size(0)
                correct += predicted.eq(clzz).sum().item()
        if epoch % 10 == 0:
            acc = 100 * correct / total

            # 记录损失（平均每个样本的损失）
            loss_history.append(epoch_loss / len(train_loader))
            print(f"LR={lr}, BS={batch_size}, Epoch={epoch + 1}/{epochs}: "
                  f"Loss={loss_history[-1]:.4f}, Acc={acc:.2f}%")
            writer.add_scalar('test accuracy', acc, epoch)
    return loss_history


# 执行多组实验
results = []
for lr in param_grid['lr']:
    for batch_size in param_grid['batch_size']:
        for epochs in param_grid['epochs']:
            print(f"\n=== Training with LR={lr}, BS={batch_size}, Epochs={epochs} ===")
            history = train_model(lr, batch_size, epochs)
            results.append({
                'lr': lr,
                'batch_size': batch_size,
                'epochs': epochs,
                'loss_history': history
            })
