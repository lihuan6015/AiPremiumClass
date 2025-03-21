import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #self.flatten = nn.Flatten()# 添加展平层
        self.layer1 = nn.Linear(64*64,64*64*2)
        self.batch_norm = nn.BatchNorm1d(64*64*2)
        self.layer2 = nn.Linear(64*64*2,4096)
        self.layer3 = nn.Linear(4096,1024)
        self.layer4 = nn.Linear(1024,40)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self,x):
        #x = self.flatten(x)
        x1 =self.layer1(x)
        x1=self.batch_norm(x1)
        x1=torch.relu(x1)
        x1=self.dropout(x1)

        x2 = self.layer2(x1)
        x2 = torch.relu(x2)
        x2=self.dropout(x2)

        x3 = self.layer3(x2)
        x3 = torch.relu(x3)
        x3=self.dropout(x3)
    


        out =self.layer4(x3)
        return out
    
class MyModel2(nn.Module):
    def __init__(self):
        super(MyModel2, self).__init__()
        self.layer1 =nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.layer2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = nn.Linear(in_features=3136,out_features=120)
        self.layer4 = nn.Linear(in_features=120,out_features=84)
        self.layer5 = nn.Linear(in_features=84,out_features=40)


    def forward(self,x):
        x1 =torch.relu(self.layer1(x))
        x1 =self.pool1(x1)
        x2 = torch.relu(self.layer2(x1))
        x2 = self.pool2(x2)
        x2 =x2.reshape(x2.size(0),-1)
        x3 = torch.relu(self.layer3(x2))
        x4 = torch.relu(self.layer4(x3))
        out = self.layer5(x4)
        return out
if __name__ == '__main__':
    model = MyModel2()
    dummy_input = torch.randn(32,3,64,64)  # 模拟输入 (batch_size=32, height=64, width=64)
    print(dummy_input.shape)
    output = model(dummy_input)
    print(output.shape)  # 应输出 torch.Size([32, 40])