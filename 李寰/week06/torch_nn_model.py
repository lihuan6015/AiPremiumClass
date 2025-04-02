import torch
import torch.nn as nn

# class RNNModel(nn.Module):
#     def __init__(self):
#         super(RNNModel, self).__init__()
#         #self.flatten = nn.Flatten()# 添加展平层
#         self.layer1 = nn.Linear(64*64,64*64*2)
#         self.batch_norm = nn.BatchNorm1d(64*64*2)
#         self.layer2 = nn.Linear(64*64*2,4096)
#         self.layer3 = nn.Linear(4096,1024)
#         self.layer4 = nn.Linear(1024,40)
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self,x):
#         #x = self.flatten(x)
#         x1 =self.layer1(x)
#         x1=self.batch_norm(x1)
#         x1=torch.relu(x1)
#         x1=self.dropout(x1)
#
#         x2 = self.layer2(x1)
#         x2 = torch.relu(x2)
#         x2=self.dropout(x2)
#
#         x3 = self.layer3(x2)
#         x3 = torch.relu(x3)
#         x3=self.dropout(x3)
#
#
#
#         out =self.layer4(x3)
#         return out
class RNN_Classifier(nn.Module):

    def __init__(self,):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=64,   # x的特征维度
            hidden_size=64*2,  # 隐藏层神经元数量 w_ht[128,64], w_hh[128,128]
            bias=True,        # 偏置[50]
            num_layers=5,     # 隐藏层层数
            batch_first=True  # 批次是输入第一个维度
        )
        self.fc = nn.Linear(64*2, 40)  # 输出层

    def forward(self, x):
        # 输入x形状应为 [batch, seq_len=64, input_size=64]
        # 若原始图像为 [batch, 64, 64]
        outputs, l_h = self.rnn(x)  # 连续运算后所有输出值
        # 取最后一个时间点的输出值
        out = self.fc(outputs[:,-1,:])
        return out

class LSTM_Classifier(nn.Module):
    def __init__(self,):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=64,   # x的特征维度
            hidden_size=64*2,  # 隐藏层神经元数量 w_ht[128,64], w_hh[128,128]
            bias=True,        # 偏置[50]
            num_layers=5,     # 隐藏层层数
            batch_first=True  # 批次是输入第一个维度
        )
        self.fc = nn.Linear(64*2, 40)  # 输出层

    def forward(self, x):
        # 输入x形状应为 [batch, seq_len=64, input_size=64]
        # 若原始图像为 [batch, 64, 64]
        outputs, l_h = self.rnn(x)  # 连续运算后所有输出值
        # 取最后一个时间点的输出值
        out = self.fc(outputs[:,-1,:])
        return out

class GRU_Classifier(nn.Module):
    def __init__(self,):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=64,   # x的特征维度
            hidden_size=64*2,  # 隐藏层神经元数量 w_ht[128,64], w_hh[128,128]
            bias=True,        # 偏置[50]
            num_layers=5,     # 隐藏层层数
            batch_first=True  # 批次是输入第一个维度
        )
        self.fc = nn.Linear(64*2, 40)  # 输出层

    def forward(self, x):
        # 输入x形状应为 [batch, seq_len=64, input_size=64]
        # 若原始图像为 [batch, 64, 64]
        outputs, l_h = self.rnn(x)  # 连续运算后所有输出值
        # 取最后一个时间点的输出值
        out = self.fc(outputs[:,-1,:])
        return out


if __name__ == '__main__':
    # model = MyModel2()
    # dummy_input = torch.randn(32,3,64,64)  # 模拟输入 (batch_size=32, height=64, width=64)
    # print(dummy_input.shape)
    # output = model(dummy_input)
    # print(output.shape)  # 应输出 torch.Size([32, 40])
    model = RNN_Classifier()
    print(model)