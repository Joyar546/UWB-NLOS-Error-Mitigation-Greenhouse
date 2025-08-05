# Import libraries
import torch
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)




# The structure of remnet
class REmodule(nn.Module):
    def __init__(self, row):  # row is K/2^n
        super(REmodule, self).__init__()
        self.conv1 = nn.Sequential(  # K*F
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),)
        self.se = SELayer(channel=16, reduction=8)
        # self.gp = nn.AvgPool1d(kernel_size=row, stride=1)  # 1*F
        # self.fc1 = nn.Sequential(
        #     nn.Linear(16, 2),
        #     nn.ReLU(),)
        # self.fc2 = nn.Sequential(
        #     nn.Linear(2, 16),
        #     nn.Sigmoid(),)
        self.conv2 = nn.Sequential(
            nn.Conv1d(row, int(row/2), kernel_size=3, padding=1),
            nn.ReLU(),)
        self.conv3 = nn.Sequential(
            nn.Conv1d(row, int(row/2), kernel_size=1),
            nn.ReLU(),)

    def forward(self, x):
        residual = x
        x0 = self.conv1(x)

        # SE block
        # out = self.gp(x0)
        # out = out.reshape(out.size(0), -1)
        # out = self.fc1(out)
        # out = self.fc2(out)
        # out = out.unsqueeze(dim=2)
        # temp = torch.Tensor(x0.shape[0], x0.shape[1], x0.shape[2]).to(device)
        # temp = temp.copy_(out)
        # out = torch.mul(x0, temp)

        out = self.se(x0)
        out = torch.add(residual, out)
        out = out.permute(0, 2, 1)
        out1 = self.conv2(out)
        out2 = self.conv3(out)
        out = torch.add(out1, out2)
        out = out.permute(0, 2, 1)

        return out


class REMNet(nn.Module):
    def __init__(self):
        super(REMNet, self).__init__()
        self.conv1 = nn.Sequential(  # K*F
            nn.Conv1d(1, 16, kernel_size=7),
            nn.ReLU(), )

        self.relayer1 = REmodule(151)
        self.relayer2 = REmodule(75)
        self.relayer3 = REmodule(37)

        self.flatten = nn.Flatten()
        self.drop = nn.Dropout()
        self.fc = nn.Linear(288, 1)

    def forward(self, x):
        x = self.conv1(x)
        out = self.relayer1(x)
        out = self.relayer2(out)
        out = self.relayer3(out)
        out = self.flatten(out)
        out = self.drop(out)
        out = self.fc(out)
        return out






