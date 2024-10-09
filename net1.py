import math
from einops.einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
 

class CrossFusionBlock(nn.Module):
    def __init__(self, channel_in = 3, channel_out = 1, middle = 64, branch1 = 16, branch2 = 32):
        super(CrossFusionBlock, self).__init__()

        channel_sum = branch1 + branch2

        self.op1 = nn.Sequential(
                    # nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.Conv2d(channel_in, middle, kernel_size=5, padding=2),
                    nn.BatchNorm2d(middle),
                    nn.ReLU(inplace=True),
        )
        self.op2 = nn.Sequential(
                    nn.Conv2d(middle, branch1, kernel_size=5, padding=2),
                    nn.BatchNorm2d(branch1),
                    nn.ReLU(inplace=True),
        )
        self.op3 = nn.Sequential(
                    nn.Conv2d(middle, branch2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(branch2),
                    nn.ReLU(inplace=True),
        )
        self.op4 = nn.Sequential(
                    nn.Conv2d(channel_sum, branch1, kernel_size=5, padding=2),
                    nn.BatchNorm2d(branch1),
                    nn.ReLU(inplace=True),
        )
        self.op5 = nn.Sequential(
                    nn.Conv2d(channel_sum, branch2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(branch2),
                    nn.ReLU(inplace=True),
        )
        self.op6 = nn.Sequential(
                    nn.Conv2d(channel_sum, channel_out, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channel_out),
                    nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.op1(x)

        x1 = self.op2(x)
        x2 = self.op3(x)
        x = torch.cat([x1,x2], dim = 1)

        x1 = self.op4(x)
        x2 = self.op5(x)
        x = torch.cat([x1,x2], dim = 1)
        return self.op6(x)



class MultiFocusTransNet(nn.Module): 
    def __init__(self):
        super(MultiFocusTransNet, self).__init__()
        
        self.cf1 = CrossFusionBlock(3, 1, 64, 16, 32)

        self.cf2 = CrossFusionBlock(6, 1, 64, 16, 32)

        self.cf3 = CrossFusionBlock(9, 1, 64, 16, 32)

        self.cf4 = CrossFusionBlock(12, 1, 64, 16, 32)

        self.cf5 = CrossFusionBlock(15, 1, 64, 16, 32)

        self.cf6 = CrossFusionBlock(18, 1, 64, 16, 32)


    def forward(self, input):
        x1 = self.cf1(input)
        x1 = x1.repeat(1,3,1,1) + input

        x2 = self.cf2(torch.cat([input, x1], dim = 1))
        x2 = x2.repeat(1,3,1,1) + input

        x3 = self.cf3(torch.cat([input, x1, x2], dim = 1))
        x3 = x3.repeat(1,3,1,1) + input

        x4 = self.cf4(torch.cat([input, x1, x2, x3], dim = 1))
        x4 = x4.repeat(1,3,1,1) + input

        x5 = self.cf5(torch.cat([input, x1, x2, x3, x4], dim = 1))
        x5 = x5.repeat(1,3,1,1) + input

        x6 = self.cf6(torch.cat([input, x1, x2, x3, x4, x5], dim = 1))
        # x6 = x6.repeat(1,3,1,1) + input
        return x6




def test():
    net = MultiFocusTransNet()
    x = torch.randn(2, 3, 480, 640)
    y = net(x)
    print(y.size())
    # torch.Size([1, 10, 32, 32])

def test_loss():
    net = MultiFocusTransNet(3,10)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.NLLLoss2d()
    # criterion = torch.nn.L1Loss()
    # criterion = torch.nn.BCELoss()
    # criterion = torch.nn.MSELoss()

    x = torch.randn(2, 3, 480, 640)
    y_label = torch.randn(2, 480, 640).long()
    y = net(x)
    print(y.size())
    loss = criterion(y, y_label)
    print(loss.item())

if __name__ == "__main__":
    test()
    # test_loss()