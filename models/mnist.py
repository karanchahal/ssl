import torch.nn as nn
import torch
import torch.nn.functional as F

class MnistNet(nn.Module):

    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(7744, 10)
    
    def forward(self, x):

        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.dropout(x)
        b_sz = x.shape[0]
        x = x.view(b_sz, -1)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)



def test():
    a = torch.randn((1,1,28,28))
    model = MnistNet()
    x = model(a)
    print(x.size())

# test()