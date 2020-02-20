import torch.nn as nn
from models.resnet import resnet18
import torch 

class Simclr(nn.Module):

    def __init__(self):
        super(Simclr, self).__init__()
        self.base = resnet18()
        l = self.get_layer_len()
        self.z_layer = nn.Sequential(
            nn.Linear(l, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )


    def get_layer_len(self):
        # TODO
        return 512

    def forward(self, x):
        h = self.base(x)
        b_size, _, _, _ = h.size()
        h = h.view(b_size, -1)
        z = self.z_layer(h)

        return z, h



def test():
    model = Simclr()
    a = torch.randn((1,3,64,64))
    z, h = model(a)
    print(z.size(), h.size())

# test()