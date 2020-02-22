import torch.nn as nn
import torch
import torch.nn.functional as F 

class Classifier(nn.Module):

    def __init__(self, model, im_size, num_classes):
        super(Classifier, self).__init__()
        self.model = model
        sample_inp = torch.randn((1,3, im_size, im_size)).cuda()
        _, x = model(sample_inp)
        b_sz = x.shape[0]
        in_channel = x.view(b_sz, -1).shape[1]

        self.linear = nn.Linear(in_channel, num_classes)

    def forward(self, x):
        _, x = self.model(x)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)