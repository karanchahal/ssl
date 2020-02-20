import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import STL10, MNIST
import torchvision.transforms as tfs 
import torchvision.models as models
from models.factory import get_model
from torch.utils.data import DataLoader
from losses.xnext import get_loss
from trainer.simclr import SimclrTrainer
import torch.optim as optim
from dataset.simclr import UnsupDataset
from losses.basic import nll_loss
from trainer.mnist import MnistTrainer

def get_stl(path='./data/'):
    dataset = STL10(root=path, split='unlabeled', folds=None, transform=None, target_transform=None, download=True)
    return dataset

def get_mnist(path='./data'):
    train_dataset = MNIST('../data', train=True, download=True, transform=tfs.Compose([
                           tfs.ToTensor(),
                           tfs.Normalize((0.1307,), (0.3081,))
                       ]))
    val_dataset = MNIST('../data', train=False, download=True, transform=tfs.Compose([
                           tfs.ToTensor(),
                           tfs.Normalize((0.1307,), (0.3081,))
                       ]))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    return train_loader, val_loader

def test_stl10():
    dataset = get_stl()
    im_size = 64

    transforms = tfs.Compose([
        tfs.Resize((512, 512)),
        tfs.RandomCrop((im_size, im_size)),
        tfs.ColorJitter(brightness=5, contrast=5, saturation=5, hue=0.5),
        tfs.ToTensor()
    ])

    unsup_dataset = UnsupDataset(dataset, transforms, im_size)
    data_loader = DataLoader(unsup_dataset, batch_size=1, shuffle=False)

    resnet18 = models.resnet18()
    model_func = get_model("simclr")
    model = model_func()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = SimclrTrainer(10, model, data_loader, data_loader, get_loss, optimizer)
    trainer.train()


def test_mnist():
    train_loader, val_loader = get_mnist()
    model = get_model("mnist")()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    trainer = MnistTrainer(10, model, train_loader, val_loader, nll_loss, optimizer)
    trainer.train()

# test_mnist()
test_stl10()