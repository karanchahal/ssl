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
from trainer.classifier import ClassifierTrainer
import torch.nn.functional as F

def get_stl(path='./data/'):

    im_size = 64
    batch_size = 2

    train_dataset = STL10(root=path, split='unlabeled', folds=None, transform=None, target_transform=None, download=True)
    linear_train_dataset = STL10(root=path, split='train', folds=None, transform=tfs.Compose([
                tfs.Resize((int(im_size*1.25), int(im_size*1.25))), # increase image size slightly
                tfs.RandomCrop((im_size, im_size)),
                tfs.ColorJitter(brightness=5, contrast=5, saturation=5, hue=0.5),
                tfs.ToTensor()
            ]), target_transform=None, download=True)
    val_dataset = STL10(root=path, split='test', folds=None, transform=tfs.Compose([
                tfs.Resize((int(im_size*1.25), int(im_size*1.25))), # increase image size slightly
                tfs.RandomCrop((im_size, im_size)),
                tfs.ColorJitter(brightness=5, contrast=5, saturation=5, hue=0.5),
                tfs.ToTensor()
            ]), target_transform=None, download=True)

    transforms = tfs.Compose([
        tfs.Resize((int(im_size*1.25), int(im_size*1.25))), # increase image size slightly
        tfs.RandomCrop((im_size, im_size)),
        tfs.ColorJitter(brightness=5, contrast=5, saturation=5, hue=0.5),
        tfs.ToTensor()
    ])

    unsup_dataset = UnsupDataset(train_dataset, transforms, im_size)

    train_loader = DataLoader(unsup_dataset, batch_size=batch_size, shuffle=True)
    linear_train_loader = DataLoader(linear_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, linear_train_loader, val_loader

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
    
    train_loader, linear_train_loader, val_loader = get_stl()
    resnet18 = models.resnet18()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("simclr")().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = SimclrTrainer(10, 5, model, train_loader, linear_train_loader, val_loader, get_loss, optimizer)
    trainer.train()


def test_mnist():
    train_loader, val_loader = get_mnist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("mnist")().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    trainer = ClassifierTrainer(10, model, train_loader, val_loader, nll_loss, optimizer)
    trainer.train()

test_mnist()
# test_stl10()