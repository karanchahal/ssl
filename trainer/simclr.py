from trainer.base import Trainer 
from models.factory import get_model
from trainer.classifier import ClassifierTrainer
from losses.basic import nll_loss
import torch.optim as optim

class SimclrTrainer(Trainer):

    def __init__(self, epochs, epochs_linear, model, train_dataloader, linear_train_loader,
                val_dataloader, loss_func, optimizer, im_size=64,
                num_classes=10):
        super(SimclrTrainer, self).__init__(epochs, model, train_dataloader, val_dataloader, loss_func, optimizer)
        self.linear_train_loader = linear_train_loader
        self.im_size = 64
        self.epochs_linear = epochs_linear
        self.num_classes = num_classes
        self.stats = {
            "train_loss" : [],
            "val_loss" : [],
            "val_acc" : []
        }

    def train(self):
        for ep in range(self.epochs):
            train_stats = self._train_1_epoch()
            self.log(train_stats, ep)
            val_stats = self.validate()
            self.log(val_stats, ep)

    def _train_1_epoch(self):
        total_loss = 0
        self.model.train()
        for (im, aug_im) in self.train_dataloader:
            im, aug_im = im.to(self.device), aug_im.to(self.device)
            z, _ = self.model(im)
            z_aug, _ = self.model(aug_im)
            loss = self.loss_func(z, z_aug)
            self.step(loss)
            total_loss += loss.item()
            break
        
        total_loss /= len(self.train_dataloader)
        self.stats["train_loss"].append(total_loss)
        return { "train_loss": total_loss }

    def validate(self):
        for param in self.model.parameters():  # freeze representation layers
            param.requires_grad = False
        self.train_linear_layer() # Train top layer.
        val_stats = self.test_linear_layer()  # Test representation quality.
        for param in self.model.parameters(): # unfreeze representation layers
            param.requires_grad = True
        return val_stats
    
    def test_linear_layer(self):
        total_loss = 0
        correct = 0
        self.model2.eval()
        for (im, tar) in self.val_dataloader:
            im, tar = im.to(self.device), tar.to(self.device)
            out = self.model2(im)
            loss = nll_loss(out, tar)
            total_loss += loss.item()
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(tar.view_as(pred)).sum().item()
            break
        
        total_loss /= len(self.val_dataloader)
        acc = float(correct*100 / len(self.val_dataloader.dataset))
        self.stats["val_loss"].append(total_loss)
        self.stats["val_acc"].append(acc)
        return { "val_loss" : total_loss, "val_acc" : acc } 

    def train_linear_layer(self):
        self.model2 = get_model('classifier')(self.model, self.im_size, self.num_classes)
        self.model2.to(self.device)
        optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, self.model2.parameters()), lr=0.01, momentum=0.9)
        self.trainer_val = ClassifierTrainer(self.epochs_linear, self.model2, self.linear_train_loader, None, nll_loss, optimizer2)
        self.trainer_val.train()

