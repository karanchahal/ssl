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


    def _train_1_epoch(self, p_bar):
        total_loss = 0
        self.model.train()
        batch_size = self.train_dataloader.batch_size
        len_train_set = len(self.train_dataloader.dataset)
        for batch_idx, (im, aug_im) in enumerate(self.train_dataloader):
            im, aug_im = im.to(self.device), aug_im.to(self.device)
            z, _ = self.model(im)
            z_aug, _ = self.model(aug_im)
            loss = self.loss_func(z, z_aug)
            self.step(loss)
            total_loss += loss.item()
            p_bar.update(batch_size)
            p_bar.set_postfix( {
                "train_loss" : total_loss / (batch_idx + 1)
            })
        
        total_loss /= len(self.train_dataloader)
        self.stats["train_loss"].append(total_loss)
        return { "train_loss": total_loss }

    def validate(self, p_bar):
        for param in self.model.parameters():  # freeze representation layers
            param.requires_grad = False
        self.train_linear_layer() # Train top layer.
        val_stats = self.test_linear_layer(p_bar)  # Test representation quality.
        for param in self.model.parameters(): # unfreeze representation layers
            param.requires_grad = True
        return val_stats
    
    def test_linear_layer(self, p_bar):
        total_loss = 0
        correct = 0
        self.model2.eval()
        batch_size = self.val_dataloader.batch_size
        len_val_set = len(self.val_dataloader.dataset)
        for batch_idx, (im, tar) in enumerate(self.val_dataloader):
            im, tar = im.to(self.device), tar.to(self.device)
            out = self.model2(im)
            loss = nll_loss(out, tar)
            total_loss += loss.item()
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(tar.view_as(pred)).sum().item()
            p_bar.update(batch_size)
            p_bar.set_postfix( {
                "val_loss" : total_loss / (batch_idx + 1),
                "val_acc" : 100.0 * correct / len_val_set
            })
        
        total_loss /= len(self.val_dataloader)
        acc = correct*100.0 / len_val_set
        self.stats["val_loss"].append(total_loss)
        self.stats["val_acc"].append(acc)
        return { "val_loss" : total_loss, "val_acc" : acc } 

    def train_linear_layer(self):
        self.model2 = get_model('classifier')(self.model, self.im_size, self.num_classes)
        self.model2.to(self.device)
        optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, self.model2.parameters()), lr=0.01, momentum=0.9)
        print('\nTraining the Linear layer now....\n')
        self.trainer_val = ClassifierTrainer(self.epochs_linear, self.model2, self.linear_train_loader, None, nll_loss, optimizer2)
        self.trainer_val.train()
        print('\nTraining finished....\n')

