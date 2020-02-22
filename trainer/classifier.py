from trainer.base import Trainer 
import torch.nn.functional as F

class ClassifierTrainer(Trainer):

    def __init__(self, epochs, model, train_dataloader, val_dataloader, loss_func, optimizer):
        super(ClassifierTrainer, self).__init__(epochs, model, train_dataloader, val_dataloader, loss_func, optimizer)
        self.stats = {
            "train_loss" : [],
            "val_loss" : [],
            "train_acc" : [],
            "val_acc" : []
        }

    def _train_1_epoch(self, p_bar):
        total_loss = 0
        correct = 0
        self.model.train()
        batch_size = self.train_dataloader.batch_size
        len_train_set = len(self.train_dataloader.dataset)

        for batch_idx, (im, tar) in enumerate(self.train_dataloader):
            im, tar = im.to(self.device), tar.to(self.device)
            out = self.model(im)
            loss = self.loss_func(out, tar)
            self.step(loss)
            total_loss += loss.item()
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(tar.view_as(pred)).sum().item()
            p_bar.update(batch_size)
            p_bar.set_postfix( {
                "train_loss" : total_loss / (batch_idx + 1),
                "train_acc" : 100.0 * correct / len_train_set
            })
        
        total_loss /= len(self.train_dataloader)
        acc = 100.0 * correct / len_train_set
        self.stats["train_loss"].append(total_loss)
        self.stats["train_acc"].append(acc)

        return { "train_loss" : total_loss, "train_acc" : acc } 

    def validate(self, p_bar):

        # case where no validation is required.
        if self.val_dataloader == None:
            return { "val_loss" : 0, "val_acc": 0 }

        total_loss = 0
        correct = 0
        batch_size = self.val_dataloader.batch_size
        len_val_set = len(self.val_dataloader.dataset)
        self.model.eval()
        for batch_idx, (im, tar) in enumerate(self.val_dataloader):
            im, tar = im.to(self.device), tar.to(self.device)
            out = self.model(im)
            loss = self.loss_func(out, tar)
            total_loss += loss.item()
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(tar.view_as(pred)).sum().item()
            p_bar.update(batch_size)
            p_bar.set_postfix( {
                "val_loss" : total_loss / (batch_idx + 1),
                "val_acc" : 100.0 * correct / len_val_set
            })
        
        total_loss /= len(self.val_dataloader)
        acc = 100.0 * correct / len_val_set
        self.stats["val_loss"].append(total_loss)
        self.stats["val_acc"].append(acc)
        return { "val_loss" : total_loss, "val_acc" : acc } 

