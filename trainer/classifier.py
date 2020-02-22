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

    def train(self):
        for ep in range(self.epochs):
            train_stats = self._train_1_epoch()
            self.log(train_stats, ep)
            if self.val_dataloader != None:
                val_stats = self.validate()
                self.log(val_stats, ep)
            else:
                val_stats = { "val_loss" : 0, "val_acc": 0 }

    def _train_1_epoch(self):
        total_loss = 0
        correct = 0
        self.model.train()
        for (im, tar) in self.train_dataloader:
            im, tar = im.to(self.device), tar.to(self.device)
            out = self.model(im)
            loss = self.loss_func(out, tar)
            self.step(loss)
            total_loss += loss.item()
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(tar.view_as(pred)).sum().item()
            break
        
        total_loss /= len(self.train_dataloader)
        acc = float(correct*100 / len(self.train_dataloader.dataset))
        self.stats["train_loss"].append(total_loss)
        self.stats["train_acc"].append(acc)

        return { "train_loss" : total_loss, "train_acc" : acc } 

    def validate(self):
        total_loss = 0
        correct = 0
        self.model.eval()
        for (im, tar) in self.val_dataloader:
            im, tar = im.to(self.device), tar.to(self.device)
            out = self.model(im)
            loss = self.loss_func(out, tar)
            total_loss += loss.item()
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(tar.view_as(pred)).sum().item()
        
        total_loss /= len(self.val_dataloader)
        acc = float(correct*100 / len(self.val_dataloader.dataset))
        self.stats["val_loss"].append(total_loss)
        self.stats["val_acc"].append(acc)
        return { "val_loss" : total_loss, "val_acc" : acc } 

    def printSummary(self, train_stats, val_stats, ep):
        print("Epoch {}, train Loss: {}, train accuracy : {} val loss : {} val accuracy : {}".format(
                ep,
                train_stats["train_loss"],
                train_stats["train_acc"],
                val_stats["val_loss"],
                val_stats["val_acc"]
            ))
