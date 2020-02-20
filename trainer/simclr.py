from trainer.base import Trainer 

class SimclrTrainer(Trainer):

    def __init__(self, epochs, model, train_dataloader, val_dataloader, loss_func, optimizer):
        super(SimclrTrainer, self).__init__(epochs, model, train_dataloader, val_dataloader, loss_func, optimizer)
        self.stats = {
            "train_loss" : [],
            "val_loss" : []
        }

    def train(self):
        for ep in range(self.epochs):
            train_loss = self._train_1_epoch()
            self.writer.add_scalar('Loss/train', train_loss, ep)
            val_loss = self.validate()
            self.writer.add_scalar('Loss/val', val_loss, ep)

    def _train_1_epoch(self):
        total_loss = 0
        self.model.train()
        for (im, aug_im) in self.train_dataloader:
            z, _ = self.model(im)
            z_aug, _ = self.model(aug_im)
            loss = self.loss_func(z, z_aug)
            self.step(loss)
            total_loss += loss.item()
            break
        
        total_loss /= len(self.train_dataloader)
        self.stats["train_loss"].append(total_loss)
        print(total_loss)
        return total_loss

    def validate(self):
        total_loss = 0
        self.model.eval()
        for (im, aug_im) in self.val_dataloader:
            z,_ = self.model(im)
            z_aug,_ = self.model(aug_im)
            loss = self.loss_func(z, z_aug)
            total_loss += loss.item()
            break
        
        total_loss /= len(self.val_dataloader)
        self.stats["val_loss"].append(total_loss)
        print(total_loss)
        return total_loss

