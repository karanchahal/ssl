from torch.utils.tensorboard import SummaryWriter
import torch
class Trainer:

    def __init__(self, epochs, model, train_dataloader, val_dataloader, loss_func, optimizer):
        self.epochs = epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.writer = SummaryWriter()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def train(self):
        raise NotImplementedError

    def log(self, stats, step):
        for key, val in stats.items():
            self.writer.add_scalar(key, val, step)
    
    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
