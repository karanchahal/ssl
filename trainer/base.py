from torch.utils.tensorboard import SummaryWriter
import torch
import sys 
from tqdm import tqdm

def init_progress_bar(train_loader):
    batch_size = train_loader.batch_size
    bar_format = "{desc}{percentage:3.0f}%"
    # bar_format += "|{bar}|"
    bar_format += " {n_fmt}/{total_fmt} [{elapsed} < {remaining}]"
    bar_format += "{postfix}"
    # if stderr has no tty disable the progress bar
    disable = not sys.stderr.isatty()
    t = tqdm(total=len(train_loader) * batch_size,
             bar_format=bar_format, disable=disable)
    if disable:
        # a trick to allow execution in environments where stderr is redirected
        t._time = lambda: 0.0
    return t

class Trainer:

    def __init__(self, epochs, model, train_dataloader, val_dataloader, loss_func, optimizer):
        self.epochs = epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.writer = SummaryWriter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):

        for ep in range(self.epochs):
            t_bar = init_progress_bar(self.train_dataloader)
            t_bar.set_description(f"Train Epoch {ep}")
            train_stats = self._train_1_epoch(t_bar)
            self.log(train_stats, ep)
            v_bar = None
            if self.val_dataloader:
                v_bar = init_progress_bar(self.val_dataloader)
                v_bar.set_description(f"Validate Epoch {ep}")
            val_stats = self.validate(v_bar)
            self.log(val_stats, ep)
            t_bar.close()
            if v_bar:
                v_bar.close()
                tqdm.clear(v_bar)

            tqdm.clear(t_bar)

    def log(self, stats, step):
        for key, val in stats.items():
            self.writer.add_scalar(key, val, step)
    
    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def validate(self):
        raise NotImplementedError

    def _train_1_epoch(self):
        raise NotImplementedError
