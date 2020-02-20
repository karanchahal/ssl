import torchvision.transforms as tfs 
from torch.utils.data import Dataset, DataLoader

class UnsupDataset(Dataset):

    def __init__(self, dataset, transform, im_size):
        self.d = dataset
        self.transform = transform

        self.im_tf = tfs.Compose([
            tfs.Resize((im_size, im_size)),
            tfs.ToTensor()
        ])
    
    def __len__(self):
        return len(self.d)
    
    def __getitem__(self, idx):
        im, _ = self.d[idx] # PIL image, -1 cause it's unlabeled
        aug_im = self.transform(im)
        return self.im_tf(im), aug_im
