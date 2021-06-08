from torch.utils import data

class MNISTDataset(data.Dataset):
    """MNIST Dataset class"""
    def __init__(self, imgs, labels, transform):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data = self.imgs[idx]
        if self.transform:
            data = self.transform(data)
        return (data, self.labels[idx])
