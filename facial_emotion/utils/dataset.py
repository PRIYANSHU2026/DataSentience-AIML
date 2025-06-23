import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image

class FERDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels = np.array(self.data['pixels'][idx].split(), dtype='float32').reshape(48, 48)
        img = Image.fromarray(pixels).convert("L")
        label = int(self.data['emotion'][idx])
        if self.transform:
            img = self.transform(img)
        return img, label
