import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import argparse

class BeijingDataset(Dataset):
    def __init__(self,
                 args: argparse,
                 phase: str):
        super(BeijingDataset, self).__init__()

        self.args = args
        self.data_dir = args.data_dir
        self.phase = phase
        self.shuffle = args.shuffle

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.x, self.y = self.split_windowing()

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        x = self.transform(x)
        x = torch.from_numpy(np.array(x).reshape(-1,9)).float()
        # y = self.transform(y)

        return dict(X=x, Y=y)

    def __len__(self):
        return len(self.x)

    def split_windowing(self, window_size=24, shift_size=24):
        if self.phase == "train":
            if self.shuffle:
                data = pd.read_csv(os.path.join(self.data_dir, 'train_True_df.csv'))
            else:
                data = pd.read_csv(os.path.join(self.data_dir, 'train_False_df.csv'))
        elif self.phase == 'valid':
            if self.shuffle:
                data = pd.read_csv(os.path.join(self.data_dir, 'valid_True_df.csv'))
            else:
                data = pd.read_csv(os.path.join(self.data_dir, 'valid_False_df.csv'))
        elif self.phase == "test":
            data = pd.read_csv(os.path.join(self.data_dir, 'test_df.csv'))

        x = data.drop(['Unnamed: 0', 'labels'], axis=1)
        y = data[['labels']]

        try:
            x = x.reset_index(drop=True)
            y = y.reset_index(drop=True)
        except:
            pass
        
        x_window = []
        y_window = []
        
        for start_idx in range(0, x.shape[0] - window_size + 1, shift_size):
            x_window.append(x.iloc[start_idx:start_idx + window_size, :])
            y_window.append(y.iloc[start_idx + window_size - 1, :])
        
        x_window = np.array(x_window)
        y_window = np.array(y_window)
        
        return x_window, y_window

