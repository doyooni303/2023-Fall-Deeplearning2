import argparse

from torch.utils.data import DataLoader
from datasets.dataset.Beijing import BeijingDataset

def make_dataloader(args: argparse, **kwargs: dict):
    train_set = BeijingDataset(args, phase="train")
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=False)
    
    valid_set = BeijingDataset(args, phase="valid")
    valid_loader = DataLoader(valid_set,
                              batch_size=args.batch_size,
                              shuffle=False)

    test_set = BeijingDataset(args, phase="test")
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False)

    return train_loader, valid_loader, test_loader