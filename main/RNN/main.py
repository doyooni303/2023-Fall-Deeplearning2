import os, sys, random, time, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from datasets import make_dataloader
from utils.utils import parser, return_result, visualization_loss_history
from tasks.train_test import Train_Test, Train_Test_Attention
from rnn import RNN_model, RNN_Attention

def main(args):
    # seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # data
    train_loader, valid_loader, test_loader = make_dataloader(args)
    
    # hyperparameter
    rnn_type= args.rnn_type
    input_size = 9
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    learning_rate = args.lr
    dropout = args.dropout
    seq_len = 24
    # weight_decay = args.weight_decay
    bidirectional = args.bidirectional
    attention = args.attention
    layer_norm = args.layer_norm
    num_epochs = args.num_epochs
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu') 
    best_model_path = os.path.join(args.ckpt_dir, rnn_type)

    ## Modeling
    if attention:
        model = RNN_Attention(input_size, hidden_size, num_layers, bidirectional, rnn_type, dropout, layer_norm, device)
    else:
        model = RNN_model(input_size, hidden_size, seq_len, num_layers, bidirectional, rnn_type, dropout, layer_norm, device)
    model = model.to(device)

    # Training and Save Weights(Parameters)
    dataloaders_dict_train = {'train': train_loader, 'val': valid_loader}

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)   # Adam or Adamw

    if attention:
        trainer = Train_Test_Attention(train_loader, valid_loader, test_loader, input_size, device)
    else:
        trainer = Train_Test(train_loader, valid_loader, test_loader, input_size, device)
    best_model, train_loss_history, val_loss_history, best_epoch = trainer.train(model, dataloaders_dict_train, criterion, num_epochs, optimizer)

    os.makedirs(os.path.join(args.ckpt_dir, rnn_type), exist_ok=True)
    best_model_ckpt = os.path.join(best_model_path, f'{args.shuffle, args.rnn_type, args.attention, args.hidden_size, args.num_layers, args.lr, args.bidirectional, args.dropout}')

    torch.save(best_model.state_dict(), f'{best_model_ckpt}.pt')

    visualization_loss_history(args, best_model_path, train_loss_history, val_loss_history, best_model_ckpt)

    # Evaluation
    model.load_state_dict(torch.load(f'{best_model_ckpt}.pt'))    # Load model weights(Parameters)

    dataloaders_dict_test = {'train': train_loader, 'val': valid_loader, 'test':test_loader}
    if attention:
        train_preds, train_y_true, train_attn_scores, val_preds, val_y_true, val_attn_scores, test_preds, test_y_true, test_attn_scores = trainer.test(model, dataloaders_dict_test)
    else:
        train_preds, train_y_true, val_preds, val_y_true, test_preds, test_y_true  = trainer.test(model, dataloaders_dict_test)

    performance = return_result(args, train_preds, train_y_true, val_preds, val_y_true, test_preds, test_y_true, best_epoch)


if __name__ == "__main__":
    args = parser().parse_args()

    args.num_epochs = 500
    args.batch_size = 128

    args.layer_norm = True

    shuffle = [False, True]
    rnn_type = ['lstm','gru']
    hidden_size = [64, 128, 256]
    num_layers = [1,2]
    learning_rate = [0.01, 0.001, 0.0001,]
    # bidirectional = [True, False]
    dropout = [0, 0.2]
    attention = [True, False]

    combinations = list(product(shuffle, rnn_type, hidden_size, num_layers, learning_rate, dropout, attention))

    for c in tqdm(combinations):
        args.shuffle = c[0]
        args.rnn_type = c[1]
        args.hidden_size = c[2]
        args.num_layers = c[3]
        args.lr = c[4]
        args.dropout = c[5]
        args.attention = c[6]

        main(args)