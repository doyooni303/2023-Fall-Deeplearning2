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
        model = RNN_model(input_size, hidden_size, num_layers, bidirectional, rnn_type, dropout, layer_norm, device)
    model = model.to(device)

    # Training and Save Weights(Parameters)
    dataloaders_dict = {'train': train_loader, 'val': valid_loader}

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if attention:
        trainer = Train_Test_Attention(train_loader, valid_loader, test_loader, input_size, device)
        best_model, train_loss_history, val_loss_history, attn_scores = trainer.train(model, dataloaders_dict, criterion, num_epochs, optimizer)
    else:
        trainer = Train_Test(train_loader, valid_loader, test_loader, input_size, device)
        best_model, train_loss_history, val_loss_history = trainer.train(model, dataloaders_dict, criterion, num_epochs, optimizer)

    os.makedirs(os.path.join(args.ckpt_dir, rnn_type), exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(best_model_path, f'{hidden_size}_{num_layers}_{bidirectional}.pt'))

    visualization_loss_history(args, best_model_path, train_loss_history, val_loss_history)

    # Evaluation
    model.load_state_dict(torch.load(os.path.join(best_model_path, f'{hidden_size}_{num_layers}_{bidirectional}.pt')))    # Load model weights(Parameters)

    if attention:
        y_pred, y_true, mse, attn_scores = trainer.test(model, test_loader)
    else:
        y_pred, y_true, mse = trainer.test(model, test_loader)

    y_true = np.array(y_true)

    performance = return_result(args, y_true, y_pred)

    print()


if __name__ == "__main__":
    args = parser().parse_args()

    args.shuffle = False

    args.rnn_type = 'gru'
    args.hidden_size = 256
    args.num_layers = 1
    args.lr = 0.0001
    args.dropout = 0.1
    args.bidirectional = False
    args.attention = False
    args.layer_norm = True
    args.num_epochs = 500
    args.batch_size = 256


    main(args)

    '''
    rnn_type = ['rnn', 'lstm', 'gru']
    hidden_size = [16, 32, 64, 128]
    num_layers = [1,2]
    bidirectional = [True, False]
    num_epochs = [1000]

    combinations = list(product(rnn_type, hidden_size, num_layers, bidirectional, num_epochs))

    for c in tqdm(combinations):
        args.rnn_type = c[0]
        args.hidden_size = c[1]
        args.num_layers = c[2]
        args.bidirectional = c[3]
        args.num_epochs = c[4]

        main(args)
    '''