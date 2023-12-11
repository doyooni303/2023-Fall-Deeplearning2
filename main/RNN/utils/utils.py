import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

def parser():
    parser = argparse.ArgumentParser()
    
    # Data/Results
    parser.add_argument('--data_dir', type=str, default='/workspace/data/')
    parser.add_argument('--ckpt_dir', type=str, default="./ckpt/")
    parser.add_argument('--output_dir', type=str, default="./results/")

    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=2023)

    # Model
    parser.add_argument('--rnn_type', type=str, default='gru', choices=['rnn','lstm','gru'])
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--layer_norm', action='store_true')

    return parser

def return_result(args, y_true, y_pred, best_epoch):
    # Regression 문제 평가 Metrics
    performance = {}
    performance['r2'] = r2_score(y_true=y_true, y_pred=y_pred)
    performance['mae'] = mean_absolute_error(y_true=y_true,y_pred=y_pred)
    performance['mse'] = mean_squared_error(y_true=y_true,y_pred=y_pred)
    performance['rmse'] = mean_squared_error(y_true=y_true,y_pred=y_pred,squared=False)
    performance['mape'] = mean_absolute_percentage_error(y_true, y_pred)

    print(f'rnn_type:{args.rnn_type}, hidden_size:{args.hidden_size}, num_layers:{args.num_layers}, bidirectional:{args.bidirectional}')
    print('R2 Score : ', round(performance['r2'],4))
    print('MAE : ', round(performance['mae'],4))
    print('MSE : ', round(performance['mse'],4))
    print('RMSE : ', round(performance['rmse'],4)) 
    print('MAPE: ', round(performance['mape'],4))

    record = [args.shuffle, args.seed, args.num_epochs, args.batch_size, args.attention,
              args.layer_norm, args.rnn_type, args.hidden_size, args.num_layers, args.lr,
              args.bidirectional, args.dropout, best_epoch, 
              performance['r2'], performance['mae'], performance['mse'], performance['rmse'], performance['mape']]
    df = pd.DataFrame([record], columns=('shuffle', 'seed', 'num_epochs', 'batch_size', 'attention',
                                         'layer_norm', 'rnn_type', 'hidden_size', 'num_layers', 'lr', 
                                         'bidirectional', 'dropout', 'best_epoch',
                                         'R2', 'MAE', 'MSE', 'RMSE', 'MAPE'))

    if os.path.exists(os.path.join(args.output_dir, 'results.csv')):
        prev_df = pd.read_csv(os.path.join(args.output_dir, 'results.csv'))
        df = pd.concat([prev_df, df])
        df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)
    else:
        df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)

    return performance

def visualization_loss_history(args, best_model_path, train_loss_history, val_loss_history, best_model_ckpt):
    plt.figure(figsize=(8,5))
    plt.title('Loss History')
    plt.plot(range(args.num_epochs), train_loss_history, c='blue', label='Train Loss')
    plt.plot(range(args.num_epochs), val_loss_history, c='red', label='Validation Loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f"{best_model_ckpt}.jpg")

# def save_predictions(args, key, labels, preds):
#     labels = pd.Series(labels[labels.columns[0]], name='Real')
#     preds = pd.Series(preds, name='Pred')
#     results = pd.concat([labels, preds], axis=1)
#     results.to_csv(os.path.join(args.ckpt_dir, f"{key}_predictions.csv"), index=False)