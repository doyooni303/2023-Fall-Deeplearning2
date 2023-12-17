import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

def parser():
    parser = argparse.ArgumentParser()
    
    # Data/Results
    parser.add_argument('--data_dir', type=str, default='/workspace/data/')
    parser.add_argument('--ckpt_dir', type=str, default="./ckpt_trainvaltest/")
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

def return_result(args, train_preds, train_y_true, val_preds, val_y_true, test_preds, test_y_true, best_epoch):

    # Regression 문제 평가 Metrics
    performance = {}
    # Train
    performance['train_mse'] = mean_squared_error(y_true=train_y_true, y_pred=train_preds)
    # val
    performance['val_r2'] = r2_score(y_true=val_y_true, y_pred=val_preds)
    performance['val_mae'] = mean_absolute_error(y_true=val_y_true, y_pred=val_preds)
    performance['val_mse'] = mean_squared_error(y_true=val_y_true, y_pred=val_preds)
    performance['val_rmse'] = mean_squared_error(y_true=val_y_true, y_pred=val_preds, squared=False)
    performance['val_mape'] = mean_absolute_percentage_error(y_true=val_y_true, y_pred=val_preds)
    # test
    performance['test_r2'] = r2_score(y_true=test_y_true, y_pred=test_preds)
    performance['test_mae'] = mean_absolute_error(y_true=test_y_true, y_pred=test_preds)
    performance['test_mse'] = mean_squared_error(y_true=test_y_true, y_pred=test_preds)
    performance['test_rmse'] = mean_squared_error(y_true=test_y_true, y_pred=test_preds, squared=False)
    performance['test_mape'] = mean_absolute_percentage_error(y_true=test_y_true, y_pred=test_preds)

    print(f'rnn_type:{args.rnn_type}, hidden_size:{args.hidden_size}, num_layers:{args.num_layers}, bidirectional:{args.bidirectional}')
    print('Test R2 Score : ', round(performance['test_r2'],4))
    print('Test MAE : ', round(performance['test_mae'],4))
    print('Test MSE : ', round(performance['test_mse'],4))
    print('Test RMSE : ', round(performance['test_rmse'],4)) 
    print('Test MAPE: ', round(performance['test_mape'],4))

    record = [args.shuffle, args.seed, args.num_epochs, args.batch_size, args.attention,
            args.layer_norm, args.rnn_type, args.hidden_size, args.num_layers, args.lr,
            args.bidirectional, args.dropout, best_epoch, 
            performance['train_mse'],
            performance['val_r2'], performance['val_mae'], performance['val_mse'], performance['val_rmse'], performance['val_mape'],
            performance['test_r2'], performance['test_mae'], performance['test_mse'], performance['test_rmse'], performance['test_mape']]
    df = pd.DataFrame([record], columns=('shuffle', 'seed', 'num_epochs', 'batch_size', 'attention',
                                        'layer_norm', 'rnn_type', 'hidden_size', 'num_layers', 'lr', 
                                        'bidirectional', 'dropout', 'best_epoch',
                                        'train_mse',
                                        'val_R2', 'val_MAE', 'val_MSE', 'val_RMSE', 'val_MAPE',
                                        'test_R2', 'test_MAE', 'test_MSE', 'test_RMSE', 'test_MAPE'))

    if os.path.exists(os.path.join(args.output_dir, 'results_trainvaltest.csv')):
        prev_df = pd.read_csv(os.path.join(args.output_dir, 'results_trainvaltest.csv'))
        df = pd.concat([prev_df, df])
        df.to_csv(os.path.join(args.output_dir, 'results_trainvaltest.csv'), index=False)
    else:
        df.to_csv(os.path.join(args.output_dir, 'results_trainvaltest.csv'), index=False)

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