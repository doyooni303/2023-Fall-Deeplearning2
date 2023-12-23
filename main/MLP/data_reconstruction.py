import pandas as pd
import os

def transform_to_B_format(df_a):
    reshaped_data = []
    unique_indices = df_a.iloc[:, 0].unique()
    
    for idx in unique_indices:
        sub_df = df_a[df_a.iloc[:, 0] == idx]
        flattened = sub_df.iloc[:, 1:-1].values.flatten()
        reshaped_row = flattened.tolist() + [sub_df['labels'].iloc[0]]
        reshaped_data.append(reshaped_row)
    
    num_unique_cols = len(df_a.columns) - 2
    column_names = [f'dim_{i // num_unique_cols}_{i % num_unique_cols}' for i in range(len(reshaped_data[0]) - 1)]
    column_names.append('labels')
    
    df_b = pd.DataFrame(reshaped_data, columns=column_names)
    df_b['index'] = unique_indices
    df_b.set_index('index', inplace=True)
    
    return df_b

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', '..', 'data')

folder_a_path = os.path.join(script_dir, 'data_reconstructed')

if not os.path.exists(folder_a_path):
    os.makedirs(folder_a_path)

file_names = ['train_True_df.csv', 'train_False_df.csv', 'valid_True_df.csv', 'valid_False_df.csv', 'test_df.csv']

for file_name in file_names:
    file_path = os.path.join(data_dir, file_name)
    df_A = pd.read_csv(file_path)
    
    df_B = transform_to_B_format(df_A)
    
    new_file_name = file_name.replace('.csv', '2.csv')
    new_file_path = os.path.join(folder_a_path, new_file_name)
    df_B.to_csv(new_file_path)
