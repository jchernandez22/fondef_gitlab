import os
import pandas as pd
from pathlib import Path
from services.create_dataset import create_dataset

def load_dataset(split=True):
    
    data_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/')
    path_dataset_folder = data_folder_path + 'dataset/dataset.csv'
    if Path(path_dataset_folder).is_file():
        df = pd.read_csv(path_dataset_folder)

    else:
        create_dataset()
        df = pd.read_csv(path_dataset_folder)

    if split:
        return df.drop(['tiempo_promedio', 'muertos'], axis=1), df[['tiempo_promedio', 'muertos']]
    
    else:
        return df

if __name__ == '__main__':

    print(load_dataset())