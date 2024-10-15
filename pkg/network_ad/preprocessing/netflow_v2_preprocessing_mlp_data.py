import shutil
import sys
sys.path.append("../..")

from network_ad.config import MLP_TEMPORARY_DATA_PATH, MLP_PREPROCESSED_DATA_PATH
from sklearn.preprocessing import OneHotEncoder
from network_ad.config import NUMERICAL_STATS_PATH, CATEGORICAL_STATS_PATH, VAL_RATIO
import json
import multiprocessing

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import joblib
from network_ad.config import MLP_CATEGORICAL_FEATURES, MLP_NUMERICAL_FEATURES, PREPROCESSED_DATA_PATH, AUTOENCODER_INPUT_DIMS
from network_ad.config import TRAIN_DATA_PATH, TEST_DATA_PATH, TEST_RATIO


def mlp_preprocess_data(mode, train_path, test_path, numerical_stats, categorical_encoders):
    """Process and save dataset to Hdf5."""
    print("\nPreprocessing data for MLP . Mode: ", mode)
    print("=====================================================")
    def load_data(mode):
        if mode == 'train':
            df = pd.read_csv(train_path)
        elif mode == 'test':
            df = pd.read_csv(test_path)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        for feature in MLP_CATEGORICAL_FEATURES:
            df[feature] = df[feature].astype(str)
        return df
        # return df.sample(frac=0.01)

    if mode == 'train':
        df = load_data('train')
        nb_train_samples = len(df) * (1 - VAL_RATIO)
        dataframe = df[:int(nb_train_samples)]

    elif mode == 'val':
        df = load_data('train')
        nb_train_samples = len(df) * (1 - VAL_RATIO)
        dataframe = df[int(nb_train_samples):]

    elif mode == 'test':
        df = load_data('test')
        dataframe = df

    temporary_h5_path = MLP_TEMPORARY_DATA_PATH
    temporary_h5_sub_paths = [PREPROCESSED_DATA_PATH / f'mlp_temporary_data_chunk_{i}.h5' for i in range(multiprocessing.cpu_count())]
    for temporary_h5_sub_path in temporary_h5_sub_paths:
        if temporary_h5_sub_path.exists():
            temporary_h5_sub_path.unlink()

    def preprocess_single_dataframe_chunk(dataframe_chunk, start_idx, mode, temporary_h5_sub_path):
        with h5py.File(temporary_h5_sub_path, 'w') as sub_f:
            sub_dataset = sub_f.create_dataset(mode, (len(dataframe_chunk), AUTOENCODER_INPUT_DIMS), dtype='f')
            dataframe_chunk.reset_index(drop=True, inplace=True)
            for idx, row in tqdm(dataframe_chunk.iterrows(), total=len(dataframe_chunk), desc=f'Preprocessing {mode} . Chunk start index: {start_idx}. Chunk size: {len(dataframe_chunk)}'):
                categorical_data_list = []
                for feature in MLP_CATEGORICAL_FEATURES:
                    encoded_feature = categorical_encoders[feature].transform([[row[feature]]])
                    categorical_data_list.append(encoded_feature)

                numerical_data = row[MLP_NUMERICAL_FEATURES].values.astype(np.float32)
                numerical_scaled = (numerical_data - numerical_stats['mean'].values) / numerical_stats['std'].values
                categorical_encoded = np.hstack(categorical_data_list).squeeze()
                features = np.hstack((numerical_scaled, categorical_encoded)).astype(np.float32)
                sub_dataset[idx] = features



    with h5py.File(temporary_h5_path, 'a') as f:
        dataset_name = mode
        n_chunks = multiprocessing.cpu_count()
        dataset_splits = np.array_split(dataframe, n_chunks)
        f.create_dataset(dataset_name, (len(dataframe), AUTOENCODER_INPUT_DIMS), dtype='f')
        start_idxs = [0]
        for i in range(1, n_chunks):
            start_idxs.append(start_idxs[i - 1] + len(dataset_splits[i - 1]))

        joblib.Parallel(n_jobs=n_chunks)(joblib.delayed(preprocess_single_dataframe_chunk)(dataset_splits[i], start_idxs[i], mode,
                                                                               temporary_h5_sub_paths[i]) for i in range(n_chunks))

        for i in range(n_chunks):
            with h5py.File(str(temporary_h5_sub_paths[i]), 'r') as h5f_sub:
                f[dataset_name][start_idxs[i]:start_idxs[i] + len(dataset_splits[i])] = h5f_sub[mode][:]
                h5f_sub.close()
            temporary_h5_sub_paths[i].unlink()

        f.close()

    print(f"Done!")

if __name__ == "__main__":

    if MLP_TEMPORARY_DATA_PATH.exists():
        MLP_TEMPORARY_DATA_PATH.unlink()

    if MLP_PREPROCESSED_DATA_PATH.exists():
        MLP_PREPROCESSED_DATA_PATH.unlink()

    numerical_stats = pd.read_csv(NUMERICAL_STATS_PATH, index_col=0)
    means = numerical_stats['mean'].values
    stds = numerical_stats['std'].values
    categorical_encoders = {}

    # Load categorical stats and fit one-hot encoders based on categorical_stats
    with open(CATEGORICAL_STATS_PATH, 'r') as f:
        categorical_stats = json.load(f)
    # Create one-hot encoders for each categorical feature
    for feature in MLP_CATEGORICAL_FEATURES:
        categorical_encoders[feature] = OneHotEncoder(categories=[categorical_stats[feature]],
                                                      sparse=False,
                                                           # sparse_output=False  #If error with sparse
                                                           handle_unknown="ignore")
        categorical_encoders[feature].fit(np.array(categorical_stats[feature]).reshape(-1, 1))




    mlp_preprocess_data('train', TRAIN_DATA_PATH, TEST_DATA_PATH, numerical_stats, categorical_encoders)
    mlp_preprocess_data('val', TRAIN_DATA_PATH, TEST_DATA_PATH, numerical_stats, categorical_encoders)
    mlp_preprocess_data('test', TRAIN_DATA_PATH, TEST_DATA_PATH, numerical_stats, categorical_encoders)

    shutil.copy(MLP_TEMPORARY_DATA_PATH, MLP_PREPROCESSED_DATA_PATH)


