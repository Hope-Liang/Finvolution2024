"""Dataset loader for LibriSpeech."""

import functools
import os
from typing import Callable, Optional, Sequence, Union

import gin
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import tqdm

# Disable tqdm globally.
# tqdm.tqdm.__init__ = functools.partialmethod(tqdm.tqdm.__init__, disable=True)


@gin.configurable
class DeepFake(Dataset):
    """The BVCC dataset."""
    
    def __init__(
        self,
        data_path: str = '../../datasets/FinvCup2024/finvcup9th_1st_ds5',
        features_folder: str = 'train_feature_w2v2',
        valid: str = 'train',
        train_on_all_data: bool = False,
        debug: bool = False,
        add_asvspoof: bool = False,
        add_cfad: bool = False,
        add_es: bool = False,
    ):
        self._data_path = data_path
        self._features_folder = features_folder
        self._train_on_all_data = train_on_all_data
        self._debug = debug
        self._valid = valid
        self._df = self._load_df(valid)
        self._num_samples = len(self._df)
        self._features, self._labels = self._load_clips()
        if add_asvspoof and valid == 'train':
            self._add_asvspoof()
        if add_cfad and valid == 'train':
            self._add_cfad()
        if add_es and valid == 'train':
            self._add_es()

    @property
    def features_shape(self) -> int:
        return self._features[0].shape

    def _load_df(self, valid: str) -> pd.DataFrame:
        filenames, labels = [], []
        with open(os.path.join(self._data_path, 'train_label.txt'), 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                if line:
                    filename, label = line.split(",")
                    filenames.append(filename)
                    labels.append(int(label))
                    if self._debug and len(filenames) == 3000:
                        break

        df = pd.DataFrame({'filenames': filenames, 'labels': labels})
        if self._train_on_all_data:
            return df

        shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        if valid == 'train':
            return df.iloc[:len(df)-1000]
        elif valid == 'val':
            return df.iloc[len(df)-1000:len(df)-10]
        return df.iloc[len(df)-10:]

    def _load_clips(self) -> list[np.ndarray]:
        """Loads the clips, applies augmentations (if so), and transforms to spectrograms"""
        features, labels = [], []
        for filename, label in tqdm.tqdm(
            zip(self._df['filenames'], self._df['labels']), total=self._num_samples, desc='Loading clips...',
        ):
            try:
                feature = np.load(os.path.join(self._data_path, self._features_folder, filename.replace('wav', 'npy')))
                features.append(feature)
                labels.append(label)
            except FileNotFoundError:
                print(f'Signal {filename} was not found.')

        return features, labels

    def _add_asvspoof(self):
        data_path = '../../datasets/FinvCup2024/ASVSpoof2021'
        df = pd.read_csv(os.path.join(data_path, 'ASVSpoof_part0_10k_labels.csv'))
        df = df.sample(frac=1.0).reset_index(drop=True)
        features_folder = self._features_folder.replace('train_', '')
        for path, label in tqdm.tqdm(zip(df['path'], df['label']), total=len(df), desc='ASVSpoof loading...'):
            filepath = os.path.join(
                data_path,
                features_folder,
                os.path.basename(path).replace('flac', 'npy')
            )
            try:
                feature = np.load(filepath)
            except OSError:
                continue
            self._features.append(feature)
            self._labels.append(label)
    
    def _add_cfad(self):
        data_path = '../../datasets/FinvCup2024/CFAD'
        df = pd.read_csv(os.path.join(data_path, 'data.csv'))
        df = df.sample(frac=0.25).reset_index(drop=True)
        features_folder = self._features_folder.replace('train_', '')
        for path, label in tqdm.tqdm(zip(df['path'], df['label']), total=len(df)):
            feature = np.load(os.path.join(
                data_path,
                features_folder,
                os.path.basename(path).replace('.wav','.npy')
            ))
            self._features.append(feature)
            self._labels.append(int(label))

    def _add_es(self):
        data_path = '../../datasets/FinvCup2024/Spanish'
        df = pd.read_csv(os.path.join(data_path, 'es_1k.csv'))
        df = df.sample(frac=1.0).reset_index(drop=True)
        for path, label in tqdm.tqdm(zip(df['path'], df['label']), total=len(df)):
            feature = np.load(os.path.join(
                data_path,
                self._features_folder,
                path.replace('/','_').replace('.wav','.npy').replace('.mp3', '.npy')
            ))
            self._features.append(feature)
            self._labels.append(int(label))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """Returns a spectrogram with label and augmentation applied."""
        return self._features[idx], self._labels[idx]
   
    def __len__(self) -> int:
        """Returns the number of speech clips in the dataset."""
        return len(self._features)
 
    def collate_fn(self, batch: list) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a batch consisting of tensors."""
        features, labels = zip(*batch)
        features = torch.FloatTensor(np.array(features))
        labels = torch.LongTensor(labels)
        return features, labels


@gin.configurable
def get_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    """Returns a dataloader of the dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )
