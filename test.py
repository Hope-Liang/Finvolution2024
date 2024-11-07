"""Test latent representation classification."""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors

import audio
import utils


class ESC50:
    
    def __init__(self, directory: str = '../../datasets/esc50'):
        self._directory = directory
        self._csv = pd.read_csv(os.path.join(directory, 'esc50.csv'))

        # Potential subset of categories that makes sense in meetings?
        self._categories = ['dog', 'vacuum_cleaner',
         'clapping', 'mouse_click', 'pouring_water', 'clock_alarm',
         'keyboard_typing', 'wind', 'brushing_teeth', 'drinking_sipping',
         'rain', 'laughing', 'breathing', 'crying_baby', 'coughing',
         'toilet_flush', 'washing_machine', 'sneezing', 'cat',
         'door_wood_creaks']

        # TODO: Add support of 5-class classification.
        self._map_class = {'dog': 'animal', 'chirping_birds': 'natural', 'vacuum_cleaner': 'interior',
                     'thunderstorm': 'natural', 'door_wood_knock': 'interior', 'can_opening': 'interior',
                     'crow': 'animal', 'clapping': 'human', 'fireworks': 'exterior', 'chainsaw': 'exterior',
                     'airplane': 'exterior', 'mouse_click': 'interior', 'pouring_water': 'natural',
                     'train': 'exterior', 'sheep': 'animal', 'water_drops': 'natural',
                     'church_bells': 'exterior', 'clock_alarm': 'interior', 'keyboard_typing': 'interior',
                     'wind': 'natural', 'footsteps': 'human', 'frog': 'animal', 'cow': 'animal',
                     'brushing_teeth': 'human', 'car_horn': 'exterior', 'crackling_fire': 'natural',
                     'helicopter': 'exterior', 'drinking_sipping': 'human', 'rain': 'natural', 'insects': 'animal',
                     'laughing': 'human', 'hen': 'animal', 'engine': 'exterior', 'breathing': 'human',
                     'crying_baby': 'human', 'hand_saw': 'exterior', 'coughing': 'human',
                     'glass_breaking': 'interior', 'snoring': 'human', 'toilet_flush': 'natural',
                     'pig': 'animal', 'washing_machine': 'interior', 'clock_tick': 'interior', 'sneezing': 'human',
                     'rooster': 'animal', 'sea_waves': 'natural', 'siren': 'exterior', 'cat': 'animal',
                     'door_wood_creaks': 'interior', 'crickets': 'natural'}

        self._unique_categories = self._csv.category.unique()

        self._idx2class = {i: class_cur for i, class_cur in enumerate(self._unique_categories)}
        self._class2idx = {class_cur: i for i, class_cur in enumerate(self._unique_categories)}
        self._count_classes = {class_cur: (
            self._csv.category == class_cur).sum() for i, class_cur in enumerate(self._unique_categories)}

        self._mag_specs, self._label_encoded = self._load_data()

    def _load_data(self):
        mag_specs, label_encoded = [], []
        for path, class_cur in tqdm.tqdm(
            zip(self._csv.filename, self._csv.category), total=len(self._csv)):

            fullpath = os.path.join(self._directory, 'audio', 'audio', path)
            sig = audio.Audio.read_wav(fullpath).resample(16000)
            sig = sig.repetitive_crop(2 * int(sig.duration * sig.rate))
            spec = utils.stft(np.squeeze(sig.samples))
            mag_specs.append(spec)
            label_encoded.append(self._class2idx[class_cur])
        return mag_specs, label_encoded

    def get_latent_representations(
        self,
        model: Any,
        latent_dim: int,
        test_size: float = 0.3,
        device: str = 'cpu',
    ):
        model.eval()
        num_examples = len(self._csv)
        features = np.zeros((num_examples, latent_dim))
        z = np.zeros((num_examples, len(self._unique_categories)))

        for i, (spec, label) in enumerate(zip(self._mag_specs, self._label_encoded)):
            x = torch.from_numpy(np.expand_dims(spec, (0, 1)))
            x = x.type(torch.float32)
            x = x.to(device)
            features[i, :] = model.encoder(x).cpu().detach().numpy().flatten()
            z[i, label] = 1

        X = np.array(features)
        X_train, X_val, z_train, z_val = sklearn.model_selection.train_test_split(
            X, z, test_size=test_size, random_state=0, stratify=z)
        return X_train, X_val, z_train, z_val

    def knn(self,
            *,
            X_train: np.ndarray,
            z_train: np.ndarray,
            X_val: np.ndarray,
            z_val: np.ndarray,
            n_neighbors: int,
            save_confusion_matrix: bool = False,
            save_dir: str = '',
           ):
        kNN_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
        kNN_model.fit(X_train, np.argmax(z_train, axis=1))

        targets = np.argmax(z_val, axis=1)
        predictions = kNN_model.predict_proba(X_val)
        top_1 = sklearn.metrics.top_k_accuracy_score(targets, predictions, k=1)
        top_3 = sklearn.metrics.top_k_accuracy_score(targets, predictions, k=3)
        if save_confusion_matrix and save_dir:
            self._save_confusion_matrix(X_val, z_val, kNN_model, save_dir)
        return top_1, top_3
    
    def _save_confusion_matrix(
        self,
        X: np.ndarray,
        z: np.ndarray,
        model: Any,
        save_dir: str,
    ) -> None:
        predictions = model.predict(X)
        targets = np.argmax(z, axis=1)
        cm = sklearn.metrics.confusion_matrix(targets, predictions)
        target_names = []
        for i in targets:
            if self._idx2class[i] not in target_names:
                target_names.append(self._idx2class[i])

        #plt.rcParams["figure.figsize"] = [40, 15]
        #plt.rcParams["figure.autolayout"] = True
        plt.imshow(cm, interpolation='none', cmap='Blues')
        for (i, j), z in np.ndenumerate(cm):
            if z != 0:
                plt.text(j, i, round(z, 2), ha='center', va='center')
        plt.xlabel("Actual label")
        plt.xticks([i for i in range(len(target_names))], target_names, rotation=90)
        plt.yticks([i for i in range(len(target_names))], target_names, rotation=0)
        plt.ylabel("Predicted label")
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
