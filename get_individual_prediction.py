
import os

import numpy as np
import torch
import tqdm
import pandas as pd


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    for layer in range(10, 20):
        model_path = f'runs/w2v_8s_es_layer{layer}'
        dataset_folder = f'test_feature_w2v2_xlsr_2b_layer{layer}'
        model = torch.jit.load(os.path.join(model_path, 'model_best.pt')).to(device)
        dataset_path = os.path.join('../../datasets/FinvCup2024/finvcup9th_2nd_ds2a', dataset_folder)
        filepaths = []
        labels_0 = []
        labels_1 = []
        all_paths = os.listdir(dataset_path)
        for filepath in tqdm.tqdm(all_paths, total=len(all_paths)):
            feature = torch.FloatTensor(np.load(os.path.join(dataset_path, filepath))).to(device).unsqueeze(0)
            label_0, label_1 = model(feature).squeeze().cpu().tolist()
            labels_0.append(label_0)
            labels_1.append(label_1)
            filepaths.append(filepath.replace('npy', 'wav'))
        results = pd.DataFrame([filepaths, labels_0, labels_1]).T
        results.to_csv(os.path.join(model_path, 'predictions.csv'), index=False, header=False)


if __name__ == '__main__':
    main()