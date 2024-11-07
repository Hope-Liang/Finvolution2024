
import os

import numpy as np
import torch
import tqdm
import pandas as pd


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    for layer in range(10, 11, 2):
        model_path = f'runs/w2v_8s_aug_layer{layer}'
        dataset_folder = f'test_feature_w2v2_xlsr_2b_layer{layer}'
        model = torch.jit.load(os.path.join(model_path, 'model_best.pt')).to(device)
        dataset_path = os.path.join('../../datasets/FinvCup2024/finvcup9th_2nd_ds2a', dataset_folder)
        filepaths = []
        labels = []
        all_paths = os.listdir(dataset_path)
        for filepath in tqdm.tqdm(all_paths, total=len(all_paths)):
            feature = torch.FloatTensor(np.load(os.path.join(dataset_path, filepath))).to(device).unsqueeze(0)
            labels.extend(np.argmax(model(feature).cpu().tolist(), axis=1))
            filepaths.append(filepath.replace('npy', 'wav'))
        results = pd.DataFrame([filepaths, labels]).T
        results.to_csv(os.path.join(model_path, 'submit.csv'), index=False, header=False)


if __name__ == '__main__':
    main()
