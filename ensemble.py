
import pandas as pd
import numpy as np

csv_paths = [
    'runs/w2v_alldatacorrect_shufflecfad_layer12/submit.csv',
    'runs/w2v_alldatacorrect_shufflecfad_layer/submit.csv',
    'runs/w2v_wavlm_baseline_layer12/submit.csv',
    'runs/w2v_wavlm_baseline_layer16/submit.csv',
    'runs/w2v_wavlm_baseline_layer20/submit.csv',
]
csv_paths = [f'runs/w2v_alldatacorrect_shufflecfad_layer{i}/submit.csv' for i in range(12, 29, 4)]
#csv_paths = ['submit_ln_sigmoid_e15.csv', 'submit_wavlm.csv', 'submit_hubert.csv']
#csv_paths = ['submit_threedatasets.csv', 'submit_threedatasets_8s.csv', 'submit_threedatasets_balanced.csv']

def main():
    dfs = [pd.read_csv(csv_path, names=['paths', 'values']) for csv_path in csv_paths]
    majority = (len(csv_paths) + 1) // 2
    
    combined_df = pd.concat(dfs).groupby('paths')['values'].sum().reset_index()
    combined_df['values'] = combined_df['values'].apply(lambda x: x // majority)
    print(combined_df.head())
    print(f'Num samples: {len(combined_df)}')
    for df in dfs:
        combined = pd.merge(df, combined_df, on='paths')
        print(f'The same values {np.mean(combined.values_x == combined.values_y):.4f}')
    combined_df.to_csv('submit.csv', header=None, index=None)

if __name__ == '__main__':
    main()