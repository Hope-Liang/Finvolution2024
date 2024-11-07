
import pandas as pd
import numpy as np

csv_paths = [f'runs/w2v_8s_es_layer{i}/predictions.csv' for i in range(10, 20)]
#csv_paths = ['submit_ln_sigmoid_e15.csv', 'submit_wavlm.csv', 'submit_hubert.csv']
#csv_paths = ['submit_ln.csv', 'submit_sigmoid.csv', 'submit_ensemble15.csv']

def main():
    dfs = [pd.read_csv(csv_path, names=['paths', 'values_0', 'values_1']) for csv_path in csv_paths]
    
    combined_df = pd.concat(dfs).groupby('paths')[['values_0', 'values_1']].sum().reset_index()
    combined_df['values'] = (combined_df.values_0 < combined_df.values_1).astype(int)
    print(combined_df[['paths', 'values']].head())
    print(f'Num samples: {len(combined_df)}')
    print(sum(combined_df.values == 1), sum(combined_df.values == 0))
    combined_df[['paths', 'values']].to_csv('submit.csv', header=None, index=None)

if __name__ == '__main__':
    main()