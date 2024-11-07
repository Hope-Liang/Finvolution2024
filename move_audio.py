import os
import shutil
import tqdm


base_path_clean = '../../datasets/FinvCup2024/CFAD/clean_version'
base_path_codec = '../../datasets/FinvCup2024/CFAD/codec_version'
base_path_noisy = '../../datasets/FinvCup2024/CFAD/noisy_version'

def copy_wav_files(src_root, dest_dir):
    # Ensure the destination directory exists
    
    # Walk through the source directory
    for root, dirs, files in tqdm.tqdm(os.walk(src_root), total=len(os.listdir(src_root))+1):
        for file in files:
            if file.endswith('.wav'):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, file)
                
                # If the destination file already exists, append a counter to the filename
                if os.path.exists(dest_path):
                    print(f'{dest_path} already exists')
                    continue
                
                # Copy the file to the destination directory
                shutil.copy2(src_path, dest_path)

# Example usage
src_root = os.path.join(base_path_clean, 'test_unseen_clean', 'real_clean')
dest_dir = '../../datasets/FinvCup2024/CFAD/test_real'

copy_wav_files(src_root, dest_dir)

