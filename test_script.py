from audio_generate import Audio


_REAL_RECORDINGS_BASE_PATH = '../../datasets/FinvCup2024/finvcup9th_1st_ds5/train'
real_recordings = []
with open('../../datasets/FinvCup2024/finvcup9th_1st_ds5/train_label.txt') as f:
    lines = f.read().splitlines()
    for line in lines:
        if line:
            path, label = line.split(',')
            if label == ' 1':
                print(path)
                sig = Audio.read_wav(
                    os.path.join(_REAL_RECORDINGS_BASE_PATH, real_recordings_paths)
                )
                sig = sig.resample(16000)
                if sig.duration > 4:
                    sig = sig.repetitive_crop(_TARGET_LENGTH // 2)
                real_recordings.append(sig)