
import os

import numpy as np
import torch
import tqdm
import pandas as pd

import random
import librosa
from audio_generate import Audio
from transformers import AutoProcessor, Wav2Vec2BertModel, Wav2Vec2Model
import audio_generate
import torchaudio

_SAMPLING_RATE = 16000
_FRAME_SIZE = _SAMPLING_RATE*4
_HOP_SIZE = _SAMPLING_RATE*2

class Wav2Vec2MMS():
    def __init__(self, model_id =  "facebook/mms-300m"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2Model.from_pretrained(model_id).to(self.device)
        self.sampling_rate = 16_000
        self.feature_buffer = []
        for layer in self.model._modules['encoder']._modules['layers']:
            layer.register_forward_hook(self.layer_hook)

    def layer_hook(self, module, input, output):
        emb = output[0].cpu().detach().numpy().squeeze().T
        self.feature_buffer.append(emb)

    def extract_features(self, waveform):
        self.feature_buffer.clear()

        _ = self.model(waveform)
        return self.feature_buffer.copy()


def main():

    ssl_model_name = 'mms-1b' # 'w2v2_xlsr_2b', 'mms-1b'

    if ssl_model_name == 'w2v2_xlsr_2b':
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR_2B
        ssl_model = bundle.get_model().to(device=device)
    elif ssl_model_name == 'mms-1b':
        ssl_model = Wav2Vec2MMS("facebook/mms-1b")
    else:
        raise ValueError('')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    for layer in range(11, 12, 2):
        model_path = f'runs/w2v_4s_mms_layer{layer}'
        model = torch.jit.load(os.path.join(model_path, 'model_best.pt')).to(device)
        
        dataset_path = '../../datasets/FinvCup2024/finvcup9th_2nd_ds2a/test'
        all_paths = os.listdir(dataset_path)

        filepaths = []
        labels = []
        
        for filepath in tqdm.tqdm(all_paths, total=len(all_paths)):
            
            full_wav_path = os.path.join(dataset_path, filepath)
            audio_object = Audio.read_wav(full_wav_path).resample(_SAMPLING_RATE)

            audio_frames = Audio.process_to_frames(audio_object, _FRAME_SIZE, _HOP_SIZE)
            
            if ssl_model_name == 'w2v2_xlsr_2b':
                frame_labels = []
                with torch.inference_mode():
                    for audio_frame in audio_frames:
                        features, _ = ssl_model.extract_features(torch.Tensor(audio_frame.samples).to(device=device))
                        feature_layer = features[layer].squeeze().T.to(device).unsqueeze(0)
                        frame_labels.extend(np.argmax(model(feature_layer).cpu().tolist(), axis=1))
            if ssl_model_name == 'mms-1b':
                frame_labels = []
                for audio_frame in audio_frames:
                    features = ssl_model.extract_features(torch.Tensor(audio_frame.samples).to(device=device))
                    feature_layer = torch.FloatTensor(features[layer]).to(device).unsqueeze(0)
                    frame_labels.extend(np.argmax(model(feature_layer).cpu().tolist(), axis=1))

            labels.append(frame_labels) #np.array(frame_labels).min())
            filepaths.append(filepath)
        results = pd.DataFrame([filepaths, labels]).T
        results.to_csv(os.path.join(model_path, 'submit.csv'), index=False, header=False)


if __name__ == '__main__':
    main()
