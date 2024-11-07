"""Generates SSL features of audio.

Example usage:
python generate_ssl_features.py --model="w2v2_xlsr_2b" --split="train" --target_duration=8
"""

import argparse
import random
import torch
import numpy as np
import os
import librosa
import tqdm
from audio_generate import Audio
import audio_generate
from transformers import AutoProcessor, Wav2Vec2BertModel, Wav2Vec2Model
import torchaudio


parser = argparse.ArgumentParser(description='Model and target length')
parser.add_argument('--model', type=str, help='SSL model to use.')
parser.add_argument('--split', type=str, help='Train or test')
parser.add_argument('--target_duration', type=float, help='Target duration of the clips, in seconds.')
parser.add_argument('--folder', type=str, help='Data folder to be used.')
parser.add_argument('--batch', type=int, help='Batch number')
parser.add_argument('--num_batches', type=int, help='Number of batches.')
args = parser.parse_args()

model_name = args.model # 'w2v2', 'w2vb2', 'w2v2_xlsr_300m', 'w2v2_xlsr_1b', 'w2v2_xlsr_2b', 'hubert_xlarge', 'wavlm_large', 'mms-300m', 'mms-1b'
split = args.split
folder = args.folder # 'finvcup9th_1st_ds5', 'finvcup9th_2nd_ds2a'

_DEFAULT_PATH = f'../../datasets/FinvCup2024/{folder}/{split}'
_DEFAULT_FEATURE_PATH = f'../../datasets/FinvCup2024/{folder}/{split}_feature_{model_name}'
_SAMPLING_RATE = 16000
_TARGET_LENGTH = int(_SAMPLING_RATE * args.target_duration)
_LAYERS_TO_USE = list(range(10, 20))
_PROB_CONCAT = 0.1
_PROB_NOISE = 0.1
_AUGMENT = True
_CONCATENATE = False

if split == 'train' and _AUGMENT and _CONCATENATE:
    _REAL_RECORDINGS_BASE_PATH = '../../datasets/FinvCup2024/finvcup9th_1st_ds5/train'
    real_recordings = []
    with open('../../datasets/FinvCup2024/finvcup9th_1st_ds5/train_label.txt') as f:
        lines = f.read().splitlines()
        for line in lines:
            if line:
                path, label = line.split(',')
                if int(label) == 1:
                    sig = Audio.read_wav(
                        os.path.join(_REAL_RECORDINGS_BASE_PATH, path)
                    )
                    sig = sig.resample(16000)
                    if sig.duration > 4:
                        sig = sig.repetitive_crop(_TARGET_LENGTH // 2)
                    real_recordings.append(sig)


    _ESC50_PATH = '../../datasets/esc50/audio/audio/16000'
    esc50_sigs = [Audio.read_wav(os.path.join(_ESC50_PATH, path)) for path in os.listdir(_ESC50_PATH)]

wav_paths = os.listdir(_DEFAULT_PATH)
wav_paths.sort()
clips_per_batch = len(wav_paths) // args.num_batches
if args.batch + 1 != args.num_batches:
    wav_paths = wav_paths[args.batch * clips_per_batch: (args.batch+1) * clips_per_batch]
else:
    wav_paths = wav_paths[args.batch * clips_per_batch:]

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model_name == 'w2vb2':
    processor = AutoProcessor.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")
    model = Wav2Vec2BertModel.from_pretrained("hf-audio/wav2vec2-bert-CV16-en").to(device=device)
    target_tensor_shape = [1024, 200]
elif model_name == 'w2v2':
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device=device)
    target_tensor_shape = [768, 399]
elif model_name == 'w2v2_xlsr_300m':
    bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
    model = bundle.get_model().to(device=device)
    target_tensor_shape = [1024, 399]
elif model_name == 'w2v2_xlsr_1b':
    bundle = torchaudio.pipelines.WAV2VEC2_XLSR_1B
    model = bundle.get_model().to(device=device)
    target_tensor_shape = [1280, 399]
elif model_name == 'w2v2_xlsr_2b':
    bundle = torchaudio.pipelines.WAV2VEC2_XLSR_2B
    model = bundle.get_model().to(device=device)
    target_tensor_shape = [1920, 399]
elif model_name == 'hubert_xlarge':
    bundle = torchaudio.pipelines.HUBERT_XLARGE
    model = bundle.get_model().to(device=device)
    target_tensor_shape = [1280, 399]
elif model_name == 'wavlm_large':
    bundle = torchaudio.pipelines.WAVLM_LARGE
    model = bundle.get_model().to(device=device)
    target_tensor_shape = [1024, 399]
elif model_name == 'mms-300m':
    model = Wav2Vec2MMS("facebook/mms-300m")
    target_tensor_shape = [1024, 399]
elif model_name == 'mms-1b':
    model = Wav2Vec2MMS("facebook/mms-1b")
    target_tensor_shape = [1280, 399]
else:
    raise ValueError()

for wav_path in tqdm.tqdm(wav_paths):
    full_wav_path = os.path.join(_DEFAULT_PATH, wav_path)
    wav_name = wav_path[:-4]
    full_feature_path = os.path.join(_DEFAULT_FEATURE_PATH, wav_name+'.npy')
    audio_object = Audio.read_wav(full_wav_path).resample(_SAMPLING_RATE)
    audio_object = audio_object.peak_normalize()
    if split == 'train' and _AUGMENT:
        gain = random.uniform(0.3, 1.0)
        audio_object = audio_object.scale(gain)
        if random.random() < 0.1:
            audio_object = audio_object.resample(random.uniform(8000, 16000)).resample(16000)
        if _CONCATENATE:
            drawn_value = random.random()
            real_speech = random.choice(real_recordings)
            real_noise = random.choice(esc50_sigs)
            if drawn_value < _PROB_CONCAT / 2:
                audio_object = audio_generate.concatenate(audio_object, real_speech)
            elif drawn_value < _PROB_CONCAT:
                audio_object = audio_generate.concatenate(real_speech, audio_object)
            elif drawn_value < _PROB_CONCAT + _PROB_NOISE / 2:
                audio_object = audio_generate.concatenate(audio_object, real_noise)
            elif drawn_value < _PROB_CONCAT + _PROB_NOISE:
                audio_object = audio_generate.concatenate(real_noise, audio_object)
    audio_object = audio_object.repetitive_crop(_TARGET_LENGTH)

    if model_name in ['w2vb2', 'w2v2']:
        inputs = processor(audio_object.samples, sampling_rate=_SAMPLING_RATE, return_tensors="pt").to(device=device)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        features = last_hidden_states.squeeze().T
        if list(features.shape) != target_tensor_shape:
            break
        np.save(full_feature_path, features.cpu().numpy())
    elif model_name in ['w2v2_xlsr_300m', 'w2v2_xlsr_1b', 'w2v2_xlsr_2b', 'hubert_xlarge', 'wavlm_large']:
        with torch.inference_mode():
            features, _ = model.extract_features(torch.Tensor(audio_object.samples).to(device=device))
        # print(len(features))
        # print(features[0].shape)
        # break
        for i in range(len(features)):
            if i not in _LAYERS_TO_USE:
                continue
            features[i] = features[i].squeeze().T
            folder_path_i = _DEFAULT_FEATURE_PATH+'_layer'+str(i)
            if not os.path.exists(folder_path_i):
                os.makedirs(folder_path_i)
            full_feature_path_i = os.path.join(folder_path_i, wav_name+'.npy')
            np.save(full_feature_path_i, features[i].cpu().numpy())
    elif model_name in ['mms-300m', 'mms-1b']:
        features = model.extract_features(torch.Tensor(audio_object.samples).to(device=device))
        # print(len(features))
        # print(features[0].shape)
        # break
        for i in range(len(features)):
            if i not in _LAYERS_TO_USE:
                continue
            folder_path_i = _DEFAULT_FEATURE_PATH+'_layer'+str(i)
            if not os.path.exists(folder_path_i):
                os.makedirs(folder_path_i)
            full_feature_path_i = os.path.join(folder_path_i, wav_name+'.npy')
            np.save(full_feature_path_i, features[i])

