from libsvm.svmutil import *
import numpy as np
import pickle
import os
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from string import ascii_uppercase

mapping = {float(k): v for k, v in enumerate(ascii_uppercase)}


def pad_audio(data, fs, T=3):
    # Calculate target number of samples
    N_tar = int(fs * T)
    # Calculate number of zero samples to append
    shape = data.shape
    # Create the target shape
    N_pad = N_tar - shape[0]
    # print("Padding with %s seconds of silence" % str(N_pad/fs) )
    shape = (N_pad,) + shape[1:]
    # Stack only if there is something to append
    if shape[0] > 0:
        if len(shape) > 1:
            return np.vstack((np.zeros(shape), data))
        else:
            return np.hstack((np.zeros(shape), data))
    else:
        return data


def load_audio(audio_file):

    samplerate, data = wavfile.read(audio_file)
    return pad_audio(data, samplerate)


def extract_features(audio_data):
    data = mfcc(audio_data, samplerate=44100, nfft=2048, winfunc=np.hamming)
    return data.ravel()


model = pickle.load(open("classifier_all.pkl", "rb"))
# prediction = loaded_classifier.predict(features.flatten().reshape(1,-1))


rootdir = "data_by_subject/"

all_results = {}
letters = {}
print("Loading audio...")
file_count = sum(len(files) for _, _, files in os.walk(rootdir))
with tqdm(total=file_count) as pbar:
    for subdir, dirs, files in os.walk(rootdir):
        if "6" in subdir or "7" in subdir or "8" in subdir:

            all_features = []

            for file in files:
                pbar.update(1)
                sample_rate, audio = wavfile.read(os.path.join(subdir, file))
                data = pad_audio(audio, sample_rate)
                label = os.path.join(subdir, file).split("/")[2]
                features = extract_features(data)
                output = [
                    mapping[k] for k in model.predict(features.flatten().reshape(1, -1))
                ]
                if all_results.get(subdir.split("/")[1][-1]):
                    if all_results[subdir.split("/")[1][-1]].get(label):
                        all_results[subdir.split("/")[1][-1]][label].append(output[0])
                    else:
                        all_results[subdir.split("/")[1][-1]][label] = [output[0]]
                else:
                    all_results[subdir.split("/")[1][-1]] = {label: [output[0]]}

print(all_results)
for k, v in all_results.items():
    for k1, v1 in v.items():
        print(f"{k1} - {v1.count(k1) / len(v1)}")
