from libsvm.svmutil import *
import numpy as np
import pickle
import os
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from string import ascii_uppercase
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
mapping = {float(k): v for k, v in enumerate(ascii_uppercase)}


# def pad_audio(data, fs, T=3):
#     # Calculate target number of samples
#     N_tar = int(fs * T)
#     # Calculate number of zero samples to append
#     shape = data.shape
#     # Create the target shape
#     N_pad = N_tar - shape[0]
#     # print("Padding with %s seconds of silence" % str(N_pad/fs) )
#     shape = (N_pad,) + shape[1:]
#     # Stack only if there is something to append
#     if shape[0] > 0:
#         if len(shape) > 1:
#             return np.vstack((np.zeros(shape), data))
#         else:
#             return np.hstack((np.zeros(shape), data))
#     else:
#         return data


# def load_audio(audio_file):

#     samplerate, data = wavfile.read(audio_file)
#     return pad_audio(data, samplerate)


# def extract_features(audio_data):
#     data = mfcc(audio_data, samplerate=44000, nfft=2048, winfunc=np.hamming)
#     return data.ravel()



# # prediction = loaded_classifier.predict(features.flatten().reshape(1,-1))


# rootdir = "data_by_subject/"

# all_results = {}
# letters = {}
# print("Loading audio...")
# file_count = sum(len(files) for _, _, files in os.walk(rootdir))
# with tqdm(total=file_count) as pbar:
#     for subdir, dirs, files in os.walk(rootdir):
#         if "6" in subdir or "7" in subdir or "8" in subdir:

#             all_features = []

#             for file in files:
#                 pbar.update(1)
#                 sample_rate, audio = wavfile.read(os.path.join(subdir, file))
#                 data = pad_audio(audio, sample_rate)
#                 label = os.path.join(subdir, file).split("/")[2]
#                 features = extract_features(data)
#                 output = [
#                     mapping[k] for k in model.predict(features.flatten().reshape(1, -1))
#                 ]
#                 if all_results.get(subdir.split("/")[1][-1]):
#                     if all_results[subdir.split("/")[1][-1]].get(label):
#                         all_results[subdir.split("/")[1][-1]][label].append(output[0])
#                     else:
#                         all_results[subdir.split("/")[1][-1]][label] = [output[0]]
#                 else:
#                     all_results[subdir.split("/")[1][-1]] = {label: [output[0]]}

# print(all_results)
# for k, v in all_results.items():
#     for k1, v1 in v.items():
#         print(f"{k1} - {v1.count(k1) / len(v1)}")

def pad_audio(data, fs, T=2.89):
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


def load_audio():
    rootdir = "data_by_subject/"
    letters = {}
    print("Loading audio...")
    file_count = sum(len(files) for _, _, files in os.walk(rootdir))
    print(file_count)
    length_sum = 0
    longest = 0
    with tqdm(total=file_count) as pbar:
        for subdir, dirs, files in tqdm(os.walk(rootdir)):
            if (
                "8" in subdir
                

            ):
                for file in files:
                    pbar.update(1)
                    sample_rate, audio = wavfile.read(os.path.join(subdir, file))
                    length_sum += len(audio) / float(sample_rate)
                    if len(audio) / float(sample_rate) > longest:
                        longest = len(audio) / float(sample_rate)

                    data = pad_audio(audio, sample_rate)
                    label = os.path.join(subdir, file).split("/")[2]
                    if letters.get(label):
                        letters[label].append(data.ravel())
                    else:
                        letters[label] = [data]
    print(f"longest sample - {longest}")
    print(f"average {length_sum/780}")
    return letters


#characters = load_audio()
mapping = {v: k for k, v in enumerate(ascii_uppercase)}
inv_map = {v: k for k, v in mapping.items()}


def extract_features(audio_data):

    # Remember that the audio data consists of raw audio wave followed by sample rate
    # so we need to only take the raw audio wave.
    output, label_output = [], []
    print("Extracting features")

    for k, v in tqdm(audio_data.items()):
        for i in v:
            label_output.append(k)
            data = mfcc(i, samplerate=44100, nfft=2048, winfunc=np.hamming)
            output.append(data.ravel())

    label_output = [mapping[k] for k in label_output]
    label_output = np.array(label_output)
    return output, label_output

model = pickle.load(open("classifier_all.pkl", "rb"))
# Define a function to load the raw audio files

#character_features, labels = extract_features(characters)
X_test, Y = pickle.load(open("../Accent-Recognition-2019-master/unseen-data", "rb"))
#Y = labels

y_pred = model.predict(X_test)
print(y_pred)
from sklearn.metrics import confusion_matrix, accuracy_score
print(accuracy_score(Y, y_pred))

df_cm = pd.DataFrame(
    confusion_matrix(Y, y_pred),
    index=[i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
    columns=[i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
)
plt.figure(figsize=(10, 7))
all_map = sns.heatmap(df_cm, annot=True)

fig = all_map.get_figure()
fig.savefig("new_map.png")