import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window

import IPython.display as ipd

from tqdm import tqdm
from python_speech_features import mfcc
from string import ascii_uppercase
import boto3
import os

import pickle
import itertools


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


def load_audio():
    rootdir = "data_by_subject/"
    letters = {}
    print("Loading audio...")
    file_count = sum(len(files) for _, _, files in os.walk(rootdir))
    print(file_count)
    with tqdm(total=file_count) as pbar:
        for subdir, dirs, files in tqdm(os.walk(rootdir)):
            if '1' in subdir or '2' in subdir or '3' in subdir or '4' in subdir or '5' in subdir:
                for file in files:
                    pbar.update(1)
                    sample_rate, audio = wavfile.read(os.path.join(subdir, file))
                    print(sample_rate, audio)
                    data = pad_audio(audio, sample_rate)
                    label = os.path.join(subdir, file).split("/")[2]
                    if letters.get(label):
                        letters[label].append(data)
                    else:
                        letters[label] = [data]
    return letters


# How to use load_audio() function
characters = load_audio()
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
            data = mfcc(i, samplerate=44000, nfft=2048, winfunc=np.hamming)
            output.append(data.ravel())

    label_output = [mapping[k] for k in label_output]
    label_output = np.array(label_output)
    return output, label_output


# Define a function to load the raw audio files

character_features, labels = extract_features(characters)


X = character_features
Y = labels


group_1 = ["C", "G", "I", "J", "L", "M", "N", "O", "S", "U", "V", "W", "Z"]
group_2 = ["A", "B", "D", "K", "P", "Q", "R", "T", "X", "Y"]
group_3 = ["E", "F", "H"]

X_1, X_2, X_3 = [], [], []
Y_1, Y_2, Y_3 = [], [], []

for i, label in enumerate(Y):
    if inv_map[label] in group_1:
        X_1.append(X[i])
        Y_1.append(label)
    if inv_map[label] in group_2:
        X_2.append(X[i])
        Y_2.append(label)
    if inv_map[label] in group_3:
        X_3.append(X[i])
        Y_3.append(label)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=30, stratify=Y
)
# X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
#     X_1, Y_1, test_size=0.3, random_state=30, stratify=Y_1
# )
# X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
#     X_2, Y_2, test_size=0.3, random_state=30, stratify=Y_2
# )
# X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
#     X_3, Y_3, test_size=0.3, random_state=30, stratify=Y_3
# )


def custom_dump_svmlight_file(X_train, Y_train, filename):

    # This function inserts the extracted features in the libsvm format
    featinds = [" " + str(i) + ":" for i in range(1, len(X_train[0]) + 1)]
    with open(filename, "w") as f:
        for ind, row in enumerate(X_train):
            f.write(
                str(Y_train[ind])
                + " "
                + "".join(
                    [
                        x
                        for x in itertools.chain.from_iterable(
                            zip(featinds, map(str, row))
                        )
                        if x
                    ]
                )
                + "\n"
            )


custom_dump_svmlight_file(X_train, y_train, "training_data")
custom_dump_svmlight_file(X_test, y_test, "test_data")
# custom_dump_svmlight_file(X_train_1, y_train_1, "training_data_1")
# custom_dump_svmlight_file(X_test_1, y_test_1, "test_data_1")
# custom_dump_svmlight_file(X_train_2, y_train_2, "training_data_2")
# custom_dump_svmlight_file(X_test_2, y_test_2, "test_data_2")
# custom_dump_svmlight_file(X_train_3, y_train_3, "training_data_3")
# custom_dump_svmlight_file(X_test_3, y_test_3, "test_data_3")

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


def training(training_data, test_data, file_output_name, params):
    X_train, Y_train = training_data
    X_test, Y_test = test_data
    clf = svm.SVC(kernel="rbf", C=params["C"], gamma=params["gamma"])
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)
    print(clf.score(X_test, Y_test))
    # print(confusion_matrix(Y_test, pred))
    # print(classification_report(Y_test, pred))
    # save the classifier

    with open(file_output_name, "wb") as fid:
        pickle.dump(clf, fid)

    return confusion_matrix(Y_test, pred)


def grid(X, y):
    C_range = np.logspace(-2, 10, 13, 32)
    gamma_range = np.logspace(-9, 3, 13, 32)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=7, test_size=0.3, random_state=42)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)

    print(
        "The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_)
    )
    return grid.best_params_

all_params = grid(X_train, y_train)
# params_1 = grid(X_train_1, y_train_1)
# params_2 = grid(X_train_2, y_train_2)
# params_3 = grid(X_train_3, y_train_3)

all_matrix = training(
    (X_train, y_train), (X_test, y_test), "classifier_all.pkl", all_params
)
# matrix_1 = training(
#     (X_train_1, y_train_1), (X_test_1, y_test_1), "classifier_1.pkl", params_1
# )
# matrix_2 = training(
#     (X_train_2, y_train_2), (X_test_2, y_test_2), "classifier_2.pkl", params_2
# )
# matrix_3 = training(
#     (X_train_3, y_train_3), (X_test_3, y_test_3), "classifier_3.pkl", params_3
# )

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df_cm = pd.DataFrame(
    all_matrix,
    index=[i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
    columns=[i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
)
plt.figure(figsize=(10, 7))
all_map = sns.heatmap(df_cm, annot=True)


# df_cm = pd.DataFrame(
#     matrix_1, index=[i for i in "".join(group_1)], columns=[i for i in "".join(group_1)]
# )
# plt.figure(figsize=(10, 7))
# map_1 = sns.heatmap(df_cm, annot=True)

# df_cm = pd.DataFrame(
#     matrix_2, index=[i for i in "".join(group_2)], columns=[i for i in "".join(group_2)]
# )
# plt.figure(figsize=(10, 7))
# map_2 = sns.heatmap(df_cm, annot=True)

# df_cm = pd.DataFrame(
#     matrix_3, index=[i for i in "".join(group_3)], columns=[i for i in "".join(group_3)]
# )
# plt.figure(figsize=(10, 7))
# map_3 = sns.heatmap(df_cm, annot=True)


fig = all_map.get_figure()
fig.savefig("all_map.png")
# fig = map_1.get_figure()
# fig.savefig("map_1.png")
# fig = map_2.get_figure()
# fig.savefig("map_2.png")
# fig = map_3.get_figure()
# fig.savefig("map_3.png")
