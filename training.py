import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import scipy.interpolate as interpol
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
from sklearn.preprocessing import StandardScaler, StandardScaler
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


def normalize(inSig,outLen):
	#This function normalizes the audio signal.
	#It first produces an interp1d structure that readily interpolates between points
	#Then it sets the size of the space to outLen=200000 points, and interp1d interpolates to fill in gaps
	#In essence, it takes every audio signal and produces a signal with outLen=200000 data points in it = normalization
    inSig = np.array(inSig)
    arrInterpol = interpol.interp1d(np.arange(inSig.size),inSig)
    arrOut = arrInterpol(np.linspace(0,inSig.size-1,outLen))
    return arrOut


def load_audio():
    rootdir = "data_by_subject/"
    letters = {}
    print("Loading audio...")
    file_count = sum(len(files) for _, _, files in os.walk(rootdir))
    print(file_count)
    with tqdm(total=file_count) as pbar:
        for subdir, dirs, files in tqdm(os.walk(rootdir)):
            if (
                "1" in subdir
                or "2" in subdir
                or "3" in subdir
                or "4" in subdir
                or "5" in subdir
            ):
                for file in files:
                    pbar.update(1)
                    sample_rate, audio = wavfile.read(os.path.join(subdir, file))
                    newSig = []
                    for i in range(len(audio)):newSig.append(audio[i])
                    newSig = normalize(newSig,200000)
                    data = pad_audio(newSig, sample_rate)
                    label = os.path.join(subdir, file).split("/")[2]
                    if letters.get(label):
                        letters[label].append(data.ravel())
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

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


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


import joblib

scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename)

custom_dump_svmlight_file(X_train, y_train, "training_data")
custom_dump_svmlight_file(X_test, y_test, "test_data")


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


# all_params = grid(X_train, y_train)
all_params = {"C": 32.0, "gamma": 0.0001220703125}


all_matrix = training(
    (X_train, y_train), (X_test, y_test), "classifier_all.pkl", all_params
)


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




fig = all_map.get_figure()
fig.savefig("all_map.png")
