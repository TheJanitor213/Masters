import sys
import warnings
import random

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os
from string import ascii_uppercase
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import scipy.interpolate as interpol
import glob
import wave
import pickle
import sklearn
import scipy.io.wavfile as wav

import numpy as np
from time import time
from sklearn.metrics import classification_report

import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import subprocess


import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from scipy import signal
import itertools
from subprocess import Popen

gnuplot_exe = r"C:\Program Files (x86)\gnuplot\bin\gnuplot.exe"
grid_py = r"C:\Users\bruce\Documents\Personal-projects\backup\gpu\tools\grid.py"
svmtrain_exe = (
    r"C:\Users\bruce\Documents\Personal-projects\backup\gpu\windows\svm-train-gpu.exe"
)
svmpredict_exe = (
    r"C:\Users\bruce\Documents\Personal-projects\backup\gpu\windows\svm-predict.exe"
)
crange = "-5,13,2"  # "1,5,2"
grange = "-15,-8,2"  # "-3,2,2"


def paramsfromexternalgridsearch(filename, crange, grange, printlines=False):
    # printlines specifies whether or not the function should print every line of the grid search verbosely
    cmd = 'python "{0}" -log2c {1} -log2g {2} -svmtrain "{3}" -gnuplot "{4}" -png grid.png "{5}"'.format(
        grid_py, crange, grange, svmtrain_exe, gnuplot_exe, filename
    )
    f = Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout

    line = ""
    while True:
        last_line = line
        line = f.readline()
        if not line:
            break
        if printlines:
            print(line)
    c, g, rate = map(float, last_line.split())
    return c, g, rate


def accuracyfromexternalpredict(
    scaled_test_file, model_file, predict_test_file, predict_output_file
):
    cmd = '"{0}" "{1}" "{2}" "{3}"'.format(
        svmpredict_exe, scaled_test_file, model_file, predict_test_file
    )
    f = Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    # f = subprocess.Popen(cmd, shell = True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

    line = ""
    while True:
        last_line = line
        line = f.readline()
        if not line:
            break

    return (
        last_line.split(" ")[3][1:-1].split("/")[0],
        last_line.split(" ")[3][1:-1].split("/")[1],
    )


def normalize(inSig, outLen):
    # This function normalizes the audio signal.
    # It first produces an interp1d structure that readily interpolates between points
    # Then it sets the size of the space to outLen=200000 points, and interp1d interpolates to fill in gaps
    # In essence, it takes every audio signal and produces a signal with outLen=200000 data points in it = normalization
    # inSig = np.array(inSig)
    arrInterpol = interpol.interp1d(np.arange(inSig.size), inSig)
    arrOut = arrInterpol(np.linspace(0, inSig.size - 1, outLen))
    return arrOut


def justpadwithzeros(inSig, rate):
    # >> > a = [1, 2, 3, 4, 5]
    # >> > np.pad(a, (2, 3), 'constant', constant_values=(4, 6))
    # array([4, 4, 1, 2, 3, 4, 5, 6, 6, 6])
    # longest is 267264
    maxlength = rate * 6.1  # 6.1s long which comes to 269010 samples at this rate
    return np.pad(inSig, (0, int(maxlength - inSig.shape[0])), mode="constant")


def writetopcklfile(outpath, data):
    with open(outpath, "wb") as f:
        pickle.dump(data, f)


def readfrompcklfile(outpath):
    with open(outpath, "rb") as f:
        return pickle.load(f)


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


def main():
    start = time()

    path = r"data_by_subject"
    files = os.listdir(path)

    features = []
    label = []
    Filenames = {}
    X = None
    Xdir = {}
    y = []

    y_unseen = []
    X_unseen = None

    from tqdm import tqdm

    # The upper bound for the data set samples is 2.89 seconds so later just pad every sequence with
    # zeros up to this length
    rate = 44100
    maxlength = rate * 2.89
    counter = 0
    unseen_counter = 0
    file_count = sum(len(files) for _, _, files in os.walk(path))
    mapping = {v: k for k, v in enumerate(ascii_uppercase)}
    inv_map = {v: k for k, v in mapping.items()}
    subjects = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    choices = ["1", "2", "3", "4", "5"]
    print(choices)
    with tqdm(total=file_count) as pbar:
        for subdir, dirs, files in tqdm(os.walk("data_by_subject/")):
            file_counter = 0
            for soundfile in files:
                if file_counter == 0 or file_counter == 9:
                    file_counter += 1
                    continue
                pbar.update(1)
                if soundfile.endswith(".wav"):
                    (rate, sig) = wav.read(subdir + "/" + soundfile)
                    newSig = np.array(sig)
                    if rate != 44100:

                        base2roreach = int(
                            np.floor(np.log2(newSig.shape[0] / rate * 44100))
                        )
                        samplestoeliminate = int(
                            np.ceil(
                                newSig.shape[0] - ((2**base2roreach * rate) / 44100)
                            )
                        )
                        newSig = newSig[samplestoeliminate:]
                        newwidth = (
                            2**base2roreach
                        )  # int(np.ceil(newSig.shape[0]/rate*44100))
                        newSig = signal.resample(newSig, newwidth)
                        rate = 44100
                        # print(rate)

                    # newSig = justpadwithzeros(newSig, rate)
                    newSig = np.pad(
                        newSig, (0, int(maxlength - newSig.shape[0])), mode="constant"
                    )

                    mfcc_feat = mfcc(
                        newSig, rate, nfft=2048, winfunc=np.hamming
                    ).ravel()
                    mfcc_feat = mfcc_feat.reshape(1, mfcc_feat.shape[0])
                    # mfcc_feat = None

                    Xdir[soundfile.replace(".wav", "")] = counter
                    label = mapping[os.path.join(subdir, soundfile).split("\\")[1]]

                    if any(sub in subdir for sub in choices):
                        if counter == 0:
                            X = mfcc_feat
                        else:
                            X = np.vstack((X, mfcc_feat))
                            pass
                        y.append(label)
                        counter += 1
                    else:
                        if unseen_counter == 0:
                            X_unseen = mfcc_feat
                        else:
                            X_unseen = np.vstack((X_unseen, mfcc_feat))
                            pass
                        y_unseen.append(label)
                        unseen_counter += 1
                file_counter += 1
    # print("maxlength = ", maxlength)
    # y.append(soundfile[:1])
    finish = time()
    print("Time to load data %.3f s" % (finish - start))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=30, stratify=y
    )
    # feature scaling in order to standardize the features
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    writetopcklfile("training_data", [X_train, y_train])
    writetopcklfile("semi-seen-data", [X_test, y_test])
    custom_dump_svmlight_file(X_train, y_train, "new_training_data")
    custom_dump_svmlight_file(X_test, y_test, "semi-seen-data")
    # custom_dump_svmlight_file(X_unseen, y_unseen, "unseen-data")

    scaler = StandardScaler().fit(X_unseen)
    X_unseen = scaler.transform(X_unseen)

    # writetopcklfile("unseen-data", [X_unseen, y_unseen])

    # c, gamma, accuracy = paramsfromexternalgridsearch(
    #     "new_training_data",
    #     crange,
    #     grange,
    #     printlines=True,
    # )
    # print(accuracy)

    # all_matrix = training(
    #     (X_train, y_train),
    #     (X_test, y_test),
    #     "classifier_all.pkl",
    #     {"C": c, "gamma": gamma},
    # )

    # df_cm = pd.DataFrame(
    #     all_matrix,
    #     index=[i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
    #     columns=[i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
    # )
    # plt.figure(figsize=(10, 7))
    # all_map = sns.heatmap(df_cm, annot=True)

    # fig = all_map.get_figure()
    # fig.savefig("semi-seen.png")

    model = pickle.load(open("classifier_all.pkl", "rb"))
    y_pred = model.predict(X_unseen)

    print("Unseen accuracy score")
    print(accuracy_score(y_unseen, y_pred))
    df_cm = pd.DataFrame(
        confusion_matrix(y_unseen, y_pred),
        index=[i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
        columns=[i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
    )
    plt.figure(figsize=(10, 7))
    all_map = sns.heatmap(df_cm, annot=True)

    fig = all_map.get_figure()
    fig.savefig("unseen_map.png")
    test_words = [
        "THE",
        "QUICK",
        "BROWN",
        "FOX",
        "JUMPED",
        "OVER",
        "LAZY",
        "DOG",
    ]
    subject_1, subject_2, subject_3 = [], [], []
    for w in test_words:
        word_1, word_2, word_3 = [], [], []
        for l in w:
            value = mapping[l]
            word_1.append(X_unseen[((value+1)*8)-1])
            word_2.append(X_unseen[((value+1)*8 + 208)-1])
            word_3.append(X_unseen[((value+1)*8 + 416)-1])
        subject_1.append(word_1)
        subject_2.append(word_2)
        subject_3.append(word_3)
    
    from spellchecker import SpellChecker
    spell = SpellChecker()
    for i in subject_1:
        prediction = model.predict(np.array(i))
        print("".join([inv_map[j] for j in list(prediction)]))
        # print(spell.correction("".join([inv_map[j] for j in list(model.predict(np.array(i)))])))
        print(spell.candidates("".join([inv_map[j] for j in list(prediction)])))
    for i in subject_2:
        prediction = model.predict(np.array(i))
        print("".join([inv_map[j] for j in list(prediction)]))
        # print(spell.correction("".join([inv_map[j] for j in list(model.predict(np.array(i)))])))
        print(spell.candidates("".join([inv_map[j] for j in list(prediction)])))
    for i in subject_3:
        prediction = model.predict(np.array(i))
        print("".join([inv_map[j] for j in list(prediction)]))
        # print(spell.correction("".join([inv_map[j] for j in list(model.predict(np.array(i)))])))
        print(spell.candidates("".join([inv_map[j] for j in list(prediction)])))




main()
