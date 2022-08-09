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

def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio


def frame_audio(audio, FFT_size=2048, hop_size=10, sample_rate=44100):
    # hop_size in ms

    audio = np.pad(audio, int(FFT_size / 2), mode="reflect")
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num, FFT_size))

    for n in range(frame_num):
        frames[n] = audio[n * frame_len : n * frame_len + FFT_size]

    return frames


def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def met_to_freq(mels):
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)

    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num + 2)
    freqs = met_to_freq(mels)

    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs


def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points) - 2, int(FFT_size / 2 + 1)))
    print("Getting filters")
    for n in tqdm(range(len(filter_points) - 2)):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(
            0, 1, filter_points[n + 1] - filter_points[n]
        )
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(
            1, 0, filter_points[n + 2] - filter_points[n + 1]
        )

    return filters


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


def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num, filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)

    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)

    return basis


def create_cepstral_coefficients(file):
    sample_rate, audio = wavfile.read(file)
    audio = pad_audio(audio, sample_rate)

    hop_size = 10  # ms
    FFT_size = 1024

    audio_framed = frame_audio(
        audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate
    )

    window = get_window("hann", FFT_size, fftbins=True)

    audio_win = audio_framed * window

    ind = 6

    audio_winT = np.transpose(audio_win)

    audio_fft = np.empty(
        (int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order="F"
    )

    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[: audio_fft.shape[0]]

    audio_fft = np.transpose(audio_fft)

    audio_power = np.square(np.abs(audio_fft))

    freq_min = 0
    freq_high = sample_rate / 2
    mel_filter_num = 10

    filter_points, mel_freqs = get_filter_points(
        freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100
    )

    filters = get_filters(filter_points, FFT_size)

    # taken from the librosa library
    enorm = 2.0 / (mel_freqs[2 : mel_filter_num + 2] - mel_freqs[:mel_filter_num])
    filters *= enorm[:, np.newaxis]

    audio_filtered = np.dot(filters, np.transpose(audio_power))
    audio_log = 10.0 * np.log10(audio_filtered)
    audio_log.shape

    dct_filter_num = 40

    dct_filters = dct(dct_filter_num, mel_filter_num)

    cepstral_coefficents = np.dot(dct_filters, audio_log)

    return cepstral_coefficents


def load_audio():
    rootdir = "../backup/dataset/"
    labels = []
    letters = {}
    print("Loading audio...")
    file_count = sum(len(files) for _, _, files in os.walk(rootdir))
    with tqdm(total=file_count) as pbar:
        for subdir, dirs, files in tqdm(os.walk(rootdir)):
            for file in files:
                pbar.update(1)
                if "chunk" in file:

                    sample_rate, audio = wavfile.read(os.path.join(subdir, file))

                    data = pad_audio(audio, sample_rate)
                    label = os.path.join(subdir, file).split("-")[3]
                    if letters.get(label):
                        letters[label].append(data)
                    else:
                        letters[label] = [data]
    return letters


# How to use load_audio() function
characters = load_audio()


def extract_features(audio_data):

    # Remember that the audio data consists of raw audio wave followed by sample rate
    # so we need to only take the raw audio wave.
    output, label_output = [], []
    print("Extracting features")
    for k, v in tqdm(audio_data.items()):
        for i in v:
            label_output.append(k)
            data = mfcc(i, samplerate=22050, nfft=2048, winfunc=np.hamming)
            output.append(np.array(data).flatten())

        # n_fft = int(samplerate * 0.02)
        # data = mfcc(audio_waves, samplerate=samplerate, nfft=2048, winfunc=np.hamming)
        # label_output.append(letter)
        # features = np.array(data)
        # output.append(features.flatten())

    mapping = {v: k for k, v in enumerate(ascii_uppercase)}
    label_output = [mapping[k] for k in label_output]
    label_output = np.array(label_output)
    return output, label_output


# Define a function to load the raw audio files

character_features, labels = extract_features(characters)


X = character_features
Y = labels


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=30, stratify=Y
)



parameters = {"C": [1, 10, 32, 100, 1000], "gamma": [0.00001, 0.000488281, 0.0001, 0.001, 0.01, 0.1]}
grid = GridSearchCV(svm.SVC(kernel="poly"), parameters, cv=5)
grid.fit(X_train, y_train)
print(grid)
print(grid.best_params_)
print(grid.best_score_)

# y_pred = clf.predict(X_test)
# print(clf.score(X_test, y_test))
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

clf = svm.SVC(kernel="poly", **grid.best_params_, class_weight='balanced')

#clf = svm.SVC(kernel="rbf", C=32.0, gamma=0.000488281)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(clf.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# save the classifier

with open("my_dumped_classifier.pkl", "wb") as fid:
    pickle.dump(clf, fid)


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
