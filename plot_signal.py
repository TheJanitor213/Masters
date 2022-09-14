
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
import numpy as np
import wave
import sys
from python_speech_features.sigproc import preemphasis, framesig
original = wave.open("2022-08-19-A-Attempts.wav","r")
spf = wave.open("dataset/B/12_.wav", "r")
fs = spf.getframerate()
# Extract Raw Audio from Wav File
original = original.readframes(-1)
original = np.fromstring(original, "int16")
# signal = spf.readframes(-1)
# signal = np.fromstring(signal, "int16")
time = np.linspace(
        0, # start
        len(original) / fs,
        num = len(original)
    )
# If Stereo
if spf.getnchannels() == 2:
    print("Just mono files")
    sys.exit(0)

def graph_signal(file):
    raw = wave.open(file, "r")
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="int16")
    f_rate = raw.getframerate()
    time = np.linspace(
        0, # start
        len(signal) / f_rate,
        num = len(signal)
    )
    return time, signal

figure, axis = plt.subplots(2, 5, figsize=(25,7))
axis[0,0].plot(*graph_signal("chunk0.wav"))
axis[0,1].plot(*graph_signal("chunk1.wav"))
axis[0,2].plot(*graph_signal("chunk2.wav"))
axis[0,3].plot(*graph_signal("chunk3.wav"))
axis[0,4].plot(*graph_signal("chunk4.wav"))
axis[1,0].plot(*graph_signal("chunk5.wav"))
axis[1,1].plot(*graph_signal("chunk6.wav"))
axis[1,2].plot(*graph_signal("chunk7.wav"))
axis[1,3].plot(*graph_signal("chunk8.wav"))
axis[1,4].plot(*graph_signal("chunk9.wav"))

axis[0,0].xlabel("Time (s)")
axis[0,1].xlabel("Time (s)")
axis[0,2].xlabel("Time (s)")
axis[0,3].xlabel("Time (s)")
axis[0,4].xlabel("Time (s)")
axis[1,0].xlabel("Time (s)")
axis[1,1].xlabel("Time (s)")
axis[1,2].xlabel("Time (s)")
axis[1,3].xlabel("Time (s)")
axis[1,4].xlabel("Time (s)")
plt.savefig('A.png', bbox_inches='tight')
# from scipy.fft import fft, fftfreq

# yf = fft(pre_emp)
# N = 22050 * TIME
# xf = fftfreq(N, 1 / 22050)

# plt.plot(xf, np.abs(yf))
# print(np.hamming)
# print(framing[0])
# #plt.plot(framing)
# plt.show()
