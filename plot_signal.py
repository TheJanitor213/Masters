import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
import numpy as np
import wave
import sys
from python_speech_features.sigproc import preemphasis, framesig
original = wave.open("/Users/brucebeck/Documents/Masters/2021-12-19-B-Bruce-fixed.wav","r")
spf = wave.open("dataset/2022-02-19-B-Bruce_again-fixed/chunk_2.wav", "r")
fs = spf.getframerate()
# Extract Raw Audio from Wav File
original = original.readframes(-1)
original = np.fromstring(original, "int16")
signal = spf.readframes(-1)
signal = np.fromstring(signal, "int16")

# If Stereo
if spf.getnchannels() == 2:
    print("Just mono files")
    sys.exit(0)

# plt.figure(1)
# plt.plot(original)
# plt.show()

# plt.figure(2)
# plt.plot(signal)
# plt.show()
pre_emp = preemphasis(signal)
target_len = 1.5
fig = plt.figure(figsize=(10,3))
ax = fig.add_subplot() 
Time = np.linspace(0, len(pre_emp) / fs, num=len(pre_emp))

#plt.xticks(np.arange(0.0, 1.0, 0.1))


fig, ax_left = plt.subplots()
ax_right = ax_left.twinx()


ax_left.plot(Time, pre_emp)
ax_left.set_xlabel('Time (seconds)')
ax_left.set_ylabel('Amplitude')

import scipy.stats as stats
import math

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 2*sigma, mu + 2*sigma, 100)
ax_right.plot(x, stats.norm.pdf(x, mu, sigma))
# t = fig.transFigure
# print(t)
# a = ax.transAxes

# rect1 = mpl.patches.Rectangle((0.4, 1), width=0.05, height=-1, color="red", transform=ax.transAxes, clip_on=False, fill=None, linewidth=1.2)
# rect2 = mpl.patches.Rectangle((0.425, 1), width=0.05, height=-1, color="blue", transform=ax.transAxes, clip_on=False, fill=None, linewidth=1.2)
# rect3 = mpl.patches.Rectangle((0.45, 1), width=0.05, height=-1, color="green", transform=ax.transAxes, clip_on=False, fill=None, linewidth=1.2)
# rect4 = mpl.patches.Rectangle((0.475, 1), width=0.05, height=-1, color="pink", transform=ax.transAxes, clip_on=False, fill=None, linewidth=1.2)
# rect5 = mpl.patches.Rectangle((0.5, 1), width=0.05, height=-1, color="yellow", transform=ax.transAxes, clip_on=False, fill=None, linewidth=1.2)
# rect6 = mpl.patches.Rectangle((0.525, 1), width=0.05, height=-1, color="black", transform=ax.transAxes, clip_on=False, fill=None, linewidth=1.2)
# ax.add_patch(rect1)
# ax.add_patch(rect2)
# ax.add_patch(rect3)
# ax.add_patch(rect4)
# ax.add_patch(rect5)
# ax.add_patch(rect6)
plt.show()


framing = framesig(pre_emp, 562.5,225)
print(framing)

