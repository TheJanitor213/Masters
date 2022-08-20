import matplotlib.pyplot as plt
import matplotlib            
print (matplotlib.rcParams['backend'])
import librosa
y, sr = librosa.load("dataset/2022-02-19-B-Bruce_again-fixed/chunk_4.wav")
y_filt = librosa.effects.preemphasis(y)
import numpy as np
S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max, top_db=None)
S_preemph = librosa.amplitude_to_db(np.abs(librosa.stft(y_filt)), ref=np.max, top_db=None)
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
import librosa.display
librosa.display.specshow(S_orig, y_axis='log', x_axis='time', ax=ax[0])
ax[0].set(title='Original signal')
ax[0].label_outer()
img = librosa.display.specshow(S_preemph, y_axis='log', x_axis='time', ax=ax[1])
ax[1].set(title='Pre-emphasized signal')
fig.colorbar(img, ax=ax, format="%+2.f dB")
y_filt_1, zf = librosa.effects.preemphasis(y[:1000], return_zf=True)

y_filt_2, zf = librosa.effects.preemphasis(y[1000:], zi=zf, return_zf=True)
np.allclose(y_filt, np.concatenate([y_filt_1, y_filt_2]))

plt.figure()
librosa.display.specshow(S_orig)
plt.colorbar()
plt.show()