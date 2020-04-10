import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import librosa
import IPython.display as ipd
import sounddevice as sd


#fmax = 7114
s, fe = librosa.load('/Users/hbp/Desktop/POLYTECH/3A/S6/SSII/SONS/SONS_PROF/coq.wav')
te = 1/fe

#sd.play(s,fe)
#status = sd.wait()

#plt.figure(figsize=(14, 5))
#librosa.display.waveplot(s, sr=fe)
#plt.title('SNCF')


#S = np.abs(np.fft.rfft(s))
#tf = np.fft.rfftfreq( len(s), 1 / fe )
#plt.figure(figsize=(14, 5))
#plt.plot(tf, S)
#plt.title('FFT')
#plt.show()

plt.show()
plt.figure(figsize=(14, 5))
D = librosa.amplitude_to_db(np.abs(librosa.stft(s)))
librosa.display.specshow(D, x_axis='time', y_axis='hz', sr = fe)
plt.show()

#fe = fe / 4
#S = np.abs(np.fft.rfft(s))
#tf = np.fft.rfftfreq( len(s), 1 / fe )
#plt.figure(figsize=(14, 5))
#plt.plot(tf, S)
#plt.title('FFT 2')
#plt.show()

#sd.play(s,fe)
#status = sd.wait()
