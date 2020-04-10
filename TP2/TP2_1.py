import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import librosa
import IPython.display as ipd
import sounddevice as sd

n = 5000
fe = 8000.0
te = 1/fe

t = np.linspace(0, n * te, n, endpoint=False)

s = 0.5 * np.sin( 2 * np.pi * 440 * t )
s2 = 0.5 * np.sin( 2 * np.pi * 4400 * t )

s = s + s2

#sd.play((s + s2), 1/te)
#status = sd.wait()

S = np.abs( np.fft.rfft(s) )
tf = np.fft.rfftfreq( len(s), 1 / fe )
# ou tf=[0:len(s)-1]*fe/len(s)

plt.figure(figsize=(14, 5))
D = librosa.amplitude_to_db(np.abs(librosa.stft(s)), ref=np.max)
librosa.display.specshow(D, x_axis='time', y_axis='hz')


plt.figure(figsize=(14, 5))
plt.plot(tf, S)
plt.title('FFT')
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(t, s)
plt.title('Sinusoide')
plt.show()
