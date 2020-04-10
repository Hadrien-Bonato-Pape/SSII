import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import librosa
import IPython.display as ipd
import sounddevice as sd
from scipy import signal

#s, fe = librosa.load('/Users/hbp/Desktop/POLYTECH/3A/S6/SSII/SONS/SONS_PROF/coq.wav')
#te = 1/fe

#sd.play(s,fe)
#status = sd.wait()


#Butterworth example

b, a = signal.butter(4, 100, 'low', analog=True)
w, h = signal.freqs(b, a)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green')
plt.show()


#Sinusoides

t = np.linspace(0, 1, 1000, False)  # 1 second
sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, sig)
ax1.set_title('10 Hz and 20 Hz sinusoids')
ax1.axis([0, 1, -2, 2])

sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')
filtered = signal.sosfilt(sos, sig)
ax2.plot(t, filtered)
ax2.set_title('After 15 Hz high-pass filter')
ax2.axis([0, 1, -2, 2])
ax2.set_xlabel('Time [seconds]')
plt.tight_layout()
plt.show()


#WAV
s, fe = librosa.load('/Users/hbp/Desktop/POLYTECH/3A/S6/SSII/SONS/SONS_PROF/coq.wav')
te = 1/fe

#sd.play(s,fe)
#status = sd.wait()

t = [i*fe for i in range(len(s))]
#sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, s)
ax1.set_title('Signal coq')

sos = signal.butter(10, 5000, 'hp', fs=1000, output='sos')
filtered = signal.sosfilt(sos, s)
ax2.plot(t, filtered)
ax2.set_title('Coq filter 15hz hp')
ax2.set_xlabel('Time [seconds]')
plt.tight_layout()
plt.show()

sd.play(filtered, fe)
status = sd.wait()






