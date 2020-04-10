import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import librosa
import IPython.display as ipd
import sounddevice as sd

n = 5000
te = 1/8000.0
t = np.linspace(0, n * te, n, endpoint=False)
s = 0.5 * np.sin( 2 * np.pi * 440 * t )

sd.play(s,1/te)
status = sd.wait()

plt.plot(t,s)
plt.title('Sinusoide')
plt.show()
