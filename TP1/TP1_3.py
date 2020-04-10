import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import librosa
import IPython.display as ipd
import sounddevice as sd

t = np.linspace(0, n * te, n, endpoint=False)
n = 5000
fe = 8000.0
te = 1/fe

s = 0.5 * np.sin( 2 * np.pi * 440 * t )
s2 = 0.5 * np.sin( 2 * np.pi * 4400 * t )

sd.play((s + s2), 1/te)
status = sd.wait()

plt.plot(t,s + s2)
plt.title('Sinusoide')
plt.show()
