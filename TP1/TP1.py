import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import librosa
import IPython.display as ipd
import sounddevice as sd


#Coq -> longeur fichier : 23190 (entete) en octets
x, fe = librosa.load('/Users/hbp/Desktop/POLYTECH/3A/S6/SSII/SONS/SONS_PROF/coq.wav')

sd.play(x,fe)
status = sd.wait()

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=fe)



# default is mono. In case of stereo:
# librosa.load('fichierStereo.wab', mono=False)


plt.title('Je m\'appelle Hadrien Bonato-Pape et je soumets le fichier Bonato-Pape.png qui est une image représentant le chronogramme d\'un son de Coq au format PNG')
plt.show()

# n=len(x)
# t = np.linspace(0, n/fe, n, endpoint=False)
#endpoint not i
#plot(t,x)
