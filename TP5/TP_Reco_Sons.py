from python_speech_features import mfcc
import librosa

s1,fe1 = librosa.load("PIANO.wav")
s2,fe2 = librosa.load("honka.wav")

mfcc1 = mfcc(s1,fe1)
mfcc2 = mfcc(s2,fe2)

print(type(mfcc1)) #<class 'numpy.ndarray'>
print(mfcc1.shape) #(175, 13)
print(mfcc2.shape) #(35, 13)
