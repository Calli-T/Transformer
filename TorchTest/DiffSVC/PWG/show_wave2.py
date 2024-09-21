import librosa.display, librosa
from matplotlib import pyplot as plt

sig, sr = librosa.load('./sample/cs1_gen.wav')
print(sr)
plt.figure()
librosa.display.waveshow(sig, alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")

plt.show()