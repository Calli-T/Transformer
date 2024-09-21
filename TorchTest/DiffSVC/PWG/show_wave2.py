import librosa.display, librosa
from matplotlib import pyplot as plt

# sig, sr = librosa.load('./sample/L-O-V-E.wav')
sig, sr = librosa.load('./outputs/L-O-V-E_gen.wav')
plt.figure()
librosa.display.waveshow(sig, sr=sr, alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")

plt.show()