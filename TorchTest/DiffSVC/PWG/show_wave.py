import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.io import wavfile
from glob import glob
from tqdm import tqdm

sns.set_style('darkgrid')

def data_loader(files):
    out = []
    for file in tqdm(files):
        fs, data = wavfile.read(file)
        out.append(data)
    out = np.array(out)
    return out

'''x_data = glob('./sample/cs1.wav')
x_data = data_loader(x_data)'''

fs, data = wavfile.read('./sample/cs3_gen.wav')
# fs, data = wavfile.read('./sample/cs1_gen.wav')
data = np.array(data)

plt.plot(data)
plt.show()