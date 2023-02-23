import numpy as np
import scipy as sc
import scipy.signal as sig
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

from scipy.fft import fft, fftfreq

sb.set_theme()

data = pd.read_csv("F0074CH2.CSV").values[17:].transpose()[3:-1]
fourier = np.abs(fft(data[1]))
fourier_freq = fftfreq(len(data[0]), np.mean(np.diff(data[0])))

freq = np.abs(fourier_freq[fourier.argmax()])
print(freq)

plt.figure(1)
plt.plot(data[0], data[1], ".")
plt.savefig("test.png")

plt.figure(2)
plt.plot(fourier_freq, fourier)
plt.savefig("test2.png")
