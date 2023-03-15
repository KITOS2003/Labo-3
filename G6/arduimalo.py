import os
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import set_theme

set_theme()


def mkdir(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass


def analisis(root):
    cuts = [None, (0, 250_000), (0, 375_000), None]
    for i, file in enumerate(sorted(os.listdir(root))):
        data = pd.read_csv(f"{root}/{file}").values.transpose()
        if cuts[i] is not None:
            low_index = np.abs(data[0] - cuts[i][0]).argmin()
            high_index = np.abs(data[0] - cuts[i][1]).argmin()
            data = data[:, low_index:high_index]
        mkdir(f"figures/arduimalo/{file[-1]}/")
        for i, column in enumerate(data[1:]):
            plt.figure(1)
            plt.plot(data[0], column, ".")
            plt.savefig(f"figures/arduimalo/{file[-1]}/{i}.pdf")
            plt.clf()
        cutting = False
        cut_indexes = []
        for i, element in enumerate(data[2]):
            if element < 500:
                cutting = True
            if cutting == True and element > 500:
                cutting = False
                cut_indexes.append(i)
        for i, cut in enumerate(pairwise(cut_indexes)):
            (low, high) = cut
            cut_data = data[:, low:high]
            for j, curve in enumerate(cut_data):
                mkdir(f"figures/arduimalo/{file[-1]}/{j}")
                tiempo = (cut_data[0] - cut_data[0][0]) / 1000
                plt.figure(1)
                plt.plot(tiempo, curve, ".", color="slateblue")
                plt.xlabel("Tiempo [s]")
                plt.ylabel("Tension [V]")
                plt.savefig(f"figures/arduimalo/{file[-1]}/{j}/{i}.pdf")
                plt.clf()


mkdir("figures/arduimalo")
analisis("arduimalo")
