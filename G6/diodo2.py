import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

sb.set_theme()

# CH2 Diodo 1n4148
R1 = 10_000


def mkdir_noexcept(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass


mkdir_noexcept("figures")


def analisis_osciloscopio(root, r_list):
    for dir in os.listdir(root):
        mkdir_noexcept("figures/{}".format(dir))
        for file in os.listdir("{}/{}".format(root, dir)):
            if "CH1" in file:
                ch1 = "{}/{}/{}".format(root, dir, file)
            if "CH2" in file:
                ch2 = "{}/{}/{}".format(root, dir, file)
            if "TEK.SET" in file:
                config = "{}/{}/{}".format(root, dir, file)

        ch1 = pd.read_csv(ch1).values[17:].transpose()[3:-1]
        ch2 = pd.read_csv(ch2).values[17:].transpose()[3:-1]

        plt.figure(1)
        plt.plot(ch1[0], ch1[1], ".", color="orange")
        plt.plot(ch2[0], ch2[1], ".", color="slateblue")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Tension [V]")
        plt.savefig("figures/{}/osciloscopio.pdf".format(dir))
        plt.clf()
        analisis(ch1, ch2, 10_000, figname="figures/{}/corriente.pdf".format(dir))


def analisis(ch1, ch2, R, figname=""):
    ch1 = np.asarray(ch1, dtype=float)
    ch2 = np.asarray(ch2, dtype=float)
    tension_diodo = ch1[1]
    corriente = ch2[1] / R
    plt.figure(2)
    plt.xlabel("Tension [V]")
    plt.ylabel("Corriente [V]")
    plt.plot(tension_diodo, corriente, ".", color="slateblue")
    plt.savefig(figname)
    plt.clf()


analisis_osciloscopio("data", [])
