import os

import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

sb.set_theme()

def mkdir_noexept(dir):
    try:
        os.mkdir(dir)
    except:
        pass

def analisis_osc(datadir, resistencias):
    mkdir_noexept("figures")
    for dir in sorted(os.listdir(datadir)):
        dir_num = int(dir.replace("ALL", ""))
        dir = datadir + "/" + dir + "/"
        for file in os.listdir(dir):
            if "CH1" in file:
                ch1 = dir + file
            if "CH2" in file:
                ch2 = dir + file
            if "TEK.SET" in file:
                config = dir + file
        ch1 = pd.read_csv(ch1).values[17:].transpose()[3:-1]
        ch2 = pd.read_csv(ch2).values[17:].transpose()[3:-1]
        with open(config, "r") as file:
            config = file.read()
        osc_error = [0, 0]
        for channel in [1, 2]:
            scale = config[config.find("CH{}".format(channel)):]
            scale = scale[scale.find("SCALE"):]
            scale = scale[:scale.find(";")]
            scale = scale.replace("SCALE ", "")
            osc_error[channel-1] = 10 * float(scale) / 255
            # Aca el error del osciloscopio, a eso se le suma 3% de ganancia
            # No me dio tiempo a propagar pero es trivial usando el modulo uncertainties
        # Grafico canal 1 y 2
        plt.figure(1)
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Tension [V]")
        plt.plot(ch1[0], ch1[1], ".", color="orange")
        plt.plot(ch2[0], ch2[1], ".", color="blue")
        plt.savefig("figures/{}.pdf".format(dir_num))
        plt.clf()
        # Grafico U_L Y U_C
        resistencia = resistencias[dir_num - 1]
        corriente = (ch2[1] - ch1[1]) / resistencia
        capacidad = 1 * 10**(-9)
        inductancia = 1
        U_L = inductancia * (corriente)**2 / 2
        U_C = capacidad * ch1[1]**2 / 2
        plt.figure(1)
        plt.plot(ch1[0], U_C, ".", color="orange")
        plt.plot(ch1[0], U_L, ".", color="blue")
        plt.savefig("figures/{}U.pdf".format(dir_num))
        plt.clf()
        # Grafico energia total
        U = U_L + U_C
        plt.figure(1)
        plt.xlabel("Tiemo[s]")
        plt.ylabel("Energia [J]")
        plt.plot(ch1[0], U, ".", color="orange")
        plt.savefig("figures/ENERGIA_TOTAL_{}.pdf".format(dir_num))
        plt.clf()


analisis_osc("data", [1100, 10_100])


# Conclusiones: la resistencia tiene que ser grande para minimizar el error propio del osciloscopio, observar la diferencia entre
# los dos graficos de la energia
