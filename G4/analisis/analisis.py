import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

# from prettytable import PrettyTable

import os

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
# from scipy.fft import fft, fftfreq

import uncertainties as u
from uncertainties import unumpy as un
# from uncertainties.umath import *

sb.set_theme()

def mkdir_noexcept(dir):
    try:
        os.mkdir(dir)
    except:
        pass

def uplot(x, y, label=""):
    x_values = un.nominal_values(x)
    x_errors = un.std_devs(x)
    y_values = un.nominal_values(y)
    y_errors = un.std_devs(y)
    plt.errorbar(x_values, y_values, y_errors, x_errors, ".", color="slateblue", label=label)

def ucurve_fit(ajuste, x, y, coefs_guess, bounds=[(), ()], absolute_sigma=False):
    x_values = un.nominal_values(x)
    y_values = un.nominal_values(y)
    y_error = un.std_devs(y)
    return curve_fit(ajuste, x_values, y_values, coefs_guess, sigma=y_error, bounds=bounds, absolute_sigma=absolute_sigma)

class DatasetRC:
    def __init__(self, resistencias, figdir="figures"):
        self.figdir = figdir
        self.resistencias = np.array(resistencias)
        self.v0 = []
        self.exp = []
        self.offset = []
        self.dirnums = []
        mkdir_noexcept(figdir)
        mkdir_noexcept(figdir + "/osciloscopio")

    def sort(self):
        dic = dict( zip(self.dirnums, list(zip(self.v0, self.exp, self.offset))) )
        self.dirnums = sorted(self.dirnums)
        sorted_values = [dic[x] for x in self.dirnums]
        sorted_values = list(zip(*sorted_values))
        self.v0 = np.array(sorted_values[0])
        self.exp = np.array(sorted_values[1])
        self.offset = np.array(sorted_values[2])

    def analisis_osc(self, ch1, ch2, fig_name="figure.pdf"):
        # Primero hay que limpiar:
        # Tomamos la diferencia de todo par de puntos consecutivos en el canal 2 y buscamos los picos
        d_ch2 = np.abs(np.diff(ch2))
        cut_indices = find_peaks(d_ch2[1], prominence=10)[0]
        # ahora de entre esos picos, buscamos uno que corresponda a un salto de 10+-1V y cortamos por ahi
        for i, index in enumerate(cut_indices):
            if 9 < d_ch2[1][index] < 11:
                if cut_indices[i + 1:].size > 0:
                    upper_cut_index = cut_indices[i + 1]
                    ch1 = ch1[:, :upper_cut_index]
                    ch2 = ch2[:, :upper_cut_index]
                ch1 = ch1[:, index + 1:]
                ch2 = ch2[:, index + 1:]

        # Ahora ajustamos: TODO errores
        time = ch2[0] - ch2[0][0]
        ch2 = np.array([time, ch2[1]])
        time = ch2[0] - ch2[0][0]
        ch1 = np.array([time, ch1[1]])

        def ajuste(t, v0, ex, v_offset):
            return v_offset + v0 * np.exp(-ex * t)

        coefs, cov = curve_fit(ajuste, ch1[0], ch1[1], [10, 1, 0], bounds=[(0, 0, -0.2), (np.inf, np.inf, 0.2)])
        # Graficamos todo
        plt.figure(1)
        x_fit = np.linspace(ch1[0][0], ch1[0][-1], 1000)
        y_fit = ajuste(x_fit, *coefs)
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Tension [V]")
        plt.plot(ch1[0], ch1[1], ".", color="slateblue", label="Canal 1")
        # plt.plot(ch2[0], ch2[1], ".", color="orange", label="Canal 2")
        plt.plot(x_fit, y_fit, color="orange", label="Ajuste")
        plt.savefig(self.figdir + "/osciloscopio/" + fig_name)
        plt.legend(loc="best")
        plt.clf()
        # Calculamos los errores y retornamos ufloats
        v0_value, ex_value, v_offset_value = coefs
        v0_error, ex_error, v_offset_error = np.sqrt(np.diag(cov))
        self.v0.append(u.ufloat(v0_value, v0_error))
        self.exp.append(u.ufloat(ex_value, ex_error))
        self.offset.append(u.ufloat(v_offset_value, v_offset_error))

    def analisis(self):
        self.sort()
        rc = 1 / self.exp
        plt.figure(2)
        plt.xlabel(r"resistencia [$\Omega$]")
        plt.ylabel("1/exponente")
        uplot(resistencias, rc)
        plt.savefig(self.figdir + "/rc_lineal.pdf")
        plt.clf()


class DatasetRL:
    def __init__(self, resistencias, figdir="rl_figures"):
        mkdir_noexcept(figdir)
        mkdir_noexcept(figdir + "/osciloscopio")
        self.resistencias = resistencias
        self.dirnums = []
        self.figdir = figdir
        self.amplitud = []
        self.exp = []

    def analisis_osc(self, ch1, ch2, fig_name="figure.pdf"):
        d_ch2 = np.abs(np.diff(ch2[1]))
        cut_indices = find_peaks(d_ch2, prominence=1)[0]
        cut_index = cut_indices[0]
        ch1 = ch1[:, cut_index:]
        ch2 = ch2[:, cut_index:]

        def ajuste(t, a, exp):
            return a * (1 - np.exp(-exp * t))

        coefs, cov = curve_fit(ajuste, ch1[0], ch1[1], [10, 1])
        x_fit = np.linspace(ch1[0][0], ch1[0][-1], 1000)
        y_fit = ajuste(x_fit, *coefs)
        fig = plt.figure(1)
        plt.xlabel("Tiempo [ms]")
        plt.ylabel("Tension [V]")
        plt.plot(1000 * ch1[0], ch1[1], ".", color="slateblue", label="Canal 1")
        # plt.plot(ch2[0], ch2[1], ".", color="red", label="Canal 2")
        plt.plot(1000 * x_fit, y_fit, color="orange", label="Ajuste")
        plt.savefig(self.figdir + "/osciloscopio/" + fig_name)
        fig.legend()
        plt.clf()
        amplitud_value, exp_value = coefs
        amplitud_error, exp_error = np.sqrt(np.diag(cov))
        self.amplitud.append(u.ufloat(amplitud_value, amplitud_error))
        self.exp.append(u.ufloat(exp_value, exp_error))

    def sort(self):
        dic = dict(zip(self.dirnums, list(zip(self.amplitud, self.exp)) ))
        self.dirnums = sorted(self.dirnums)
        sorted_values = [dic[x] for x in self.dirnums]
        sorted_values = list(zip(*sorted_values))
        self.amplitud = np.array(sorted_values[0])
        self.exp = np.array(sorted_values[1])

    def analisis(self):
        self.sort()

        def ajuste_lineal(x, a, b):
            return a * x + b

        coefs, cov = ucurve_fit(ajuste_lineal, self.resistencias[1:], self.exp[1:], [1, 0])
        x_fit = np.linspace(self.resistencias[0].nominal_value, self.resistencias[-1].nominal_value, 1000)
        y_fit = ajuste_lineal(x_fit, *coefs)
        plt.figure(1)
        plt.xlabel(r"Resistencia [$\Omega$]")
        plt.ylabel(r"Exponente [$s^{-1}$]")
        plt.plot(x_fit, y_fit, color="orange", label="Ajuste")
        uplot(resistencias, self.exp, label="Datos")
        plt.savefig(self.figdir + "/r_exp.pdf")
        plt.clf()
        errors = np.sqrt(np.diag(cov))
        print("valor de la pendiente: {}+-{} H^-1".format(coefs[0], errors[0]))
        print("valor de la ordenada: {}+-{} Omhs/H".format(coefs[1], errors[1]))


def analisis_osciloscopio(root, dataset):
    dir_list = os.listdir(root)
    for dir_name in dir_list:
        dir = root + "/" + dir_name + "/"
        file_list = os.listdir(dir)
        for file in file_list:
            if "CH1" in file:
                ch1 = pd.read_csv(dir + file).values
                # ch1_header = ch1[:17]
                ch1 = ch1[17:].transpose()[3:-1]
            elif "CH2" in file:
                ch2 = pd.read_csv(dir + file).values
                # ch2_header = ch1[:17]
                ch2 = ch2[17:].transpose()[3:-1]
        dataset.analisis_osc(ch1, ch2, fig_name=dir_name.replace("ALL", "") + ".pdf")
        dataset.dirnums.append(int(dir_name.replace("ALL", "")))
    return dataset


resistencias = [1000, 3162, 10101, 31600, 101000, 316000, 1001000, 3162000, 10001000]
resistencias = un.uarray(resistencias, [0] * len(resistencias))
dataset_rc = DatasetRC(resistencias, figdir="rc_figures")
dataset_rc = analisis_osciloscopio("data/rc", dataset_rc)
dataset_rc.analisis()


resistencias = [ 500, 1000, 1500, 2000, 2500, 3000, 3500, 4500, 5000 ]
resistencias = un.uarray(resistencias, [0] * len(resistencias))
dataset_rl = DatasetRL(resistencias)
dataset_rl = analisis_osciloscopio("data/rl", dataset_rl)
dataset_rl.analisis()
