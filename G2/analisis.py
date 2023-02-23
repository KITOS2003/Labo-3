import numpy as np
import scipy as sc
import scipy.signal as sig
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

import os

from scipy.optimize import curve_fit

sb.set_theme()
datasets_ls = os.listdir("data")

voltajes1 = []
fases1 = []
voltajes2 = []
fases2 = []
voltajes_err1 = []
fases_err1 = []
voltajes_err2 = []
fases_err2 = []
frecuencias = []
frecuencias_err = []
for dataset in datasets_ls:
    frequency = float(dataset)
    ch1 = pd.read_csv("data/" + dataset + "/F0000CH1.CSV").values[17:].transpose()[3:-1]
    ch2 = pd.read_csv("data/" + dataset + "/F0000CH2.CSV").values[17:].transpose()[3:-1]
    if frequency == 1.6:
        ch1 = [ ch1[0][250:-30], ch1[1][250:-30] ]
        ch2 = [ ch2[0][250:-30], ch2[1][250:-30] ]

    def f(t, w, v0, phi): 
        return v0*np.sin(w*t + phi)

    fit1, cov1 = curve_fit(f, ch1[0], ch1[1], [2*np.pi*frequency, 1, 0])
    fit2, cov2 = curve_fit(f, ch2[0], ch2[1], [2*np.pi*frequency, 1, 0])
    error1 = np.sqrt(np.diag(cov1))
    error2 = np.sqrt(np.diag(cov2))

    freq_recover1 = fit1[0]/(2*np.pi)
    freq_recover2 = fit2[0]/(2*np.pi)

    # print("frecuencias recuperadas para %s: %fHz, %fHz" % (dataset, freq_recover1, freq_recover2))

    x_fit = np.linspace(ch1[0][0], ch1[0][-1], 1000)
    phi1 = np.array([fit1[2]] * len(ch1[0]))
    phi2 = np.array([fit2[2]] * len(ch2[0]))
    y_fit1 = [f(t, fit1[0], fit1[1], fit1[2]) for t in x_fit]
    y_fit2 = [f(t, fit2[0], fit2[1], fit2[2]) for t in x_fit]

    plt.figure(1)
    plt.grid("on")
    plt.xlabel("tiempo [s]")
    plt.ylabel("voltaje [V]")
    plt.plot(ch1[0], ch1[1])
    plt.plot(x_fit, y_fit1)
    plt.plot(x_fit, y_fit2)
    plt.plot(ch2[0], ch2[1])
    plt.savefig("figures/%fHz.png"%(frequency))
    plt.clf()
    voltajes1.append(fit1[1])
    voltajes_err1.append(error1[1])
    voltajes2.append(fit2[1])
    voltajes_err2.append(error2[1])
    fases1.append(fit1[2])
    fases_err1.append(error1[2])
    fases2.append(fit2[2])
    fases_err2.append(error2[2])
    frecuencias.append(fit2[0])
    frecuencias_err.append(error2[0])

transferencia       = np.array(voltajes2) / np.array(voltajes1)
transferencia_error = np.sqrt((np.array(voltajes_err2)/np.array(voltajes1))**2 + (np.array(voltajes2)*np.array(voltajes_err1)/(np.array(voltajes1)**2))**2)

dif_fase = np.array(fases2) - np.array(fases1)
dif_fase_err = np.sqrt(np.array(fases_err2)**2 + np.array(fases_err1)**2)

f = lambda w, w0: 1/np.sqrt(1 + (w/w0)**2)
transferencia_fit, transferencia_cov = curve_fit(f, frecuencias, transferencia, [1600], transferencia_error, absolute_sigma = True)
transferencia_fit_err = np.sqrt(np.diag(transferencia_cov))

frecuencia_corte = transferencia_fit[0] / (2*np.pi)
frecuencia_corte_err = transferencia_error[0] / (2*np.pi)

print("Valor de la frecuencia de corte: %f+-%fHz"%(frecuencia_corte, frecuencia_corte_err))
x_fit = np.logspace(np.log10(np.min(frecuencias)), np.log10(np.max(frecuencias)), num = 1000)
t_fit = [f(w, 2*np.pi*frecuencia_corte) for w in x_fit]

plt.figure(2)
plt.grid("on")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Funcion de transferencia")
plt.xscale("log")
plt.plot(x_fit, t_fit, color = "blue")
plt.errorbar(frecuencias, transferencia, transferencia_error, frecuencias_err, ".", color = "orange", ecolor = "red")
plt.savefig("voltajes.png")
plt.clf()

f = lambda w, w0: -np.arctan(w/w0)
fase_fit =  f(x_fit, 2*np.pi*frecuencia_corte)

plt.grid("on")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Desfasaje [Rad]")
plt.xscale("log")
plt.plot(x_fit, fase_fit, color = "blue")
plt.errorbar(frecuencias, dif_fase, dif_fase_err, frecuencias_err, ".", color = "orange", ecolor = "red")
plt.savefig("fase.png")
plt.clf()

atenuacion = 20*np.log10(transferencia)
atenuacion_err = 20/(np.log(10)*transferencia)
              
w_w0 = frecuencias/(2*np.pi*frecuencia_corte)
x_fit = np.logspace( np.log10(np.min(w_w0)), np.log10(np.max(w_w0)), 1000)
a_fit = -10*np.log10(1 + x_fit**2)

plt.grid("on")
plt.xlabel(r"$\omega/\omega_0$")
plt.ylabel("Atenuacion [dB]")
plt.xscale("log")
plt.plot(x_fit, a_fit, color = "blue")
plt.errorbar(w_w0, atenuacion, 0, frecuencias_err/(2*np.pi*frecuencia_corte), ".", color = "orange", ecolor = "red")
plt.savefig("atenuacion.png")



