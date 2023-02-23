import numpy as np
import scipy as sc
import scipy.signal as sig
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.optimize import curve_fit

def err(value, percentage, const):
    c1 = percentage / 100
    c2 = const / 100
    return c1 * value + c2

def osciloscopio_resistencia():
    # Medido en Volts
    voltaje = [0.48, 1.01, 1.52, 2.01, 2.50, 3.03, 3.53, 4.03, 4.53, 5.04, 5.55, 6.04, 6.54, 7.04, 7.55, 8.02, 8.53, 9.03, 9.52, 9.91]
    # Medido en micro Amperes
    corriente =   [0.5,  1.1,  1.6,  2.2,  2.7,  3.2,  3.8,  4.4,  4.9,  5.4,  6.0,  6.5,  7.1,  7.7,  8.2,  8.8,  9.3,  9.9,  10.4, 10.9]


    error_corriente = [err(x, 1, 2) for x in corriente]
    error_voltaje = [err(x, 3, 0) for x in voltaje]

    f = lambda i, r: r*i

    fit, cov = curve_fit(f, corriente, voltaje)

    error = np.sqrt(np.diag(cov))

    x_fit = np.linspace(0, 11, 1000)
    y_fit = [f(x, fit[0]) for x in x_fit]

    print("valor de la resistencia total %f +- %f MOmhs"%(fit[0], error[0]))

    sb.set_theme()

    plt.figure()
    plt.grid("on")
    plt.xlabel("Corriente $[\mu A]$")
    plt.ylabel("Voltaje $[V]$")
    plt.errorbar(corriente, voltaje, error_voltaje, error_corriente, ".", color = "orange", ecolor = "red")
    plt.plot(x_fit, y_fit, color = "blue")
    plt.savefig("4.png")


osciloscopio_resistencia()
