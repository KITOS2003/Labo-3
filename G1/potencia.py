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

def potencia_bateria():
    # Escala 2V
    voltaje =     [ 0.166, 0.179, 0.193, 0.205, 0.218, 0.230, 0.241, 0.253, 0.263, 0.274, 
                    0.285, 0.373, 0.442, 0.448, 0.454, 0.460, 0.466, 0.471, 0.477, 0.482, 
                    0.487, 0.493, 0.498, 0.503, 0.507, 0.512, 0.517, 0.521, 0.526, 0.530, 
                    0.534, 0.539, 0.543, 0.581, 0.613, 0.640, 0.664 ]

    # Escala 20Omhs
    resistencia = [ 10.2,  11.2,  12.2, 13.2, 14.2, 15.2, 16.2, 17.2, 18.2, 19.2,
                    20.2,  30.2,  40.2, 41.2, 42.2, 43.2, 44.2, 45.2, 46.2, 47.2, 48.2, 
                    49.2,  50.2,  51.2, 52.2, 53.2, 54.2, 55.2, 56.2, 57.2, 58.2, 59.1, 
                    60.1,  70.2,  80.1, 90.1, 100.1 ]

    voltaje = np.array(voltaje)
    resistencia = np.array(resistencia)

    err_resistencia = np.array([err(x, 0.8, 3) for x in resistencia])
    err_voltaje =     np.array([err(x, 0.5, 0.1) for x in voltaje])

    potencia = voltaje**2 / resistencia

    err_potencia = np.sqrt(((2*voltaje/resistencia)*err_voltaje)**2 + (((voltaje/resistencia)**2)*err_resistencia)**2)

    potencia *= 1000
    err_potencia *= 1000

    f = lambda rc, e, ri: rc*(e/(ri + rc))**2

    fit, cov = curve_fit(f, resistencia, potencia, [1, 55], err_potencia, absolute_sigma = True)

    error = np.sqrt(np.diag(cov))

    print("Valor de la resistencia interna de la bateria %f +- %f Omhs"%(fit[1], error[1]))
    print("valor del voltaje emitido por la fuente %f +- %f"%(fit[0], error[0]))

    x_fit = np.linspace(9, 100, 1000)
    y_fit = [f(x, fit[0], fit[1]) for x in x_fit]

    sb.set_theme()

    plt.figure()
    plt.xlabel("Resistencia $[\Omega]$")
    plt.ylabel("Potencia $[mW]$")
    plt.errorbar(resistencia, potencia, err_potencia, err_resistencia, '.', ecolor = "red", color = "orange")
    plt.plot(x_fit, y_fit, color = "blue")
    plt.savefig("3.png")

potencia_bateria()
