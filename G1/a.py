import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.optimize import curve_fit

voltajes1 = [4.67, 4.28, 3.95, 3.67, 3.43, 3.21, 3.03, 2.86, 2.71, 2.58]
resistencias1 = [1, 1.99, 2.99, 3.98, 4.98, 5.96, 6.96, 7.96, 8.96, 9.92]


def err(value, percentage, const):
    c1 = percentage / 100
    c2 = const / 100
    return c1 * value + c2


def voltimetro(voltajes, resistencias):
    err_voltajes = [err(x, 0.5, 1) for x in voltajes]
    err_resistencias = [err(x, 0.5, 1) for x in resistencias]

    def f(r, rv, e0):
        return rv * e0 / (r + rv)

    fit, cov = curve_fit(f, resistencias, voltajes, sigma=err_voltajes)
    errors = np.sqrt(np.diag(cov))

    print("resistencia del voltimetro %f +- %f MOmhs" % (fit[0], errors[0]))
    print("voltaje de la fuente %f +- %f V" % (fit[1], errors[1]))

    sb.set_theme()
    plt.rcParams["text.usetex"] = False
    plt.figure()
    plt.grid("on")
    plt.xlabel("Voltaje [V]")
    plt.ylabel(r"Resistencia $[M\,\Omega]$")
    y_fit = np.linspace(0, 10, 1000)
    x_fit = [f(x, fit[0], fit[1]) for x in y_fit]
    plt.plot(x_fit, y_fit)
    plt.errorbar(
        voltajes,
        resistencias,
        err_resistencias,
        err_voltajes,
        ".",
        color="orange",
        ecolor="red",
        capsize=2,
    )
    plt.savefig("1.png")


voltimetro(voltajes1, resistencias1)

corriente = [50.7, 25.7, 17.2, 12.9, 10.4, 8.6]
voltaje = [57.4, 29.1, 19.4, 14.6, 11.7, 9.8]


def amperimetro():
    err_corriente = [err(x, 1.5, 1) for x in corriente]
    err_voltaje = [err(x, 0.5, 1) for x in voltaje]

    def f(x, a, b):
        return a * x + b

    fit, cov = curve_fit(f, corriente, voltaje, sigma=err_voltaje)

    errors = np.sqrt(np.diag(cov))

    print("Valor de la resistencia %f +- %f Omhs" % (fit[0], errors[0]))

    x_fit = np.linspace(7, 51, 1000)
    y_fit = np.array([f(x, fit[0], fit[1]) for x in x_fit])

    sb.set_theme()
    plt.figure()
    plt.xlabel("Corriente [mA]")
    plt.ylabel("Voltaje [mV]")
    plt.plot(x_fit, y_fit)
    plt.errorbar(
        corriente,
        voltaje,
        err_voltaje,
        err_corriente,
        ".",
        color="orange",
        ecolor="red",
        capsize=2,
    )
    plt.savefig("2.png")


amperimetro()
