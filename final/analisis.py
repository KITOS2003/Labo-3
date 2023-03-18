import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from seaborn import set_theme  # type: ignore
from uncertainties import unumpy as unp

set_theme()

# R4= 9x100 + 10x10 + 7x0.1

# R1=1000, R2=1000 R3=1000
# Equilirio 10x10 + 7x0.1
mediciones1 = [
    (-0.3, -2.2),
    (-0.2, -1.4),
    (-0.1, -0.7),
    (0.0, 0.0),
    (0.1, 0.8),
    (0.2, 1.5),
    (0.3, 2.3),
]
err_mediciones1 = [(0.05, 0.1)] * 7
# R1=2000, R2=1000, R3= 2x1000 + 6x1
# mediciones2 = [
#     (-0.4, -2.8),
#     (-0.3, -2.1),
#     (-0.2, -1.4),
#     (-0.1, -0.7),
#     (0.0, 0.0),
#     (0.1, 0.6),
#     (0.2, 1.3),
#     (0.3, 2),
# ]

# R1=3000, R2=1000, R3= 3x1000 + 10x1 + 1x0.1
mediciones3 = [
    (-0.4, -2.3),
    (-0.3, -1.7),
    (-0.2, -1.1),
    (-0.1, -0.5),
    (0.0, 0.0),
    (0.1, 0.6),
    (0.2, 1.1),
    (0.3, 1.7),
]
err_mediciones3 = [(0.05, 0.1)] * 8
# R1=1x10_000, R2=1000, R3= 1x10_000 + 2x10 + 3x1 + 6x0.1
mediciones4 = [
    (-4.0, -10.0),
    (-3.0, -7.5),
    (-2.0, -5.0),
    (-1.0, -2.5),
    (-0.4, -1),
    (-0.3, -0.8),
    (-0.2, -0.5),
    (-0.1, -0.2),
    (0.0, 0.0),
    (0.1, 0.2),
    (0.2, 0.5),
    (0.3, 0.7),
    (1.0, 2.5),
    (2.0, 5.0),
    (3.0, 7.4),
]
err_mediciones4 = [(0.05, 0.3)] * 15
# R1=1x1_000, R2=3x1_000, R3= 2x100 + 6x10 + 6x10 + 8x1 + 6x1 + 3x0.1
mediciones5 = [
    (-0.4, -2.4),
    (-0.3, -1.9),
    (-0.2, -1.3),
    (-0.1, -0.7),
    (0.0, 0.0),
    (0.1, 0.4),
    (0.2, 0.9),
    (0.3, 1.5),
]
err_mediciones5 = [(0.05, 0.1)] * 8

# R1=1x1_000, R2=2x1_000, R3= 4x100 + 7x10 + 3x10 + 1x1 + 2x0.1
mediciones6 = [
    (-0.3, -2.2),
    (-0.2, -1.5),
    (-0.1, -0.8),
    (0.0, -0.1),
    (0.1, 0.5),
    (0.2, 1.2),
    (0.3, 1.9),
    (0.4, 2.5),
]
err_mediciones6 = [(0.05, 0.1)] * 8

# R1=2x1_000, R2=1_000, R3= 1x100 + 5x1 + 9x0.1
mediciones7 = [
    (-0.3, -1.9),
    (-0.2, -1.2),
    (-0.1, -0.6),
    (0.0, 0.0),
    (0.1, 0.7),
    (0.2, 1.4),
    (0.3, 2.0),
]
err_mediciones7 = [(0.05, 0.1)] * 7

# R1=1_000, R2=1x10_000, R3= 9x10 + 5x1 + 5x1 + 3x0.1
mediciones8 = [
    (-4, -10.0),
    (-3, -7.5),
    (-2, -5.0),
    (-1, -2.5),
    (0.0, 0.0),
    (1, 2.5),
    (2, 5.0),
    (3, 7.5),
    (4, 10.0),
]
err_mediciones8 = [(0.05, 0.2)] * 9


def plot_data(mediciones, err_mediciones, figname="test.png"):
    mediciones = tuple(zip(*mediciones))
    err_mediciones = tuple(zip(*err_mediciones))
    resistencias = mediciones[0]
    tensiones = np.array(mediciones[1]) / 1_000  # Eran mV
    err_resistencias = np.array(err_mediciones[0])
    err_tensiones = np.array(err_mediciones[1]) / 1_000

    def ajuste(x, S, b):
        return (30.8 / 1000) * S * x + b

    coefs, cov = curve_fit(
        ajuste, resistencias, tensiones, sigma=err_tensiones, absolute_sigma=True
    )
    errcoefs = np.sqrt(np.diag(cov))
    x_fit = np.linspace(resistencias[0], resistencias[-1], 1000)
    y_fit = ajuste(x_fit, *coefs)

    # plt.ticklabel_format(axis="both", style="sci")
    plt.figure(100)
    plt.ylabel(r"$\Delta V_{AB}$ [mV]")
    plt.xlabel(r"$\Delta R_4$")
    plt.plot(x_fit, y_fit, color="orange")
    plt.plot(resistencias, tensiones, ".", color="slateblue")
    plt.savefig(figname)
    plt.clf()
    return coefs[0], errcoefs[0]


def plot_sensibilidad(R1R2, pendientes, errores):
    def ajuste(x, a):
        return a * (x / (1 + x) ** 2)

    def ajuste_log(x, a):
        return ajuste(np.log10(x), a)

    def ajuste_mejor(x, a, b):
        alpha = x / (x + 1)
        return a / (1 + b * alpha)

    coefs, cov = curve_fit(
        ajuste,
        R1R2,
        pendientes,
        [1 / (4 * 7.2)],
        sigma=np.array(errores),
        absolute_sigma=True,
    )

    coefs2, cov2 = curve_fit(
        ajuste_mejor,
        R1R2,
        pendientes,
        [1 / (4 * 7.2), 1],
        sigma=np.array(errores),
        absolute_sigma=True,
    )
    max_value = coefs[0] / 4
    x = np.linspace(np.min(R1R2), np.max(R1R2), 100_000)
    print(
        f"Ancho de la campana: {2*(x[np.abs(ajuste_log(x, *coefs) - max_value / 2).argmin()]-1) }"
    )

    errcoefs = np.sqrt(np.diag(cov))

    print(coefs, errcoefs)
    plt.figure(3141516)
    plt.xscale("log")
    plt.xlabel("$R_1/R_2$")
    plt.ylabel("S")
    x_fit = np.linspace(np.min(R1R2), np.max(R1R2), 1000)
    y_fit = ajuste(x_fit, *coefs)
    plt.plot(x_fit, y_fit, color="orange", label="Modelo ideal")
    x_fit = np.linspace(np.min(R1R2), np.max(R1R2), 1000)
    y_fit = ajuste_mejor(x_fit, *coefs2)
    plt.plot(x_fit, y_fit, color="orange", label="Modelo mejorado")
    plt.errorbar(
        R1R2,
        pendientes,
        yerr=np.array(errores),
        fmt=".",
        color="slateblue",
        label="Datos",
    )
    plt.legend()
    plt.savefig("pendientes.pdf")
    plt.clf()


R1 = np.array([1, 3, 10, 1, 1, 2, 1]) * 1000
R2 = np.array([1, 1, 1, 3, 2, 1, 10]) * 1000

R1err = np.array([10] * 7)
R2err = np.array([10] * 7)

uR1 = unp.uarray(R1, R1err)
uR2 = unp.uarray(R2, R2err)
uR1R2 = uR1 / uR2

R1R2 = [1, 3, 10, 1 / 3, 1 / 2, 2, 1 / 10]
pendientes = []
errores = []

# pendientes.append(plot_data(mediciones2, "2.png"))
pendientes.append(plot_data(mediciones1, err_mediciones1, "1.png")[0])
pendientes.append(plot_data(mediciones3, err_mediciones3, "3.png")[0])
pendientes.append(plot_data(mediciones4, err_mediciones4, "4.png")[0])
pendientes.append(plot_data(mediciones5, err_mediciones5, "5.png")[0])
pendientes.append(plot_data(mediciones6, err_mediciones6, "6.png")[0])
pendientes.append(plot_data(mediciones7, err_mediciones7, "7.png")[0])
pendientes.append(plot_data(mediciones8, err_mediciones8, "8.png")[0])

errores.append(plot_data(mediciones1, err_mediciones1, "1.png")[1])
errores.append(plot_data(mediciones3, err_mediciones3, "3.png")[1])
errores.append(plot_data(mediciones4, err_mediciones4, "4.png")[1])
errores.append(plot_data(mediciones5, err_mediciones5, "5.png")[1])
errores.append(plot_data(mediciones6, err_mediciones6, "6.png")[1])
errores.append(plot_data(mediciones7, err_mediciones7, "7.png")[1])
errores.append(plot_data(mediciones8, err_mediciones8, "8.png")[1])

plot_sensibilidad(R1R2, pendientes, errores)
