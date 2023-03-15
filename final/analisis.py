import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from seaborn import set_theme

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


def plot_data(mediciones, figname="test.pdf"):
    mediciones = tuple(zip(*mediciones))
    resistencias = mediciones[0]
    tensiones = mediciones[1]

    def ajuste(x, a, b):
        return a * x + b

    coefs, cov = curve_fit(ajuste, resistencias, tensiones)
    x_fit = np.linspace(resistencias[0], resistencias[-1], 1000)
    y_fit = ajuste(x_fit, *coefs)

    plt.figure(100)
    plt.plot(x_fit, y_fit, color="orange")
    plt.plot(resistencias, tensiones, ".", color="slateblue")
    plt.savefig(figname)
    plt.clf()
    return coefs[0]


def plot_sensibilidad(R1R2, pendientes):
    def ajuste(x, a):
        return a * (x / (1 + x) ** 2)

    coefs, cov = curve_fit(ajuste, R1R2, pendientes, [1 / (4 * 7.2)])
    x_fit = np.linspace(np.min(R1R2), np.max(R1R2), 1000)
    y_fit = ajuste(x_fit, *coefs)

    plt.figure(3141516)
    plt.xscale("log")
    plt.xlabel("$R_1/R_2$")
    plt.ylabel("pendientes")
    plt.plot(x_fit, y_fit, color="orange")
    plt.plot(R1R2, pendientes, ".", color="slateblue")
    plt.savefig("pendientes.pdf")


R1R2 = [1, 3, 10, 1 / 3, 1 / 2, 2, 1 / 10]
pendientes = []

# pendientes.append(plot_data(mediciones2, "2.pdf"))
pendientes.append(plot_data(mediciones1, "1.pdf"))
pendientes.append(plot_data(mediciones3, "3.pdf"))
pendientes.append(plot_data(mediciones4, "4.pdf"))
pendientes.append(plot_data(mediciones5, "5.pdf"))
pendientes.append(plot_data(mediciones6, "6.pdf"))
pendientes.append(plot_data(mediciones7, "7.pdf"))
pendientes.append(plot_data(mediciones8, "8.pdf"))

plot_sensibilidad(R1R2, pendientes)
