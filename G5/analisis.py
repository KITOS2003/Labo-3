import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from seaborn import set_theme


R1 = [100.1, 100.1, 100.1]
R2 = [100.3, 5100.0, 2500.0]
R3 = [50.9, 1.9, 3.1]
R4_A = [52.5, 85.7, 72.5]

set_theme()


def mkdir_noexcept(dir):
    try:
        os.mkdir(dir)
    except:
        pass


def analisis(
    resistencia, tension, resistencia_error, tension_error, index=0, fig_dir="figures/1"
):
    def ajuste_lineal(x, a, b):
        return a * x + b

    mkdir_noexcept(fig_dir)
    resistencia = np.array(resistencia) / R4_A[index]
    tension = np.array(tension) / 5.11
    tension /= 1000

    coef_list = []
    error_list = []
    print("-----------------------------")
    for i in range(3, len(resistencia) - 1):
        x = resistencia[:i]
        y = tension[:i]
        coefs, cov = curve_fit(ajuste_lineal, x, y, [1, 0])
        error = np.sqrt(np.diag(cov))
        print(
            "Valor de la pendiente para {} puntos: {} +- {}".format(
                i, coefs[0], error[0]
            )
        )
        x_fit = np.linspace(x[0], x[-1], 1000)
        y_fit = ajuste_lineal(x_fit, *coefs)

        plt.figure(1)
        plt.grid("on")
        plt.xlabel(r"$\Delta R/R_{eq}$")
        plt.ylabel(r"$\Delta V/V_{0}$")
        plt.plot(x_fit, y_fit, color="orange")
        plt.errorbar(x, y, 0, 0, ".", color="slateblue")
        plt.savefig(fig_dir + "/{}.pdf".format(i))
        plt.clf()

        residuos = y - ajuste_lineal(x, *coefs)
        plt.figure(2)
        plt.xlabel(r"$\Delta R/R_{eq}$")
        plt.ylabel(r"Residuos")
        plt.plot(x, residuos, ".", color="slateblue")
        plt.savefig(fig_dir + "/{}_res.pdf".format(i))
        plt.clf()
        coef_list.append(coefs)
        error_list.append(error)

    pendientes = list(zip(*coef_list))[0]
    pendientes_err = list(zip(*error_list))[0]
    plt.figure(1)
    plt.xlabel("Numero de puntos")
    plt.ylabel("Pendiente")
    indices = np.array(tuple(zip(*enumerate(pendientes)))[0]) + 3
    plt.errorbar(indices, pendientes, pendientes_err, 0, ".", color="slateblue")
    plt.savefig(fig_dir + "pendientes.pdf")


mkdir_noexcept("figures")
resistencia_1 = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
]
tension_1 = [
    0.5,
    2.5,
    4.9,
    7.3,
    9.7,
    12.1,
    14.5,
    16.9,
    19.3,
    21.6,
    24.3,
    26.6,
    28.9,
    31.3,
    33.7,
    36.0,
]
analisis(
    resistencia_1,
    tension_1,
    [0] * len(resistencia_1),
    [0] * len(tension_1),
    fig_dir="figures/1",
)

resistencia_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
tension_2 = [
    0.1,
    1.2,
    2.3,
    3.4,
    4.5,
    5.5,
    6.5,
    7.5,
    8.4,
    9.3,
    10.3,
    11.2,
    12.0,
    12.9,
    13.8,
    14.6,
]
analisis(
    resistencia_2,
    tension_2,
    [0] * len(resistencia_2),
    [0] * len(tension_2),
    index=1,
    fig_dir="figures/2",
)

resistencia_3 = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
tension_3 = [-13.2, -10.3, -7.4, -4.6, -2.0, 0.4, 3.0, 5.5, 7.9, 10.3, 12.7]
analisis(
    resistencia_3,
    tension_3,
    [0] * len(resistencia_3),
    [0] * len(tension_3),
    index=2,
    fig_dir="figures/3",
)
