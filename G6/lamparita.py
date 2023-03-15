import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from utils import error_voltimetro

# protek 506


def mkdir(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass


mkdir("figures")
mkdir("figures/lamparita")

mediciones = [
    # escala mV
    (0.0, 4.5),
    (0.045, 18.5),
    (0.095, 35.5),
    # escala v
    (0.202, 55.6),
    (0.435, 70.9),
    (0.917, 94.6),
    (1.383, 115.7),
    (1.86, 134.8),
    (2.34, 152.4),
    (2.842, 168.9),
    (3.32, 183.9),
    (3.808, 198.1),
    (4.78, 224.2),
    (4.29, 211.4),
    (4.93, 228.0),
]

R = 1

mediciones = list(zip(*mediciones))
tensiones = np.array(mediciones[0])
corrientes = np.array(mediciones[1]) / R

tensiones_error = error_voltimetro(tensiones)

sb.set_theme()

plt.figure(1)
plt.xlabel("Tension [V]")
plt.ylabel("Corriente [mA]")
plt.plot(tensiones, corrientes, ".", color="slateblue")
plt.savefig("figures/lamparita.pdf")
plt.clf()


def hipotesis(tensiones, corrientes):
    corrientes /= 1000
    tensiones_error = error_voltimetro(tensiones)
    corrientes_error = error_voltimetro(corrientes)
    tensiones.sort()
    corrientes.sort()
    potencia_fr = (tensiones * corrientes) ** (1 / 4)
    interpolacion = UnivariateSpline(corrientes, potencia_fr, s=1)
    x = np.linspace(corrientes[0], corrientes[-1], 10_000)
    y = interpolacion(x)

    print(f"Coeficientes de la interpolacion {interpolacion.get_coeffs()}")

    plt.figure(100)
    plt.xlabel("Corriente [A]")
    plt.ylabel(r"${(\mathrm{IV})}^\frac{1}{4}$ [$\mathrm{W}^{\frac{1}{4}}$]")
    plot1 = plt.errorbar(
        corrientes, potencia_fr, 0, 0, ".", color="slateblue", label="Mediciones"
    )
    (plot2,) = plt.plot(x, y, color="orange", label="Linea polinomica suave")
    plt.legend(handles=[plot1, plot2])
    plt.savefig("figures/lamparita/hipotesis1.pdf")

    integral = interpolacion.antiderivative(1)

    def ajuste(I, R0, alpha):
        return R0 * I + alpha * integral(I)

    coefs, cov = curve_fit(ajuste, corrientes, tensiones, [30, 0.1], tensiones_error)
    errors = np.sqrt(np.diag(cov))
    print(f"{coefs  = }")
    print(f"{errors = }")
    x_fit = x
    y_fit = ajuste(x_fit, *coefs)

    plt.figure(200)
    plt.xlabel("Corriente [A]")
    plt.ylabel("Tension [V]")
    plot1 = plt.errorbar(
        corrientes,
        tensiones,
        tensiones_error,
        corrientes_error,
        " ",
        color="slateblue",
        label="Mediciones",
    )
    (plot2,) = plt.plot(x_fit, y_fit, color="orange", label="Ajuste")
    plt.legend(handles=[plot1, plot2])
    plt.savefig("figures/lamparita/hipotesis2.pdf")
    plot_residuos(
        corrientes, tensiones, corrientes_error, tensiones_error, ajuste, coefs
    )


def plot_residuos(x, y, x_err, y_err, func, coefs):
    residuos = y - func(x, *coefs)
    plt.figure(131415)
    plt.xlabel("Corriente [A]")
    plt.ylabel("residuos [V]")
    plt.errorbar(x, residuos, y_err, x_err, "ko", color="slateblue")
    plt.savefig("figures/residuos.pdf")


def potencia(tensiones, corrientes):
    tensiones = np.asarray(tensiones, dtype=float)
    potencia = tensiones * np.asarray(corrientes, dtype=float)
    plt.figure(1)
    plt.xlabel("Tension [V]")
    plt.ylabel("Potencia [mW]")
    plt.plot(tensiones, potencia + tensiones, ".")
    plt.savefig("figures/lamparita/potencia.pdf")


potencia(tensiones, corrientes)
hipotesis(tensiones, corrientes)
