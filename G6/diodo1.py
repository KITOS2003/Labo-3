import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.optimize import curve_fit

from utils import error_voltimetro

sb.set_theme()

try:
    os.mkdir("figures")
except:
    pass

R1 = 100
R2 = 1_000

mediciones1 = [
    (0.0, 0.018),
    (0.002, 0.417),
    (0.015, 0.504),
    (0.056, 0.565),
    (0.118, 0.602),
    (0.195, 0.626),
    (0.257, 0.643),
    (0.365, 0.656),
    (0.545, 0.675),
    (0.729, 0.689),
    (0.824, 0.694),
    (0.918, 0.700),
    (1.108, 0.709),
    (1.301, 0.715),
    (1.501, 0.723),
    (1.696, 0.728),
    (1.792, 0.730),
    (2.294, 0.741),
]

mediciones2 = [
    (-0.0050, -5.050),
    (-0.0045, -4.540),
    (-0.0040, -4.03),
    (-0.0032, -3.540),
    (-0.0027, -3.038),
    (-0.0023, -2.526),
    (-0.0018, -2.02),
    (-0.0014, -1.519),
    (-0.0009, -1.02),
    (-0.0004, -0.519),
]

fig_name = "diodo1.pdf"

mediciones1 = list(zip(*mediciones1))
mediciones2 = list(zip(*mediciones2))
corriente = (np.array(mediciones2[0]) / R2).tolist() + (
    np.array(mediciones1[0]) / R1
).tolist()
tensiones_diodo = list(mediciones2[1]) + list(mediciones1[1])

error_diodo = error_voltimetro(tensiones_diodo)
error_corriente = error_voltimetro(np.array(mediciones1[0])) / R1

sb.set_theme()


def ajuste(v_diodo, I0, VT):
    return I0 * (np.exp(v_diodo / VT) - 1)


coefs, cov = curve_fit(
    ajuste, tensiones_diodo, corriente, [0.01, 0.026]  # , error_corriente
)
errors = np.sqrt(np.diag(cov))
print("Valor de la corriente de filtrado {}+-{}".format(coefs[0], errors[0]))
print("Valor de VT {}+-{}".format(coefs[1] / 2, errors[1] / 2))

x_fit = np.linspace(tensiones_diodo[0], tensiones_diodo[-1], 1000)
y_fit = ajuste(x_fit, *coefs)

plt.figure(1)
# plt.yscale("log")
plt.xlabel("Tension [V]")
plt.ylabel("Corriente [A]")
plt.plot(x_fit, y_fit, color="orange")
plt.plot(tensiones_diodo, corriente, ".", color="slateblue")
# plt.errorbar(
#     tensiones_diodo, corriente, error_corriente, error_diodo, ".", color="slateblue"
# )
plt.savefig("figures/{}".format(fig_name))
