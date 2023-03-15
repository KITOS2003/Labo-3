import numpy as np

# Error del Orotek 506: 0.005 + 2d
def error_voltimetro(medicion):
    return 0.005 * np.abs(medicion) + 0.002
