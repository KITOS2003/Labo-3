import numpy as np
import matplotlib.pyplot as plt
from seaborn import set_theme

set_theme()

mediciones = [
    (0.90, 177.51),
    (0.86, 124.19),
    (0.84, 95.75),
    (0.82, 77.98),
    (0.81, 65.66),
    (0.80, 57.10),
    (0.79, 50.21),
    (0.77, 33.99),
    (0.74, 18.66),
    (0.72, 12.89),
    (0.70, 9.78),
    (0.69, 6.44),
    (0.65, 3.33),
    (0.64, 2.22),
    (0.63, 1.56),
    (0.62, 1.11),
    (0.61, 0.89),
    (0.60, 0.67),
]

mediciones = list(zip(*mediciones))
tensiones = mediciones[0]
corrientes = mediciones[1]


plt.figure(1)
plt.plot(tensiones, corrientes, ".")
plt.savefig("figures/arduino.pdf")
