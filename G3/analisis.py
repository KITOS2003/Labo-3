import numpy as np
import scipy as sc
import scipy.signal as sig
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

from prettytable import PrettyTable

import os

from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq

class HorrorEstadistico(Exception):
    pass

class measurement:

    def __init__(self, *args):
        self.values  = []
        self.errors = []
        if args != ():
            measurement_list = list(args[0])
            measurement_list = list(zip(*measurement_list))
            self.values = measurement_list[0]
            self.errors = measurement_list[1]
    
    def __getitem__(self, i):
        if isinstance(i, slice):
            result = measurement()
            result.values = self.values[i]
            result.errors = self.errors[i]
            return result
        return self.values[i], self.errors[i]
    
    def __add__(self, x):
        result = measurement()
        result.values = (np.array(self.values) + np.array(x.values)).tolist()
        result.errors = np.sqrt(np.array(self.errors)**2 + np.array(x.errors)**2).tolist()
        return result
    
    def __sub__(self, x):
        result = measurement()
        result.values = (np.array(self.values) - np.array(x.values)).tolist()
        result.errors = np.sqrt(np.array(self.errors)**2 + np.array(x.errors)**2).tolist()
        return result
    
    def __mul__(self, x):
        result = measurement()
        result.values = (np.array(self.values) * np.array(x.values)).tolist()
        result.errors = np.sqrt((np.array(x.values)*np.array(self.errors))**2 + (np.array(self.values)*np.array(x.values))**2).tolist()
        return result
    
    def __truediv__(self, x):
        result = measurement()
        result.values = (np.array(self.values) / np.array(x.values)).tolist()
        result.errors = np.sqrt((np.array(self.errors)/np.array(x.values))**2 + (np.array(self.values)*np.array(x.errors)/np.array(x.values)**2)**2).tolist()
        return result
    
    def pow(self, x):
        result = measurement()
        a = np.array(self.values)
        b = np.array(x.values)
        da = np.array(self.errors)
        db = np.array(x.errors)
        result.values = (a**b).tolist()
        result.errors = np.sqrt( (b*a**(b-np.array([1]*len(b)))*da)**2 + (np.log(a)*(a**b)*db)**2 ).tolist()
        return result
    
    def append(self, value, error):
        self.values.append(value)
        self.errors.append(error)
    
    def sort(self):
        dic = dict(zip(self.values, self.errors))
        self.values.sort()
        self.errors = [dic[x] for x in self.values]
    
    def plot(measurement1, measurement2):
        plt.errorbar(measurement1.values, measurement2.values, measurement2.errors, measurement1.errors, ".", color = "orange", ecolor = "orange")


class dataset:
    
    def __init__(self):
        self.frecuencia = measurement()      
        self.fase =       measurement()      
        self.tension =   measurement()      
        self.dirnum = []
        self.r2 = []
        self.chi2nu = []
    
    def append(self, measurements):
        self.frecuencia.append(measurements[0], measurements[1])
        self.fase.append(measurements[2], measurements[3])
        self.tension.append(measurements[4], measurements[5])
    
    def print_table(self):
        table = PrettyTable()
        table.field_names = ["Set de datos", "frecuencia [Hz]", "fase [rad]", "tension [V]", "r2", "chi2nu"]
        for i in range(len(self.frecuencia.values)):
            table.add_row(["%sCH1"%(self.dirnum[i]), "%f+-%f"%self.frecuencia[i], "%f+-%f"%self.fase[i], "%f+-%f"%self.tension[i], self.r2[i], self.chi2nu[i]])
        print(table)
    
    def adjust(self):
        # ordenar los datos
        aux = list(zip(self.fase, self.tension, self.dirnum))
        dic = dict(zip(self.frecuencia, aux))
        self.frecuencia.sort()
        aux = [dic[x] for x in self.frecuencia]
        aux = list(zip(*aux))
        self.fase = measurement(aux[0])
        self.tension = measurement(aux[1])
        self.dirnum = list(aux[2])
        # convertir a rms
        self.tension.values = (np.abs(np.array(self.tension.values))/np.sqrt(2)).tolist()
        self.tension.errors = (np.array(self.tension.errors)/np.sqrt(2)).tolist()
        # arreglar la fase
        self.fase.values = np.unwrap(self.fase.values, period = np.pi)


def mkdir_noexcept(dir_name):
    try:
        os.mkdir(dir_name)
    except:
        pass


def calc_r2(xdata, ydata, f, coefs):
    residuos = np.array(ydata)-np.array([f(x, *coefs) for x in xdata])
    total    = np.array(ydata)-np.array([np.mean(ydata)]*len(ydata)) 
    return 1-np.sum(residuos**2)/np.sum(total**2)


def calc_chi2nu(xdata, ydata, yerror, f, coefs):
    residuos = np.array(ydata)-np.array([f(x, *coefs) for x in xdata])
    chi2 = np.sum((residuos/np.array(yerror))**2)
    return chi2/(len(ydata)-len(coefs)-1)


def measurement_r2(x_measurement, y_measurement, f, coefs):
    xdata = x_measurement.values
    ydata = y_measurement.values
    residuos = np.array(ydata)-np.array([f(x, *coefs) for x in xdata])
    total    = np.array(ydata)-np.array([np.mean(ydata)]*len(ydata)) 
    return 1-np.sum(residuos**2)/np.sum(total**2)


def measurement_chi2nu(x_measurement, y_measurement, f, coefs):
    xdata = x_measurement.values
    ydata = y_measurement.values
    yerror = y_measurement.errors
    residuos = np.array(ydata)-np.array([f(x, *coefs) for x in xdata])
    chi2 = np.sum((residuos/np.array(yerror))**2)
    return chi2/(len(ydata)-len(coefs)-1)


def caracterize_sin(data, osc_error, fguess=0):
    max = np.max(data[1])
    data_errors = np.abs(np.array(data[1], dtype=float)) * (3 / 100) + osc_error
    if fguess == 0:
        fourier = np.abs(fft(data[1]))
        fourier_freq = fftfreq(len(data[0]), np.mean(np.diff(data[0])))
        fguess = np.abs(fourier_freq[fourier.argmax()])

    def f(t, frecuencia, fase, amplitud):
        return amplitud * np.cos(2 * np.pi * frecuencia * t + fase)

    fit, cov = curve_fit(f, data[0], data[1], [fguess, 0, max], sigma=data_errors, absolute_sigma=True,
                         bounds=([4 * fguess / 5, -np.pi, -np.inf], [6 * fguess / 5, np.pi, np.inf]))
    r2 = calc_r2(data[0], data[1], f, fit)
    chi2nu = calc_chi2nu(data[0], data[1], data_errors, f, fit)
    if chi2nu > 5:
        raise HorrorEstadistico
    errors = np.sqrt(np.diag(cov))
    x_fit = np.linspace(data[0][0], data[0][-1], 1000)
    y_fit = [f(t, fit[0], fit[1], fit[2]) for t in x_fit]
    plt.xlabel("Frecuencia estimada: %f Hz"%(fguess))
    plt.ylabel("Amplitud %f V"%(fit[2]))
    plt.plot(x_fit, y_fit)
    plt.errorbar(data[0], data[1], data_errors, 0, ".")
    return fit[0], errors[0], fit[1], errors[1], fit[2], errors[2], fguess, r2, chi2nu


def get_osciloscope_error(config, channel):
    aux = config[config.find(channel):]
    aux = aux[aux.find("SCALE"):]
    aux = aux[:aux.find(";")]
    aux = aux.replace("SCALE ", "")
    base = float(aux[:aux.find("E")])
    exp = float(aux[aux.find("E"):].replace("E", ""))
    scale = base*10**exp
    return 8*scale/255


def analisis_datos_osclioscopio(datadir_path, figdir, freq_corte_campana=3000):
    mkdir_noexcept(figdir)
    mkdir_noexcept(figdir+"/osciloscopio")
    sets_descartados = []
    datadir_ls = os.listdir(datadir_path)
    dataset_ch1 = dataset()
    dataset_ch2 = dataset()
    for i, dir in enumerate(datadir_ls):
        dir_num = dir.replace("ALL", "")
        ch1 = pd.read_csv(datadir_path+"/"+dir+"/F%sCH1.CSV"%(dir_num)).values[17:].transpose()[3:-1]
        ch2 = pd.read_csv(datadir_path+"/"+dir+"/F%sCH2.CSV"%(dir_num)).values[17:].transpose()[3:-1]
        with open(datadir_path+"/"+dir+"/F%sTEK.SET"%(dir_num), "r") as file:
            config = file.read()
        ch1_error=get_osciloscope_error(config, "CH1")
        ch2_error=get_osciloscope_error(config, "CH2")
        plt.figure(1)
        try:
            aux2 = caracterize_sin(ch2, ch2_error)
            aux1 = caracterize_sin(ch1, ch1_error, fguess = aux2[6])
        except HorrorEstadistico:
            plt.clf()
            sets_descartados.append(dir_num)
            continue
        dataset_ch2.append(aux2)
        dataset_ch1.append(aux1)
        dataset_ch2.r2.append(aux2[7])
        dataset_ch1.r2.append(aux1[7])
        dataset_ch2.chi2nu.append(aux2[8])
        dataset_ch1.chi2nu.append(aux1[8])
        dataset_ch1.dirnum.append(dir_num)
        dataset_ch2.dirnum.append(dir_num)
        plt.savefig(figdir+"/osciloscopio/%s.png"%(dir_num))
        plt.clf()
    dataset_ch1.fase = dataset_ch1.fase - dataset_ch2.fase
    dataset_ch1.adjust()
    dataset_ch2.adjust()
    if sets_descartados != []:
        print("descartados los sets de datos {}".format(sets_descartados))
    return dataset_ch1, dataset_ch2


def plot_fit(measurement1, measurement2, x_fit, y_fit, out_name):
    plt.grid("on")
    plt.plot(x_fit, y_fit, color = "blue")
    measurement.plot(measurement1, measurement2)
    plt.savefig(out_name)
    plt.clf()


def calc_potencia(tension, resistencia_carga):   
    potencia = np.array(tension.values)**2/resistencia_carga
    potencia_error = np.sqrt(((2*np.array(tension.values))/resistencia_carga * np.array(tension.errors))**2 + ((np.array(tension.values)/resistencia_carga)**2 * 0.1)**2 ) 
    return measurement(zip(potencia, potencia_error))


def rlc_serie_analisis(ch1, ch2, figdir="figures", freq_corte_campana=5000, resistencia_carga = 0):
    # hallar la frecuencia a la cual cortar el diagrama campana
    for i, x in enumerate(ch1.frecuencia.values):
        if x > freq_corte_campana:
            indice_corte = i-1
            break
    # ajuste de la tension
    tension_ajuste_angular = lambda w, w0, a, Q: a/np.sqrt(1/Q**2+(w**2-w0**2)**2/((w0**2)*(w**2)))
    tension_ajuste = lambda freq, freq0, a, Q: tension_ajuste_angular(2*np.pi*freq, 2*np.pi*freq0, a, Q)
    freq_guess_index = np.argmax(ch1.tension.values)
    freq_guess = ch1.frecuencia.values[freq_guess_index]
    freq_min = ch1.frecuencia.values[freq_guess_index-1]
    freq_max = ch1.frecuencia.values[freq_guess_index+1]
    tension_coefs, tension_cov = curve_fit(tension_ajuste, ch1.frecuencia.values[:indice_corte], ch1.tension.values[:indice_corte], [freq_guess, 0.1, 10], sigma = ch1.tension.errors[:indice_corte], absolute_sigma=True, bounds = ([freq_min, 0, 0],[freq_max, np.inf, np.inf]))
    freq_resonancia, coef_a, factor_calidad = tension_coefs
    freq_resonancia_error, coef_a_error, factor_calidad_error = np.sqrt(np.diag(tension_cov))
    # ajuste de la fase
    fase_ajuste = lambda f, b: np.arctan(b*(freq_resonancia/f-f/freq_resonancia))
    fase_coefs, fase_cov = curve_fit(fase_ajuste, ch1.frecuencia.values[:indice_corte], ch1.fase.values[:indice_corte], [20], sigma = ch1.fase.errors[:indice_corte], absolute_sigma=True, bounds = ([0],[np.inf]))
    coef_b = fase_coefs[0]
    coef_b_error = np.sqrt(np.diag(fase_cov))[0]
    # Parametros estadisticos
    tension_r2     = measurement_r2(ch1.frecuencia, ch1.tension, tension_ajuste, tension_coefs)
    tension_chi2nu = measurement_chi2nu(ch1.frecuencia, ch1.tension, tension_ajuste, tension_coefs)
    # Reportar resultados
    print("Valor de  la frecuencia de resonancia %f+-%f Hz"%(freq_resonancia, freq_resonancia_error))
    print("Valor del factor de calidad %f+-%f"%(factor_calidad, factor_calidad_error))
    print("Valor del coeficiente A %f+-%f V"%(coef_a, coef_a_error))
    print("Valor del coeficiente B %f+-%f V"%(coef_b, coef_b_error))
    print("Valor del R2 del ajuste de la tension %f"%(tension_r2))
    print("Valor del chi2nu del ajuste de la tension %f"%(tension_chi2nu))
    print("--------------------------------------------------------------------------------------")
    freq_freq0 = measurement()
    freq_freq0.values = ch1.frecuencia.values / freq_resonancia
    freq_freq0.errors = ch1.frecuencia.errors / freq_resonancia
    f_aux1 = lambda freq_freq0: tension_ajuste(freq_freq0 * freq_resonancia, *tension_coefs)
    f_aux2 = lambda freq_freq0: fase_ajuste(freq_freq0 * freq_resonancia, *fase_coefs)
    # Campana de la tension
    x_fit = np.linspace(freq_freq0.values[0], freq_freq0.values[indice_corte], 1000)
    y_fit = [f_aux1(f) for f in x_fit]
    plt.xlabel(r"$\omega/\omega_0$")
    plt.ylabel("Tension RMS [V]")
    plot_fit(freq_freq0[:indice_corte], ch1.tension[:indice_corte], x_fit, y_fit, figdir+"/tension.pdf")
    # Diagrama de bode de la tension
    x_fit = np.logspace(np.log10(freq_freq0.values[0]), np.log10(freq_freq0.values[-1]), 1000)
    y_fit = [f_aux1(f) for f in x_fit]
    plt.xscale("log")
    plt.xlabel(r"$\omega/\omega_0$")
    plt.ylabel("Tension RMS [V]")  
    plot_fit(freq_freq0, ch1.tension, x_fit, y_fit, figdir+"/tension_bode.pdf")
    # Diagrama de bode de la potencia
    potencia = calc_potencia(ch1.tension, resistencia_carga)
    potencia.values = (1000*np.array(potencia.values)).tolist()
    potencia.errors = (1000*np.array(potencia.errors)).tolist()
    y_fit = 1000*np.array(y_fit)**2/resistencia_carga
    plt.xscale("log")
    plt.xlabel(r"$\omega/\omega_0$")
    plt.ylabel("Potencia disipada [mW]")
    plot_fit(freq_freq0, potencia, x_fit, y_fit, figdir+"/potencia.pdf")   
    # Campana de la fase
    x_fit = np.linspace(freq_freq0.values[0], freq_freq0.values[indice_corte], 1000)
    y_fit = [f_aux2(f) for f in x_fit]
    plt.xlabel(r"$\omega/\omega_0$")
    plt.ylabel("Diferencia de fase [rad]")
    plot_fit(freq_freq0[:indice_corte], ch1.fase[:indice_corte], x_fit, y_fit, figdir+"/fase.pdf")
    # Diagrama de bode de la fase
    x_fit = np.logspace(np.log10(freq_freq0.values[0]), np.log10(freq_freq0.values[-1]), 1000)
    y_fit = [f_aux2(f) for f in x_fit]
    plt.xscale("log")
    plt.xlabel(r"$\omega/\omega_0$")
    plt.ylabel("Diferencia de fase [rad]")
    plot_fit(freq_freq0, ch1.fase, x_fit, y_fit, figdir+"/fase_bode.pdf")
    # Atenuacion:
    tension_0 = np.max(ch2.tension.values)
    atenuacion = measurement()
    atenuacion.values = (20*np.log10(tension_0 * np.array(ch1.tension.values)**(-1))).tolist()
    atenuacion.errors = ((20/np.log(10)) * (np.array(ch1.tension.errors)/np.array(ch1.tension.values))).tolist()
    x_fit = np.logspace(np.log10(freq_freq0.values[0]), np.log10(freq_freq0.values[-1]), 1000)
    y_fit = [f_aux1(f) for f in x_fit]
    y_fit = 20*np.log10(tension_0 * np.array(y_fit)**(-1))
    plt.xscale("log")
    plt.xlabel(r"$\omega/\omega_0$")
    plt.ylabel("Atenuacion [dB]")
    plot_fit(freq_freq0, atenuacion, x_fit, y_fit, figdir+"/atenuacion.pdf")


def rlc_paralelo_analisis(ch1, ch2, figdir="figures", freq_corte_campana=5000, resistencia_carga=0):
    for i, x in enumerate(ch1.frecuencia.values):
        if x > freq_corte_campana:
            indice_corte = i - 1
            break
    # ajuste de la tension
    freq_guess = ch1.frecuencia.values[np.argmin(ch1.tension.values)]
    freq_min = freq_guess - 100
    freq_max = freq_guess + 100

    def func_aux(w, w0, Q):
        return (1 / Q**2) + (w / w0 - w0 / w)**2

    def tension_ajuste_angular(w, w0, Q, a, b):
        return a / np.sqrt((b + (w / w0)**2 / func_aux(w, w0, Q))**2 + (
                           1 / Q**2) * ((Q**2 * (w / w0 - w0 / w) + w0 / w) / func_aux(w, w0, Q))**2)

    def tension_ajuste(f, f0, Q, a, b):
        return tension_ajuste_angular(2 * np.pi * f, 2 * np.pi * f0, Q, a, b)

    tension_coefs, tension_cov = curve_fit(tension_ajuste,
                                           ch1.frecuencia.values[:indice_corte], ch1.tension.values[:indice_corte],
                                           [freq_guess, 20, 1, 1], sigma=ch1.tension.errors[:indice_corte],
                                           bounds=[(freq_min, 0, 0, 0), (freq_max, np.inf, np.inf, np.inf)])
    freq_resonancia, factor_calidad, coef_a, coef_b = tension_coefs
    freq_resonancia_error, factor_calidad_error, coef_a_error, coef_b_error = np.sqrt(np.diag(tension_cov))
    tension_r2     = measurement_r2(ch1.frecuencia, ch1.tension, tension_ajuste, tension_coefs)
    tension_chi2nu = measurement_chi2nu(ch1.frecuencia, ch1.tension, tension_ajuste, tension_coefs)

    # Ajuste de la fase
    def fase_ajuste(f, f0, Q, B):
        return np.arctan((Q * (f / f0 - f0 / f) + (1 / Q) * f0 / f) / (B * (1 / Q**2 + (f / f0 - f0 / f)**2) + (f0 / f)**2))

    fase_coefs = freq_resonancia, factor_calidad, coef_b
    # Reportar valores
    print("Valor de  la frecuencia de resonancia %f+-%f Hz" % (freq_resonancia, freq_resonancia_error))
    print("Valor del factor de calidad %f+-%f" % (factor_calidad, factor_calidad_error))
    # print("Valor del coeficiente A %f+-%f V"%(coef_a, coef_a_error))
    print("Valor del coeficiente B %f+-%f V" % (coef_b, coef_b_error))
    print("Valor del R2 del ajuste de la tension %f" % (tension_r2))
    print("Valor del chi2nu del ajuste de la tension %f" % (tension_chi2nu))
    print("--------------------------------------------------------------------------------------")
    freq_freq0 = measurement()
    freq_freq0.values = ch1.frecuencia.values / freq_resonancia
    freq_freq0.errors = ch1.frecuencia.errors / freq_resonancia

    def f_aux1(freq_freq0):
        tension_ajuste(freq_freq0 * freq_resonancia, *tension_coefs)

    def f_aux2(freq_freq0):
        fase_ajuste(freq_freq0 * freq_resonancia, *fase_coefs)
    # Campana de la tension
    x_fit = np.linspace(freq_freq0.values[0], freq_freq0.values[indice_corte], 1000)
    y_fit = [f_aux1(x) for x in x_fit]
    plt.xlabel(r"$\omega/\omega_0$")
    plt.ylabel("Tension RMS [V]")
    plot_fit(freq_freq0[:indice_corte], ch1.tension[:indice_corte], x_fit, y_fit, figdir+"/tension.pdf")
    # Diagrama de bode de la tension
    x_fit = np.logspace(np.log10(freq_freq0.values[0]), np.log10(freq_freq0.values[-1]), 1000)
    y_fit = [f_aux1(x) for x in x_fit]
    plt.xscale("log")
    plt.xlabel(r"$\omega/\omega_0$")
    plt.ylabel("Tension RMS [V]")
    plot_fit(freq_freq0, ch1.tension, x_fit, y_fit, figdir+"/tension_bode.pdf")
    # Diagrama de bode de la potencia
    potencia = calc_potencia(ch1.tension, resistencia_carga)
    potencia.values = (1000*np.array(potencia.values)).tolist()
    potencia.errors = (1000*np.array(potencia.errors)).tolist()
    y_fit = 1000*np.array(y_fit)**2/resistencia_carga
    plt.xscale("log")
    plt.xlabel(r"$\omega/\omega_0$")
    plt.ylabel("Potencia disipada [mW]")
    plot_fit(freq_freq0, potencia, x_fit, y_fit, figdir+"/potencia.pdf")   
    # Campana de la fase
    x_fit = np.linspace(freq_freq0.values[0], freq_freq0.values[indice_corte], 1000)
    y_fit = [f_aux2(x) for x in x_fit]
    plt.xlabel(r"$\omega/\omega_0$")
    plt.ylabel("Diferencia de fase [rad]")
    plot_fit(freq_freq0[:indice_corte], ch1.fase[:indice_corte], x_fit, y_fit, figdir+"/fase.pdf")
    # Diagrama de bode de la fase
    x_fit = []
    y_fit = []
    plt.xscale("log")
    plt.xlabel(r"$\omega/\omega_0$")
    plt.ylabel("Diferencia de fase [rad]")
    plot_fit(freq_freq0, ch1.fase, x_fit, y_fit, figdir+"/fase_bode.pdf")
    # Atenuacion
    tension_0 = np.max(ch2.tension.values)
    atenuacion = measurement()
    atenuacion.values = (20*np.log10(tension_0 * np.array(ch1.tension.values)**(-1))).tolist()
    atenuacion.errors = ((20/np.log(10)) * (np.array(ch1.tension.errors)/np.array(ch1.tension.values))).tolist()
    x_fit = np.logspace(np.log10(freq_freq0.values[0]), np.log10(freq_freq0.values[-1]), 1000)
    y_fit = [f_aux1(f) for f in x_fit]
    y_fit = 20*np.log10(tension_0 * np.array(y_fit)**(-1))
    plt.xscale("log")
    plt.xlabel(r"$\omega/\omega_0$")
    plt.ylabel("Atenuacion [dB]")
    plot_fit(freq_freq0, atenuacion, x_fit, y_fit, figdir+"/atenuacion.pdf")


sb.set_theme()


print("----------------------------------- Set de datos 1 -----------------------------------")
dataset1_ch1, dataset1_ch2 = analisis_datos_osclioscopio("data1", "figures1")
print("Mediciones Canal 1:")
dataset1_ch1.print_table()
print("Mediciones Canal 2:")
dataset1_ch2.print_table()

print("----------------------------------- Set de datos 2 -----------------------------------")
dataset2_ch1, dataset2_ch2 = analisis_datos_osclioscopio("data2", "figures2")
print("Mediciones Canal 1:")
dataset2_ch1.print_table()
print("Mediciones Canal 2:")
dataset2_ch2.print_table()

print("----------------------------------- Set de datos 3 -----------------------------------")
dataset3_ch1, dataset3_ch2 = analisis_datos_osclioscopio("data3", "figures3")
print("Mediciones Canal 1:")
dataset3_ch1.print_table()
print("Mediciones Canal 2:")
dataset3_ch2.print_table()

print("----------------------------------- Set de datos 4 -----------------------------------")
dataset4_ch1, dataset4_ch2 = analisis_datos_osclioscopio("data4", "figures4")
print("Mediciones Canal 1:")
dataset4_ch1.print_table()
print("Mediciones Canal 2:")
dataset4_ch2.print_table()

print("----------------------------------- Set de datos 5 -----------------------------------")
dataset5_ch1, dataset5_ch2 = analisis_datos_osclioscopio("data5", "figures5")
print("Mediciones Canal 1:")
dataset5_ch1.print_table()
print("Mediciones Canal 2:")
dataset5_ch2.print_table()

rlc_serie_analisis(dataset1_ch1, dataset1_ch2, figdir = "figures1", resistencia_carga = 101)
rlc_serie_analisis(dataset2_ch1, dataset2_ch2, figdir = "figures2", resistencia_carga = 1101)
rlc_serie_analisis(dataset3_ch1, dataset3_ch2, figdir = "figures3", resistencia_carga = 10101)

rlc_paralelo_analisis(dataset4_ch1, dataset4_ch2, figdir = "figures4", freq_corte_campana = 2400, resistencia_carga = 1000)
rlc_paralelo_analisis(dataset5_ch1, dataset5_ch2, figdir = "figures5", freq_corte_campana = 2400, resistencia_carga = 1000)

